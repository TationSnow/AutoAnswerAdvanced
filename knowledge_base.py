"""基于 SQLite 和向量相似度的本地题库。

题库以清洗后的题目作为稳定身份，以答案文字作为跨乱序页面的权威值；
选项编号仅用于展示和当前页面临时绑定。
"""
import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from answer_strategy import AnswerParser
from models import (
    AnswerCandidate,
    OptionItem,
    QuestionSnapshot,
    QuestionType,
    normalize_for_match,
    text_similarity,
)


logger = logging.getLogger(__name__)
SCHEMA_VERSION = 2


class KnowledgeBase:
    """SQLite 题库仓储。"""

    def __init__(
        self,
        db_path: str,
        embedding_client: Any = None,
        similarity_threshold: float = 0.85,
        question_text_threshold: float = 0.65,
    ) -> None:
        """初始化数据库并创建新结构。"""
        self.db_path = db_path
        self.embedding_client = embedding_client
        self.similarity_threshold = similarity_threshold
        self.question_text_threshold = question_text_threshold
        self.answer_parser = AnswerParser()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """创建带超时和行工厂的数据库连接。"""
        connection = sqlite3.connect(self.db_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _init_db(self) -> None:
        """创建新题库结构，并拒绝误用旧测试结构。"""
        connection = self._connect()
        try:
            table = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='questions'"
            ).fetchone()
            if table is not None:
                columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(questions)")
                }
                if "external_id" not in columns or "answer_texts_json" not in columns:
                    raise RuntimeError(
                        "检测到旧版测试题库，请按已确认方案删除 question_bank.db 后重启"
                    )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    schema_version INTEGER NOT NULL DEFAULT 2,
                    external_id TEXT,
                    question_hash TEXT NOT NULL UNIQUE,
                    question_text TEXT NOT NULL,
                    question_type TEXT NOT NULL,
                    options_json TEXT NOT NULL,
                    answer TEXT,
                    answer_labels_json TEXT,
                    answer_texts_json TEXT,
                    detail TEXT,
                    source TEXT,
                    verified INTEGER NOT NULL DEFAULT 0,
                    embedding BLOB,
                    embedding_dim INTEGER,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_qhash ON questions(question_hash)"
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_external_type
                ON questions(external_id, question_type)
                WHERE external_id IS NOT NULL
                """
            )
            connection.execute(
                """
                INSERT INTO schema_meta(key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )
            connection.commit()
        finally:
            connection.close()

    def schema_version(self) -> int:
        """返回当前题库版本。"""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
            return int(row["value"]) if row else 0
        finally:
            connection.close()

    def question_hash(self, question: QuestionSnapshot) -> str:
        """生成不依赖选项顺序和页面装饰的稳定哈希。"""
        if question.external_id:
            identity = "%s|id:%s" % (
                question.question_type.value,
                question.external_id,
            )
        else:
            identity = "%s|question:%s" % (
                question.question_type.value,
                normalize_for_match(question.question),
            )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]

    def add(self, question: QuestionSnapshot, answer: AnswerCandidate) -> bool:
        """新增题目；已有验证答案不会被未验证结果覆盖。"""
        if (
            not question.is_valid
            or not question.question
            or not answer.is_valid
            or not answer.labels
            and not answer.texts
        ):
            return False

        question_hash = self.question_hash(question)
        connection = self._connect()
        try:
            existing = connection.execute(
                "SELECT * FROM questions WHERE question_hash = ?",
                (question_hash,),
            ).fetchone()
            if existing is not None:
                if not self._same_question(existing, question):
                    return False
                if answer.verified:
                    self._update_row(connection, existing["id"], question, answer)
                    connection.commit()
                return False

            embedding = self._embed_question(question)
            now = datetime.now().isoformat(timespec="seconds")
            embedding_blob = embedding.tobytes() if embedding is not None else None
            embedding_dim = int(embedding.shape[0]) if embedding is not None else None
            connection.execute(
                """
                INSERT INTO questions
                (schema_version, external_id, question_hash, question_text,
                 question_type, options_json, answer, answer_labels_json,
                 answer_texts_json, detail, source, verified, embedding,
                 embedding_dim, hit_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    SCHEMA_VERSION,
                    question.external_id,
                    question_hash,
                    question.question,
                    question.question_type.value,
                    self._serialize_options(question.options),
                    answer.display_answer,
                    json.dumps(answer.labels, ensure_ascii=False),
                    json.dumps(answer.texts, ensure_ascii=False),
                    answer.detail,
                    answer.source,
                    1 if answer.verified else 0,
                    embedding_blob,
                    embedding_dim,
                    now,
                    now,
                ),
            )
            connection.commit()
            return True
        finally:
            connection.close()

    def search(self, question: QuestionSnapshot) -> Optional[AnswerCandidate]:
        """依次执行题号、稳定哈希和向量语义检索。"""
        if not question.is_valid or not question.question:
            return None

        connection = self._connect()
        try:
            if question.external_id:
                rows = connection.execute(
                    """
                    SELECT * FROM questions
                    WHERE external_id = ? AND question_type = ?
                    """,
                    (question.external_id, question.question_type.value),
                ).fetchall()
                for row in rows:
                    if self._question_guard(row, question):
                        hit = self._candidate_from_row(
                            row, question, "题库(题号·%s)" % self._verify_label(row)
                        )
                        if hit is not None:
                            self._record_hit(connection, row["id"])
                            return hit

            question_hash = self.question_hash(question)
            row = connection.execute(
                "SELECT * FROM questions WHERE question_hash = ?",
                (question_hash,),
            ).fetchone()
            if row is not None and self._question_guard(row, question):
                hit = self._candidate_from_row(
                    row, question, "题库(精确·%s)" % self._verify_label(row)
                )
                if hit is not None:
                    self._record_hit(connection, row["id"])
                    return hit

            return self._semantic_search(connection, question)
        finally:
            connection.close()

    def _semantic_search(
        self,
        connection: sqlite3.Connection,
        question: QuestionSnapshot,
    ) -> Optional[AnswerCandidate]:
        """执行向量语义检索，并再次校验答案能否绑定当前选项。"""
        if self.embedding_client is None:
            return None
        query_embedding = self.embedding_client.embed_one(question.question)
        if query_embedding is None:
            return None
        query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(-1)

        best_row = None
        best_similarity = 0.0
        rows = connection.execute(
            """
            SELECT * FROM questions
            WHERE embedding IS NOT NULL AND question_type = ?
            """,
            (question.question_type.value,),
        ).fetchall()
        for row in rows:
            if row["embedding_dim"] != int(query_embedding.shape[0]):
                continue
            stored = np.frombuffer(row["embedding"], dtype=np.float32).reshape(-1)
            similarity = float(np.dot(query_embedding, stored))
            if similarity > best_similarity:
                best_similarity = similarity
                best_row = row

        if best_row is None or best_similarity < self.similarity_threshold:
            return None
        candidate = self._candidate_from_row(
            best_row,
            question,
            "题库(相似%.2f·%s)" % (best_similarity, self._verify_label(best_row)),
            best_similarity,
        )
        if candidate is None:
            return None
        self._record_hit(connection, best_row["id"])
        return candidate

    def _candidate_from_row(
        self,
        row: sqlite3.Row,
        question: QuestionSnapshot,
        source: str,
        similarity: float = 1.0,
    ) -> Optional[AnswerCandidate]:
        """将保存的文字答案绑定到当前页面的选项编号。"""
        texts = self._load_json_list(row["answer_texts_json"])
        labels = self._load_json_list(row["answer_labels_json"])
        if texts:
            candidate = self.answer_parser.bind_texts(
                question,
                texts,
                detail=row["detail"] or "",
                source=source,
                verified=bool(row["verified"]),
                match_id=row["id"],
                similarity=similarity,
            )
        else:
            candidate = self.answer_parser.from_values(
                question,
                labels,
                [],
                detail=row["detail"] or "",
                source=source,
                verified=bool(row["verified"]),
                match_id=row["id"],
                similarity=similarity,
            )
        return candidate if candidate.is_valid else None

    def update_feedback(
        self,
        match_id: Optional[int],
        question: QuestionSnapshot,
        answer: AnswerCandidate,
    ) -> bool:
        """使用屏幕捕获到的真实答案覆盖题库记录。"""
        if not match_id or not answer.is_valid:
            return False
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT id FROM questions WHERE id = ?",
                (match_id,),
            ).fetchone()
            if row is None:
                return False
            self._update_row(connection, match_id, question, answer)
            connection.commit()
            return True
        finally:
            connection.close()

    def update_by_id(
        self,
        match_id: int,
        answer: AnswerCandidate,
        question: QuestionSnapshot,
    ) -> bool:
        """兼容旧调用命名。"""
        return self.update_feedback(match_id, question, answer)

    def _update_row(
        self,
        connection: sqlite3.Connection,
        row_id: int,
        question: QuestionSnapshot,
        answer: AnswerCandidate,
    ) -> None:
        """更新答案快照和当前选项信息。"""
        now = datetime.now().isoformat(timespec="seconds")
        connection.execute(
            """
            UPDATE questions
            SET question_text = ?,
                question_type = ?,
                options_json = ?,
                answer = ?,
                answer_labels_json = ?,
                answer_texts_json = ?,
                detail = ?,
                source = ?,
                verified = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                question.question,
                question.question_type.value,
                self._serialize_options(question.options),
                answer.display_answer,
                json.dumps(answer.labels, ensure_ascii=False),
                json.dumps(answer.texts, ensure_ascii=False),
                answer.detail,
                answer.source,
                1 if answer.verified else 0,
                now,
                row_id,
            ),
        )

    def _embed_question(self, question: QuestionSnapshot) -> Optional[np.ndarray]:
        """生成归一化题目向量。"""
        if self.embedding_client is None:
            return None
        embedding = self.embedding_client.embed_one(question.question)
        if embedding is None:
            return None
        return np.asarray(embedding, dtype=np.float32).reshape(-1)

    def _same_question(
        self,
        row: sqlite3.Row,
        question: QuestionSnapshot,
    ) -> bool:
        """防止错误题号覆盖语义不同的题目。"""
        return text_similarity(row["question_text"], question.question) >= self.question_text_threshold

    def _question_guard(
        self,
        row: sqlite3.Row,
        question: QuestionSnapshot,
    ) -> bool:
        """校验相同题号下题干仍然一致。"""
        similarity = text_similarity(row["question_text"], question.question)
        if similarity < self.question_text_threshold:
            logger.warning(
                "题号 %s 的题干相似度过低 (%.2f)，拒绝使用题库答案",
                question.external_id,
                similarity,
            )
            return False
        return True

    def _record_hit(self, connection: sqlite3.Connection, row_id: int) -> None:
        """更新命中计数。"""
        connection.execute(
            "UPDATE questions SET hit_count = hit_count + 1 WHERE id = ?",
            (row_id,),
        )
        connection.commit()

    @staticmethod
    def _serialize_options(options: Sequence[OptionItem]) -> str:
        """保存有序选项，保留标签与文字。"""
        return json.dumps(
            [option.to_dict() for option in options],
            ensure_ascii=False,
        )

    @staticmethod
    def _load_json_list(value: Optional[str]) -> List[str]:
        """安全读取 JSON 字符串列表。"""
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return []
        if not isinstance(parsed, list):
            return []
        return [str(item).strip() for item in parsed if str(item).strip()]

    @staticmethod
    def _verify_label(row: sqlite3.Row) -> str:
        """生成来源展示中的验证状态。"""
        return "已验" if row["verified"] else "AI"

    def stats(self) -> Dict[str, int]:
        """返回题库统计。"""
        connection = self._connect()
        try:
            total = connection.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
            verified = connection.execute(
                "SELECT COUNT(*) FROM questions WHERE verified = 1"
            ).fetchone()[0]
            hits = connection.execute(
                "SELECT COALESCE(SUM(hit_count), 0) FROM questions"
            ).fetchone()[0]
            return {
                "total": int(total),
                "verified": int(verified),
                "total_hits": int(hits),
            }
        finally:
            connection.close()

