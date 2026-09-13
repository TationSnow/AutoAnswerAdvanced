"""SQLite + 向量语义检索的题库。"""
import os
import re
import json
import sqlite3
import hashlib
import logging
from datetime import datetime
from typing import Dict, Optional, List
import numpy as np

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', '', text or '')
    text = re.sub(r'[，。？！、；：""''（）【】《》]', '', text)
    return text.lower()


class KnowledgeBase:
    def __init__(self, db_path: str, embedding_client=None,
                 similarity_threshold: float = 0.85):
        self.db_path = db_path
        self.embedding_client = embedding_client
        self.similarity_threshold = similarity_threshold
        self._init_db()

    # ------------------------------------------------------------------
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_hash TEXT UNIQUE,
                    question_text TEXT NOT NULL,
                    options_json TEXT,
                    question_type TEXT,
                    answer TEXT,
                    detail TEXT,
                    source TEXT,
                    verified INTEGER DEFAULT 0,
                    embedding BLOB,
                    embedding_dim INTEGER,
                    hit_count INTEGER DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_qhash ON questions(question_hash)"
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def _hash(self, question: str, options: Dict) -> str:
        norm = normalize_text(question)
        opt_norm = '|'.join(
            f"{k}:{normalize_text(v)}" for k, v in sorted((options or {}).items())
        )
        raw = norm + '#' + opt_norm
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:32]

    # ------------------------------------------------------------------
    def add(self, question: str, options: Dict, question_type: str,
            answer: str, detail: str, source: str = "LLM",
            verified: bool = False) -> bool:
        """新增或更新题目。返回 True 表示新增，False 表示已存在或未写入。"""
        if not question or not answer or answer == '?':
            return False

        q_hash = self._hash(question, options)
        emb = None
        if self.embedding_client is not None:
            emb = self.embedding_client.embed_one(question)

        now = datetime.now().isoformat(timespec='seconds')
        options_json = json.dumps(options or {}, ensure_ascii=False)
        emb_blob = emb.tobytes() if emb is not None else None
        emb_dim = int(emb.shape[0]) if emb is not None else None

        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT id FROM questions WHERE question_hash = ?",
                        (q_hash,))
            row = cur.fetchone()

            if row:
                # 已有记录：如果新来的带 verified=True，才覆盖，避免污染
                cur.execute("""
                    UPDATE questions
                    SET answer = CASE WHEN ? = 1 THEN ? ELSE answer END,
                        detail = CASE WHEN ? = 1 THEN ? ELSE detail END,
                        source = CASE WHEN ? = 1 THEN ? ELSE source END,
                        verified = CASE WHEN ? = 1 THEN 1 ELSE verified END,
                        embedding = COALESCE(?, embedding),
                        embedding_dim = COALESCE(?, embedding_dim),
                        updated_at = ?
                    WHERE id = ?
                """, (1 if verified else 0, answer,
                      1 if verified else 0, detail,
                      1 if verified else 0, source,
                      1 if verified else 0,
                      emb_blob, emb_dim, now, row[0]))
                conn.commit()
                return False

            cur.execute("""
                INSERT INTO questions
                (question_hash, question_text, options_json, question_type,
                 answer, detail, source, verified, embedding, embedding_dim,
                 hit_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """, (q_hash, question, options_json, question_type,
                  answer, detail, source, 1 if verified else 0,
                  emb_blob, emb_dim, now, now))
            conn.commit()
            return True
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def search(self, question: str, options: Dict) -> Optional[Dict]:
        """先精确 hash 命中，再向量语义检索。"""
        if not question:
            return None

        # 1) 精确 hash
        q_hash = self._hash(question, options)
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, answer, detail, source, verified
                FROM questions WHERE question_hash = ?
            """, (q_hash,))
            row = cur.fetchone()
            if row:
                cur.execute(
                    "UPDATE questions SET hit_count = hit_count + 1 WHERE id = ?",
                    (row[0],)
                )
                conn.commit()
                return {
                    "answer": row[1],
                    "detail": row[2],
                    "source": f"题库(精确·{'已验' if row[4] else 'AI'})",
                    "match_id": row[0],
                    "similarity": 1.0,
                }
        finally:
            conn.close()

        # 2) 向量语义检索
        if self.embedding_client is None:
            return None
        query_emb = self.embedding_client.embed_one(question)
        if query_emb is None:
            return None

        best = None
        best_sim = 0.0

        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, answer, detail, source, embedding, embedding_dim, verified
                FROM questions WHERE embedding IS NOT NULL
            """)
            for row in cur.fetchall():
                blob, dim = row[4], row[5]
                if blob is None or dim != int(query_emb.shape[0]):
                    continue
                stored = np.frombuffer(blob, dtype=np.float32)
                sim = float(np.dot(query_emb, stored))
                if sim > best_sim:
                    best_sim = sim
                    best = (row[0], row[1], row[2], row[3], row[6])

            if best and best_sim >= self.similarity_threshold:
                cur.execute(
                    "UPDATE questions SET hit_count = hit_count + 1 WHERE id = ?",
                    (best[0],)
                )
                conn.commit()
                return {
                    "answer": best[1],
                    "detail": best[2],
                    "source": f"题库(相似{best_sim:.2f}·{'已验' if best[4] else 'AI'})",
                    "match_id": best[0],
                    "similarity": best_sim,
                }
        finally:
            conn.close()

        return None

    # ------------------------------------------------------------------
    def update_by_id(self, match_id: int, answer: str, detail: str,
                     verified: bool = True):
        """用屏幕捕获到的真实答案覆盖题库记录。"""
        if not match_id:
            return
        now = datetime.now().isoformat(timespec='seconds')
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                UPDATE questions
                SET answer = ?, detail = ?,
                    source = '屏幕捕获', verified = 1, updated_at = ?
                WHERE id = ?
            """, (answer, detail, now, match_id))
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def stats(self) -> Dict:
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM questions")
            total = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM questions WHERE verified = 1")
            verified = cur.fetchone()[0]
            cur.execute("SELECT COALESCE(SUM(hit_count), 0) FROM questions")
            hits = cur.fetchone()[0]
            return {"total": total, "verified": verified, "total_hits": hits}
        finally:
            conn.close()