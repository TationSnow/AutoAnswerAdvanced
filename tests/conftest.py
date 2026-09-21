"""测试公共夹具。"""
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from models import OptionItem, QuestionSnapshot, QuestionType


def make_question(
    question_type: QuestionType,
    options: Sequence[Tuple[str, str]],
    question: str = "示例题目内容？",
    external_id: Optional[str] = None,
) -> QuestionSnapshot:
    """创建不依赖屏幕和 OCR 的题目快照。"""
    option_items = [
        OptionItem(
            label=label,
            text=text,
            box=[
                [10, 100 + index * 30],
                [300, 100 + index * 30],
                [300, 124 + index * 30],
                [10, 124 + index * 30],
            ],
            center=(155, 112 + index * 30),
        )
        for index, (label, text) in enumerate(options)
    ]
    return QuestionSnapshot(
        external_id=external_id,
        question=question,
        question_type=question_type,
        options=option_items,
        raw_text=question,
        confidence=0.95,
        is_valid=True,
    )


class FakeEmbeddingClient:
    """使用稳定向量代替真实 Embedding 服务。"""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def embed_one(self, text: str) -> Optional[np.ndarray]:
        self.calls.append(text)
        if not text:
            return None
        vec = np.zeros(8, dtype=np.float32)
        for index, char in enumerate(text):
            vec[index % len(vec)] += float(ord(char) % 97)
        norm = float(np.linalg.norm(vec))
        return vec if norm == 0 else vec / norm


@pytest.fixture
def fake_embedding_client() -> FakeEmbeddingClient:
    """提供可检查调用次数的假 Embedding 客户端。"""
    return FakeEmbeddingClient()

