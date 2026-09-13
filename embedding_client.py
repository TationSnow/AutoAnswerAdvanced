"""Embedding client via LMStudio OpenAI-compatible /v1/embeddings.

直接用 requests，避开 OpenAI SDK 默认发送 encoding_format=base64
导致 LMStudio 返回 502 的兼容性问题。
"""
import logging
from typing import List, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)


class EmbeddingClient:
    def __init__(self, base_url: str, model: str, api_key: str = "sk-local",
                 timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._dim: Optional[int] = None

    # ------------------------------------------------------------------
    @property
    def dim(self) -> int:
        if self._dim is None:
            v = self.embed_one("warmup")
            self._dim = len(v) if v is not None else 0
        return self._dim

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-8 else vec

    # ------------------------------------------------------------------
    def _post(self, payload: dict) -> Optional[dict]:
        try:
            resp = requests.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                logger.error(
                    "Embedding HTTP %s: %s",
                    resp.status_code, resp.text[:200]
                )
                return None
            return resp.json()
        except requests.exceptions.Timeout:
            logger.error("Embedding 请求超时")
            return None
        except Exception as e:
            logger.error(f"Embedding 请求异常: {e}")
            return None

    # ------------------------------------------------------------------
    def embed_one(self, text: str) -> Optional[np.ndarray]:
        if not text:
            return None
        data = self._post({"model": self.model, "input": text})
        if not data:
            return None
        try:
            vec = np.asarray(data["data"][0]["embedding"], dtype=np.float32)
            return self._normalize(vec)
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Embedding 响应解析失败: {e}")
            return None

    # ------------------------------------------------------------------
    def embed_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        if not texts:
            return []
        # LMStudio 对批量 input 支持不一定稳定，这里逐条调用更保险
        return [self.embed_one(t) for t in texts]