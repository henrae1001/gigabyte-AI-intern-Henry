from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from gigabyte_rag.models import Chunk


DEFAULT_DIM = 2048


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class HashingVectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: list[list[float]], dim: int = DEFAULT_DIM):
        self.chunks = chunks
        self.vectors = vectors
        self.dim = dim

    @classmethod
    def build(cls, chunks: list[Chunk], dim: int = DEFAULT_DIM) -> "HashingVectorIndex":
        vectors = [embed_text(_index_text(chunk), dim) for chunk in chunks]
        return cls(chunks, vectors, dim)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": self.dim,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "vectors": self.vectors,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "HashingVectorIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in payload["chunks"]]
        return cls(chunks, payload["vectors"], int(payload["dim"]))

    def search(self, query: str, top_k: int = 5, model_filter: str | None = None) -> list[SearchResult]:
        query_vector = embed_text(query, self.dim)
        scored = []
        normalized_filter = model_filter.lower() if model_filter else None
        for idx, chunk in enumerate(self.chunks):
            if normalized_filter and normalized_filter not in chunk.model.lower() and chunk.model != "ALL":
                continue
            score = dot(self.vectors[idx], query_vector) + _keyword_boost(query, chunk)
            if score > 0:
                scored.append(SearchResult(chunk, score))
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]


def embed_text(text: str, dim: int = DEFAULT_DIM) -> list[float]:
    vector = [0.0] * dim
    tokens = _features(text)
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(dot(vector, vector))
    if norm > 0:
        vector = [value / norm for value in vector]
    return vector


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _index_text(chunk: Chunk) -> str:
    return " ".join([chunk.model, chunk.section, chunk.text, *chunk.aliases])


def _features(text: str) -> list[str]:
    text = _normalize(text)
    word_tokens = re.findall(r"[a-z0-9.+#-]+|[\u4e00-\u9fff]", text)
    features: list[str] = []
    features.extend(word_tokens)
    features.extend(_char_ngrams(text, 2))
    features.extend(_char_ngrams(text, 3))
    return features


def _char_ngrams(text: str, n: int) -> list[str]:
    compact = re.sub(r"\s+", "", text)
    return [compact[i : i + n] for i in range(max(0, len(compact) - n + 1))]


def _normalize(text: str) -> str:
    replacements = {
        "顯卡": "顯示晶片 gpu",
        "螢幕": "顯示器 display screen",
        "解析度": "顯示器 display resolution",
        "刷新率": "顯示器 display refresh rate",
        "接口": "連接埠 ports",
        "連接孔": "連接埠 ports",
        "重量": "重量 weight",
        "電池": "電池 battery",
        "充電器": "變壓器 adapter charger",
        "變壓器": "變壓器 adapter charger",
        "處理器": "中央處理器 cpu processor",
    }
    text = text.lower()
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _keyword_boost(query: str, chunk: Chunk) -> float:
    normalized_query = _normalize(query)
    boost = 0.0

    model_suffix = chunk.model.split()[-1].lower() if chunk.model != "ALL" else ""
    if model_suffix and model_suffix in normalized_query:
        boost += 0.20

    section = chunk.section.replace("比較", "").lower()
    if section in normalized_query:
        boost += 0.80

    for alias in chunk.aliases:
        alias = alias.lower()
        if alias and alias in normalized_query:
            boost += 0.35

    if chunk.model == "ALL" and any(term in normalized_query for term in ["compare", "comparison", "difference", "差異", "比較"]):
        boost += 0.50

    return boost
