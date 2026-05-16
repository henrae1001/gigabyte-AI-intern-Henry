from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class SpecSection:
    model: str
    section: str
    values: list[str]
    source_url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Chunk:
    id: str
    model: str
    section: str
    text: str
    source_url: str
    aliases: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

