from __future__ import annotations

import json
import re
from pathlib import Path

from gigabyte_rag.models import Chunk


ALIASES = {
    "作業系統": ["os", "operating system", "windows", "作業系統"],
    "中央處理器": ["cpu", "processor", "中央處理器", "處理器"],
    "顯示晶片": ["gpu", "graphics", "display card", "顯示晶片", "顯卡"],
    "顯示器": ["display", "screen", "monitor", "resolution", "refresh rate", "panel", "螢幕", "顯示器", "解析度", "刷新率", "面板"],
    "記憶體": ["memory", "ram", "記憶體"],
    "儲存裝置": ["storage", "ssd", "m.2", "儲存", "硬碟"],
    "鍵盤種類": ["keyboard", "rgb", "鍵盤"],
    "連接埠": ["ports", "i/o", "io", "left side", "right side", "usb", "thunderbolt", "hdmi", "連接埠", "接口"],
    "音效": ["audio", "speaker", "microphone", "dolby atmos", "音效", "喇叭", "麥克風"],
    "通訊": ["wifi", "wi-fi", "lan", "bluetooth", "network", "wireless", "通訊", "網路", "藍牙"],
    "視訊鏡頭": ["webcam", "camera", "windows hello", "視訊鏡頭", "鏡頭"],
    "安全裝置": ["security", "tpm", "ptt", "安全"],
    "電池": ["battery", "wh", "電池"],
    "變壓器": ["adapter", "charger", "power adapter", "power supply", "變壓器", "充電器", "電源供應器"],
    "尺寸": ["dimension", "size", "尺寸"],
    "重量": ["weight", "kg", "重量", "多重", "多重啊"],
    "顏色": ["color", "colour", "顏色"],
}


def build_chunks(specs: dict[str, object]) -> list[Chunk]:
    chunks: list[Chunk] = []
    source_url = str(specs["source_url"])
    for model in specs["models"]:  # type: ignore[index]
        model_name = str(model["model"])
        model_suffix = model_name.split()[-1]
        for section in model["sections"]:
            section_name = str(section["section"])
            values = [str(value) for value in section["values"]]
            alias_text = ALIASES.get(section_name, [])
            text = _format_chunk_text(model_name, section_name, values)
            chunk_id = f"{model_suffix.lower()}-{_slugify(section_name)}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    model=model_name,
                    section=section_name,
                    text=text,
                    source_url=source_url,
                    aliases=[model_suffix.lower(), model_name.lower(), *alias_text],
                )
            )
    chunks.extend(_build_comparison_chunks(specs))
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_chunks(path: Path) -> list[Chunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**item) for item in data]


def _build_comparison_chunks(specs: dict[str, object]) -> list[Chunk]:
    rows: dict[str, dict[str, list[str]]] = {}
    source_url = str(specs["source_url"])
    for model in specs["models"]:  # type: ignore[index]
        model_name = str(model["model"])
        for section in model["sections"]:
            rows.setdefault(str(section["section"]), {})[model_name] = [str(v) for v in section["values"]]

    chunks = []
    for section, by_model in rows.items():
        if len(by_model) < 2:
            continue
        lines = [f"{section} comparison across AORUS MASTER 16 AM6H variants:"]
        for model_name, values in by_model.items():
            lines.append(f"{model_name}: {'; '.join(values)}")
        chunks.append(
            Chunk(
                id=f"compare-{_slugify(section)}",
                model="ALL",
                section=f"{section}比較",
                text="\n".join(lines),
                source_url=source_url,
                aliases=["compare", "comparison", "difference", "差異", "比較", *ALIASES.get(section, [])],
            )
        )
    return chunks


def _format_chunk_text(model: str, section: str, values: list[str]) -> str:
    bullet_values = "\n".join(f"- {value}" for value in values)
    return f"Model: {model}\nSection: {section}\nSpecs:\n{bullet_values}"


def _slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff-]+", "", value)
    return value.strip("-")
