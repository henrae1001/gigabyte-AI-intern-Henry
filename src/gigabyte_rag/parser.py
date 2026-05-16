from __future__ import annotations

import json
import re
from pathlib import Path

from gigabyte_rag.config import SOURCE_URL
from gigabyte_rag.models import SpecSection


MODEL_ORDER = ["BXH", "BYH", "BZH"]
MODEL_NAMES = {suffix: f"AORUS MASTER 16 {suffix}" for suffix in MODEL_ORDER}


BASE_SPECS: dict[str, list[str]] = {
    "作業系統": [
        "Windows 11 Pro (GIGABYTE recommends Windows 11 Pro for business.)",
        "Windows 11 Home",
        "UEFI Shell OS",
    ],
    "中央處理器": [
        "Intel Core Ultra 9 Processor 275HX (36MB cache, up to 5.4 GHz, 24 cores, 24 threads)",
    ],
    "顯示器": [
        '16" 16:10',
        "OLED WQXGA (2560x1600) 240Hz, 1ms, DCI-P3 100%, 500nits (peak), 1,000,000:1",
        "NVIDIA G-SYNC",
        "NVIDIA Advanced Optimus",
        "VESA DisplayHDR True Black 500",
        "VESA ClearMR 10000",
        "Pantone Validated",
        "TUV Rheinland Low Blue Light",
        "Dolby Vision",
    ],
    "記憶體": [
        "Up to 64GB DDR5 5600MHz",
        "2x SO-DIMM sockets for expansion",
    ],
    "儲存裝置": [
        "1x PCIe Gen5 M.2 slot",
        "1x PCIe Gen4x4 M.2 slot",
        "Up to 4TB PCIe NVMe M.2 SSD",
    ],
    "鍵盤種類": [
        "3-zone RGB Backlit Keyboard, Up to 1.7mm Key-travel (Support N-Key)",
    ],
    "連接埠": [
        "Left Side: 1 x DC in; 1 x RJ-45; 1 x HDMI 2.1; 1 x Type-A support USB3.2 Gen2; 1 x Type-C with Thunderbolt 5 (support USB4, DisplayPort 2.1 and Power Delivery 3.0)",
        "Right Side: 1 x Type-A support USB3.2 Gen2; 1 x Type-C with Thunderbolt 4 (support USB4, DisplayPort 1.4 and Power Delivery 3.0); 1 x MicroSD (UHS-II); 1 x Audio Jack support mic / headphone combo",
    ],
    "音效": [
        "4x 2W speakers",
        "Microphone",
        "Dolby Atmos",
        "Smart Amp Technology",
    ],
    "通訊": [
        "WIFI 7 (802.11be 2x2)",
        "LAN: 1G",
        "Bluetooth v5.4",
    ],
    "視訊鏡頭": [
        "FHD (1080p) IR Webcam",
        "Built-in array Microphone",
        "Support Windows Hello Face Authentication",
    ],
    "安全裝置": [
        "Firmware-based TPM, supports Intel Platform Trust Technology (Intel PTT)",
    ],
    "電池": ["Li-ion 99Wh"],
    "變壓器": ["330W AC Adapter"],
    "尺寸": ["357 x 254 x 23~29.9 mm"],
    "重量": ["~2.5 kg"],
    "顏色": ["Dark Tide"],
}

DEFAULT_GPU_SPECS: dict[str, list[str]] = {
    "BXH": [
        "NVIDIA GeForce RTX 5090 Laptop GPU",
        "24GB GDDR7",
        "175W Maximum Graphics Power with Dynamic Boost",
        "AI Boost: 1797 MHz (1597 MHz Boost Clock + 200 MHz OC)",
    ],
    "BYH": [
        "NVIDIA GeForce RTX 5080 Laptop GPU",
        "16GB GDDR7",
        "175W Maximum Graphics Power with Dynamic Boost",
        "AI Boost: 1902 MHz (1702 MHz Boost Clock + 200 MHz OC)",
    ],
    "BZH": [
        "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
        "12GB GDDR7",
        "140W Maximum Graphics Power with Dynamic Boost",
        "AI Boost: 1962 MHz (1762 MHz Boost Clock + 200 MHz OC)",
    ],
}


def download_html(url: str = SOURCE_URL, timeout: float = 30.0) -> str:
    import httpx

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
    }
    response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return response.text


def save_html(html: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def load_html(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_specs(html: str, source_url: str = SOURCE_URL) -> dict[str, list[dict[str, object]]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    table_specs = _extract_table_specs(soup, source_url)
    if table_specs:
        return table_specs

    text = soup.get_text("\n")
    text = _normalize_text(text)
    gpu_specs = _extract_gpu_variants(text) or DEFAULT_GPU_SPECS
    specs = _build_specs(gpu_specs, source_url)
    validate_specs(specs)
    return specs


def seed_specs(source_url: str = SOURCE_URL) -> dict[str, list[dict[str, object]]]:
    specs = _build_specs(DEFAULT_GPU_SPECS, source_url)
    validate_specs(specs)
    return specs


def validate_specs(specs: dict[str, object]) -> None:
    models = specs.get("models")
    if not isinstance(models, list):
        raise ValueError("Invalid specs: missing models list")

    expected_models = {MODEL_NAMES[suffix] for suffix in MODEL_ORDER}
    actual_models = {str(model.get("model")) for model in models if isinstance(model, dict)}
    missing_models = sorted(expected_models - actual_models)
    if missing_models:
        raise ValueError(f"Invalid specs: missing model variants: {', '.join(missing_models)}")

    required_sections = {"顯示晶片", "顯示器", "連接埠", "電池", "重量", "變壓器"}
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("Invalid specs: model item is not an object")
        model_name = str(model.get("model"))
        sections = model.get("sections")
        if not isinstance(sections, list):
            raise ValueError(f"Invalid specs: {model_name} missing sections list")
        by_name = {str(section.get("section")): section for section in sections if isinstance(section, dict)}
        missing_sections = sorted(required_sections - set(by_name))
        if missing_sections:
            raise ValueError(f"Invalid specs: {model_name} missing sections: {', '.join(missing_sections)}")
        for section_name in required_sections:
            values = by_name[section_name].get("values")
            if not isinstance(values, list) or not values or any(not str(value).strip() for value in values):
                raise ValueError(f"Invalid specs: {model_name} section {section_name} is empty")


def write_specs_json(specs: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")


def write_specs_markdown(specs: dict[str, object], path: Path) -> None:
    lines = [f"# AORUS MASTER 16 AM6H Product Specs", ""]
    for model in specs["models"]:  # type: ignore[index]
        lines.extend([f"## {model['model']}", ""])
        for section in model["sections"]:
            values = "\n".join(f"- {value}" for value in section["values"])
            lines.extend([f"### {section['section']}", values, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _extract_gpu_variants(text: str) -> dict[str, list[str]] | None:
    pattern = re.compile(
        r"NVIDIA\s+GeForce\s+RTX\s+(5090|5080|5070\s+Ti)\s+Laptop\s+GPU\s+"
        r"(\d+GB\s+GDDR7)\s+"
        r"(\d+W\s+Maximum\s+Graphics\s+Power\s+with\s+Dynamic\s+Boost)\*?\s+"
        r"AI\s+Boost\s*:\s*(\d+\s+MHz\s+\(\d+\s+MHz\s+Boost\s+Clock\s+\+\s+200\s+MHz\s+OC\))",
        flags=re.IGNORECASE,
    )
    matches = pattern.findall(text)
    if len(matches) < 3:
        return None

    by_gpu = {match[0].replace(" ", ""): match for match in matches}
    mapping = {"BXH": "5090", "BYH": "5080", "BZH": "5070Ti"}
    extracted: dict[str, list[str]] = {}
    for suffix, gpu_key in mapping.items():
        match = by_gpu.get(gpu_key)
        if not match:
            return None
        gpu_name, memory, power, boost = match
        extracted[suffix] = [
            f"NVIDIA GeForce RTX {gpu_name} Laptop GPU",
            memory,
            power,
            f"AI Boost: {boost}",
        ]
    return extracted


def _extract_table_specs(soup, source_url: str) -> dict[str, list[dict[str, object]]] | None:
    by_model: dict[str, dict[str, list[str]]] = {suffix: {} for suffix in MODEL_ORDER}
    for table in soup.find_all("table"):
        header_map = _table_header_map(table)
        if not header_map:
            continue
        for row in table.find_all("tr"):
            cells = [_clean_cell_text(cell.get_text("\n")) for cell in row.find_all(["th", "td"])]
            cells = [cell for cell in cells if cell]
            if len(cells) < 2:
                continue
            section = _canonical_section(cells[0])
            if not section or _row_is_header(cells):
                continue
            for suffix, index in header_map.items():
                if index >= len(cells):
                    continue
                values = _split_values(cells[index])
                if values:
                    by_model[suffix][section] = values

    if not any(sections for sections in by_model.values()):
        return None
    specs = _build_specs_from_sections(by_model, source_url)
    try:
        validate_specs(specs)
    except ValueError:
        return None
    return specs


def _table_header_map(table) -> dict[str, int]:
    for row in table.find_all("tr"):
        cells = [_clean_cell_text(cell.get_text(" ")) for cell in row.find_all(["th", "td"])]
        if len(cells) < 2:
            continue
        mapping: dict[str, int] = {}
        for index, cell in enumerate(cells):
            upper = cell.upper()
            for suffix in MODEL_ORDER:
                if suffix in upper:
                    mapping[suffix] = index
        if len(mapping) == len(MODEL_ORDER):
            return mapping
    return {}


def _build_specs_from_sections(
    by_model: dict[str, dict[str, list[str]]],
    source_url: str,
) -> dict[str, list[dict[str, object]]]:
    models: list[dict[str, object]] = []
    for suffix in MODEL_ORDER:
        model_name = MODEL_NAMES[suffix]
        sections = []
        merged = _merge_with_defaults(by_model.get(suffix, {}), suffix)
        for section_name, values in merged.items():
            sections.append(SpecSection(model_name, section_name, values, source_url).to_dict())
        models.append({"model": model_name, "sections": sections})
    return {"source_url": source_url, "models": models}  # type: ignore[return-value]


def _merge_with_defaults(parsed: dict[str, list[str]], suffix: str) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for section, values in BASE_SPECS.items():
        merged[section] = parsed.get(section, values)
        if section == "中央處理器":
            merged["顯示晶片"] = parsed.get("顯示晶片", DEFAULT_GPU_SPECS[suffix])
    for section, values in parsed.items():
        merged.setdefault(section, values)
    return merged


def _canonical_section(section: str) -> str | None:
    normalized = re.sub(r"\s+", "", section).lower()
    aliases = {
        "os": "作業系統",
        "作業系統": "作業系統",
        "cpu": "中央處理器",
        "processor": "中央處理器",
        "中央處理器": "中央處理器",
        "顯示晶片": "顯示晶片",
        "gpu": "顯示晶片",
        "graphics": "顯示晶片",
        "display": "顯示器",
        "顯示器": "顯示器",
        "螢幕": "顯示器",
        "memory": "記憶體",
        "記憶體": "記憶體",
        "storage": "儲存裝置",
        "儲存裝置": "儲存裝置",
        "keyboard": "鍵盤種類",
        "鍵盤種類": "鍵盤種類",
        "ports": "連接埠",
        "io": "連接埠",
        "i/o": "連接埠",
        "連接埠": "連接埠",
        "audio": "音效",
        "音效": "音效",
        "communications": "通訊",
        "通訊": "通訊",
        "webcam": "視訊鏡頭",
        "視訊鏡頭": "視訊鏡頭",
        "security": "安全裝置",
        "安全裝置": "安全裝置",
        "battery": "電池",
        "電池": "電池",
        "adapter": "變壓器",
        "變壓器": "變壓器",
        "dimensions": "尺寸",
        "尺寸": "尺寸",
        "weight": "重量",
        "重量": "重量",
        "color": "顏色",
        "colour": "顏色",
        "顏色": "顏色",
    }
    for key, canonical in aliases.items():
        if key in normalized:
            return canonical
    return None


def _row_is_header(cells: list[str]) -> bool:
    joined = " ".join(cells).upper()
    return all(suffix in joined for suffix in MODEL_ORDER)


def _clean_cell_text(text: str) -> str:
    text = _normalize_text(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_values(text: str) -> list[str]:
    parts = re.split(r"\s*(?:\n|;|•|●)\s*", text)
    values = [part.strip(" -") for part in parts if part.strip(" -")]
    return values or ([text.strip()] if text.strip() else [])


def _build_specs(gpu_specs: dict[str, list[str]], source_url: str) -> dict[str, list[dict[str, object]]]:
    specs: dict[str, list[dict[str, object]]] = {"source_url": source_url, "models": []}  # type: ignore[assignment]
    models: list[dict[str, object]] = []
    for suffix in MODEL_ORDER:
        sections = []
        for section, values in BASE_SPECS.items():
            sections.append(SpecSection(MODEL_NAMES[suffix], section, values, source_url).to_dict())
            if section == "中央處理器":
                sections.append(
                    SpecSection(MODEL_NAMES[suffix], "顯示晶片", gpu_specs[suffix], source_url).to_dict()
                )
        models.append({"model": MODEL_NAMES[suffix], "sections": sections})

    specs["models"] = models  # type: ignore[index]
    return specs


def _normalize_text(text: str) -> str:
    replacements = {
        "®": "",
        "™": "",
        "×": "x",
        "DCIP-3": "DCI-P3",
        "TÜV": "TUV",
        "\xa0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines)
