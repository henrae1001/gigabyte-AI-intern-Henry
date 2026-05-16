from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from gigabyte_rag.config import DEFAULT_MODEL_PATH, MODEL_DIR


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    filename: str
    bytes_hint: str
    reason: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/main/{self.filename}"


MODEL_SPECS: dict[str, ModelSpec] = {
    "qwen2.5-3b-q4_k_m": ModelSpec(
        name="qwen2.5-3b-q4_k_m",
        repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        bytes_hint="about 2.1 GB",
        reason="Best quality target for a narrow RAG assistant while still fitting a 4GB VRAM budget with Q4_K_M and n_ctx=2048.",
    ),
    "qwen2.5-1.5b-q4_k_m": ModelSpec(
        name="qwen2.5-1.5b-q4_k_m",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        bytes_hint="about 1.1 GB",
        reason="Lower-memory fallback for CPU-only or very tight VRAM environments.",
    ),
}


def destination_for(spec: ModelSpec, output_dir: Path = MODEL_DIR) -> Path:
    if spec.name == "qwen2.5-3b-q4_k_m":
        return DEFAULT_MODEL_PATH
    return output_dir / spec.filename


def download_model(
    spec: ModelSpec,
    destination: Path,
    *,
    force: bool = False,
    progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    if destination.exists() and not force:
        return destination

    headers = {}
    mode = "wb"
    existing = temp_path.stat().st_size if temp_path.exists() and not force else 0
    if existing:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    with httpx.stream("GET", spec.url, headers=headers, follow_redirects=True, timeout=None) as response:
        response.raise_for_status()
        if existing and response.status_code != 206:
            existing = 0
            mode = "wb"

        total = _content_length(response)
        if total is not None:
            total += existing

        downloaded = existing
        with temp_path.open(mode) as handle:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if progress:
                    progress(downloaded, total)

    temp_path.replace(destination)
    return destination


def resolve_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model {name!r}. Valid choices: {valid}") from exc


def _content_length(response: httpx.Response) -> int | None:
    value = response.headers.get("Content-Length")
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None
