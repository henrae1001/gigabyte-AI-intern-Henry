from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from gigabyte_rag.chunking import build_chunks, save_chunks
from gigabyte_rag.config import CHUNKS_JSON_PATH, INDEX_PATH, RAW_HTML_PATH, SOURCE_URL, SPECS_JSON_PATH, SPECS_MD_PATH
from gigabyte_rag.parser import download_html, load_html, parse_specs, save_html, seed_specs, write_specs_json, write_specs_markdown
from gigabyte_rag.vector_index import HashingVectorIndex, SearchResult


@dataclass(frozen=True)
class RetrievalMetrics:
    seconds: float
    result_count: int


def ingest(
    *,
    source_url: str = SOURCE_URL,
    raw_html_path: Path = RAW_HTML_PATH,
    specs_json_path: Path = SPECS_JSON_PATH,
    specs_md_path: Path = SPECS_MD_PATH,
    chunks_json_path: Path = CHUNKS_JSON_PATH,
    index_path: Path = INDEX_PATH,
    use_cached_html: bool = False,
    use_seed: bool = False,
) -> None:
    if use_seed:
        specs = seed_specs(source_url)
    else:
        if use_cached_html:
            cached_html_path = _resolve_cached_html_path(raw_html_path)
            html = load_html(cached_html_path)
        else:
            html = download_html(source_url)
            save_html(html, raw_html_path)

        specs = parse_specs(html, source_url)
    write_specs_json(specs, specs_json_path)
    write_specs_markdown(specs, specs_md_path)

    chunks = build_chunks(specs)
    save_chunks(chunks, chunks_json_path)
    index = HashingVectorIndex.build(chunks)
    index.save(index_path)


def retrieve(
    question: str,
    *,
    index_path: Path = INDEX_PATH,
    top_k: int = 5,
    model_filter: str | None = None,
    min_score: float = 0.65,
) -> tuple[list[SearchResult], RetrievalMetrics]:
    start = time.perf_counter()
    index = HashingVectorIndex.load(index_path)
    results = [result for result in index.search(question, top_k=top_k, model_filter=model_filter) if result.score >= min_score]
    elapsed = time.perf_counter() - start
    return results, RetrievalMetrics(elapsed, len(results))


def _resolve_cached_html_path(raw_html_path: Path) -> Path:
    if raw_html_path.exists():
        return raw_html_path

    html_files = sorted(raw_html_path.parent.glob("*.html"))
    if len(html_files) == 1:
        return html_files[0]
    if not html_files:
        raise FileNotFoundError(
            f"No cached HTML found. Expected {raw_html_path} or one .html file under {raw_html_path.parent}."
        )
    choices = ", ".join(str(path) for path in html_files)
    raise FileExistsError(
        f"Multiple cached HTML files found under {raw_html_path.parent}. "
        f"Please keep one file or use the default path {raw_html_path}. Found: {choices}"
    )
