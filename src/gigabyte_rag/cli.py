from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from gigabyte_rag.config import DEFAULT_MODEL_PATH, INDEX_PATH
from gigabyte_rag.llm import (
    build_prompt,
    estimate_prompt_tokens,
    heuristic_answer,
    measure_stream,
    stream_with_llama_cli,
    stream_with_llama_cpp,
    stream_with_llama_server,
)
from gigabyte_rag.pipeline import ingest, retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="AORUS MASTER 16 AM6H pure Python RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Parse specs and build vector index")
    ingest_parser.add_argument("--cached-html", action="store_true", help="Use data/raw HTML if it already exists")
    ingest_parser.add_argument("--seed", action="store_true", help="Build from embedded official spec summary without downloading")

    model_parser = subparsers.add_parser("download-model", help="Download a recommended GGUF model into models/")
    model_parser.add_argument("--model", choices=["qwen2.5-1.5b-q4_k_m", "qwen2.5-3b-q4_k_m"], default="qwen2.5-3b-q4_k_m")
    model_parser.add_argument("--output", help=f"Destination path. Default: {DEFAULT_MODEL_PATH}")
    model_parser.add_argument("--force", action="store_true", help="Redownload even when the destination already exists")

    ask_parser = subparsers.add_parser("ask", help="Ask a question with retrieval and optional llama.cpp generation")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.add_argument("--min-score", type=float, default=0.65)
    ask_parser.add_argument("--model-filter", help="Filter retrieval to BXH, BYH, BZH, or full model text")
    ask_parser.add_argument("--debug", action="store_true")
    ask_parser.add_argument("--model-path", help="Path to a GGUF model for llama.cpp")
    ask_parser.add_argument("--llama-cli-path", help="Use official llama-cli / llama-cli.exe with --model-path")
    ask_parser.add_argument("--llama-server-url", help="Use a running llama-server OpenAI-compatible endpoint")
    ask_parser.add_argument("--n-ctx", type=int, default=2048)
    ask_parser.add_argument("--n-gpu-layers", type=int, default=-1)
    ask_parser.add_argument("--max-tokens", type=int, default=256)
    ask_parser.add_argument("--temperature", type=float, default=0.1)
    ask_parser.add_argument("--no-llm", action="store_true", help="Return top retrieved context without llama.cpp")

    bench_parser = subparsers.add_parser("bench", help="Run retrieval/generation benchmark from JSONL")
    bench_parser.add_argument("--questions", default="eval/questions.jsonl")
    bench_parser.add_argument("--output", help="Output JSON path. Default depends on --no-llm or --model-path")
    bench_parser.add_argument("--model-path")
    bench_parser.add_argument("--llama-cli-path")
    bench_parser.add_argument("--llama-server-url")
    bench_parser.add_argument("--n-ctx", type=int, default=2048)
    bench_parser.add_argument("--n-gpu-layers", type=int, default=-1)
    bench_parser.add_argument("--max-tokens", type=int, default=128)
    bench_parser.add_argument("--temperature", type=float, default=0.1)
    bench_parser.add_argument("--top-k", type=int, default=5)
    bench_parser.add_argument("--min-score", type=float, default=0.65)
    bench_parser.add_argument("--no-llm", action="store_true")

    args = parser.parse_args()
    if args.command == "ingest":
        try:
            ingest(use_cached_html=args.cached_html, use_seed=args.seed)
        except Exception as exc:
            if exc.__class__.__name__ != "HTTPStatusError":
                raise
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", "unknown")
            print(
                f"Failed to download official page: HTTP {status_code}. "
                "Use `uv run gigabyte-rag ingest --seed` for the embedded official spec summary, "
                "or save one HTML file under data/raw/ and run with --cached-html.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc
        print(f"Index built at {INDEX_PATH}")
    elif args.command == "download-model":
        _download_model(args)
    elif args.command == "ask":
        _ask(args)
    elif args.command == "bench":
        _bench(args)


def _ask(args: argparse.Namespace) -> None:
    _ensure_index()
    results, retrieval_metrics = retrieve(args.question, top_k=args.top_k, model_filter=args.model_filter, min_score=args.min_score)
    prompt = build_prompt(args.question, results)
    if args.debug:
        _print_debug(results, retrieval_metrics.seconds, estimate_prompt_tokens(prompt))

    if args.no_llm or not _has_generation_backend(args):
        started = time.perf_counter()
        print(heuristic_answer(args.question, results))
        total = time.perf_counter() - started
        print(json.dumps({"retrieval_seconds": retrieval_metrics.seconds, "generation_seconds": total, "prompt_estimated_tokens": estimate_prompt_tokens(prompt)}, indent=2))
        return

    stream = _generation_stream(args, prompt, max_tokens=args.max_tokens)
    _, metrics = measure_stream(stream)
    print(
        json.dumps(
            {
                "retrieval_seconds": retrieval_metrics.seconds,
                "ttft_seconds": metrics.ttft_seconds,
                "total_generation_seconds": metrics.total_seconds,
                "output_tokens": metrics.output_tokens,
                "tokens_per_second": metrics.tokens_per_second,
                "prompt_estimated_tokens": estimate_prompt_tokens(prompt),
            },
            indent=2,
        )
    )


def _download_model(args: argparse.Namespace) -> None:
    from gigabyte_rag.download_model import destination_for, download_model, resolve_model_spec

    spec = resolve_model_spec(args.model)
    destination = Path(args.output) if args.output else destination_for(spec)
    print(f"Downloading {spec.repo_id}/{spec.filename}")
    print(f"Target: {destination}")
    print(f"Size: {spec.bytes_hint}")
    print(f"Reason: {spec.reason}")

    last_percent = -1

    def progress(downloaded: int, total: int | None) -> None:
        nonlocal last_percent
        if total:
            percent = int(downloaded / total * 100)
            if percent != last_percent and (percent % 5 == 0 or percent == 100):
                last_percent = percent
                print(f"{percent:3d}% ({downloaded / (1024 ** 3):.2f} / {total / (1024 ** 3):.2f} GiB)", flush=True)
        else:
            gib = downloaded / (1024 ** 3)
            print(f"downloaded {gib:.2f} GiB", flush=True)

    path = download_model(spec, destination, force=args.force, progress=progress)
    print(f"Model ready: {path}")


def _bench(args: argparse.Namespace) -> None:
    _ensure_index()
    output_path = args.output or _default_benchmark_output(args)
    rows = []
    with open(args.questions, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            question = item["question"]
            results, retrieval_metrics = retrieve(
                question,
                top_k=args.top_k,
                model_filter=item.get("model_filter"),
                min_score=args.min_score,
            )
            answer = ""
            generation_metrics = None
            if args.no_llm or not _has_generation_backend(args):
                prompt = build_prompt(question, results)
                started = time.perf_counter()
                answer = heuristic_answer(question, results)
                generation_metrics = {
                    "total_seconds": time.perf_counter() - started,
                    "prompt_estimated_tokens": estimate_prompt_tokens(prompt),
                }
            else:
                prompt = build_prompt(question, results)
                answer, metrics = measure_stream(
                    _generation_stream(args, prompt, max_tokens=int(item.get("max_tokens", args.max_tokens))),
                    echo=False,
                )
                generation_metrics = {
                    "ttft_seconds": metrics.ttft_seconds,
                    "total_seconds": metrics.total_seconds,
                    "output_tokens": metrics.output_tokens,
                    "tokens_per_second": metrics.tokens_per_second,
                    "prompt_estimated_tokens": estimate_prompt_tokens(prompt),
                }

            rows.append(
                {
                    "id": item.get("id"),
                    "question": question,
                    "expected": item.get("expected"),
                    "expected_top_id": item.get("expected_top_id"),
                    "expected_chunk_ids": item.get("expected_chunk_ids"),
                    "expected_refusal": item.get("expected_refusal"),
                    "top_id_hit": _top_id_hit(results, item.get("expected_top_id")),
                    "expected_hit_at_k": _expected_hit_at_k(results, item.get("expected_chunk_ids")),
                    "refusal_hit": _refusal_hit(results, item.get("expected_refusal")),
                    "answer": answer,
                    "top_chunks": [
                        {"id": result.chunk.id, "model": result.chunk.model, "section": result.chunk.section, "score": result.score}
                        for result in results
                    ],
                    "retrieval_seconds": retrieval_metrics.seconds,
                    "generation": generation_metrics,
                }
            )

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    print(f"Wrote benchmark results to {output_path}")


def _generation_stream(args: argparse.Namespace, prompt: str, *, max_tokens: int):
    if args.llama_server_url:
        return stream_with_llama_server(
            prompt,
            args.llama_server_url,
            max_tokens=max_tokens,
            temperature=args.temperature,
        )
    if args.llama_cli_path:
        return stream_with_llama_cli(
            prompt,
            args.llama_cli_path,
            args.model_path,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            max_tokens=max_tokens,
            temperature=args.temperature,
        )
    return stream_with_llama_cpp(
        prompt,
        args.model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        max_tokens=max_tokens,
        temperature=args.temperature,
    )


def _has_generation_backend(args: argparse.Namespace) -> bool:
    if args.llama_server_url:
        return True
    return bool(args.model_path)


def _default_benchmark_output(args: argparse.Namespace) -> str:
    if args.no_llm or not _has_generation_backend(args):
        return "eval/results_retrieval.json"

    model_path = (args.model_path or "").lower()
    if "1.5b" in model_path or "1_5b" in model_path:
        return "eval/results_1.5b.json"
    if "3b" in model_path:
        return "eval/results_3b.json"
    if args.llama_server_url:
        return "eval/results_llama_server.json"
    if args.llama_cli_path:
        return "eval/results_llama_cli.json"
    return "eval/results_llm.json"


def _top_id_hit(results, expected_top_id: str | None) -> bool | None:
    if not expected_top_id:
        return None
    return bool(results and results[0].chunk.id == expected_top_id)


def _expected_hit_at_k(results, expected_chunk_ids: list[str] | None) -> bool | None:
    if not expected_chunk_ids:
        return None
    returned_ids = {result.chunk.id for result in results}
    return any(chunk_id in returned_ids for chunk_id in expected_chunk_ids)


def _refusal_hit(results, expected_refusal: bool | None) -> bool | None:
    if expected_refusal is None:
        return None
    return (len(results) == 0) == expected_refusal


def _ensure_index() -> None:
    if not INDEX_PATH.exists():
        print("Index not found. Run `uv run gigabyte-rag ingest` first.", file=sys.stderr)
        raise SystemExit(2)


def _print_debug(results, retrieval_seconds: float, prompt_estimated_tokens: int) -> None:
    print(f"retrieval_seconds={retrieval_seconds:.4f}")
    print(f"prompt_estimated_tokens={prompt_estimated_tokens}")
    for idx, result in enumerate(results, start=1):
        print(f"[{idx}] score={result.score:.4f} id={result.chunk.id} model={result.chunk.model} section={result.chunk.section}")
        print(result.chunk.text)
        print()


if __name__ == "__main__":
    main()
