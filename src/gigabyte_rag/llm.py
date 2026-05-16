from __future__ import annotations

import time
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from urllib.parse import urljoin

from gigabyte_rag.vector_index import SearchResult


SYSTEM_PROMPT = """You are a precise GIGABYTE laptop hardware assistant.
Answer in the user's language. Traditional Chinese is preferred when the user asks in Chinese.
Use only the provided context. If the context is insufficient, say you cannot confirm it from the official specs.
Keep answers concise and include the cited model/section names."""


@dataclass(frozen=True)
class GenerationMetrics:
    ttft_seconds: float | None
    total_seconds: float
    output_tokens: int

    @property
    def tokens_per_second(self) -> float:
        if self.total_seconds <= 0:
            return 0.0
        return self.output_tokens / self.total_seconds


def build_prompt(question: str, results: list[SearchResult]) -> str:
    context_blocks = []
    for idx, result in enumerate(results, start=1):
        chunk = result.chunk
        context_blocks.append(
            "\n".join(
                [
                    f"[{idx}] model={chunk.model}",
                    f"section={chunk.section}",
                    f"score={result.score:.4f}",
                    chunk.text,
                    f"source={chunk.source_url}",
                ]
            )
        )
    context = "\n\n".join(context_blocks) if context_blocks else "No retrieved context."
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"


def estimate_prompt_tokens(text: str) -> int:
    """Small dependency-free token estimate for debug and benchmark output."""
    ascii_chars = sum(ch.isascii() for ch in text)
    cjk_chars = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    other_chars = max(0, len(text) - ascii_chars - cjk_chars)
    return max(1, round(ascii_chars / 4 + cjk_chars * 1.1 + other_chars / 3))


def stream_with_llama_cpp(
    prompt: str,
    model_path: str,
    *,
    n_ctx: int = 2048,
    n_gpu_layers: int = -1,
    max_tokens: int = 256,
    temperature: float = 0.1,
) -> Iterator[str]:
    from llama_cpp import Llama

    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    stream = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["</s>", "<|eot_id|>", "<|im_end|>"],
        stream=True,
    )
    for event in stream:
        token = event["choices"][0].get("text", "")
        if token:
            yield token


def build_llama_cli_command(
    llama_cli_path: str,
    model_path: str,
    prompt: str,
    *,
    n_ctx: int = 2048,
    n_gpu_layers: int = -1,
    max_tokens: int = 256,
    temperature: float = 0.1,
) -> list[str]:
    return [
        llama_cli_path,
        "-m",
        model_path,
        "-p",
        prompt,
        "-c",
        str(n_ctx),
        "-ngl",
        str(n_gpu_layers),
        "-n",
        str(max_tokens),
        "--temp",
        str(temperature),
        "--no-display-prompt",
    ]


def stream_with_llama_cli(
    prompt: str,
    llama_cli_path: str,
    model_path: str,
    *,
    n_ctx: int = 2048,
    n_gpu_layers: int = -1,
    max_tokens: int = 256,
    temperature: float = 0.1,
) -> Iterator[str]:
    command = build_llama_cli_command(
        llama_cli_path,
        model_path,
        prompt,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None

    try:
        while True:
            piece = process.stdout.read(1)
            if piece:
                yield piece
                continue
            if process.poll() is not None:
                break
    finally:
        if process.poll() is None:
            process.terminate()

    stderr = "" if process.stderr is None else process.stderr.read()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"llama-cli failed with exit code {return_code}: {stderr.strip()}")


def stream_with_llama_server(
    prompt: str,
    server_url: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.1,
) -> Iterator[str]:
    import json

    import httpx

    endpoint = urljoin(server_url.rstrip("/") + "/", "v1/completions")
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stop": ["</s>", "<|eot_id|>", "<|im_end|>"],
    }
    with httpx.stream("POST", endpoint, json=payload, timeout=None) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            token = event.get("choices", [{}])[0].get("text", "")
            if token:
                yield token


def measure_stream(stream: Iterator[str], *, echo: bool = True) -> tuple[str, GenerationMetrics]:
    start = time.perf_counter()
    first_token_at: float | None = None
    output_tokens = 0
    pieces: list[str] = []

    for piece in stream:
        if first_token_at is None:
            first_token_at = time.perf_counter()
        output_tokens += 1
        pieces.append(piece)
        if echo:
            print(piece, end="", flush=True)

    total = time.perf_counter() - start
    if echo:
        print()
    ttft = None if first_token_at is None else first_token_at - start
    return "".join(pieces), GenerationMetrics(ttft, total, output_tokens)


def heuristic_answer(question: str, results: list[SearchResult]) -> str:
    if not results:
        return "無法從官方規格資料確認答案。"
    top = results[0].chunk
    values = "\n".join(line for line in top.text.splitlines() if line.startswith("- "))
    if not values:
        values = top.text
    if _looks_english(question):
        return f"From the official specs, {top.model} / {top.section}:\n{values}\nSource: {top.source_url}"
    return f"根據官方規格，{top.model} 的「{top.section}」為：\n{values}\n來源：{top.source_url}"


def _looks_english(text: str) -> bool:
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    return ascii_letters > cjk
