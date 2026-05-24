# GIGABYTE AORUS MASTER 16 AM6H RAG

Lightweight RAG assistant for the GIGABYTE AORUS MASTER 16 AM6H official spec page. It supports Traditional Chinese and English questions, and is designed for resource-limited consumer laptops.

Source: https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp

## Requirements

- Python 3.10-3.12
- `uv`
- Optional: GGUF model for LLM generation

## Step 1. Clone project

```bash
git clone <repo-url>
cd gigabyte-AI-intern-Henry
```

## Step 2. Install uv

Skip this step if `uv` is already installed.

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
$env:Path = "$HOME\.local\bin;$env:Path"
```

Linux / macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

WSL:

```bash
# Run this inside the cloned repository.
pwd
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

If you already installed `uv` in Windows, WSL still needs its own Linux-side `uv` installation.

Check:

```bash
uv --version
```

## Step 3. Install dependencies

```bash
uv sync
```

Check CLI:

```bash
uv run gigabyte-rag --help
```

Note: `uv sync` installs Python dependencies and the `gigabyte-rag` CLI. It does not download GGUF models or create `data/` and `indexes/`.

## Step 4. Build spec data and Vector Index

Recommended for a fresh setup:

```bash
uv run gigabyte-rag ingest --seed
```

Try live official page:

```bash
uv run gigabyte-rag ingest
```

If the official site returns HTTP 403, use `--seed`.

If you saved the official HTML manually, place it here:

```text
data/raw/aorus_master_16_am6h.html
```

`--cached-html` also accepts one `.html` file under `data/raw/`, so the original browser download name is fine if it is the only HTML file in that folder.

Then run:

```bash
uv run gigabyte-rag ingest --cached-html
```

## Step 5. Test Retrieval without model

Traditional Chinese query:

```bash
uv run gigabyte-rag ask "AORUS MASTER 16 BXH 的 GPU 規格是什麼？" --model-filter BXH --debug --no-llm
```

Shorter prompt with only the top retrieved chunk:

```bash
uv run gigabyte-rag ask "AORUS MASTER 16 BXH 的 GPU 規格是什麼？" --model-filter BXH --debug --no-llm --top-k 1
```

This usually keeps only the first `bxh-顯示晶片` chunk, making the prompt shorter.

English query:

```bash
uv run gigabyte-rag ask "What ports are on the right side?" --debug --no-llm
```

Model comparison:

```bash
uv run gigabyte-rag ask "Compare the GPU differences between BXH, BYH, and BZH." --debug --no-llm
```

Refusal test:

```bash
uv run gigabyte-rag ask "Does the official spec mention the laptop price?" --debug --no-llm
```

## Step 6. Download model

4GB VRAM target model:

```bash
uv run gigabyte-rag download-model
```

Output path:

```text
models/qwen2.5-3b-instruct-q4_k_m.gguf
```

Low-memory fallback model:

```bash
uv run gigabyte-rag download-model --model qwen2.5-1.5b-q4_k_m
```

Output path:

```text
models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

GGUF files are ignored by Git.

## Step 7. Run LLM generation

3B target model:

```bash
uv run gigabyte-rag ask "AORUS MASTER 16 BXH 的 GPU 規格是什麼？" --model-filter BXH --model-path models/qwen2.5-3b-instruct-q4_k_m.gguf --n-gpu-layers 0 --max-tokens 96 --temperature 0.1
```

WSL with NVIDIA GPU requires WSL GPU driver support. Check first:

```bash
nvidia-smi
```

If `nvidia-smi` works, try `--n-gpu-layers -1`. If VRAM is not enough, lower `--n-gpu-layers`.

1.5B fallback model with smaller retrieval context:

```bash
uv run gigabyte-rag ask "AORUS MASTER 16 BXH 的 GPU 規格是什麼？" --model-filter BXH --top-k 1 --model-path models/qwen2.5-1.5b-instruct-q4_k_m.gguf --n-gpu-layers 0 --max-tokens 96 --temperature 0.1
```

LLM English ports query:

```bash
uv run gigabyte-rag ask "What ports are on the right side?" --debug --model-path models/qwen2.5-1.5b-instruct-q4_k_m.gguf --max-tokens 96 --temperature 0.1
```

LLM model comparison:

```bash
uv run gigabyte-rag ask "Compare the GPU differences between BXH, BYH, and BZH." --debug --model-path models/qwen2.5-1.5b-instruct-q4_k_m.gguf --max-tokens 128 --temperature 0.1
```

LLM refusal example:

```bash
uv run gigabyte-rag ask \
  "Does the official spec mention the laptop price?" \
  --debug \
  --model-path models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --max-tokens 96 \
  --temperature 0.1
```

Expected behavior: the retrieved context is empty or insufficient, so the model should answer that the price cannot be confirmed from the official specs.

Example WSL CPU output:

```text
retrieval_seconds=0.0314
prompt_estimated_tokens=109
I cannot confirm it from the official specs. The context provided does not contain any information about the laptop price.
{
  "retrieval_seconds": 0.031419492999702925,
  "ttft_seconds": 13.83816276699963,
  "total_generation_seconds": 18.089723839999806,
  "output_tokens": 96,
  "tokens_per_second": 5.306880350916459,
  "prompt_estimated_tokens": 109
}
```

Metrics:

- `retrieval_seconds`
- `ttft_seconds`
- `total_generation_seconds`
- `output_tokens`
- `tokens_per_second`
- `prompt_estimated_tokens`

## Step 8. Run benchmark

Benchmark has two dimensions:

- Data source: `--seed` or `--cached-html`
- Generation mode: `--no-llm` or `--model-path`

Retrieval-only evaluation with seed data:

```bash
uv run gigabyte-rag ingest --seed
uv run gigabyte-rag bench --questions eval/questions.jsonl --no-llm
```

Default output:

```text
eval/results_retrieval.json
```

Retrieval-only evaluation with official cached HTML:

```bash
uv run gigabyte-rag ingest --cached-html
uv run gigabyte-rag bench --questions eval/questions.jsonl --output eval/results_retrieval_html.json --no-llm
```

LLM evaluation with seed data:

```bash
uv run gigabyte-rag ingest --seed
uv run gigabyte-rag bench --questions eval/questions.jsonl --model-path models/qwen2.5-1.5b-instruct-q4_k_m.gguf --max-tokens 96 --temperature 0.1
```

Default output for the command above:

```text
eval/results_1.5b.json
```

LLM evaluation with official cached HTML:

```bash
uv run gigabyte-rag ingest --cached-html
uv run gigabyte-rag bench --questions eval/questions.jsonl --output eval/results_1.5b_html.json --model-path models/qwen2.5-1.5b-instruct-q4_k_m.gguf --max-tokens 96 --temperature 0.1
```

3B model evaluation:

```bash
uv run gigabyte-rag ingest --seed
uv run gigabyte-rag bench --questions eval/questions.jsonl --model-path models/qwen2.5-3b-instruct-q4_k_m.gguf --max-tokens 96 --temperature 0.1
```

Default output:

```text
eval/results_3b.json
```

If `--output` is provided, the CLI always uses the explicit path.

See report:

```text
eval/results.md
```

## Step 9. Run tests

```bash
uv run python -m compileall src
uv run python -m unittest discover -s tests
```

## Model Choice

Primary model:

- `Qwen2.5-3B-Instruct GGUF Q4_K_M`
- `n_ctx=2048`
- Try `n_gpu_layers=-1` on 4GB VRAM; reduce it if VRAM is not enough

Fallback model:

- `Qwen2.5-1.5B-Instruct GGUF Q4_K_M`
- Recommended with `--top-k 1`

Reason: the domain is narrow. Retrieval provides the official spec context, and the SLM only formats the answer. 3B Q4_K_M targets 4GB VRAM. 1.5B Q4_K_M is for CPU-only or safer low-memory runs.

## Troubleshooting

`error: Failed to spawn: gigabyte-rag`:

- Make sure you are in the project root with `pyproject.toml`.
- Run `uv sync` first.
- Check with `uv run gigabyte-rag --help`.

`llama-server: command not found`:

- Main flow does not require `llama-server`.
- `uv sync` does not install `llama-server`.
- Use Step 7 with `--model-path`, or install official `llama.cpp` binary yourself.

`tools/llama-cli.exe` not found:

- Main flow does not require `llama-cli.exe`.
- This repo does not include `llama-cli` / `llama-cli.exe` binary.
- Use Step 7 with `--model-path`, or pass the correct `llama-cli` path yourself.
