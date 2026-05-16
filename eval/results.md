# Evaluation Results

Run date：2026-05-12

執行指令：

```powershell
uv run gigabyte-rag ingest --seed
uv run gigabyte-rag bench --questions eval/questions.jsonl --output eval/results.json --no-llm
```

## Quantitative Baseline

這個 baseline 量測純 Python `Retrieval` pipeline 與 deterministic fallback answer。因為 repo 不提交 GGUF model，所以此結果尚未包含真實 `llama.cpp` generation 的 `TTFT` / `TPS`。

| Metric | Result |
| --- | ---: |
| Questions | 6 |
| Average retrieval latency | 0.0119 s |
| Expected chunk Hit@K | 5 / 5 answerable questions |
| Top-1 exact id hit | 2 / 2 questions with strict top-1 labels |
| Refusal cases | 1 / 6 |
| Refusal precision | 1 / 1 out-of-scope questions |
| Prompt token estimate range | 106 - 847 |

`Refusal case` 是詢問 price。官方規格頁沒有提供 price，因此系統在預設 `--min-score 0.65` 下不回傳 context，並拒答。

`Hit@K` 使用 `expected_chunk_ids`，避免三個型號規格完全相同的欄位被誤判。例如 `連接埠` 與 `顯示器` 在 BXH/BYH/BZH 內容相同，任一型號 chunk 或 comparison chunk 都是可接受 evidence。

## Qualitative Analysis

- `Correctness`：GPU、battery、display、ports 等 structured fields 可由 key-value chunks 直接回答。
- `Groundedness`：回答會附上 model / section / source，debug mode 可看到 retrieved chunks。
- `Citation usefulness`：每個 chunk 保留 `source_url`，回答能回指官方規格來源。
- `Multilingual robustness`：alias rerank 讓 `GPU`、`顯示晶片`、`顯卡`、`ports`、`連接埠` 等混合問法能命中同一欄位。
- `Refusal behavior`：低於 `min_score` 的問題不交給 fallback answer 硬答，降低 hallucination。

## Generation Metrics

放入本地 GGUF model 後可執行：

```powershell
uv run gigabyte-rag ask "AORUS MASTER 16 BXH 的 GPU 規格是什麼？" --model-path models/qwen2.5-3b-instruct-q4_k_m.gguf
```

CLI 會輸出：

- `ttft_seconds`
- `total_generation_seconds`
- `output_tokens`
- `tokens_per_second`
- `prompt_estimated_tokens`

建議 model 是 `Qwen2.5-3B-Instruct GGUF Q4_K_M`。若 4GB VRAM 壓力較大，改用 `Qwen2.5-1.5B-Instruct GGUF Q4_K_M`。

## Local Generation Baseline

本機已下載：

```text
models/qwen2.5-3b-instruct-q4_k_m.gguf
models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

3B CPU fallback 測試指令：

```powershell
uv run gigabyte-rag ask "AORUS MASTER 16 BXH 的 GPU 規格是什麼？" --model-filter BXH --model-path models/qwen2.5-3b-instruct-q4_k_m.gguf --n-gpu-layers 0 --max-tokens 96 --temperature 0.1
```

3B 結果：

| Metric | Result |
| --- | ---: |
| retrieval_seconds | 0.0120 s |
| ttft_seconds | 47.4822 s |
| total_generation_seconds | 51.6565 s |
| output_tokens | 92 |
| tokens_per_second | 1.7810 |
| prompt_estimated_tokens | 404 |

1.5B 低記憶體 fallback 測試指令：

```powershell
uv run gigabyte-rag ask "AORUS MASTER 16 BXH 的 GPU 規格是什麼？" --model-filter BXH --top-k 1 --model-path models/qwen2.5-1.5b-instruct-q4_k_m.gguf --n-gpu-layers 0 --max-tokens 96 --temperature 0.1
```

1.5B 結果：

| Metric | Result |
| --- | ---: |
| retrieval_seconds | 0.0125 s |
| ttft_seconds | 1.4441 s |
| total_generation_seconds | 3.4773 s |
| output_tokens | 92 |
| tokens_per_second | 26.4571 |
| prompt_estimated_tokens | 190 |

1.5B GPU offload attempt 測試指令：

```powershell
uv run gigabyte-rag ask "AORUS MASTER 16 BXH 的 GPU 規格是什麼？" --model-filter BXH --top-k 1 --model-path models/qwen2.5-1.5b-instruct-q4_k_m.gguf --n-gpu-layers -1 --max-tokens 96 --temperature 0.1
```

目前環境使用 CPU wheel，因此 `n_gpu_layers=-1` 沒有帶來 GPU acceleration：

| Metric | Result |
| --- | ---: |
| retrieval_seconds | 0.0121 s |
| ttft_seconds | 3.4981 s |
| total_generation_seconds | 5.7378 s |
| output_tokens | 92 |
| tokens_per_second | 16.0341 |
| prompt_estimated_tokens | 190 |

此成績是 CPU wheel baseline，不代表 4GB VRAM GPU offload 的最終速度。面談展示時可用支援 GPU 的 `llama.cpp` build 或 CUDA-enabled `llama-cpp-python` wheel 重跑相同命令，比較 TTFT / TPS 差異。1.5B fallback 適合搭配較小 `top-k` 使用，減少小模型在長 context 下混入非目標欄位。
