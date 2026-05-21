# Evaluation Results

本報告整理三組 benchmark 輸出：

- `eval/results.json`：Retrieval-only baseline，不載入 GGUF model。
- `eval/results_1.5b.json`：`Qwen2.5-1.5B-Instruct GGUF Q4_K_M` generation benchmark。
- `eval/results_3b.json`：`Qwen2.5-3B-Instruct GGUF Q4_K_M` generation benchmark。

所有測試都使用同一組 benchmark questions：`eval/questions.jsonl`。題型涵蓋單一規格查詢、跨型號比較、英文查詢、繁體中文查詢，以及資料來源缺失時的 refusal。

## Summary

| Run | Questions | Avg Retrieval Latency | Hit@K | Strict Top-1 Hit | Refusal Precision | Avg TTFT | Avg TPS | Avg Total Generation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Retrieval-only | 6 | 0.0340 s | 5 / 5 | 2 / 2 | 1 / 1 | N/A | N/A | N/A |
| Qwen2.5 1.5B Q4_K_M | 6 | 0.0390 s | 5 / 5 | 2 / 2 | 1 / 1 | 24.19 s | 3.36 tok/s | 29.90 s |
| Qwen2.5 3B Q4_K_M | 6 | 0.0407 s | 5 / 5 | 2 / 2 | 1 / 1 | 48.20 s | 1.67 tok/s | 58.25 s |

## Retrieval Quality

| Metric | Result |
| --- | ---: |
| Answerable questions | 5 |
| Expected chunk Hit@K | 5 / 5 |
| Strict Top-1 exact id hit | 2 / 2 |
| Out-of-scope questions | 1 |
| Refusal precision | 1 / 1 |
| Prompt token estimate range | 106 - 847 |

Retrieval 表現穩定，五題 answerable questions 都有命中 expected chunks。價格問題屬於官方規格未提供的資訊，系統沒有回傳 context，refusal 判定正確。

## Per-Question Metrics

### Retrieval-only

| ID | Retrieval Latency | Hit@K | Refusal Hit | Prompt Tokens |
| --- | ---: | ---: | ---: | ---: |
| q001 | 0.0350 s | true | N/A | 401 |
| q002 | 0.0384 s | true | N/A | 847 |
| q003 | 0.0339 s | true | N/A | 318 |
| q004 | 0.0356 s | true | N/A | 629 |
| q005 | 0.0311 s | true | N/A | 682 |
| q006 | 0.0301 s | N/A | true | 106 |

### Qwen2.5 1.5B Q4_K_M

| ID | Retrieval Latency | TTFT | Total Generation | Output Tokens | TPS | Hit@K / Refusal |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| q001 | 0.0325 s | 22.96 s | 29.12 s | 94 | 3.23 | Hit |
| q002 | 0.0569 s | 32.83 s | 38.25 s | 96 | 2.51 | Hit |
| q003 | 0.0358 s | 20.19 s | 25.82 s | 95 | 3.68 | Hit |
| q004 | 0.0372 s | 23.68 s | 29.59 s | 96 | 3.24 | Hit |
| q005 | 0.0365 s | 31.49 s | 37.22 s | 95 | 2.55 | Hit |
| q006 | 0.0352 s | 13.96 s | 19.38 s | 96 | 4.95 | Refusal |

### Qwen2.5 3B Q4_K_M

| ID | Retrieval Latency | TTFT | Total Generation | Output Tokens | TPS | Hit@K / Refusal |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| q001 | 0.0365 s | 46.98 s | 56.83 s | 87 | 1.53 | Hit |
| q002 | 0.0600 s | 62.41 s | 72.31 s | 96 | 1.33 | Hit |
| q003 | 0.0348 s | 39.13 s | 48.73 s | 96 | 1.97 | Hit |
| q004 | 0.0351 s | 50.88 s | 60.81 s | 96 | 1.58 | Hit |
| q005 | 0.0398 s | 61.75 s | 72.43 s | 90 | 1.24 | Hit |
| q006 | 0.0382 s | 28.05 s | 38.40 s | 92 | 2.40 | Refusal |

## Qualitative Analysis

### Retrieval

- `Data Parsing` 與 key-value first `Chunking` 適合產品規格表，因為問題通常會對應到明確欄位，例如 `GPU`、`display`、`ports`、`battery`。
- `comparison chunks` 對跨型號問題有效，例如 BXH / BYH / BZH 的 GPU 差異。
- alias rerank 對 bilingual query 有幫助，讓 `GPU`、`顯示晶片`、`顯卡`、`ports`、`連接埠` 等不同問法能命中同一類規格。
- `min_score` refusal 對 out-of-scope question 有效。價格問題沒有 official spec context，因此拒答是正確行為。

### Generation

- 1.5B 的 latency 明顯優於 3B。平均 `TTFT` 約為 3B 的一半，平均 `TPS` 約為 3B 的兩倍。
- 3B 輸出較慢，若在 CPU fallback 環境執行，互動體驗較差。
- 1.5B 較適合作為資源受限環境的 fallback，尤其是 CPU-only 或 VRAM 壓力較大的情境。
- 3B 仍是 4GB VRAM 目標模型候選，但需要 CUDA-enabled `llama.cpp` / `llama-cpp-python` 與適當 `n_gpu_layers` 才能更公平評估。

## Recommended Interpretation

目前瓶頸不是 retrieval，而是 local generation。Retrieval latency 約 0.03 - 0.04 秒，遠低於 LLM generation latency。這代表系統架構合理：CPU-side `Vector Index` 很輕，VRAM / compute budget 可以留給 `llama.cpp` generation。

在 4GB VRAM 的實際部署建議：

- 首選：`Qwen2.5-3B-Instruct GGUF Q4_K_M`，搭配 GPU offload 測試。
- fallback：`Qwen2.5-1.5B-Instruct GGUF Q4_K_M`，適合 CPU-only 或 latency-sensitive demo。
- 保持 `n_ctx=2048`，避免 KV cache 過大。
- 若 prompt 太長，降低 `top-k`，尤其 1.5B model 可優先使用 `--top-k 1`。

## Next Steps

- 在 WSL + NVIDIA GPU 可用環境下重跑 3B benchmark，記錄 GPU offload 後的 `TTFT` / `TPS`。
- 補充 `n_gpu_layers` sweep，例如 `0`、`16`、`32`、`-1`。
- 將 benchmark output 加入硬體資訊，例如 CPU、GPU、RAM、VRAM、WSL distro。
- 針對 generation answer 做人工 correctness label，區分 retrieval hit 與 final answer correctness。
- 調整 prompt，讓 refusal answer 更短，避免模型在沒有 context 時繼續補充不必要內容。
