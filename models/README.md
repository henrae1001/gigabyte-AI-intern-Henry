# Models

此資料夾只放本機下載的 GGUF model。大型模型檔已被 `.gitignore` 排除，請不要提交到 Git。

## 目前專案使用的模型

4GB VRAM 目標模型：

```text
models/qwen2.5-3b-instruct-q4_k_m.gguf
```

下載：

```bash
uv run gigabyte-rag download-model
```

低記憶體 / CPU-only fallback：

```text
models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

下載：

```bash
uv run gigabyte-rag download-model --model qwen2.5-1.5b-q4_k_m
```

## 使用建議

3B 目標模型：

```bash
uv run gigabyte-rag ask "AORUS MASTER 16 BXH 的 GPU 規格是什麼？" --model-filter BXH --model-path models/qwen2.5-3b-instruct-q4_k_m.gguf --n-gpu-layers 0 --max-tokens 96 --temperature 0.1
```

1.5B fallback 建議縮小 context：

```bash
uv run gigabyte-rag ask "AORUS MASTER 16 BXH 的 GPU 規格是什麼？" --model-filter BXH --top-k 1 --model-path models/qwen2.5-1.5b-instruct-q4_k_m.gguf --n-gpu-layers 0 --max-tokens 96 --temperature 0.1
```

## 4GB VRAM 策略

- 使用 `Q4_K_M` quantization，降低權重大小。
- 使用 `n_ctx=2048` 控制 KV cache。
- 優先嘗試 `n_gpu_layers=-1`；若 VRAM 不足，降低 `n_gpu_layers`。
- Retrieval 與 vector index 在 CPU / RAM 執行，把 VRAM 留給 `llama.cpp` generation。
- 若環境只有 CPU wheel，仍可執行，只是 TTFT / TPS 會較慢。

## 本機 benchmark 摘要

目前記錄在 `eval/results.md`：

| Model | Setting | TTFT | TPS |
| --- | --- | ---: | ---: |
| Qwen2.5-3B Q4_K_M | `top_k=5`, `n_gpu_layers=0` | 47.48 s | 1.78 |
| Qwen2.5-1.5B Q4_K_M | `top_k=1`, `n_gpu_layers=0` | 1.44 s | 26.46 |
| Qwen2.5-1.5B Q4_K_M | `top_k=1`, `n_gpu_layers=-1` on CPU wheel | 3.50 s | 16.03 |

`n_gpu_layers=-1` 這筆是 CPU wheel 環境下的 offload attempt，不代表真正 GPU acceleration 成績。若在有 CUDA-enabled `llama-cpp-python` 或 GPU 版 `llama.cpp` binary 的機器上，請用相同指令重跑。
