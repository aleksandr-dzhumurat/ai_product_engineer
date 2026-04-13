# vLLM CPU Setup Guide (Ubuntu, Alibaba Cloud ECS)

Tested on: Ubuntu 22.04, 2 vCPUs, 4 GiB RAM, x86_64

---

## 1. Install Prerequisites

### Install `uv` (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.bashrc
```

> If `snap` is available you can also use `snap install astral-uv --classic`.

### Install `jq` (JSON parser, needed to fetch latest version)

```bash
apt install jq
```

### Install TCMalloc (memory allocator for better CPU performance)

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends libtcmalloc-minimal4
```

On **Ubuntu 24.04** the package was renamed — if the above fails:

```bash
sudo apt-get install -y --no-install-recommends libgoogle-perftools4
# or
sudo apt-get install -y google-perftools
```

Then find the actual path:

```bash
find /usr/lib -iname "*tcmalloc*"
```

---

## 2. Create a Python Virtual Environment

```bash
sudo apt-get install -y python3.12-dev
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
```

---

## Docker

```shell
sudo usermod -aG docker $USER
newgrp docker
docker-compose up -d
```

## 3. Install vLLM (GPU Wheel)

Check the Nvidia version

```shell
nvidia-smi
```

Based on available GPU inslall vllm

```shell
uv pip install vllm --torch-backend cu130
```



## 3. Install vLLM (CPU Wheel)

Find the correct CPU wheel URL — note the filename uses `manylinux_2_34`, not `manylinux_2_35`:

```bash
# Check available CPU wheels
curl -s https://api.github.com/repos/vllm-project/vllm/releases?per_page=5 \
  | jq -r '.[].assets[].browser_download_url' | grep cpu
```

Install the latest x86_64 CPU wheel:

```bash
uv pip install "https://github.com/vllm-project/vllm/releases/download/v0.22.1/vllm-0.22.1%2Bcpu-cp38-abi3-manylinux_2_34_x86_64.whl" \
  --torch-backend cpu
```

> **Note:** Replace `0.22.1` with the latest version from the releases page if needed.

---

## 4. Configure LD_PRELOAD (TCMalloc + Intel OpenMP)

Find the library paths:

```bash
sudo find / -iname "*libtcmalloc_minimal.so.4"
sudo find / -iname "*libiomp5.so"
```

Set `LD_PRELOAD` (use the system TCMalloc and venv OpenMP):

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/root/.venv/lib/libiomp5.so
```

To make permanent:

```bash
echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/root/.venv/lib/libiomp5.so' >> ~/.bashrc
source ~/.bashrc
```

---

## 5. Free Up Memory Before Starting

Check what is using RAM:

```bash
ps aux --sort=-%mem | head -11
```

Kill any heavy processes to free RAM (vLLM needs ~2.4 GB free):

```bash
# Example: kill a bot process
pkill -9 -f run_bot.py

# Check free memory after
free -h
```

> You need at least **2 GB available** before loading the model.

---

## 6. Run vLLM

Optimized command for 2 vCPU / 4 GiB machine:

```bash
VLLM_CPU_KVCACHE_SPACE=2 \
VLLM_CPU_OMP_THREADS_BIND=0-1 \
vllm serve Qwen/Qwen3-0.6B \
  --dtype=bfloat16 \
  --max-model-len 512 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 512
```

| Variable / Flag | Value | Purpose |
|---|---|---|
| `VLLM_CPU_KVCACHE_SPACE` | `2` | 2 GB for KV cache |
| `VLLM_CPU_OMP_THREADS_BIND` | `0-1` | Use both vCPUs |
| `--dtype` | `bfloat16` | Best dtype for CPU |
| `--max-model-len` | `512` | Short context saves RAM |
| `--max-num-seqs` | `1` | One request at a time |
| `--max-num-batched-tokens` | `512` | Limits batch memory |

> Model loads from HuggingFace (~1.4 GiB download). First start takes 2–3 minutes.

---

## 7. Verify the Server is Running

Open the firewall port (Alibaba Cloud ECS Security Group: add inbound TCP rule for port 8000), then test:

```bash
# Health check
curl http://<YOUR_SERVER_IP>:8000/health

# List models
curl http://<YOUR_SERVER_IP>:8000/v1/models

# Chat completion
curl http://<YOUR_SERVER_IP>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

Alternatively, use an SSH tunnel to avoid opening the port publicly:

```bash
# On your local machine
ssh -L 8000:localhost:8000 root@<YOUR_SERVER_IP>

# Then curl localhost
curl http://localhost:8000/health
```

---

## Resource Summary

| Component | RAM |
|---|---|
| Qwen3-0.6B weights (bfloat16) | ~1.2 GB |
| KV cache | ~0.2 GB |
| vLLM runtime overhead | ~0.5 GB |
| OS + system processes | ~0.5 GB |
| **Total needed** | **~2.4 GB** |

> **Expected performance:** 5–15 seconds per response on 2 vCPUs. Single request at a time.


# Hardware Notes: H100 80GB + Qwen3-30B-A3B-Instruct-2507

## Instance

| | |
|---|---|
| GPU | 1× NVIDIA H100 80 GB SXM |
| Config file | `configs/vllm/h100.json` |

## Memory Budget

| Component | Size |
|---|---|
| Model weights (30B params × 2 bytes, bfloat16) | ~60 GB |
| GPU memory utilization cap (90% of 80 GB) | 72 GB |
| Available for KV cache | ~12 GB |

## vLLM Config (`configs/vllm/h100.json`)

| Setting | Value | Rationale |
|---|---|---|
| `VLLM_MODEL` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | MoE: 30B total params, 3.3B active per token |
| `VLLM_DTYPE` | `bfloat16` | H100 native BF16 tensor cores; best accuracy/throughput |
| `VLLM_MAX_MODEL_LEN` | `8192` | Conservative: only ~12 GB left for KV cache after weights |
| `VLLM_MAX_NUM_SEQS` | `32` | Each concurrent sequence holds KV cache slots |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `8192` | Matches max context length |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.90` | 10% headroom for CUDA runtime overhead |

## Starting vLLM

```bash
bash scripts/start_vllm.sh --config h100
```

## Tuning Tips

- **OOM on startup** → reduce `VLLM_GPU_MEMORY_UTILIZATION` to `0.85` or `VLLM_MAX_MODEL_LEN` to `4096`
- **Too few concurrent users** → reduce `VLLM_MAX_MODEL_LEN` to free more KV cache for additional sequences
- **Higher throughput needed** → consider `VLLM_DTYPE: float8` (experimental; halves weight memory to ~30 GB, freeing ~30 GB for KV cache and more sequences)


## Starting the Agent Server

The agent FastAPI server runs on port 8001. Start it on the remote host:

```bash
make run-agent-server
# equivalent: uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001
```

Port 8001 is already forwarded by `make tunnel`, so after the tunnel is up the server is reachable locally at **http://localhost:8001**.

Verify:
```bash
curl http://localhost:8001/health
```

> Keep both vLLM (port 8000) and the agent server (port 8001) running in separate `screen` or `tmux` windows. Only restart the agent server when you change agent code — vLLM startup takes ~2.5 min.

---

# Gotchas

## vLLM

### Config & startup

| Symptom | Root cause | Fix |
|---|---|---|
| `POST /chat/completions HTTP/1.1 404` | `VLLM_BASE_URL` in `.env` missing `/v1` suffix; LangChain appends `/chat/completions` literally | Set `VLLM_BASE_URL=http://localhost:8000/v1` |
| OOM on startup | `VLLM_MAX_MODEL_LEN` too large — KV cache reservation exceeds free GPU RAM | Reduce `VLLM_MAX_MODEL_LEN` to `4096`; or reduce `VLLM_GPU_MEMORY_UTILIZATION` to `0.85` |
| Startup takes 2.5 min | vLLM loads all 30B weight shards and pre-allocates KV cache blocks | Normal; only restart when you change vLLM config — keep the process alive across agent code changes |
| vLLM runs on CPU despite H100 present (`cpu_model_runner`, `CPU_ATTN backend`, `device_config=cpu` in logs; `nvidia-smi` shows 0 MiB used) | `uv pip install vllm` resolved a CPU-only torch build; CUDA driver present but `torch.cuda.is_available()` returns `False` | Force-reinstall torch with CUDA: `uv pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall` then restart vLLM |
| `AttributeError: 'Qwen2Tokenizer' object has no attribute 'all_special_tokens_extended'` on startup | `transformers 5.x` removed `all_special_tokens_extended`; vLLM 0.10.x was built against transformers 4.x. Check with `uv run python -c "import transformers; print(transformers.__version__)"` | Pin transformers to 4.x: `uv pip install "transformers>=4.45,<5.0"` then restart vLLM |
| `ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'` on startup (in `utils/hub.py` or `tokenization_utils_tokenizers.py`) | Latest transformers 4.x (4.51+) imports `is_offline_mode` from `huggingface_hub`, but it was removed from `huggingface_hub`'s public `__init__` long ago — both 0.36.x and 1.x lack it. Downgrading `huggingface_hub` only shifts which file fails first. | Pin transformers to the version vLLM 0.10.x was built against: `uv pip install "transformers==4.47.0" "huggingface_hub>=0.23.2,<1.0"`. If the entire 4.x series still fails, upgrade vLLM: change `"vllm>=0.9.0,<0.11"` → `"vllm>=0.11.0,<0.12"` in `pyproject.toml` and reinstall. |

### GPU pre-flight checklist

Run these three commands before starting vLLM. All three must pass.

```bash
# 1. Driver visible
nvidia-smi
# ✓ shows H100 80GB, MiB usage 0 (nothing running yet)
# ✗ "command not found" → install nvidia driver

# 2. CUDA toolkit
nvcc --version
# ✓ shows CUDA 13.x

# 3. PyTorch compiled with CUDA  ← most often wrong
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# ✓ True  12.4  (or any non-None CUDA version)
# ✗ False None  → torch is CPU-only; reinstall (see below)
```

**Fix when step 3 fails:**
```bash
# uv pip install without --force-reinstall silently does nothing if torch is
# already installed (outputs "Checked 1 package in 20ms" and exits).
# Always use --force-reinstall when switching to a CUDA build:
uv pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall

# Verify, then restart vLLM
uv run python -c "import torch; print(torch.cuda.is_available())"  # must be True
bash scripts/start_vllm.sh --config h100
```

**How to confirm vLLM is on GPU from startup logs:**
```
# GPU (correct):
[cuda_model_runner.py:...] Starting to load model...
GPU KV cache size: 123,456 tokens

# CPU (wrong):
[cpu_model_runner.py:...] Starting to load model...
device_config=cpu
Using HND KV cache layout for CPU_ATTN backend.
```

### Throughput & concurrency

| Symptom | Root cause | Fix |
|---|---|---|
| Grafana "waiting" queue grows without bound | Input RPS > achieved RPS; queue grows at rate `(input − achieved)` req/s until load test ends | Increase `VLLM_MAX_NUM_SEQS` + reduce `VLLM_MAX_MODEL_LEN` to fit more sequences in KV cache |
| Grafana "running" line hard-capped below target concurrency | `VLLM_MAX_NUM_SEQS` is the hard ceiling; Grafana green line never exceeds it | Raise `VLLM_MAX_NUM_SEQS` proportionally when reducing `VLLM_MAX_MODEL_LEN` |
| `VLLM_MAX_NUM_BATCHED_TOKENS` equal to `VLLM_MAX_MODEL_LEN` → prefill bottleneck | Only 1 sequence can be in the prefill phase per scheduler step | Set `VLLM_MAX_NUM_BATCHED_TOKENS` ≥ 2× `VLLM_MAX_MODEL_LEN` |
| Load test NaN latency (`ok: 0`) from vLLM side | All requests queued, none completing within the 30 s timeout window | Fix concurrency first; run at lower RPS (`--rps 2`) as sanity check |

### Capacity planning arithmetic

```
max_achievable_RPS ≈ VLLM_MAX_NUM_SEQS / avg_latency_seconds

Example (baseline):
  VLLM_MAX_NUM_SEQS=32, avg_latency≈11s → max ≈ 32/11 ≈ 3 RPS   (SLO needs 10)

Example (tuned):
  VLLM_MAX_NUM_SEQS=64, avg_latency≈11s → max ≈ 64/11 ≈ 6 RPS   (closer; verify empirically)
```

**Why reducing `VLLM_MAX_MODEL_LEN` enables more concurrency:**
KV cache is pre-allocated as `num_blocks × block_size × 2 × num_layers × num_heads × head_dim × dtype_bytes`.
Halving `VLLM_MAX_MODEL_LEN` halves the blocks reserved per sequence → same GPU RAM fits 2× more concurrent sequences.
SQL prompts are 500–2000 tokens; `VLLM_MAX_MODEL_LEN=4096` is safe.

**⚠️ Empirical correction (load test @ 12 RPS, Grafana observation):**
At `MAX_MODEL_LEN=4096`, KV cache stayed at **20–25%** utilization throughout the load test — KV cache is **not** the bottleneck. The dominant latency component (P95 lifecycle breakdown) was **queue wait**, not prefill or decode. Reducing `MAX_MODEL_LEN` to 2048 would change nothing meaningful. The actual ceilings are:
1. **GPU compute throughput** — fixes itself as you reduce incoming RPS to match completion rate
2. **Agent server uvicorn backlog** — at 12 RPS × 12.6 s/req ≈ 150 concurrent connections, the FastAPI server drops connections (`client_errors`); fix with more `--workers` or an async connection pool
3. **Prefix cache hit rate was 80–90%** — system prompt + schema is reused across requests; this is already helping significantly and requires no config change

### Metrics

| Metric | Unit in Prometheus | Grafana expression |
|---|---|---|
| KV cache usage | fraction 0–1 | `vllm:kv_cache_usage_perc * 100` with `"unit": "percent"` |
| Running sequences | count | `vllm:num_requests_running` |
| Waiting sequences | count | `vllm:num_requests_waiting` |
| Throughput | tokens/s | `rate(vllm:prompt_tokens_total[1m])` + `rate(vllm:generation_tokens_total[1m])` |

## Agent server (FastAPI)

| Symptom | Root cause | Fix |
|---|---|---|
| Load test: all `client_errors` at high RPS | Sync `def` endpoint fills uvicorn thread pool (~32 threads); new connections are dropped when pool is exhausted | Make endpoint `async def` + `loop.run_in_executor(None, ...)` |
| Still dropping connections despite async endpoint | Single uvicorn worker; all threads share one pool | Add `--workers 4` (`make run-agent-server` already sets `AGENT_WORKERS=4`) |

## Docker

| Symptom | Root cause | Fix |
|---|---|---|
| `docker-compose up -d` → `PermissionError: [Errno 13] Permission denied` on `/var/run/docker.sock` | Current user is not in the `docker` group | `sudo usermod -aG docker $USER && newgrp docker` (or log out/in); then retry |

## Langfuse

| Symptom | Root cause | Fix |
|---|---|---|
| `AttributeError: 'LangchainCallbackHandler' object has no attribute 'langfuse'` | Newer Langfuse removed the `.langfuse` public attribute from `LangchainCallbackHandler` | Use `@observe(name="agent_run")` + `langfuse_context.update_current_trace()` inside the observed function |
| Tags not visible in Langfuse UI | `langfuse_context` is a Python context variable; LangChain `CallbackHandler` does NOT set it, so calls inside LangGraph nodes are silent no-ops | Set tags from the calling code (server / test script) via `@observe`, not from inside the graph |

---

# Baseline Eval Results (`results/eval_baseline.json`)

30 questions from `evals/eval_set.jsonl`, each run through the full agent (generate → execute → verify → revise loop, cap=3).

## Per-iteration pass rate

| Stop point | Passed | Total | Pass rate |
|---|---|---|---|
| iter 0 — generate_sql only | 11 | 30 | 36.7% |
| iter 1 — after 1 revise | 12 | 30 | 40.0% |
| iter 2 — after 2 revises | 12 | 30 | 40.0% |

Wall clock: **363 s** (~6 min for 30 questions, avg ~12 s/question = ~2 LLM calls each).

## Is the loop earning its keep?

The loop fixed **1 question** (student_club / Art and Design Department names). 36.7% → 40% is a 3.3 pp gain.

Root cause of the flat curve: the **verifier is too permissive**. It accepts semantically wrong answers whenever the SQL runs without error and returns rows that look superficially plausible. Examples:

| Question | Failure mode |
|---|---|
| toxicology carcinogenic % | Used `LIKE '%carcinogenic%'` instead of `label = '+'`; returns a number, verifier OK |
| financial crimes 1995 | Used column `A14` (1994 crimes) instead of `A15` (1995); plausible number, verifier OK |
| formula_1 fastest lap | Returned milliseconds instead of formatted time string; verifier OK |
| thrombosis Ig G / UA normal range | Hardcoded wrong numeric ranges from model knowledge; verifier OK |

The revise loop architecture is correct, but it can only fire when `verify_ok=False`. Stricter verify prompts — explicitly checking that returned column names and value types match the question's intent — would trigger more revise cycles and push the pass rate up.

---

# Test vllm

```shell
curl -s http://localhost:7000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "messages": [{"role": "user", "content": "How many male clients in '\''Hl.m. Praha'\'' district?"}],
    "max_tokens": 200
  }' | python3 -m json.tool
  ```