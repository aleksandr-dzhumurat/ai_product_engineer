# LLM ops

[Youtube|Where Does the Memory Go? (7B)](https://www.youtube.com/watch?v=zSNk0FC3rr8)

- **14 GB** — model weights  
- **14 GB** — gradients (peak)  
- **84 GB** — optimizer states (Adam, mixed precision)  
- **40 GB** — activations


# LLM optimization tools

| Category | Purpose | Examples |
| --- | --- | --- |
| **Memory/Speed Hacks** | Speed up training/inference | FlashAttention, xFormers, DeepSpeed |
| **Efficient Inference** | Serve models faster + at scale | vLLM, TensorRT-LLM, TGI |
| **Compression** | Reduce model size | Quantization (GPTQ, AWQ), Pruning |

## 🔁 1. **FlashAttention** — Faster Attention

- **What:** A memory-efficient implementation of attention using CUDA kernel fusion.
- **Why:** Standard attention scales poorly (`O(n²)` in memory); FlashAttention reduces memory overhead and increases speed.
- **Result:** You can train or run **longer sequences** (e.g., 8K, 32K tokens) on the same hardware.
- **Used in:** GPT-NeoX, LLaMA 2+, Mistral, OpenChat

✅ Speeds up training & inference

✅ Reduces GPU memory consumption

---

## 🚀 2. **vLLM** — Efficient LLM Inference Engine

VLLM explained

- **What:** A high-performance inference engine for LLMs focused on **serving at scale**.
- **Key feature:** **PagedAttention**, which allows dynamic batching of requests with varying lengths, avoiding GPU memory waste.
- **Supports:** HuggingFace models (OPT, LLaMA, Mistral, Falcon, etc.)
- **Use case:** Serve LLMs in production with high throughput.

✅ Up to **10x higher throughput** than naïve HuggingFace pipelines

✅ Plug-and-play with many LLMs

---

## 🧠 3. **Quantization Methods** — Smaller & Faster Models

Quantization reduces the **precision** of weights (from FP32 to INT8, INT4, etc.) to make models run faster and use less memory — often **without a big accuracy loss**.

### Popular Quantization Tools:

| Tool | Description |
| --- | --- |
| **GPTQ** | Post-training quantization, used in LLaMA models |
| **AWQ** | Outlier-aware quantization, reduces degradation |
| **BitsAndBytes** | HuggingFace-compatible 8-bit/4-bit loading |
| **ExLlama / ExLlamaV2** | Fast 4-bit inference on GPUs |

✅ Load 13B+ models on consumer GPUs (e.g., RTX 3090)

✅ Useful for edge deployment, local inference

---

## 🧰 4. **DeepSpeed** — End-to-End Training Optimizer

- **What:** A full training optimization library for LLMs from Microsoft.
- **Features:**
    - **ZeRO Offload**: shard model states across GPUs
    - **Activation checkpointing**: reduce memory
    - **FP16/bfloat16 training**: lower precision = faster training
- **Use case:** Train massive models like Bloom, GPT-J, etc.

✅ Enables model training that would otherwise not fit into memory

---

## 🧱 5. **TensorRT-LLM** (NVIDIA)

- Optimized LLM inference engine that compiles models to **GPU-native kernels**.
- Integrates quantization (INT8, FP8), FlashAttention, and graph optimizations.
- **Highly performant** on NVIDIA hardware (H100, A100, L4, etc.)

---

## 📡 6. **TGI** (Text Generation Inference)

- **What:** HuggingFace’s scalable serving system for LLMs.
- **Includes:** Token streaming, batching, quantization support.
- **Used for:** Open-source LLM deployments like Mistral or Falcon

---

## 💡 When to Use What?

| Use Case | Tool(s) |
| --- | --- |
| Run LLMs on limited hardware | GPTQ, AWQ, BitsAndBytes |
| Fast inference in production | vLLM, TensorRT-LLM, TGI |
| Efficient training on multiple GPUs | DeepSpeed, FlashAttention, FSDP |
| Serve 4-bit quantized models | ExLlama, llama.cpp (GGUF) |

# Model inference

| Формат | Где уместен | Преимущества |
| --- | --- | --- |
| GGUF | Локальный запуск LLM, без GPU | Быстро, мало памяти, легко запускать |
| ONNX | Кросс-фреймворк, production-инфраструктура | Универсальность, поддержка оптимизаций |
| TorchScript | PyTorch-only продакшен | Высокая скорость, вне Python |
| SavedModel | TensorFlow-проекты | Полная поддержка TF-фичей |
| TFLite | Мобильные и IoT | Минимальный размер, низкое потребление |
| OpenVINO IR | Intel устройства, edge inference | Отличная производительность на Intel |

## 🔧 **BentoML**, **LitAPI**, and **LitServe** Explained

These are modern tools in the **LLM deployment ecosystem**—each focused on **model serving**, **scaling**, and **production readiness**. Let’s go through them one by one and then tie them together with your example of using **vLLM** with **LitServe**.

---

### 🥡 **BentoML**

**What it is:**

A **framework for serving ML models** in production. Think of it as a full-service platform to:

- Package ML models (from PyTorch, TensorFlow, HuggingFace, etc.)
- Build **inference APIs**
- Deploy models to **local**, **cloud**, **Kubernetes**, or even **serverless** environments

**Key features:**

- Model packaging (`bentoml.Model`)
- API definition using FastAPI or Starlette-style syntax
- Supports custom runners (for things like Triton or vLLM)
- Scalable serving with containerization and Yatai (its model registry & deployment platform)

📦 *BentoML = model packaging + serving + deployment infrastructure*

---

### ⚡ **LitAPI** & **LitServe** (from **Lightning AI**)

These are **newer, lightweight serving components** built by the creators of PyTorch Lightning.

### 🔹 **LitAPI**

- A **minimal web server framework** to define your model’s HTTP interface.
- You define "endpoints" (`@app.get("/generate")`, etc.), very similar to FastAPI.
- Focused on being fast, lightweight, and production-ready.

### 🔹 **LitServe**

- A **serving layer** that wraps your LitAPI into a **scalable inference server**.
- Supports autoscaling, batching, and request queueing.
- Can integrate with **different inference engines**, like HuggingFace pipelines, Triton, or **vLLM**.

📡 *LitServe = production-grade wrapper for LitAPI apps with scaling & performance optimizations*

---

### 🤖 **vLLM** — the Inference Engine

- Highly optimized **inference engine for LLMs** (like LLaMA, Mistral, etc.).
- Known for:
    - **PagedAttention** – efficient memory usage
    - **Continuous batching** – serving multiple users concurrently
    - Native HuggingFace support

💡 *Think of it like a high-performance engine under the hood, built for fast LLM inference.*

---

## 🔄 Putting It All Together

Here’s what your sentence is saying, with explanation:

> "LitServe is not a replacement of vLLM, but you could use vLLM as the Inference Engine, and LitServe as the Inference Server, serving that engine."
> 
- vLLM does **fast inference** (engine).
- LitServe **wraps it in a scalable HTTP server** (inference API + request routing).
- This means you can:
    - Load your model using `from vllm import LLM`
    - Wrap it with a **LitAPI**
    - Deploy it using **LitServe** for **autoscaling, batching, and endpoint serving**

---

## ✅ Summary

| Component | Role in Pipeline | Example Tool |
| --- | --- | --- |
| Inference Engine | Runs the model fast | `vLLM`, `Transformers`, `Triton` |
| Inference API | Defines HTTP/REST interface | `LitAPI`, `FastAPI`, `Flask` |
| Inference Server | Scales and serves the API | `LitServe`, `BentoML`, `TGI` |
| Deployment | Sends it to cloud / prod | `Lightning`, `BentoML`, `Kubernetes` |

---

Let me know if you want a working code example of wrapping `vLLM` with `LitAPI` + `LitServe`, or a comparison with **BentoML** vs **LitServe**!

Regarding ONNX

| Role | Tool/Format |
| --- | --- |
| **Model Format** | ✅ ONNX, TorchScript, SavedModel |
| **Inference Engine** | ✅ ONNX Runtime, TensorRT, vLLM |
| **API Layer** | ✅ FastAPI, LitAPI, BentoML APIs |
| **Inference Server** | ✅ BentoML, LitServe, TGI, Triton |
| **Deployment Infra** | ✅ Kubernetes, Docker, Lightning AI |

---

# Refs

LoRA fine tuning: [Collab notebook](https://colab.research.google.com/#fileId=https://huggingface.co/agents-course/notebooks/blob/main/bonus-unit1/bonus-unit1.ipynb)

[Designing Data-Intensive applications](https://cloud.mail.ru/public/Xwje/GsKPVbagV)

[DevOps for data science](https://do4ds.com/chapters/intro.html)

[Prompt handbook](https://www.linkedin.com/feed/update/ugcPost:7208896181089808384)