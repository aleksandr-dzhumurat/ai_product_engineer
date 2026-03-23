# Fine-tuning IBM Granite 4.0 Micro on APIGen-MT-5k via Nebius Serverless

**Goal:** Fine-tune `ibm-granite/granite-4.0-micro` (3B) on the Salesforce `APIGen-MT-5k` multi-turn function-calling dataset using QLoRA on a single NVIDIA H100, via Nebius Serverless Jobs.

> **Important note:** Verify exact CLI flag names and YAML schema fields against the current live documentation at [docs.nebius.com/serverless/quickstart/jobs#cli-2](https://docs.nebius.com/serverless/quickstart/jobs#cli-2) before running in production.

---

## Dataset overview: APIGen-MT-5k

The dataset (`Salesforce/APIGen-MT-5k`) contains **5,000 multi-turn agent trajectories** in ShareGPT format, designed for training function-calling / agentic models. Each record has:

```json
{
  "conversations": [
    { "from": "human",         "value": "user query" },
    { "from": "function_call", "value": "tool arguments (JSON)" },
    { "from": "observation",   "value": "tool result" },
    { "from": "gpt",           "value": "agent response" }
  ],
  "system": "system prompt with domain policy",
  "tools":  "tool descriptions (OpenAI function schema)"
}
```

> **Access requirement:** The dataset is **gated**. You must log in to HuggingFace and accept the usage terms at [huggingface.co/datasets/Salesforce/APIGen-MT-5k](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k) before your `HF_TOKEN` can download it. License: CC-BY-NC-4.0.

---

## 2. Training script (`train.py`)

This script:
- Loads APIGen-MT-5k directly from HuggingFace (gated, requires `HF_TOKEN`)
- Converts the ShareGPT-format conversations to Granite's chat template format, injecting `tools` and `system` fields
- Fine-tunes with QLoRA (4-bit) using TRL's `SFTTrainer`
- Saves the LoRA adapter to the S3-mounted output directory

```python
import os
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

HF_TOKEN   = os.environ["HF_TOKEN"]
MODEL_NAME = os.environ.get("MODEL_NAME", "ibm-granite/granite-4.0-micro")
DATA_PATH  = os.environ.get("DATA_PATH", "Salesforce/APIGen-MT-5k")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/mnt/output")

# ── Load dataset ────────────────────────────────────────────────────────────
# Load from HuggingFace Hub (requires HF_TOKEN with dataset access granted)
raw = load_dataset(DATA_PATH, token=HF_TOKEN, split="train")

# ── Convert ShareGPT → Granite chat-template format ─────────────────────────
ROLE_MAP = {
    "human":         "user",
    "gpt":           "assistant",
    "function_call": "assistant",   # tool-call turn
    "observation":   "tool",        # tool result
}

def sharegpt_to_chat(example):
    """Convert one APIGen-MT record to a list of chat messages."""
    messages = []

    # System prompt (contains domain policy)
    if example.get("system"):
        messages.append({"role": "system", "content": example["system"]})

    for turn in example["conversations"]:
        role    = ROLE_MAP.get(turn["from"], turn["from"])
        content = turn["value"]

        # function_call turns carry JSON tool arguments — wrap them properly
        if turn["from"] == "function_call":
            try:
                tool_args = json.loads(content)
            except json.JSONDecodeError:
                tool_args = {"raw": content}
            messages.append({
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": tool_args}],
                "content": "",
            })
        else:
            messages.append({"role": role, "content": content})

    return {"messages": messages}

dataset = raw.map(sharegpt_to_chat, remove_columns=raw.column_names)

# ── Tokenizer ────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

def format_example(example):
    """Apply Granite's chat template to produce a single training string."""
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

# ── QLoRA / model ────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    # Granite 4.0 Micro uses standard attention projection names
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# ── Training config ──────────────────────────────────────────────────────────
training_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=int(os.environ.get("NUM_EPOCHS", "3")),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,       # effective batch = 16
    learning_rate=float(os.environ.get("LEARNING_RATE", "2e-4")),
    bf16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    max_seq_length=4096,                 # APIGen-MT trajectories can be long
    dataset_text_field=None,             # we use formatting_func instead
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=training_config,
    tokenizer=tokenizer,
    formatting_func=format_example,
)

trainer.train()
trainer.save_model(os.path.join(OUTPUT_DIR, "final-adapter"))
print("Fine-tuning complete. Granite adapter saved to", OUTPUT_DIR)
```

---

## 3. Dockerfile

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN pip install --no-cache-dir \
    transformers==4.44.0 \
    trl==0.9.6 \
    peft==0.12.0 \
    datasets==2.20.0 \
    accelerate==0.33.0 \
    bitsandbytes==0.43.1

COPY train.py /app/train.py
WORKDIR /app
CMD ["python", "train.py"]
```

```bash
docker build -t cr.nebius.cloud/<registry-id>/granite-apigen:v1 .
docker push cr.nebius.cloud/<registry-id>/granite-apigen:v1
```

---

## 4. Job spec (`finetune-job.yaml`)

```yaml
name: granite-micro-apigen-mt-qlora
image: cr.nebius.cloud/<registry-id>/granite-apigen:v1
resources:
  preset: gpu-h100-sxm
  gpuCount: 1
env:
  - name: MODEL_NAME
    value: "ibm-granite/granite-4.0-micro"
  - name: HF_TOKEN
    value: "hf_xxxxxxxxxxxxxxxxxxxxxxxxx"   # ← replace; use secretRef in production
  - name: DATA_PATH
    value: "Salesforce/APIGen-MT-5k"        # loaded from HF Hub, not S3
  - name: LEARNING_RATE
    value: "2e-4"
  - name: NUM_EPOCHS
    value: "3"
  - name: OUTPUT_DIR
    value: "/mnt/output"
volumes:
  - name: model-output
    objectStorage:
      bucket: my-training-data
      prefix: runs/granite-apigen-mt-001/
    mountPath: /mnt/output
command: ["python", "train.py"]
```

---

## GPU selection rationale

| Model | Params | Recommended GPU | Strategy |
|---|---|---|---|
| `granite-4.0-micro` | 3B | 1× H100 (80 GB) | QLoRA 4-bit — fits easily, fast iteration |
| `granite-4.0-tiny-preview` | 7B MoE | 1–2× H100 | QLoRA 4-bit; install transformers from source |
| `granite-4.0-h-1b` | 1B | 1× L40S (48 GB) | QLoRA or even full fine-tune |
| `granite-4.0-h-350m` | 350M | 1× L40S | Full fine-tune possible |
