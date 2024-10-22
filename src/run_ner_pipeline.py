import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch

# CRITICAL: Set this immediately after importing torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

root_data_dir = os.getenv('DATA_DIR', '/srv/data')
cache_dir=os.path.join(root_data_dir, "models")

from huggingface_hub import snapshot_download

print('Download all model files to local folder')
local_folder = snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    cache_dir=cache_dir,
    local_dir=cache_dir  # optional: forces a local folder
)

tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER", cache_dir=cache_dir)
model = AutoModelForTokenClassification.from_pretrained("dslim/distilbert-NER", cache_dir=cache_dir)

print('Model loading started')
ner = pipeline(
    'ner', model=model, tokenizer=tokenizer,
    aggregation_strategy="simple", device='cpu'
)

sample_text = """SoftBank Vision Fund 2 is leading the round, a Series C, with iPod “father” and Nest co-founder Tony Fadell (by way of Future Shape), Blisce, French entrepreneur Xavier Niel, Mirabaud, Cassius and Evolution — all previous backers — also participating. (Previous investors in the company also include DeepMind co-founders Mustafa Suleyman and Demis Hassabis, notable given the company’s early focus on data science and recommendation algorithms.) Prior to this round Dice had raised around $45 million, according to PitchBook estimates."""

print(ner(sample_text)[:5])

from sentence_transformers import SentenceTransformer

print('Loading transformers...')
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_dir, device='cpu')
print('Running encoding')
embeddings = model.encode(["Hello world", "Hi there"], batch_size=32)

print([e.shape for e in embeddings])
