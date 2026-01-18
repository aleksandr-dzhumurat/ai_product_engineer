# NLP (LLM)

The **encoder**, **decoder**, and **transformer** are components of the Transformer architecture, which is widely used in natural language processing tasks like translation, summarization, and question answering. Here's how they relate to each other:

---

### 1. **Transformer**

[https://www.youtube.com/watch?v=7fvxOgliYRw](https://www.youtube.com/watch?v=7fvxOgliYRw) - transformer explained

The **Transformer** is the overarching architecture introduced in the paper "Attention is All You Need" by Vaswani et al. It is a model designed to process sequences of data, such as sentences, by focusing on the relationships between elements in the sequence using attention mechanisms.

A full Transformer model typically consists of:

- **Encoder**: Processes the input sequence.
- **Decoder**: Generates the output sequence, often conditioned on the encoder's output.

---

### 2. **Encoder**

The **encoder** is responsible for transforming the input sequence into a meaningful representation that captures its structure and relationships. It processes all input tokens in parallel, allowing for efficient computation.

- **Components of the Encoder**:
    - **Input Embeddings**: Converts input tokens into vectors.
    - **Positional Encoding**: Adds information about the position of tokens in the sequence.
    - **Self-Attention Mechanism**: Allows the encoder to focus on relevant parts of the input sequence while processing each token.
    - **Feedforward Neural Network**: Applies a transformation to the attention outputs for each token.
    - **Layer Normalization** and **Residual Connections**: Help stabilize and improve training.
- **Output**: The encoder produces a sequence of context-aware representations for the input tokens.

---

### 3. **Decoder**

The **decoder** takes the encoder's output and generates the target sequence (e.g., translated text, predictions). It works in an autoregressive manner, processing one token at a time and considering previously generated tokens.

- **Components of the Decoder**:
    - **Input Embeddings**: Converts target tokens into vectors.
    - **Positional Encoding**: Adds positional information to the token embeddings.
    - **Masked Self-Attention**: Ensures the decoder can only attend to tokens generated so far, preserving the autoregressive nature.
    - **Encoder-Decoder Attention**: Allows the decoder to attend to the encoder's outputs, incorporating information from the input sequence.
    - **Feedforward Neural Network**: Processes the combined attention outputs for each token.
    - **Layer Normalization** and **Residual Connections**.
- **Output**: The decoder produces a probability distribution over the vocabulary for the next token, enabling sequential generation.

---

### 4. **Relations Between Encoder, Decoder, and Transformer**


1. **Encoders**
    
    An encoder-based Transformer takes text (or other data) as input and outputs a dense representation (or embedding) of that text.
    
    - **Example**: BERT from Google
    - **Use Cases**: Text classification, semantic search, Named Entity Recognition
    - **Typical Size**: Millions of parameters
2. **Decoders**
    
    A decoder-based Transformer focuses **on generating new tokens to complete a sequence, one token at a time**.
    
    - **Example**: Llama from Meta
    - **Use Cases**: Text generation, chatbots, code generation
    - **Typical Size**: Billions (in the US sense, i.e., 10^9) of parameters

3. **Seq2Seq (Encoder–Decoder)**
    
    A sequence-to-sequence Transformer *combines* an encoder and a decoder. The encoder first processes the input sequence into a context representation, then the decoder generates an output sequence.
    
    - **Example**: T5, BART,
    - **Use Cases**: Translation, Summarization, Paraphrasing
    - **Typical Size**: Millions of parameters

- **Encoder-Decoder Interaction**:
    - The encoder processes the input sequence into a rich, contextual representation.
    - The decoder uses this representation (via encoder-decoder attention) along with its own token history to generate the output sequence.
- **Shared Attention Mechanisms**:
    - Both the encoder and decoder use self-attention to model intra-sequence relationships.
    - The decoder additionally uses encoder-decoder attention to link input and output sequences.
- **Transformer Model**:
    - A **Transformer Encoder** model, like BERT, uses only the encoder to generate representations for tasks like classification and question answering.
    - A **Transformer Decoder** model, like GPT, uses only the decoder to generate text.
    - A **Full Transformer** model, like in machine translation, uses both the encoder and decoder.


While English has an estimated 600,000 words, an LLM might have a vocabulary of around 32,000 tokens (as is the case with Llama 2). Tokenization often works on sub-word units that can be combined.

the difference between a Base Model vs. an Instruct Model:

- *A Base Model* is trained on raw text data to predict the next token.
- An *Instruct Model* is fine-tuned specifically to follow instructions and engage in conversations. For example, `SmolLM2-135M` is a base model, while `SmolLM2-135M-Instruct` is its instruction-tuned variant.

To make a Base Model behave like an instruct model, we need to **format our prompts in a consistent way that the model can understand**. This is where chat templates come in.
---

### Summary Table

| **Component** | **Input** | **Output** | **Attention Mechanisms** | **Usage** |
| --- | --- | --- | --- | --- |
| **Encoder** | Input sequence | Contextual representations | Self-attention | Processing input sequences |
| **Decoder** | Encoder outputs + target tokens | Generated output sequence | Self-attention + Encoder-Decoder attention | Generating output sequences |
| **Transformer** | Full model (Encoder + Decoder) | Task-specific results | Combines both components | Machine translation, text generation |

**BERT (Bidirectional Encoder Representations from Transformers)** is a transformer-based machine learning model designed for natural language processing (NLP) tasks. It was introduced by Google AI in 2018 in the paper [“BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”](https://arxiv.org/abs/1810.04805).

---

### Key Characteristics of BERT:

1. **Transformer-Based Architecture**:
    - BERT is built on the **encoder** part of the transformer architecture.
    - It uses self-attention mechanisms to model relationships between all words in a sentence, capturing both left and right contexts.
2. **Bidirectional Contextual Understanding**:
    - Unlike earlier models like GPT or traditional RNNs that process text left-to-right or right-to-left, BERT processes text bidirectionally.
    - This allows BERT to understand the full context of a word by looking at both its preceding and following words simultaneously.
3. **Pre-training and Fine-tuning**:
    - **Pre-training**: BERT is pre-trained on large amounts of text data using self-supervised learning tasks, such as:
        - **Masked Language Modeling (MLM)**: Randomly masks words in a sentence and trains the model to predict them based on context.
        - **Next Sentence Prediction (NSP)**: Trains the model to understand relationships between sentences by predicting if one sentence follows another.
    - **Fine-tuning**: After pre-training, BERT can be fine-tuned on specific NLP tasks (e.g., classification, named entity recognition, and question answering) with task-specific labeled data.
4. **Model Variants**:
    - **BERT-base**: 12 layers (transformer blocks), 110 million parameters.
    - **BERT-large**: 24 layers, 340 million parameters.
    - Variants like **DistilBERT**, **ALBERT**, and **RoBERTa** offer lighter or optimized versions of BERT.

---

### Applications of BERT:

BERT can be applied to a wide range of NLP tasks, including:

- **Text Classification**: Sentiment analysis, spam detection.
- **Named Entity Recognition (NER)**: Identifying entities like names, dates, and locations in text.
- **Question Answering**: Extracting answers from documents or passages (e.g., SQuAD dataset).
- **Text Summarization**: Generating summaries of long documents.
- **Machine Translation**: Translating text from one language to another.

---

### Advantages of BERT:

1. **Contextual Representations**: Understands the meaning of a word based on its full sentence context.
2. **Pre-trained on Large Data**: Leverages knowledge from extensive pre-training on diverse corpora (e.g., Wikipedia, BooksCorpus).
3. **Versatile**: Can be fine-tuned for a variety of downstream tasks with minimal task-specific changes.

---

### Example Workflow:

1. Pre-training: BERT learns language representations from a large text corpus.
2. Fine-tuning: Tailors BERT for a specific task, such as:
    - Input: Sentence pairs (e.g., "What is BERT?" and "BERT is a transformer model.")
    - Output: Task-specific results, such as classification probabilities or extracted answers.

---

In essence, **BERT revolutionized NLP** by introducing a deep bidirectional transformer architecture, enabling models to achieve state-of-the-art performance across many tasks.

# Attention

**Attention** is a mechanism in machine learning, particularly in natural language processing (NLP) and computer vision, that enables models to focus on specific parts of the input data when making predictions. It is a way to prioritize important elements of the input, improving the model's understanding and performance.

---

### **Key Idea of Attention**

Attention works by assigning different levels of importance, or weights, to various elements of the input. For example, in a sentence, not all words contribute equally to the meaning of a specific word or phrase. Attention allows the model to focus more on the relevant words and less on the irrelevant ones.

---

### **Types of Attention**

1. **Self-Attention (or Intra-Attention)**:
    - Used to relate different parts of the same input sequence.
    - Example: In the sentence "The cat sat on the mat," self-attention helps the model understand that "cat" is the subject of "sat."
    - Self-attention is a key component of the Transformer architecture, including models like BERT and GPT.
2. **Cross-Attention**:
    - Used to relate two different sequences, such as a query and a document.
    - Example: In machine translation, cross-attention helps the model focus on the relevant words in the source sentence while generating the target sentence.

---

### **How Attention Works (High-Level Steps)**

1. **Query, Key, and Value Representations**:
    - Each input element (e.g., a word) is transformed into three vectors:
        - **Query (Q)**: What are we looking for?
        - **Key (K)**: What information does this element have?
        - **Value (V)**: The actual information content.
    - These vectors are learned through training.
2. **Similarity Scores**:
    - The query is compared with all keys in the input sequence to calculate a similarity score (e.g., dot product).
    - This score represents how relevant each input element is to the query.
3. **Attention Weights**:
    - The similarity scores are converted into probabilities using the softmax function.
    - These probabilities (attention weights) indicate the relative importance of each element.
4. **Weighted Sum of Values**:
    - The attention weights are used to compute a weighted sum of the value vectors.
    - This produces a context vector that aggregates relevant information based on the attention mechanism.

---

### **Attention in Transformers**

In the **Transformer** model:

- Self-attention is applied multiple times (multi-head self-attention) to allow the model to focus on different aspects of the input simultaneously.
- This enables the model to capture complex dependencies between words in a sequence, regardless of their position.

---

### **Why is Attention Important?**

1. **Captures Long-Range Dependencies**:
    - Traditional models like RNNs struggle with long sequences. Attention can effectively model relationships between distant elements.
2. **Contextual Understanding**:
    - Attention enables models to understand how words relate to each other in a given context (e.g., "bank" can mean a financial institution or a riverbank depending on the context).
3. **Parallelization**:
    - Attention-based models like Transformers can process all input elements simultaneously, making them faster and more efficient than sequential models like RNNs.

---

### **Real-World Example**

Imagine translating the sentence:
*"She gave him a book because he asked for it."*

- While generating the word "it" in the translated sentence, the model uses attention to determine that "it" refers to "a book."

---

### **Applications of Attention**

1. **Machine Translation**: Focus on relevant parts of a source sentence while generating translations.
2. **Text Summarization**: Identify key sentences or words for concise summaries.
3. **Question Answering**: Focus on relevant parts of a passage to extract answers.
4. **Image Captioning**: Focus on specific regions of an image to generate accurate descriptions.

---

Attention has become a cornerstone of modern machine learning, enabling the success of powerful models like Transformers, BERT, and GPT. Its ability to focus on relevant data has revolutionized tasks across NLP, computer vision, and beyond.

# LLM fine tuning and optimization

[Model inference](https://www.notion.so/Model-inference-1f79c76f79e8808e86c7e30e222a3dc4?pvs=21) 

| Method | Full Name | Purpose / Idea |
| --- | --- | --- |
| **SFT** | Supervised Fine-Tuning | Train on labeled examples to adapt the LLM to a specific task |
| **PEFT** | Parameter-Efficient Fine-Tuning | Fine-tune a small part of the model to save compute and memory |
| **RLHF** | Reinforcement Learning from Human Feedback | Align model with human preferences using reward models |
| **DDP** | Distributed Data Parallel | Parallel training across multiple GPUs (speed up training) |
| **FSDP** | Fully Sharded Data Parallel | Memory-efficient training of large models across GPUs |

[The Ultimate Guide to Fine-Tuning LLMs from Basics to Breakthroughs](https://arxiv.org/html/2408.13296v1)

🎯 Task-Specific Fine-Tuning: Adapts a pre-trained LLM for a specific task by training on relevant data.

📚 Domain-Specific Fine-Tuning: Tailors a LLM to understand and generate text in a specialized field by training on domain-specific data.

🤏 Parameter-Efficient Fine-Tuning (PEFT): Adapts LLMs by only training a small subset of (additional) parameters.

🧊 Low-Rank Adaptation (LoRA): Freezes original weights and applies changes via low-rank matrices.

4️⃣ QLoRA: LoRA extension that uses 4-bit quantization for better memory efficiency.

🧭 Weight-Decomposed Low-Rank Adaptation (DoRA): Fine-tunes models by decomposing weight updates into magnitude and direction, applying LoRA to the latter to achieve full fine-tuning performance.

🌓 Half Fine-Tuning (HFT): balances knowledge retention and new learning by freezing half of an LLM's parameters per fine-tuning round.

🧠 Mixture of Experts (MoE): Combines outputs from specialized subnetworks ("experts") for computation.

🦙 Lamini Memory Tuning: Uses a Mixture of Experts (adapters) as individual memory banks to store specific information and reduce hallucinations.

🤖 Mixture of Agents (MoA): Combines outputs of multiple LLM agents (like MoE but with agents).

🛡️ Proximal Policy Optimisation (PPO): Reinforcement learning method used to train agents to make optimal decisions through reward signals.

🗣️ Direct Preference Optimisation (DPO): Directly aligns LLMs with human preferences using a classification objective on preferred and rejected responses.

✂️ Pruning: Reduces LLM size and complexity by eliminating unnecessary components.

### 📘 1. **SFT — Supervised Fine-Tuning**

- **What it is:** Standard fine-tuning of an LLM on a task using labeled input–output pairs.
- **Use case:** Adapt a base model (like LLaMA or Mistral) to chat, summarize, translate, etc.
- **Example:** Train GPT on `(question, answer)` pairs to create a customer support bot.

---

### ⚙️ 2. **PEFT — Parameter-Efficient Fine-Tuning**

- **What it is:** Techniques that fine-tune only a **small part** of the LLM to save memory and compute.
- **Popular PEFT methods:**
    - **LoRA** (Low-Rank Adaptation): Inject trainable matrices into transformer layers
    - **Prefix Tuning**: Add trainable tokens to the start of input
    - **Adapters**: Add small MLP blocks to layers
- **Use case:** You want to fine-tune a 7B+ model on a laptop or a single A100.

✅ **Pros:** Fast, low-cost, avoids catastrophic forgetting

❌ **Cons:** Less flexible than full fine-tuning for deep changes

---

### 🎯 3. **RLHF — Reinforcement Learning from Human Feedback**

- **What it is:** Align a language model's behavior with human preferences by:
    1. Training a reward model on human rankings
    2. Fine-tuning the LLM using **reinforcement learning** (e.g., PPO)
- **Example:** ChatGPT's alignment phase is done with RLHF.

✅ Produces more aligned and human-like outputs

❌ Requires human-labeled data and careful tuning

---

## 🖥️ Training Strategies for Scale

Now let’s cover **how** we train these models efficiently on big hardware.

---

### 🔁 4. **DDP — Distributed Data Parallel** (from PyTorch)

- **What it is:** A way to **parallelize training across multiple GPUs** by copying the model to each GPU and syncing gradients.
- **Use case:** You want to train a model faster by splitting the batch across 4 or more GPUs.
- Built into PyTorch (`torch.nn.parallel.DistributedDataParallel`)

✅ Simple to set up, good performance

❌ Each GPU holds a full copy of the model (uses more memory)

---

### 💡 5. **FSDP — Fully Sharded Data Parallel** (also PyTorch)

- **What it is:** Like DDP, but **shards model weights and optimizer states** across GPUs, reducing memory usage.
- Use for **very large models** (13B+, 65B, etc.) that can’t fit on one GPU even during training.

✅ Enables training of giant models on fewer GPUs

❌ More complex setup, not always faster for small models

---

## 🔄 How These Work Together

Here’s a typical stack:

- Use **PEFT (e.g., LoRA)** or **SFT** to fine-tune your model
- If the model is big, use **FSDP** or **DDP** to distribute training
- If you want to align with user preferences → use **RLHF**

---

## 📌 Example Workflow

Let’s say you’re adapting LLaMA 13B to customer support:

1. Start with **SFT** on customer chat logs (QA pairs)
2. Use **LoRA (PEFT)** to save memory
3. Train on 4 GPUs using **FSDP** for efficiency
4. Collect human preferences and apply **RLHF** to improve tone

