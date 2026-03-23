# NLP (LLM)

LLMs generate text Token by token:

* Predict probabilities for every possible next token options (~128 tokens)
* Sample something from this distribution

For each input token we have an embedding. Then vectors are pushed into transformers block to get internal representation for all these tokens.
THen take vector one by one and push to LM head which is a linear transformation (matrix multiply). LM head produces logits.
Softmax transforms logits to probabilities, then sampling happens.
If distribution is flat - hallucinations happens.

What is temperature?

$$\left( \frac{e^{x_1/T}}{\sum_t e^{x_t/T}}, \cdots, \frac{e^{x_V/T}}{\sum_t e^{x_t/T}} \right)$$

If $T$ is small - distribution is sharper (more reliable otput), otherwise it is flattener (more random).


---

The **encoder**, **decoder**, and **transformer** are components of the Transformer architecture, which is widely used in natural language processing tasks like translation, summarization, and question answering. Here's how they relate to each other:

---

**1. Transformer**

[https://www.youtube.com/watch?v=7fvxOgliYRw](https://www.youtube.com/watch?v=7fvxOgliYRw) - transformer explained

The **Transformer** is the overarching architecture introduced in the paper "Attention is All You Need" by Vaswani et al. It is a model designed to process sequences of data, such as sentences, by focusing on the relationships between elements in the sequence using attention mechanisms.

A full Transformer model typically consists of:

The Transformer consists of stacked layers of self-attention and feedforward networks.

Encoder (Processes Input)
1. Multi-Head Self-Attention → Lets each word focus on others.
2. Feedforward Neural Network → Applies non-linear transformations.
3. Layer Normalization & Residual Connections → Helps with training stability.

Decoder (Generates Output)
1. Masked Multi-Head Self-Attention → Prevents attending to future words.
2. Encoder-Decoder Attention → Lets the decoder focus on relevant encoder outputs.
3. Feedforward Neural Network.

RNN processes a sequence step by step → slow, struggles to retain long-range dependencies.

Transformer processes the entire sequence in parallel via the attention mechanism.

```shell
Input
  ↓
[Multi-Head Self-Attention] + Residual connection
  ↓
Layer Norm
  ↓
[Feed-Forward Network] + Residual connection
  ↓
Layer Norm
  ↓
Output
```

**FFN** — two linear layers with an activation function (typically GELU):

$$\text{FFN}(x) = \max(0,\ xW_1 + b_1)W_2 + b_2$$

---

**2. Encoder**

The **encoder** is responsible for transforming the input sequence into a meaningful representation that captures its structure and relationships. It processes all input tokens in parallel, allowing for efficient computation.

- **Components of the Encoder**:
    - **Input Embeddings**: Converts input tokens into vectors.
    - **Positional Encoding**: Adds information about the position of tokens in the sequence.
    - **Self-Attention Mechanism**: Allows the encoder to focus on relevant parts of the input sequence while processing each token.
    - **Feedforward Neural Network**: Applies a transformation to the attention outputs for each token.
    - **Layer Normalization** and **Residual Connections**: Help stabilize and improve training.
- **Output**: The encoder produces a sequence of context-aware representations for the input tokens.

---

**3. Decoder**

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

**4. Relations Between Encoder, Decoder, and Transformer**


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

**Summary Table**

| **Component** | **Input** | **Output** | **Attention Mechanisms** | **Usage** |
| --- | --- | --- | --- | --- |
| **Encoder** | Input sequence | Contextual representations | Self-attention | Processing input sequences |
| **Decoder** | Encoder outputs + target tokens | Generated output sequence | Self-attention + Encoder-Decoder attention | Generating output sequences |
| **Transformer** | Full model (Encoder + Decoder) | Task-specific results | Combines both components | Machine translation, text generation |

## 3. Transformer Architecture: Encoder vs Decoder

### Фундаментальное различие

**BERT (Encoder-only)** vs **GPT (Decoder-only)** — это не просто "разные модели", это разные философии.

#### BERT: Bidirectional Attention

```text
Input: "The cat [MASK] on the mat"

Attention pattern (все видят всё):
     The  cat  [MASK]  on  the  mat
The   ●────●────●────●────●────●
cat   ●────●────●────●────●────●
[MASK]●────●────●────●────●────●  ← Видит ВЕСЬ контекст!
on    ●────●────●────●────●────●
the   ●────●────●────●────●────●
mat   ●────●────●────●────●────●

Предсказание [MASK] использует:
- Левый контекст: "The cat"
- Правый контекст: "on the mat"
```

#### GPT: Causal Attention

```text
Input: "The cat sits on the"

Attention pattern (только прошлое):
         The  cat  sits  on   the
The      ●    ✗    ✗    ✗    ✗
cat      ●────●    ✗    ✗    ✗
sits     ●────●────●    ✗    ✗
on       ●────●────●────●    ✗
the      ●────●────●────●────●

При генерации следующего слова видит только предыдущие!
```

**Когда что использовать:**

| Задача | BERT | GPT |
|--------|------|-----|
| Classification | ✅ Отлично | ⚠️ Можно |
| NER | ✅ Отлично | ❌ Плохо |
| Text Generation | ❌ Невозможно | ✅ Отлично |
| Q&A (extractive) | ✅ Отлично | ⚠️ Можно |
| Q&A (generative) | ❌ | ✅ Отлично |

**BERT (Bidirectional Encoder Representations from Transformers)** is a transformer-based machine learning model designed for natural language processing (NLP) tasks. It was introduced by Google AI in 2018 in the paper [“BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”](https://arxiv.org/abs/1810.04805).

---

**Key Characteristics of BERT:**

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

**Applications of BERT:**

BERT can be applied to a wide range of NLP tasks, including:

- **Text Classification**: Sentiment analysis, spam detection.
- **Named Entity Recognition (NER)**: Identifying entities like names, dates, and locations in text.
- **Question Answering**: Extracting answers from documents or passages (e.g., SQuAD dataset).
- **Text Summarization**: Generating summaries of long documents.
- **Machine Translation**: Translating text from one language to another.

---

**Advantages of BERT:**

1. **Contextual Representations**: Understands the meaning of a word based on its full sentence context.

# Word2Vec: "bank" всегда одинаковый вектор
"I went to the bank" → [0.23, -0.45, 0.67, ...]
"River bank is beautiful" → [0.23, -0.45, 0.67, ...]  # Тот же!

# BERT: "bank" зависит от контекста
```
"I went to the bank" → [0.12, 0.89, -0.34, ...]  # Финансы
"River bank is beautiful" → [-0.45, 0.23, 0.91, ...]  # Берег
```

2. **Pre-trained on Large Data**: Leverages knowledge from extensive pre-training on diverse corpora (e.g., Wikipedia, BooksCorpus).
3. **Versatile**: Can be fine-tuned for a variety of downstream tasks with minimal task-specific changes.

---

**Example Workflow:**

1. Pre-training: BERT learns language representations from a large text corpus.
2. Fine-tuning: Tailors BERT for a specific task, such as:
    - Input: Sentence pairs (e.g., "What is BERT?" and "BERT is a transformer model.")
    - Output: Task-specific results, such as classification probabilities or extracted answers.

---

In essence, **BERT revolutionized NLP** by introducing a deep bidirectional transformer architecture, enabling models to achieve state-of-the-art performance across many tasks.

# Attention

**Attention** is a mechanism in machine learning, particularly in natural language processing (NLP) and computer vision, that enables models to focus on specific parts of the input data when making predictions. It is a way to prioritize important elements of the input, improving the model's understanding and performance.

---

**Key Idea of Attention**

Attention works by assigning different levels of importance, or weights, to various elements of the input. For example, in a sentence, not all words contribute equally to the meaning of a specific word or phrase. Attention allows the model to focus more on the relevant words and less on the irrelevant ones.

---

**Types of Attention**

1. **Self-Attention (or Intra-Attention)**:
    - Used to relate different parts of the same input sequence.
    - Example: In the sentence "The cat sat on the mat," self-attention helps the model understand that "cat" is the subject of "sat."
    - Self-attention is a key component of the Transformer architecture, including models like BERT and GPT.
2. **Cross-Attention**:
    - Used to relate two different sequences, such as a query and a document.
    - Example: In machine translation, cross-attention helps the model focus on the relevant words in the source sentence while generating the target sentence.

---

### Три типа Attention

Многие путают bidirectional attention и cross-attention. Это **разные** механизмы!

#### 1. Self-Attention (Bidirectional) — BERT

```text
Q, K, V все из ОДНОЙ последовательности

Input: "The cat sits"
Query из: "The cat sits"
Key из: "The cat sits"
Value из: "The cat sits"
```

#### 2. Self-Attention (Causal) — GPT

```text
Q, K, V из одной последовательности + causal mask

Треугольная маска запрещает смотреть в будущее
```

#### 3. Cross-Attention — Encoder-Decoder

```text
Query из DECODER
Key, Value из ENCODER (другая последовательность!)

Encoder input: "The cat sits" → Encoder outputs
                                       ↓
Decoder: "Le chat" → Cross-Attention → смотрит на Encoder outputs
```

**Сравнение:**

| Тип | Q из | K, V из | Маска | Модели |
|-----|------|---------|-------|--------|
| Bidirectional Self | Той же seq | Той же seq | Нет | BERT |
| Causal Self | Той же seq | Той же seq | Треугольная | GPT |
| Cross | Decoder | Encoder | Нет | T5, BART |

---

**How Attention Works (High-Level Steps)**

1. **Query, Key, and Value Representations**:
    - Each token (input element, e.g., a word) is transformed into three vectors:
        - **Query (Q)**: What are we looking for?
        - **Key (K)**: What information does this element have?
        - **Value (V)**: The actual information content.
    - These vectors are learned through training.
2. **Similarity Scores**:
    - The query is compared with all keys in the input sequence to calculate a similarity score (e.g., dot product).
    - This score represents how relevant each input element is to the query.

$$\text{score} = \frac{Q \cdot K^\top}{\sqrt{d_k}}$$

3. **Attention Weights**:
    - The similarity scores are converted into probabilities using the softmax function.
    - These probabilities (attention weights) indicate the relative importance of each element.
4. **Weighted Sum of Values**:
    - The attention weights are used to compute a weighted sum of the value vectors.
    - This produces a context vector that aggregates relevant information based on the attention mechanism.

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) \cdot V$$

**Intuition:** Q — "what I'm looking for", K — "what I offer", V — "what I give if I'm chosen".

Compute Attention Scores. We measure the importance of each word **relative to all others** using the **dot product** of the Query and Key vectors:

$$\text{Attention Score} = Q \cdot K^T$$

This gives a matrix where each row represents how much **one word attends to others**.

To normalize the scores into a probability distribution, we apply a **softmax function**:

$$\alpha_{ij} = \frac{\exp(Q_i \cdot K_j^T)}{\sum_k \exp(Q_i \cdot K_k^T)}$$

Now, these values represent the **attention weights** — higher means more focus on that word.

Each word's Value vector is weighted by these attention scores:

$$Z_i = \sum_j \alpha_{ij} V_j$$

This results in a new vector representation for each word, incorporating information from all other words **based on their relevance**.

---

**Attention in Transformers**

In the **Transformer** model:

- Self-attention is applied multiple times (multi-head self-attention) to allow the model to focus on different aspects of the input simultaneously.
- This enables the model to capture complex dependencies between words in a sequence, regardless of their position.

---

**Why is Attention Important?**

1. **Captures Long-Range Dependencies**:
    - Traditional models like RNNs struggle with long sequences. Attention can effectively model relationships between distant elements.
2. **Contextual Understanding**:
    - Attention enables models to understand how words relate to each other in a given context (e.g., "bank" can mean a financial institution or a riverbank depending on the context).
3. **Parallelization**:
    - Attention-based models like Transformers can process all input elements simultaneously, making them faster and more efficient than sequential models like RNNs.

---

**Real-World Example**

Imagine translating the sentence:
*"She gave him a book because he asked for it."*

- While generating the word "it" in the translated sentence, the model uses attention to determine that "it" refers to "a book."

---

**Applications of Attention**

1. **Machine Translation**: Focus on relevant parts of a source sentence while generating translations.
2. **Text Summarization**: Identify key sentences or words for concise summaries.
3. **Question Answering**: Focus on relevant parts of a passage to extract answers.
4. **Image Captioning**: Focus on specific regions of an image to generate accurate descriptions.

---

Attention has become a cornerstone of modern machine learning, enabling the success of powerful models like Transformers, BERT, and GPT. Its ability to focus on relevant data has revolutionized tasks across NLP, computer vision, and beyond.

# Search and Retrieval

## 1. Bi-encoder vs Cross-encoder

### Bi-encoder (двойной энкодер)

Два текста кодируются **независимо** — каждый превращается в вектор, а потом сравниваются через cosine similarity или dot product.

```
Текст A → [Encoder] → вектор A ─┐
                                  ├─ cosine_sim → score
Текст B → [Encoder] → вектор B ─┘
```

**Плюсы:**
- Векторы можно вычислить заранее и сохранить в индекс (FAISS, Annoy)
- Поиск по миллиону документов за миллисекунды
- Масштабируется

**Минусы:**
- Не видит взаимодействие между текстами — контекст одного не влияет на кодирование другого
- Точность ниже, чем у cross-encoder

**Где используется:** семантический поиск, ANN-индексы, первый этап в двухступенчатом пайплайне (retrieval).

---

### Cross-encoder

Оба текста подаются в модель **вместе** — конкатенируются и обрабатываются совместно. Модель видит взаимодействие между ними с первого слоя.

```
[CLS] Текст A [SEP] Текст B [SEP] → [Encoder] → score
```

**Плюсы:**
- Гораздо точнее — attention видит связи между словами обоих текстов
- Учитывает контекст и пересечения смыслов

**Минусы:**
- Нельзя предвычислить — нужно прогонять каждую пару заново
- O(N) операций на запрос → не масштабируется на большие коллекции

**Где используется:** reranking — берём топ-100 от bi-encoder, перебираем их через cross-encoder.

---

### Типичный пайплайн в production

```
Запрос
  ↓
Bi-encoder → ANN-поиск → топ-100 кандидатов
  ↓
Cross-encoder → reranking → топ-10 финальных результатов
```


# 1M context 

2020: GPT-3 → 2K контекст
2023: GPT-4 → 32K контекст (128K extended)
2024: Claude 3 → 200K контекст
2024: Gemini 1.5 → 1M контекст
2025: Gemini 2.0 → 2M контекст (announced)

Для N = 1M токенов:

Память для attention matrix:
1M × 1M × 4 bytes (float32) = 4 TB (только для одного слоя!)

Вычисления:
Query @ Key^T = 1M × 1M умножений
Это нужно делать для КАЖДОГО слоя (например, 80+ слоёв)

На обычном GPU: невозможно!

Gemini НЕ использует "честный" full self-attention O(N²)
Вместо этого: умная комбинация sparse patterns, compression, distribution
Результат: effective 1M контекст, не literal 1M×1M attention

1. ✅ Sparse Attention - не весь контекст со всем
2. ✅ Ring Attention - распределение по устройствам
3. ✅ Flash Attention - эффективные вычисления
4. ✅ MQA/GQA - экономия KV cache
5. ✅ Hierarchical processing - уровни абстракции
6. ✅ KV compression - квантизация, pooling
7. ✅ MoE - selective activation
8. ✅ Огромные вычислительные ресурсы (TPU v5)

Даже с 1M контекста, модель эффективно использует ~10-20K токенов

Остальное:
- Background context
- Reference material
- "На всякий случай"

Но не активно участвует в каждом шаге генерации

## Проблема "lost in the middle":

Модель хуже "помнит" середину длинного контекста
Начало и конец - хорошо
Середина - хуже

Решения:
- Recency bias
- Position embeddings
- Attention boosting для важных частей

Вероятная архитектура

1. Input: 1M токенов

2. Chunking:
   Разбить на чанки по 10K токенов (100 чанков)

3. Local processing (Flash Attention):
   Каждый chunk обрабатывается локально
   Efficient O(chunk_size²) с Flash Attention

4. Hierarchical aggregation:
   Level 1: Local (10K токенов)
   Level 2: Regional (100K токенов)  
   Level 3: Global (1M токенов)

5. Sparse global attention:
   Только между important tokens
   Longformer-style patterns

6. Ring Attention:
   Distributed across many accelerators (TPU v5)
   KV cache rotation между устройствами

7. MQA/GQA:
   Shared K, V для экономии памяти

8. KV cache compression:
   INT4 quantization для старых токенов
   Pooling для distant context

9. MoE routing:
   Разные experts для разных частей контекста


Технологии, которые использует Gemini
1. Sparse Attention (Разреженное внимание)
Идея
Не все токены должны смотреть на все токены. Большинство connections не нужны.
Виды sparse attention
A. Local Attention (Sliding Window)
Каждый токен смотрит только на ближайших соседей

Пример: окно размера W = 512

Token 1000 видит: [500-1500]  (не весь контекст!)

Визуализация:
       Token 0   Token 500  Token 1000  Token 1500
Token 0    ●        -           -           -
Token 500  -        ●           ●           -
Token 1000 -        ●           ●           ●
Token 1500 -        -           ●           ●

Сложность: O(N × W) вместо O(N²)
B. Strided Attention (с пропусками)
Каждый k-й токен

Token 0:    смотрит на [0, 100, 200, 300, ...]
Token 1:    смотрит на [1, 101, 201, 301, ...]

Sparse pattern:
●  -  -  -  ●  -  -  -  ●  -  -  -
-  ●  -  -  -  ●  -  -  -  ●  -  -
-  -  ●  -  -  -  ●  -  -  -  ●  -
C. Block-Sparse Attention
Разбиваем на блоки, attention между блоками

[Block 1] ← → [Block 2] ← → [Block 3]
   ↕              ↕              ↕
[Block 4]     [Block 5]     [Block 6]

Attention только между соседними блоками
D. Longformer-style attention
Комбинация:
1. Local attention (sliding window)
2. Global attention для special tokens ([CLS], important phrases)
3. Dilated attention (с увеличивающимися пропусками)

Attention pattern:
    0   1   2   3   4   5   6   7   8
0   ●   ●   ●   -   -   -   -   -   -   ← Local (window=3)
1   ●   ●   ●   ●   -   -   -   -   -
2   ●   ●   ●   ●   ●   -   -   -   -
3   -   ●   ●   ●   ●   ●   -   -   -
4   ●   -   ●   ●   ●   ●   ●   -   -   ← Global token
5   -   -   -   ●   ●   ●   ●   ●   -
...
Эффект для Gemini
Вместо: 1M × 1M attention
Реально: 1M × W где W << 1M

Если W = 4096:
Память: 1M × 4K = 4B элементов (вместо 1T!)
Сокращение в ~250,000 раз! 🚀

2. Ring Attention (Кольцевое внимание)
Идея от Google (2023)
Разбить длинную последовательность на чанки и обрабатывать их последовательно через кольцо устройств.
Архитектура
Последовательность 1M токенов разбита на чанки по 10K

GPU 1: Chunk 1  [0-10K]
GPU 2: Chunk 2  [10K-20K]
GPU 3: Chunk 3  [20K-30K]
...
GPU 100: Chunk 100 [990K-1M]

Процесс (по кольцу):
Round 1: GPU_i обрабатывает свой chunk с локальным attention
Round 2: Передаём KV-cache соседнему GPU
Round 3: GPU_i обрабатывает чужой chunk
...
После полного круга: каждый chunk "увидел" все остальные
Псевдокод
for round in 1..num_chunks:
    // Local computation
    Q_local = compute_query(chunk_i)
    K_local = compute_key(chunk_i)
    V_local = compute_value(chunk_i)
    
    // Attention с текущими K, V
    attn_output = attention(Q_local, K_current, V_current)
    
    // Rotate K, V to next GPU (ring communication)
    send(K_local, V_local, to=next_gpu)
    K_current, V_current = receive(from=prev_gpu)
Преимущества
✅ Каждый GPU хранит только свой chunk
✅ Полный attention между всеми токенами (не sparse!)
✅ Memory distributed: O(N/num_devices)
✅ Коммуникация efficient (высокая пропускная способность между GPU)
Сложность
Compute: O(N²) всё равно, но distributed
Memory per device: O(N/D) где D = количество устройств
Communication: O(N² / D) - нужна высокая bandwidth

Для 1M токенов на 100 GPU:
Каждый GPU: 10K токенов
Memory per GPU: управляемо!

3. Multi-Query Attention (MQA) и Grouped-Query Attention (GQA)
Проблема стандартного MHA
Multi-Head Attention (MHA):

Каждая head имеет свои Q, K, V

Heads = 32
d_model = 512
d_head = 512 / 32 = 16

Memory для KV cache:
N × heads × d_head × 2 (K и V) = N × 32 × 16 × 2 = N × 1024

Для N = 1M: 1B параметров только для KV cache!
Multi-Query Attention (MQA)
Идея: Разные Q для каждой head, но ОБЩИЕ K и V

Heads = 32
Q: 32 разных матриц
K, V: ОДНА общая матрица

Memory для KV cache:
N × d_head × 2 = N × 16 × 2 = N × 32

Сокращение в 32 раза! 🚀
Grouped-Query Attention (GQA) - компромисс
Группируем heads

Total heads = 32
Groups = 8
Heads per group = 4

Каждая группа имеет общие K, V

Memory для KV cache:
N × groups × d_head × 2 = N × 8 × 16 × 2 = N × 256

Сокращение в 4 раза
Но лучше качество, чем MQA
Используется в Gemini
Gemini, вероятно, использует GQA:
- Меньше памяти для KV cache
- Позволяет caching для длинного контекста
- Качество почти как MHA

4. Flash Attention (оптимизация вычислений)
Проблема стандартного attention
Standard attention (materialized):

1. Compute Q @ K^T → store N×N matrix
2. Softmax → requires full matrix
3. Multiply by V

Peak memory: O(N²)
Flash Attention решение
Идея: Не храним полную attention matrix!

Вычисляем по блокам (tiling):
1. Разбиваем Q, K, V на блоки
2. Для каждого блока Q:
   - Загружаем нужные блоки K, V
   - Вычисляем attention
   - Сразу aggregируем результат
   - Удаляем промежуточные данные
3. Никогда не храним полную N×N матрицу!

Memory: O(N) вместо O(N²) 🚀
Speed: 2-4x faster (меньше memory I/O)
Визуализация
Standard:
Q (N×d) @ K^T (d×N) = Attention Matrix (N×N) ← ОГРОМНО!
    ↓
Softmax
    ↓
@ V (N×d)

Flash Attention (tiled):
For block_i in Q_blocks:
    For block_j in K_blocks:
        attention_block = Q_i @ K_j^T  ← Только блок!
        softmax_block = softmax(attention_block)
        output_i += softmax_block @ V_j
        # Удаляем attention_block из памяти
Для Gemini 1M контекста
Без Flash Attention:
Memory = 1M × 1M × 4 bytes = 4TB ❌

С Flash Attention:
Memory = 1M × block_size × 4 bytes
Если block_size = 256: 1M × 256 × 4 = 1GB ✅

Сокращение в ~4000 раз!

5. Hierarchical / Nested Attention
Идея
Обрабатываем контекст иерархически на разных уровнях абстракции.
Архитектура
Level 1 (Low): Обрабатываем мелкие блоки (512 токенов)
    ↓ Compress
Level 2 (Mid): Обрабатываем сжатые представления (64K токенов)
    ↓ Compress
Level 3 (High): Обрабатываем всю последовательность (1M токенов)

Аналогия: Пирамида изображений
- Низкий уровень: детали (пиксели)
- Средний уровень: регионы (паттерны)
- Высокий уровень: полная картина (контекст)
Пример для текста
Исходный текст: 1M токенов

Level 1: Local attention в блоках по 512 токенов
    "The cat sat on the mat" → embedding_1
    "It was a sunny day"     → embedding_2
    ...

Level 2: Attention между embeddings блоков (2000 блоков)
    [embedding_1, embedding_2, ..., embedding_2000]
    Attention между ними

Level 3: Global representation
    Итоговое понимание всего контекста

При генерации:
- Level 3 даёт общий контекст
- Level 2 даёт релевантные секции
- Level 1 даёт точные детали

6. KV Cache Compression
Проблема
KV cache растёт линейно с длиной последовательности

Standard KV cache для 1M токенов:
K: 1M × d_model
V: 1M × d_model

Огромная память!
Решение: Сжатие KV
A. Quantization (квантизация)
FP16 → INT8 → INT4

FP16: 2 bytes per value
INT4: 0.5 bytes per value

Сжатие в 4 раза! 🚀
B. Pooling старых токенов
Недавние токены: полное разрешение
Старые токены: сжатые

Tokens 0-1000:    сжать в 100 embeddings (pooling)
Tokens 1000-2000: сжать в 100 embeddings
...
Tokens 999000-1M: полное разрешение (1000 tokens)

Total KV: 100 × 999 + 1000 = ~100K вместо 1M
Сжатие в 10 раз!
C. Adaptive selection
Храним только "важные" токены в полном разрешении

Importance scoring:
- Attention weights
- Gradient norms
- Manual markers (headings, keywords)

Top 10% tokens: полное KV
Bottom 90% tokens: сжатое или удалённое

7. Mixture of Experts (MoE)
Связь с длинным контекстом
Не все части модели нужны для всех токенов

MoE позволяет:
- Активировать только релевантные experts для данной части контекста
- Эффективнее использовать память
- Параллелизовать обработку разных частей контекста

Пример:
Tokens 0-100K    (English technical): Expert 1, 3, 5
Tokens 100K-200K (Code Python):       Expert 2, 4, 7
Tokens 200K-1M   (English narrative): Expert 1, 6, 8

Каждый expert работает независимо

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


**📘 1. SFT — Supervised Fine-Tuning**

- **What it is:** Standard fine-tuning of an LLM on a task using labeled input–output pairs.
- **Use case:** Adapt a base model (like LLaMA or Mistral) to chat, summarize, translate, etc.
- **Example:** Train GPT on `(question, answer)` pairs to create a customer support bot.

---

**⚙️ 2. PEFT — Parameter-Efficient Fine-Tuning**

- **What it is:** Techniques that fine-tune only a **small part** of the LLM to save memory and compute.
- **Popular PEFT methods:**
    - **LoRA** (Low-Rank Adaptation): Inject trainable matrices into transformer layers
    - **Prefix Tuning**: Add trainable tokens to the start of input
    - **Adapters**: Add small MLP blocks to layers
- **Use case:** You want to fine-tune a 7B+ model on a laptop or a single A100.

✅ **Pros:** Fast, low-cost, avoids catastrophic forgetting

❌ **Cons:** Less flexible than full fine-tuning for deep changes

---

**🎯 3. RLHF — Reinforcement Learning from Human Feedback**

- **What it is:** Align a language model's behavior with human preferences by:
    1. Training a reward model on human rankings
    2. Fine-tuning the LLM using **reinforcement learning** (e.g., PPO)
- **Example:** ChatGPT's alignment phase is done with RLHF.

✅ Produces more aligned and human-like outputs

❌ Requires human-labeled data and careful tuning

---

**🖥️ Training Strategies for Scale**

Now let’s cover **how** we train these models efficiently on big hardware.

---

**🔁 4. DDP — Distributed Data Parallel** (from PyTorch)

- **What it is:** A way to **parallelize training across multiple GPUs** by copying the model to each GPU and syncing gradients.
- **Use case:** You want to train a model faster by splitting the batch across 4 or more GPUs.
- Built into PyTorch (`torch.nn.parallel.DistributedDataParallel`)

✅ Simple to set up, good performance

❌ Each GPU holds a full copy of the model (uses more memory)

---

**💡 5. FSDP — Fully Sharded Data Parallel** (also PyTorch)

- **What it is:** Like DDP, but **shards model weights and optimizer states** across GPUs, reducing memory usage.
- Use for **very large models** (13B+, 65B, etc.) that can’t fit on one GPU even during training.

✅ Enables training of giant models on fewer GPUs

❌ More complex setup, not always faster for small models

---

**🔄 How These Work Together**

Here’s a typical stack:

- Use **PEFT (e.g., LoRA)** or **SFT** to fine-tune your model
- If the model is big, use **FSDP** or **DDP** to distribute training
- If you want to align with user preferences → use **RLHF**

---

**📌 Example Workflow**

Let’s say you’re adapting LLaMA 13B to customer support:

1. Start with **SFT** on customer chat logs (QA pairs)
2. Use **LoRA (PEFT)** to save memory
3. Train on 4 GPUs using **FSDP** for efficiency
4. Collect human preferences and apply **RLHF** to improve tone


## Knowledge Distillation

Core Idea

A small model (student) learns from a large model (teacher) rather than from hard labels.

```
Teacher (large) → soft predictions (soft labels)
                            ↓
Student (small) ← learns to mimic
```

Train smaller model to mimic larger:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{student}} + (1 - \alpha) \cdot \mathrm{KL}(T_{\text{teacher}}, T_{\text{student}})$$


**Process:**
1. Train student model to mimic teacher's embeddings
2. Loss: MSE(student_embed, teacher_embed) + task_loss

**Result:** Smaller model (e.g., 7B → 400M parameters) with similar quality

**Example:** distilbert-base → 40% smaller, 97% quality

---

Why Soft Labels Are Better Than Hard Labels

Hard label: $[0, 0, 1, 0, 0]$ — carries little information.

Soft label from teacher: $[0.01, 0.02, 0.85, 0.08, 0.04]$ — carries rich information about inter-class similarity. For example: "a cat is more similar to a dog than to an airplane."

---

Temperature

To make soft labels "softer", a temperature $T$ is applied inside the softmax:

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

When $T > 1$, the distribution becomes smoother — the student sees more information about the class structure.

---

Distillation Loss

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{hard}} + (1 - \alpha) \cdot \mathcal{L}_{\text{soft}}$$

$$\mathcal{L}_{\text{hard}} = \text{CrossEntropy}(\text{student logits},\ \text{true labels})$$

$$\mathcal{L}_{\text{soft}} = \text{KL}(\text{student}_{\text{soft}},\ \text{teacher}_{\text{soft}}) \quad \text{(at temperature } T\text{)}$$

$\alpha$ is a hyperparameter (typically $0.1$–$0.5$). After training, temperature is reset to $T = 1$.

---

Types of Distillation

**Response-based** — the student mimics the teacher's final predictions.

**Feature-based** — the student mimics intermediate activations (hidden layers of the teacher).

**Relation-based** — the student learns to reproduce pairwise relationships between examples (distance matrix).

---

Applications

- **DistilBERT** — 97% of BERT's quality at 40% smaller size and 60% faster inference
- **TinyBERT** — distillation at the level of attention matrices
- **Quantization + distillation** — standard production deployment pipeline

---

Distillation ≠ simply training a small model. Identical architectures trained from scratch vs. via distillation yield different quality — distillation is a fully-fledged training method in its own right.


### Self-Attention Mechanism


**Input:** Sequence of vectors X = [x₁, ..., x_n]


**Step 1: Linear Projections**

Create Query, Key, Value representations:
$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

Where W^Q, W^K, W^V ∈ ℝ^{d×d_k}

**Intuition:**
- **Query:** "What am I looking for?"
- **Key:** "What do I contain?"
- **Value:** "What information do I have?"


**Step 2: Scaled Dot-Product Attention**

Attention Scores

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$


**Steps:**
1. Compute attention logits: $QK^T$ (each query $\times$ all keys)
2. Scale by $\sqrt{d_k}$
3. Apply softmax (convert to probabilities)
4. Weighted sum of values

**Why $\sqrt{d_k}$ Scaling:**
- Dot products grow with dimension: $\text{Var}(q \cdot k) = d_k$
- Without scaling: Large $d_k \to$ extreme softmax saturation
- Gradients vanish when softmax outputs $\approx [0, 0, \dots, 1, 0]$

**Complexity:** O(n² × d) where n = sequence length, d = dimension


### Multi-Head Attention


**Motivation:** Different heads capture different relationships (syntax, semantics, position)

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Where:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Parameters:**
- h: Number of heads (typical: 8-16)
- d_k = d_model / h (dimension per head)

**Example Specializations:**
- Head 1: Subject-verb agreement
- Head 2: Semantic similarity
- Head 3: Positional relationships
- Head 4: Coreference resolution

**Benefits:**
- Richer representations
- Ensemble of attention patterns
- Same computation cost (split d_model across heads)

### Positional Encoding


**Problem:** Self-attention is permutation-invariant—needs position information

**Sinusoidal Encoding (Original):**
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

**Properties:**
- Deterministic
- Unique for each position
- Relative position encodable: PE_{pos+k} is linear function of PE_{pos}
- Extrapolates to longer sequences

**Learned Positional Embeddings:**
- Trainable position embeddings (like BERT)
- Can't extrapolate beyond training length
- Often performs better in practice

**Relative Positional Encoding:**
- Encode relative distances (T5, DeBERTa)
- Better length generalization
- More parameter-efficient


### Tokenization

#### Byte Pair Encoding (BPE)

Algorithm:
1. Start with character vocabulary
2. Iteratively merge most frequent pair
3. Repeat until vocabulary size reached

**Example:**
```
Initial: ["l", "o", "w", "e", "r"]
Merge "e"+"r" → "er"
Merge "er"+" " → "er "
...
Final: ["low", "er", " ", "wide", "st"]
```

**Used by:** GPT-2, GPT-3, RoBERTa

#### WordPiece

Difference from BPE: Merges based on likelihood increase, not frequency

**Used by:** BERT, DistilBERT

#### SentencePiece

Key Feature: Language-agnostic, treats whitespace as character

Algorithms: BPE or Unigram LM

Used by: T5, XLNet, multilingual models

**Benefits:**
- No pre-tokenization needed
- Works for languages without spaces
- Reversible

**References:**
- [Attention Is All You Need - Paper](https://arxiv.org/abs/1706.03762)

