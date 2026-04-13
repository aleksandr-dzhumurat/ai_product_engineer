# Week 03

## Summary of the Lecture on Neural Networks and Activation Functions

### 1. Introduction to Neural Networks
- Context: The lecture, delivered by Tatiana, a PhD candidate in AI research at Queen Mary University of London, explores the fundamentals of neural networks, their training, and applications to tasks like classification.
- Plan:
  - Build intuition from logistic regression to neural networks.
  - Cover training principles, activation functions, and regularization techniques.
  - Use PyTorch for practical implementation.

---

### 2. From Logistic Regression to Neural Networks
- Logistic Regression Review:
  - Used for binary classification.
  - Formula: $P = \sigma(\mathbf{w}^T \mathbf{x} + b)$, where $\sigma$ is the sigmoid function:
    $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
  - Optimizes parameters $\mathbf{w}, b$ by minimizing binary cross-entropy loss.
- Limitations of Logistic Regression:
  - Assumes a linear decision boundary.
  - Fails on non-linearly separable data (e.g., XOR problem).

- Kernel Methods and Feature Transformation:
  - Idea: Transform inputs to a higher-dimensional space to achieve linear separability.
  - Problem: Kernels are fixed, and hand-picking the optimal kernel is impractical for complex data.

- Neural Network Construction:
  - Stacked Logistic Regressions: Each neuron applies a logistic regression (linear transformation + sigmoid). Hidden layers transform features iteratively.
  - Matrix Formulation: Input $\mathbf{x} \in \mathbb{R}^n$, weights $\mathbf{W} \in \mathbb{R}^{m \times n}$, bias $\mathbf{b} \in \mathbb{R}^m$. Activation: $\mathbf{z} = \mathbf{Wx} + \mathbf{b}$, then apply non-linearity (e.g., sigmoid).
  - Architecture Terminology:
    - Input Layer: Raw features.
    - Hidden Layers: Feature extractors (non-linear transformations).
    - Output Layer: Final classification/regression (e.g., logistic or linear).

---

### 3. Training Neural Networks
- Loss Function:
  - Binary Classification: Cross-entropy loss.
  - Regression: Mean squared error (MSE).
- Gradient Descent: Update rules for weights and biases:
  $$\mathbf{W} \leftarrow \mathbf{W} - \alpha \frac{\partial L}{\partial \mathbf{W}}, \quad \mathbf{b} \leftarrow \mathbf{b} - \alpha \frac{\partial L}{\partial \mathbf{b}}$$
  where $\alpha$ is the learning rate.
- Backpropagation: Computes gradients via the chain rule.
  - Forward Pass: Compute activations layer-by-layer.
  - Backward Pass: Propagate errors from output to input, updating gradients for weights and biases.
- Stochastic Gradient Descent (SGD): Trains on mini-batches to handle large datasets.

---

### 4. Activation Functions
- Purpose: Introduces non-linearity to model complex patterns. Without it, stacked layers reduce to a single linear transformation.

- Sigmoid Function:
  - Formula: $\sigma(z) = \frac{1}{1 + e^{-z}}$
  - Issues: Vanishing gradients (derivatives near 0 for large $|z|$); output range $[0,1]$ leads to saturation.

- Hyperbolic Tangent (tanh):
  - Formula: $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
  - Output range $[-1, 1]$, centering activations around 0.
  - Still suffers from vanishing gradients and computational overhead (exponents).

- Rectified Linear Unit (ReLU):
  - Formula: $\text{ReLU}(z) = \max(0, z)$
  - Advantages: No exponents (fast); derivative is 1 for $z > 0$ (mitigates vanishing gradients).
  - Issues: Dying ReLU problem (neurons may become inactive for $z < 0$); potential exploding gradients.

- Variants of ReLU:
  - Leaky ReLU: $\text{LeakyReLU}(z) = \max(0.01z, z)$, prevents dead neurons.
  - Parametric ReLU (PReLU): Learnable slope for $z < 0$.

- Recommendation: Use ReLU as default for hidden layers, especially in deep networks.

---

### 5. Key Takeaways
- Neural Networks: Stacked non-linear transformations enable modeling of complex decision boundaries.
- Training: Requires careful initialization, backpropagation, and optimization to avoid vanishing/exploding gradients.
- Activation Functions: Sigmoid and tanh are obsolete for hidden layers; ReLU and variants are standard.
- Practical Tips:
  - Normalize inputs to stabilize training.
  - Use softmax for multi-class classification.
  - Address exploding gradients via learning rate tuning, weight initialization, and batch norm.

---

### 6. Next Steps
- Practice: Implement a neural network using PyTorch.
- Topic: Regularization techniques (dropout, batch normalization).

---

## Summary of the Lecture: Building a Fully Connected Neural Network in PyTorch

### Key Concepts and Objectives
- Goal: Construct a fully connected neural network to approximate a cosine function ($\cos(x)$) for a regression task.
- Tools: PyTorch is used for building and training the network.
- Core Idea: A neural network is composed of linear layers ($\mathbf{W}\mathbf{x} + \mathbf{b}$) and activation functions (tanh, ReLU, etc.), stacked to form a computational graph for backpropagation.

---

### PyTorch Building Blocks
1. Linear Layers: Defined via `nn.Linear(in_features, out_features, bias=True)`. Example: `linear_layer = nn.Linear(5, 3)`.
2. Activation Functions: Can be classes (`nn.Tanh()`) or functions (`F.tanh()`). Prefer class-based for sequential models.
3. Tensors and Gradients: Tensors track gradients via `grad_fn`, enabling automatic differentiation.

---

### Network Construction
- Sequential API:
  ```python
  nn.Sequential(
      nn.Linear(5, 5),
      nn.Tanh(),
      nn.Linear(5, 1)
  )
  ```
- Example Architecture: Three layers (1 → 5 → 5 → 1 neuron) with `nn.Tanh()` activations for cosine approximation.

---

### Training Process
1. Loss Function — MSE for regression:
   $$\mathcal{L} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
   Implemented via `nn.MSELoss()`.

2. Optimizer: SGD via `torch.optim.SGD(model.parameters(), lr=0.01)`.

3. Data Handling: Mini-batching with `DataLoader`; move model and data to GPU with `.to(device)`.

4. Training Loop:
   ```python
   for epoch in range(epochs):
       for batch in dataloader:
           preds = model(batch.X)
           loss = criterion(preds, batch.y)
           loss.backward()
           optimizer.step()
           optimizer.zero_grad()
   ```

---

### Regularization: Dropout
- Purpose: Prevent overfitting by randomly "dropping" neurons during training.
- Implementation: `nn.Dropout(p=0.5)` after specific layers.
- Inference Correction: Multiply outputs by `p` during inference to scale down expectations.

---

### Practical Tips and Gotchas
- Shuffling Data: Ensures uniform sampling to avoid bias.
- Batch Size: Trade-off between memory and gradient quality. Larger batches reduce noise but use more memory.
- Debugging: Use `.detach().numpy()` for visualization.

---

### Conclusion and Recommendations
- Model Tuning: Experiment with learning rate, batch size, layer width/depth, activation functions.
- Evaluation: Test on unseen data with uniform distribution.
- Scaling: For large models, rely on empirical scaling laws to extrapolate hyperparameters.

---

# Week 04

## Recurrent Neural Networks (RNNs)

### Introduction to RNNs
- Motivation: Unlike feedforward networks, RNNs process sequential data (text, audio) by maintaining a "memory" of past inputs through hidden states.
- Key Idea: RNNs update their hidden state at each time step:
  $$h_t = f(W h_{t-1} + U x_t + b)$$
  where $h_t$ is the hidden state at time $t$, $x_t$ is the input, and $f$ is an activation function (e.g., tanh).

### RNN Architecture
- Input is converted into vector embeddings (e.g., word2vec) of dimension $D$.
- Hidden states $h_t$ propagate through time, carrying context from previous steps.
- Output: $y_t = V h_t + b_y$.
- Task-Specific Adaptations: For text classification, the final hidden state (or average) is passed to a fully connected layer.

### Training RNNs
- Backpropagation Through Time (BPTT): Gradients computed via chain rule across time steps.
  - Vanishing Gradient Problem: ReLU/sigmoid can exacerbate this.
  - Solution: tanh is preferred for gradient stability.
- Gradient Clipping: Limits gradient magnitudes to prevent exploding gradients.

---

### Tokenization

Converting raw text to numeric tokens is a core preprocessing step. The vocabulary size directly affects model capacity and efficiency.

#### Word-Level Tokenization
Build a vocabulary of frequent words. Unknown words are mapped to a special token (e.g., `<UNK>`). Simple but produces large vocabularies and cannot handle rare words.

#### Character-Level Tokenization
Treats every character as a token. Handles any word but produces very long sequences, leading to inefficiency and difficulty learning long-range dependencies.

#### Byte-Pair Encoding (BPE)
The dominant approach for modern LLMs (GPT-2, GPT-3, RoBERTa).

Algorithm:
1. Start with a character-level vocabulary.
2. Iteratively count all adjacent token pairs.
3. Merge the most frequent pair into a new token.
4. Repeat until the target vocabulary size is reached.

Example:
```
Initial: ["l", "o", "w", "e", "r"]
Merge "e" + "r" → "er"
Merge "er" + " " → "er "
...
Final: ["low", "er", " ", "wide", "st"]
```

Advantages: handles unknown words through subword decomposition; balances vocabulary size with efficiency; flexible for multilingual use.

#### WordPiece
Used by BERT, DistilBERT. Similar to BPE but merges based on the likelihood increase of the language model rather than raw frequency. This tends to produce more linguistically meaningful subwords.

#### SentencePiece
Used by T5, XLNet, and multilingual models.
- Language-agnostic: treats whitespace as an ordinary character, requiring no pre-tokenization.
- Supports both BPE and Unigram LM algorithms.
- Reversible: the original text can be recovered exactly.
- Works for languages without spaces (Chinese, Japanese).

#### Summary

| Method | Vocabulary | Handles OOV? | Used by |
|---|---|---|---|
| Word-level | Large | No (`<UNK>`) | Early NLP |
| Character-level | Small (~256) | Yes | Limited use |
| BPE | ~32K–50K | Yes (subwords) | GPT-2/3, RoBERTa |
| WordPiece | ~30K | Yes | BERT, DistilBERT |
| SentencePiece | Configurable | Yes | T5, XLNet |

While English has an estimated 600,000 words, an LLM might have a vocabulary of around 32,000 tokens (as with Llama 2). Subword tokenization lets the model represent rare words by combining familiar pieces.

---

### Practical Example: RNN Language Model
- Dataset: IMDB movie reviews.
- Preprocessing: Tokenize text with word-level tokenization; add special tokens (`<EOS>`).
- Model Architecture:
  - Embedding Layer: Maps token IDs to dense vectors.
  - LSTM Layers: Stacked recurrent layers for long-term dependencies.
  - Output Layer: Fully connected layer producing logits over vocabulary.
- Training: Cross-entropy loss minimized with Adam; batches padded to align sequence lengths.
- Inference: Generate sequences by sampling (top-k or softmax with temperature) from the model's probability distribution.

### Challenges and Improvements
- Limitations: RNNs struggle with long-range dependencies due to fading memory.
- Solutions: LSTM/GRU with gating mechanisms; bidirectional RNNs; transformers.

### Conclusion
- RNNs provide a foundational approach for sequence modeling but are outperformed by attention-based models in practice.
- Understanding BPE and embedding layers is critical for building effective language models.

---

# Week 05

## Summary: From RNNs to Transformers and Attention in Machine Translation

### Abstract
This lecture transitions from RNNs to attention-based architectures, focusing on machine translation. It covers RNN limitations, encoder-decoder design, the attention mechanism, and the shift toward Transformers.

---

### 1. Machine Translation as Conditional Language Modeling
- Task Definition: Given source sentence in $L_1$, generate target sentence in $L_2$:
  $$Y = \arg\max \sum_{i=1}^m P(y_i | y_1, \dots, y_{i-1}, X, \theta)$$
- Vocabulary: Different vocabularies for source and target. Special tokens BOS/EOS required.

---

### 2. RNN-Based Encoder-Decoder Architecture

#### Encoder
- Objective: Encode the source sentence into a fixed-dimensional context vector $h_E$.
- Implementation: Embed tokens → process through RNN (GRU/LSTM) → pass final hidden state $h_n^E$ to decoder.

#### Decoder
- Objective: Generate target sequence using the context vector and its own hidden states.
- Initialized with $h_0^D = h_n^E$; generates tokens via:
  $$y_t = \arg\max P(y_t | y_1, \dots, y_{t-1}, h_t^D, \theta)$$

#### Training with Teacher Forcing
- During training, ground-truth tokens $y_{t-1}$ are fed as inputs at step $t$ (rather than the decoder's own predictions).
- Avoids error propagation in early training; loss is categorical cross-entropy.

---

### 3. Limitations of RNNs in Machine Translation
1. Fixed Context Vector: All source information compressed into a single vector → information loss for long sequences.
2. Vanishing Gradients: Hard to retain context from early time steps.
3. Sequential Processing: Inefficient for parallel computation.

---

### 4. Attention Mechanism

#### Key Idea
Instead of using only $h_n^E$, the decoder dynamically weighs all encoder hidden states $h_i^E$ at each decoding step — allowing it to "look back" at the entire source at every generation step.

#### How Attention Works

Each token is transformed into three learned vector representations:

- **Query (Q)** — "What am I looking for?"
- **Key (K)** — "What do I contain / what do I offer?"
- **Value (V)** — "What information do I actually give if chosen?"

This separation gives the model flexibility: what a token looks for does not have to be the same as what it offers, which does not have to be the same as what it contributes.

**Step 1 — Compute alignment scores:**
A function $f_\text{align}(h_t^D, h_i^E)$ scores how relevant encoder state $i$ is to decoding step $t$. This can be a trainable network or a simple dot product:
$$\text{score} = \frac{Q \cdot K^\top}{\sqrt{d_k}}$$
The $\sqrt{d_k}$ scaling prevents extreme softmax saturation when dimension is large.

**Step 2 — Normalize to attention weights:**
$$\alpha_i = \text{softmax}_i\!\left(f_\text{align}(h_t^D, h_i^E)\right), \quad \alpha_{ij} = \frac{\exp(Q_i \cdot K_j^T)}{\sum_k \exp(Q_i \cdot K_k^T)}$$

**Step 3 — Compute the context vector (weighted sum of values):**
$$c_t = \sum_i \alpha_i h_i^E, \qquad Z_i = \sum_j \alpha_{ij} V_j$$

The full scaled dot-product attention formula:
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) \cdot V$$

#### Implementation Variants
- **Concatenation Attention:** Context vector is concatenated to decoder input:
  $$\tilde{x}_t = [x_t \| c_t]$$
- **Dot Product Attention:** $\alpha_i = \text{softmax}_i(h_t^D \cdot h_i^E)$ — no learned parameters.

#### Impact
- Resolves the fixed-context problem; enables accurate modeling of long sequences.
- Example: when generating "morning" in English, the decoder attends to "Morgen" in German.
- Attention weights are directly visualizable as alignment maps between source and target tokens.

#### Why Attention Matters
1. **Long-range dependencies:** Models relationships between distant elements without relying on sequential memory.
2. **Contextual understanding:** "bank" meaning depends on context — attention captures which other tokens are relevant.
3. **Parallelization:** Attention-based models process all elements simultaneously, unlike sequential RNNs.

#### Real-World Example
Translating "She gave him a book because he asked for it" — when generating "it", attention identifies "a book" as the referent by attending to the correct source token.

---

### 5. Transition to Transformers
- The attention mechanism decouples sequential processing from dependency modeling.
- Transformers replace RNNs entirely, using self-attention and parallel computation.
- Key Components: Multi-Head Attention, positional encodings.
- Advantages: Scalability, parallelization, superior performance on long-range dependencies.

---

### 6. Practical Implementation: RNN with Attention for Machine Translation
- Dataset: Synthetic date-formatting task ("March 13, 2023" → "2023-03-13").
- Steps:
  1. Generate source/target pairs; tokenize; encode as token indices.
  2. Encoder: RNN with embedding and GRU layers. Decoder: RNN with attention + linear layer.
  3. Train with teacher forcing; minimize cross-entropy via gradient descent.
  4. Inference: Greedy/beam search until EOS.

---

### Mathematical Formulation Summary

$$P(Y|X) = \prod_{i=1}^m P(y_i | y_1, \dots, y_{i-1}, X)$$
$$\alpha_i = \text{softmax}_i(f_\text{align}(h_t^D, h_i^E))$$
$$c_t = \sum_i \alpha_i h_i^E$$

---

# Week 06

## Overview

This week covers the full Transformer architecture, how LLMs generate text, the BERT vs GPT distinction, attention types, efficient fine-tuning, knowledge distillation, search/retrieval architectures, large-context techniques, and training-time regularization.

---

## 1. Introduction to Transformers

- Motivation: RNNs suffer from sequential computation, vanishing gradients, and poor GPU utilization. Transformers address all three by enabling parallel processing and replacing recurrence with self-attention.
- Core Ideas:
  - Attention mechanisms are generalized to **self-attention**: every token can attend to every other token in the input simultaneously.
  - The Transformer architecture (2017, "Attention is All You Need") replaces RNNs entirely, using stacked encoder-decoder blocks.

```
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

Feed-Forward Network — two linear layers with activation (typically GELU):
$$\text{FFN}(x) = \max(0,\ xW_1 + b_1)W_2 + b_2$$

RNN processes a sequence step by step → slow, struggles with long-range dependencies.  
Transformer processes the entire sequence in parallel via attention.

---

## 2. Self-Attention: Mathematical Deep Dive

### Linear Projections

Given input sequence $X = [x_1, \ldots, x_n]$, create Query, Key, Value representations:
$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$
where $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$.

The model learns three separate roles for each token. Naively doing token-vs-token dot products directly would work, but separate projections give the model flexibility — what a token looks for (Q) can differ from what it offers (K) and what it contributes (V).

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Steps:
1. Compute attention logits: $QK^T$ (each query × all keys).
2. Scale by $\sqrt{d_k}$ — dot products grow with dimension: $\text{Var}(q \cdot k) = d_k$; without scaling, large $d_k$ causes extreme softmax saturation and gradient vanishing.
3. Apply softmax (convert to probabilities).
4. Weighted sum of values.

Complexity: $O(n^2 \times d)$ where $n$ = sequence length, $d$ = dimension.

### Three Types of Attention

Many people conflate these — they are distinct mechanisms:

#### Bidirectional Self-Attention (BERT-style)
Q, K, V all come from the **same sequence**; no mask applied. Every token attends to every other token.

```
Input: "The cat [MASK] on the mat"
     The  cat  [MASK]  on  the  mat
The   ●────●────●────●────●────●
cat   ●────●────●────●────●────●
[MASK]●────●────●────●────●────●  ← sees FULL context
on    ●────●────●────●────●────●
```

#### Causal Self-Attention (GPT-style)
Q, K, V from the same sequence + a triangular mask that prevents attending to future positions.

```
Input: "The cat sits on the"
         The  cat  sits  on  the
The      ●    ✗    ✗    ✗   ✗
cat      ●────●    ✗    ✗   ✗
sits     ●────●────●    ✗   ✗
on       ●────●────●────●   ✗
the      ●────●────●────●───●
```

#### Cross-Attention (Encoder–Decoder)
Query comes from the **decoder**; Key and Value come from the **encoder** output (a different sequence). This is the only place encoder and decoder "talk" to each other.

```
Encoder input: "The cat sits" → encoder outputs
                                        ↓
Decoder: "Le chat" → Cross-Attention ← looks at encoder outputs
```

| Type | Q from | K, V from | Mask | Models |
|---|---|---|---|---|
| Bidirectional Self | Same seq | Same seq | None | BERT |
| Causal Self | Same seq | Same seq | Triangular | GPT, LLaMA |
| Cross | Decoder | Encoder | None | T5, BART |

---

## 3. Multi-Head Attention

### Motivation
A single attention head can only capture one type of relationship at a time. Multiple heads let the model attend to different things simultaneously — syntactic, semantic, positional, coreference, etc.

### Architecture

Each head gets its own $W^Q_i, W^K_i, W^V_i$ — it projects Q, K, V into a different subspace and runs attention independently:

```
head_i = Attention(Q·W_i^Q,  K·W_i^K,  V·W_i^V)
```

Outputs are concatenated and projected:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

Parameters:
- $h$: number of heads (typical: 8–16)
- $d_k = d_\text{model} / h$ (dimension per head)
- Same total computation cost as single-head (dimension split across heads)

### Example Head Specializations
For "A cute teddy bear is reading":
- Head 1 — syntactic relationships (subject–verb)
- Head 2 — semantic similarity ("cute" → "teddy bear")
- Head 3 — positional proximity
- Head 4 — coreference resolution

---

## 4. Transformer Encoder Block

Structure:
1. **Self-Attention:** Each token's embedding attends to all others, capturing contextual information.
2. **Residual Connection + LayerNorm:** $x_i = x_i^\text{embed} + \text{Attention}(x_i^\text{embed})$. LayerNorm stabilizes training by normalizing per-token embeddings.
3. **Feed-Forward Network (FFN):** Position-wise MLP with nonlinearities.
4. **Repeat:** Stacked encoder blocks propagate contextualized embeddings.

---

## 5. Transformer Decoder Block

Differs from encoder by including:
1. **Masked Self-Attention:** Prevents attending to future tokens during autoregressive generation.
2. **Cross-Attention:** Decoder attends to encoder outputs (relevant for translation, summarization).
3. **FFN:** Same as encoder.

---

## 6. Positional Encoding

Problem: Self-attention is permutation-invariant — it has no inherent notion of order.

### Fixed Sinusoidal Encodings (Original Transformer)
$$PE_{(pos,\, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos,\, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$
where $pos$ is position, $d$ is embedding dimension.

Properties:
- Deterministic, unique per position.
- Relative position encodable: $PE_{pos+k}$ is a linear function of $PE_{pos}$.
- Extrapolates to sequences longer than training length.

### Learned Positional Embeddings (BERT-style)
- Trainable position embeddings; cannot extrapolate beyond training length.
- Often performs better in practice for fixed-length tasks.

### Relative Positional Encoding (T5, DeBERTa)
- Encode relative distances rather than absolute positions.
- Better length generalization; more parameter-efficient.

---

## 7. Encoder vs Decoder Models — Architecture Philosophy

The encoder and decoder are not just "parts of a model" — they represent different design philosophies:

### Encoder-Only (e.g., BERT)
- Bidirectional attention — every token sees the full context.
- Pre-trained with Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).
- Output: dense contextual representations (embeddings), not generated text.
- Typical size: hundreds of millions of parameters.
- Use cases: text classification, NER, extractive QA, semantic search.

### Decoder-Only (e.g., GPT, LLaMA)
- Causal attention — each token only sees previous tokens.
- Pre-trained to predict the next token.
- Output: generated sequences, one token at a time.
- Typical size: billions of parameters.
- Use cases: text generation, chatbots, code generation.

### Encoder–Decoder / Seq2Seq (e.g., T5, BART)
- Encoder processes the full source; decoder generates the target attending to encoder outputs via cross-attention.
- Use cases: translation, summarization, paraphrasing.

| Component | Input | Output | Attention | Usage |
|---|---|---|---|---|
| Encoder | Input sequence | Contextual representations | Bidirectional self | Processing input |
| Decoder | Encoder outputs + target tokens | Generated output | Causal self + cross | Generating output |
| Full Transformer | Both | Task-specific | All three types | Translation, generation |

| Task | BERT | GPT |
|---|---|---|
| Classification | ✅ Excellent | ⚠️ Possible |
| NER | ✅ Excellent | ❌ Poor |
| Text Generation | ❌ Impossible | ✅ Excellent |
| Extractive Q&A | ✅ Excellent | ⚠️ Possible |
| Generative Q&A | ❌ | ✅ Excellent |

---

## 8. BERT Deep Dive

BERT (Bidirectional Encoder Representations from Transformers) was introduced by Google AI in 2018.

### Key Characteristics
1. **Transformer-Based Architecture:** Built on the encoder; uses bidirectional self-attention.
2. **Bidirectional Context:** Understands each word by looking at both preceding and following words simultaneously — unlike GPT (left-to-right) or traditional RNNs.
3. **Pre-training Tasks:**
   - **MLM (Masked Language Modeling):** Randomly masks words; trains the model to predict them from context.
   - **NSP (Next Sentence Prediction):** Trains the model to determine if two sentences are consecutive.
4. **Fine-Tuning:** After pre-training, BERT is fine-tuned on specific tasks with labeled data.

### Contextual vs. Static Embeddings
```
# Word2Vec: "bank" always the same vector
"I went to the bank"     → [0.23, -0.45, 0.67, ...]
"River bank is beautiful"→ [0.23, -0.45, 0.67, ...]  # Identical!

# BERT: "bank" depends on context
"I went to the bank"     → [0.12, 0.89, -0.34, ...]  # Finance
"River bank is beautiful"→ [-0.45, 0.23, 0.91, ...]  # Geography
```

### Model Variants
- BERT-base: 12 transformer blocks, 110M parameters.
- BERT-large: 24 layers, 340M parameters.
- DistilBERT, ALBERT, RoBERTa: lighter or optimized versions.

### Applications
Text classification, NER, extractive QA (SQuAD), summarization, machine translation.

---

## 9. LLM Training Pipeline: From Pre-Training to Alignment

### 9.1 Pre-Training

**Objective:** LLMs are pre-trained on the *next-token prediction* task using large unlabeled text corpora, minimizing cross-entropy loss over each predicted token.

**What the model learns:**
- Broad language knowledge, grammar, world facts.
- Multilingual ability (if trained on multilingual data).
- Basic reasoning and coding skills (if data includes code).
- **In-context learning** — the ability to perform tasks like translation or factual Q&A from examples in the prompt, without any gradient updates.

**Multi-Stage Pre-Training Pipeline:**

| Stage | Sequence Length | Goal |
|---|---|---|
| Initial pre-training | 2K–4K tokens | Foundational language knowledge |
| Long-context continuing pre-training | Up to millions of tokens | Improve positional encoding robustness, attention over long inputs |

The second stage extends the model's effective context window without restarting training from scratch.

---

### 9.2 How Text Generation Works

LLMs generate text token by token:
1. For each input token, create an embedding vector.
2. Push all vectors through transformer blocks to get internal representations.
3. Take the last vector and push through the **LM head** — a linear transformation (matrix multiply) that produces **logits** over the vocabulary (~128K tokens for modern models).
4. **Softmax** transforms logits to probabilities.
5. **Sample** from this distribution.

If the distribution is too flat → hallucinations. If too sharp → repetitive output.

### Temperature

Temperature $T$ controls the sharpness of the output distribution:
$$\left(\frac{e^{x_1/T}}{\sum_t e^{x_t/T}},\ \cdots,\ \frac{e^{x_V/T}}{\sum_t e^{x_t/T}}\right)$$

- Small $T$ → sharper distribution (more reliable, less creative).
- Large $T$ → flatter distribution (more random, more creative).

### Base Model vs. Instruct Model
- **Base Model:** Trained on raw text to predict the next token. (e.g., `SmolLM2-135M`)
- **Instruct Model:** Fine-tuned to follow instructions and engage in conversations. (e.g., `SmolLM2-135M-Instruct`)
- Chat templates are used to format prompts consistently for instruct models.

**Standard LLM post-training pipeline:**
```
Pre-training (next-token prediction on raw text)
  ↓
Supervised Fine-Tuning / SFT (instruction following)
  ↓
Preference Alignment (RLHF or LVR)
  ↓
Deployed instruct model
```

---

## 10. Efficient Fine-Tuning

Full fine-tuning of large LLMs updates every weight — prohibitively expensive in memory and compute. Parameter-efficient methods adapt only a small subset of parameters.

### Adapter Layers
Insert small trainable bottleneck modules between pre-trained layers. The original weights stay frozen; only the adapter learns the task-specific transformation. Reduces trainable parameters significantly but can risk slight information loss through the bottleneck.

### LoRA (Low-Rank Adaptation)
- **Idea:** For a frozen weight matrix $W \in \mathbb{R}^{D \times D}$, inject two low-rank matrices $A \in \mathbb{R}^{D \times R}$ and $B \in \mathbb{R}^{R \times D}$ (with $R \ll D$). The adapted forward pass becomes:
  $$\text{output} = Wx + ABx$$
- Only $A$ and $B$ are trained, requiring $\sim 2DR$ parameters instead of $D^2$.
- **No inference overhead:** after training, $AB$ is merged into $W$, so the deployed model is the same size.
- **Typical application:** Applied to the key and value projection matrices inside self-attention layers. Effective for style transfer, domain adaptation, and task specialization.

### Bias Tuning
Freeze all weights; train only bias terms. Reduces trainable parameters to less than 1% of the full model. Weakest of the three methods but extremely cheap.

### Transfer Learning (General Pattern)
The same principle applies outside LLMs. In image classification, pre-train on ImageNet (1000 classes), then fine-tune only the final linear layer for a target task (e.g., 10 classes). Freezing early layers preserves low-level feature extraction learned during pre-training.

### Comparison

| Method | Trainable params | Inference overhead | Typical use |
|---|---|---|---|
| Full fine-tuning | 100% | None | Small models / abundant compute |
| Adapter Layers | ~1–5% | Small (extra layers) | Task-specific adapters |
| LoRA | ~0.1–1% | None (merged) | LLMs, style/domain transfer |
| Bias Tuning | <1% | None | Ultra-cheap adaptation |

---

## 11. LLM Fine-Tuning and Optimization — Full Reference

| Method | Full Name | Purpose |
|---|---|---|
| SFT | Supervised Fine-Tuning | Train on labeled examples to adapt LLM to a specific task |
| PEFT | Parameter-Efficient Fine-Tuning | Fine-tune a small part of the model to save compute |
| RLHF | Reinforcement Learning from Human Feedback | Align model with human preferences using reward models |
| LVR | Learning from Verifiable Rewards | Align model using automated correctness signals (math, code) |
| DDP | Distributed Data Parallel | Parallel training across multiple GPUs |
| FSDP | Fully Sharded Data Parallel | Memory-efficient training of large models |

### SFT — Supervised Fine-Tuning
Train on labeled (input, output) pairs to give the model structured, assistant-like behaviour (summarizing, translating, answering questions).

**How it works:** Teacher forcing is applied — during training, the model receives ground-truth tokens as decoder inputs rather than its own previous predictions. This avoids compounding errors in early training. The model outputs tokens only conditioned on the prompt's final tokens, not the entire input context.

**Key historical models:**
- **T5 (2020):** Encoder-decoder model trained on curated tasks (translation, summarization) with fixed input formats.
- **T0:** Extended T5 to handle arbitrary natural-language task descriptions without fixed format constraints — an early step toward general instruction following.

**Limitations:**
- Relies on curated datasets; quality of SFT is bounded by data quality.
- Struggles with ambiguous prompts; does not ask clarifying questions.
- Can produce well-formatted but unhelpful answers.
- Requires diverse task and prompt formats during training to generalize robustly.

### PEFT — Parameter-Efficient Fine-Tuning
Fine-tune only a small part of the model. Methods include:
- **LoRA:** Inject low-rank trainable matrices into transformer layers (see above).
- **Prefix Tuning:** Add trainable tokens to the start of the input.
- **Adapters:** Add small MLP blocks between layers.

✅ Fast, low-cost, avoids catastrophic forgetting.  
❌ Less flexible than full fine-tuning for deep architectural changes.

### RLHF — Reinforcement Learning from Human Feedback

**Pipeline:**
1. **Sample:** Pre-trained LLM generates multiple candidate answers for a prompt using temperature sampling.
2. **Rank:** Human annotators rank the candidates for helpfulness, correctness, and clarity.
3. **Reward Model:** Trained on pairwise comparisons to assign a scalar reward score to each answer. Uses pairwise ranking losses (e.g., $\ell_1$-regularized comparisons).
4. **Policy Optimization:** The LLM is fine-tuned (typically with PPO) to maximize the reward model's output, shifting the distribution toward human-preferred responses.

Used in: ChatGPT's alignment phase.  
✅ Produces more aligned, human-like outputs.  
❌ Requires expensive human annotation; risk of **reward hacking** (model learns to be verbose or superficially pleasing rather than correct); can reduce response diversity.

### LVR — Learning from Verifiable Rewards

An alternative to RLHF for tasks where correctness is objectively measurable (math problems, code generation, formal proofs).

**Mechanism:** Instead of human annotators, automated tests verify the answers (e.g., executing generated code against test cases). The LLM is fine-tuned to maximize the passing rate.

✅ No human annotation cost; objective signal; scales to large datasets.  
❌ Only applicable where automated verification is possible. May fail to capture nuanced reasoning — the model could produce a correct answer via an invalid reasoning path without penalty.

### DDP — Distributed Data Parallel (PyTorch)
Copy the model to each GPU; split the batch; sync gradients after each step.  
`torch.nn.parallel.DistributedDataParallel`  
✅ Simple setup, good performance.  
❌ Each GPU holds a full model copy (higher memory).

### FSDP — Fully Sharded Data Parallel (PyTorch)
Shards model weights and optimizer states across GPUs. Enables training of 13B+ models that cannot fit on a single GPU.  
✅ Enables training of giant models on fewer GPUs.  
❌ More complex setup.

### Typical Production Stack
```
LLaMA 13B base model
  → SFT on customer chat logs (QA pairs)
  → LoRA (PEFT) to save memory
  → FSDP across 4 GPUs for efficiency
  → RLHF to align tone with user preferences
  (or LVR if the task output is verifiable, e.g. code/math)
```

### Key Limitations Across Methods
- **SFT generalization:** Requires diverse prompt formats in training data.
- **Human feedback reliability:** Subjective preferences introduce noise and bias.
- **RLHF reward hacking:** Model may learn superficially pleasing behaviours instead of genuinely correct ones.
- **Efficiency vs. performance:** LoRA and similar methods trade some performance for computational savings.

---

## 12. Knowledge Distillation

### Core Idea
A small **student** model learns from a large **teacher** model, not from hard (one-hot) labels.

```
Teacher (large) → soft predictions (soft labels)
                            ↓
Student (small) ← learns to mimic
```

Result: Smaller model (e.g., 7B → 400M parameters) with similar quality.

### Why Soft Labels Are Better
Hard label: $[0, 0, 1, 0, 0]$ — carries little information.  
Soft label from teacher: $[0.01, 0.02, 0.85, 0.08, 0.04]$ — encodes inter-class similarity ("cat is more similar to dog than to airplane").

### Temperature in Distillation
To make soft labels "softer", apply temperature $T$ inside the softmax:
$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
When $T > 1$, the distribution smooths out — the student sees richer information about class structure. After training, temperature is reset to $T = 1$.

### Distillation Loss
$$\mathcal{L} = \alpha \cdot \mathcal{L}_\text{hard} + (1 - \alpha) \cdot \mathcal{L}_\text{soft}$$
$$\mathcal{L}_\text{hard} = \text{CrossEntropy}(\text{student logits},\ \text{true labels})$$
$$\mathcal{L}_\text{soft} = \text{KL}(\text{student}_\text{soft},\ \text{teacher}_\text{soft}) \quad \text{at temperature } T$$
$\alpha$ is a hyperparameter (typically 0.1–0.5).

### Types of Distillation
- **Response-based:** Student mimics teacher's final predictions.
- **Feature-based:** Student mimics intermediate activations (hidden layers).
- **Relation-based:** Student learns to reproduce pairwise distances between examples.

### Real Examples
- **DistilBERT:** 97% of BERT quality at 40% smaller, 60% faster.
- **TinyBERT:** Distillation at the level of attention matrices.
- Quantization + distillation is a standard production deployment pipeline.

Note: Distillation ≠ simply training a small model from scratch. Identical architectures trained from scratch vs. via distillation yield different quality — distillation is a fully-fledged training method.

---

## 13. Label Smoothing

### Motivation
Standard training uses hard (one-hot) targets:
$$\text{correct class} \to 1.0, \quad \text{all others} \to 0.0$$

Hard targets reward the model for pushing the correct logit to $+\infty$ and all others to $-\infty$, causing:
- **Poor calibration** — model is more confident than warranted.
- **Poor generalization** — model memorizes rather than learning smooth representations.

### Definition
Label smoothing redistributes a small probability mass $\varepsilon$ (typically 0.1, from the original Transformer paper):
$$y_\text{smooth} = (1 - \varepsilon) \cdot y_\text{hard} + \frac{\varepsilon}{V}$$

Per token:
$$y_\text{smooth}(k) = (1 - \varepsilon) \cdot \mathbf{1}[k = \text{target}] + \frac{\varepsilon}{V}$$

Concrete example for $\varepsilon = 0.1$, $V = 30{,}000$:
$$\text{correct token} \to 0.9 + \frac{0.1}{30000} \approx 0.9000033$$
$$\text{wrong tokens} \to 0.0 + \frac{0.1}{30000} \approx 0.0000033$$

### Loss Function
With label smoothing (soft target):
$$\mathcal{L} = -\sum_{k} y_\text{smooth}(k) \cdot \log p_\text{model}(k)$$

Standard cross-entropy (only the correct-class term survives):
$$\mathcal{L} = -\log p_\text{model}(\text{target})$$

### Effect on Metrics

| Metric | Effect | Reason |
|---|---|---|
| Perplexity | Gets worse | Smoothing reduces confidence on correct tokens; loss can never reach 0 |
| BLEU | Gets better | Smoother distributions → more diverse, natural outputs |

$$\text{Perplexity} = \exp(\mathcal{L}_\text{cross-entropy})$$

Label smoothing deliberately increases loss, so perplexity rises — but the model generalizes better, so BLEU improves. The two metrics optimize for different things.

### In the Transformer Context
At each decoder step, the model predicts the next token:

Hard target — correct token is `"ours"` (index 472):
$$p_\text{target} = [0,\ 0,\ \ldots,\ \underbrace{1.0}_{\text{index } 472},\ \ldots,\ 0]$$

Soft target — with $\varepsilon = 0.1$:
$$p_\text{target} = \left[\frac{0.1}{29999},\ \ldots,\ \underbrace{0.9}_{\text{index } 472},\ \ldots,\ \frac{0.1}{29999}\right]$$

### Why This Matters for Translation
In translation, there is rarely a single correct output — "Un ours" and "L'ours" may both be valid. Hard targets pretend exactly one token is correct. Label smoothing implicitly acknowledges translation ambiguity, keeping the model uncertain enough to consider synonyms, which improves BLEU.

**Bottom line:** Label smoothing trades raw likelihood for better generalization — a model slightly less certain, but generating text closer to what humans actually write.

---

## 14. Search and Retrieval

### Bi-Encoder
Two texts are encoded **independently** — each becomes a vector, then similarity is computed with cosine or dot product.

```
Text A → [Encoder] → vector A ─┐
                                 ├─ cosine_sim → score
Text B → [Encoder] → vector B ─┘
```

**Pros:** Vectors can be precomputed and stored in an index (FAISS, Annoy); search over millions of documents in milliseconds; scales well.  
**Cons:** No cross-text interaction — the context of A does not influence the encoding of B; lower accuracy than cross-encoder.

Use: Semantic search, ANN indexes, **first stage** in a two-stage pipeline (retrieval).

### Cross-Encoder
Both texts are concatenated and processed jointly — the model sees interactions from the very first layer.

```
[CLS] Text A [SEP] Text B [SEP] → [Encoder] → score
```

**Pros:** Much more accurate — attention sees the connections between words of both texts; captures context and meaning overlap.  
**Cons:** Cannot precompute — must run every pair through the model; $O(N)$ operations per query → does not scale to large collections.

Use: **Reranking** — take top-100 from bi-encoder, reorder with cross-encoder.

### Typical Production Pipeline

```
Query
  ↓
Bi-encoder → ANN search → top-100 candidates
  ↓
Cross-encoder → reranking → top-10 final results
```

---

## 15. Scaling to 1M+ Context

### Context Length Evolution

| Year | Model | Context |
|---|---|---|
| 2020 | GPT-3 | 2K |
| 2023 | GPT-4 | 32K (128K extended) |
| 2024 | Claude 3 | 200K |
| 2024 | Gemini 1.5 | 1M |
| 2025 | Gemini 2.0 | 2M (announced) |

### Why Naive Attention Fails at 1M Tokens

For $N = 1M$ tokens, the attention matrix requires:
$$1M \times 1M \times 4\ \text{bytes (float32)} = 4\ \text{TB per layer}$$

With 80+ layers, this is physically impossible on current hardware. Modern long-context models use a combination of approximations:

### Key Techniques

**1. Sparse Attention — not every token attends to every other token.**

- *Local / Sliding Window:* Each token attends only to a fixed-size neighborhood $W$. Complexity: $O(N \times W)$ instead of $O(N^2)$.
- *Strided:* Each token attends to every $k$-th position.
- *Block-Sparse:* Attention only between neighboring blocks.
- *Longformer-style:* Local window + global tokens (e.g., `[CLS]`) that all tokens can attend to.

**2. Ring Attention (Google, 2023)** — distributed processing for full attention.

Sequence is split into chunks across devices. Each device processes its chunk, then rotates KV-cache to the next device in a ring. After a full cycle, every chunk has "seen" all others.

- Memory per device: $O(N/D)$ where $D$ = number of devices.
- Still achieves full (non-sparse) attention, just distributed.

**3. Flash Attention** — memory-efficient exact attention.

Standard attention materializes the full $N \times N$ matrix. Flash Attention computes attention in tiles, never storing the full matrix:
- Memory: $O(N)$ instead of $O(N^2)$.
- Speed: 2–4× faster due to reduced memory I/O.

Without Flash Attention for 1M tokens: 4TB. With Flash Attention (block size 256): ~1GB.

**4. Multi-Query Attention (MQA) / Grouped-Query Attention (GQA)**

Standard MHA: 32 heads, each with its own Q, K, V. KV-cache = $N \times 32 \times d_\text{head} \times 2$.

MQA: All heads share one K and one V. KV-cache reduced by 32×.

GQA (compromise, used in Gemini): Group heads so each group shares K, V. Balances memory saving with quality.

**5. KV Cache Compression**

- *Quantization:* FP16 → INT4 reduces cache size 4×.
- *Pooling of old tokens:* Recent tokens at full resolution; distant tokens compressed into fewer embeddings.
- *Adaptive selection:* Keep only "important" tokens at full resolution; compress or drop the rest.

**6. Hierarchical Attention**

Process context at multiple levels of abstraction:
```
Level 1: Local attention within 512-token blocks
  ↓ Compress
Level 2: Attention between block embeddings (~2000 blocks)
  ↓ Compress
Level 3: Global representation of the full context
```

**7. Mixture of Experts (MoE)**

Different parts of the context activate different experts. Enables parallel, selective processing and reduces per-token compute.

### Practical Reality of Long Context

Even with 1M context, models effectively utilize only ~10–20K tokens per generation step. The rest functions as background reference.

**Lost-in-the-middle problem:** Models remember the beginning and end of long contexts well, but performance degrades for content in the middle. Mitigations include recency bias, position embedding modifications, and attention boosting for important regions.

---

## 16. Practical Implementation (HuggingFace)

- Model loading: `AutoModelForCausalLM.from_pretrained` with positional embeddings and layer normalization.
- GPTNeoX Layer Blocks: Self-attention with QKV projections, rotary positional encodings (RoPE), and MLP layers.
- Inspecting attention heads reveals matrix multiplications for QKV generation and attention weight computation.

---

## References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [The Ultimate Guide to Fine-Tuning LLMs](https://arxiv.org/html/2408.13296v1)
- [Distributed Compute in Transformer](https://ailzhang.github.io/posts/distributed-compute-in-transformer/)
- [Transformer Explained (video)](https://www.youtube.com/watch?v=7fvxOgliYRw)