
[![Лекция 03 vol 2. Векторизация текста Word2Vec Transformers](http://img.youtube.com/vi/csqW3HF_3p8/0.jpg)](http://www.youtube.com/watch?v=csqW3HF_3p8 "Лекция 03 vol 2. Векторизация текста Word2Vec Transformers")

1️⃣ Начать можно с практики — продуктовые кейсы от ребят из FinAI (делают ассистента для команды саппорта).
Блог свежий, много интересных применений: в основном RAG, немного про обучение эффективных узкоспециализированных SLM.
Пишут обо всём — от инфры до аналитики продуктовых экспериментов.
👉 fin.ai/research (https://fin.ai/research/)

Дальше идут углубленные материалы

2️⃣ Основа и фундамент — (https://cme295.stanford.edu/syllabus/) свежий курс CME295 от Стэнфорда.
Курс про трансформеры: лекции, видосы, нормальная структура материала.
Для YouTube роликов я использую notebooklm (https://notebooklm.google.com/) — удобно вытаскивать конспекты и делать инфографику.

3️⃣ Вопросы экзамена по CME295 (https://cme295.stanford.edu/exams/midterm.pdf) идеальны для собесов.
Хорошая подборка: всё разбито по пунктам, покрывает практически весь курс. Отличный инструмент для подготовки.

4️⃣ Чтобы приземлить теорию — курс по LLM (https://huggingface.co/learn/llm-course/en/chapter6/1) от HuggingFace, особенно классная часть про токенайзеры + много практики про деплой джобов в HuggingFace Cloud.

5️⃣ Если хочется побольше инженерных деталей:
* Nebius LLM Engineering (https://github.com/Nebius-Academy/LLM-Engineering-Essentials) — упор на метрики
* LLMOps Essential (https://github.com/Nebius-Academy/LLMOps-Essentials) — полезный материал ( прикольные штуки типа деплоя в Kubernetes)
* Unsloth [fine-tuning guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)

6️⃣ Если вкатывани в LLM тяжело идет, начните с базы.
Стэнфорд выложил CS230 — отличный фундаментальный курс по DL
* слайды (https://cs230.stanford.edu/syllabus/)
* видосы (https://www.youtube.com/playlist?list=PLoROMvodv4rNRRGdS0rBbXOUGA0wjdh1X)

---

Материала хватает минимум на пару недель плотного погружения.
Идеально для новогодних каникул 🎄✨


Transformers
* video explanation https://youtu.be/ECR4oAwocjs 
* blog post  https://poloclub.github.io/transformer-explainer/


Обучение эмбеддингов на своих внутренних данных

* почему это важно https://fin.ai/research/finetuning-retrieval-for-fin/
* как обучать https://arxiv.org/abs/2512.21021

# Word2Vec: "bank" всегда одинаковый вектор
"I went to the bank" → [0.23, -0.45, 0.67, ...]
"River bank is beautiful" → [0.23, -0.45, 0.67, ...]  # Тот же!

# BERT: "bank" зависит от контекста
```
"I went to the bank" → [0.12, 0.89, -0.34, ...]  # Финансы
"River bank is beautiful" → [-0.45, 0.23, 0.91, ...]  # Берег
```

📚 **Источники:**
- [Efficient Estimation of Word Representations in Vector Space (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) — Word2Vec
- [GloVe: Global Vectors for Word Representation (Pennington et al., 2014)](https://nlp.stanford.edu/pubs/glove.pdf)
- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)

---

## 3. Transformer Architecture: Encoder vs Decoder

### Фундаментальное различие

**BERT (Encoder-only)** vs **GPT (Decoder-only)** — это не просто "разные модели", это разные философии.

#### BERT: Bidirectional Attention

```
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

```
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

📚 **Источники:**
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — оригинальный Transformer
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Improving Language Understanding by Generative Pre-Training (Radford et al., 2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1
- [The Illustrated Transformer by Jay Alammar](http://jalammar.github.io/illustrated-transformer/) — отличная визуализация

---

## 4. Attention Mechanisms: Self vs Cross

### Три типа Attention

Многие путают bidirectional attention и cross-attention. Это **разные** механизмы!

#### 1. Self-Attention (Bidirectional) — BERT

```
Q, K, V все из ОДНОЙ последовательности

Input: "The cat sits"
Query из: "The cat sits"
Key из: "The cat sits"
Value из: "The cat sits"
```

#### 2. Self-Attention (Causal) — GPT

```
Q, K, V из одной последовательности + causal mask

Треугольная маска запрещает смотреть в будущее
```

#### 3. Cross-Attention — Encoder-Decoder

```
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



- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — описание всех трёх типов
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) — код с комментариями
- [Transformers explained](https://www.linkedin.com/posts/nicole-koenigstein_transformers-the-definitive-guide-activity-7411413196646846466-PZpQ?utm_source=share&utm_medium=member_ios&rcm=ACoAABHcLTkB9ZRrPOB4NW-jmLGXwC1oz0SS_hY)
- [CS25 Transformers intro](https://youtu.be/XfpMkf4rD6E?si=A0ckxe7ZkndQxWEe)
- [NLP interview questions](https://www.linkedin.com/posts/sumanth077_top-50-llm-interview-questions-a-comprehensive-activity-7400863663253028864-2oPM)
- [encoders vs decoders](https://www.linkedin.com/posts/mary-newhauser_not-all-llms-generate-text-most-people-share-7402121282898739201-mOSi/)
- [self-attention-from-scratch](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html?utm_source=substack&utm_medium=email)
- [Self attention](https://youtu.be/Bg8Y5q1OiP0)
- [Transformers cheatsheet](https://github.com/afshinea/stanford-cme-295-transformers-large-language-models/blob/main/en/cheatsheet-transformers-large-language-models.pdf)

## LLM datasets

- [amazon-reviews-2023](https://amazon-reviews-2023.github.io/)


---

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