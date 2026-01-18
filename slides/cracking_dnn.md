# Basics

# Exploding/Vanishing Gradients

## Вопрос
Объясни exploding/vanishing gradients? Как с этим бороться?

## Ответ

Давай разложим проблему exploding и vanishing gradients структурированно, как на собесе по deep learning.

### 1️⃣ Что такое градиенты

В нейросетях обучение идёт через обратное распространение ошибки (backprop):

$$w \gets w - \eta \frac{\partial L}{\partial w}$$

где $\frac{\partial L}{\partial w}$ — градиент

Если градиенты слишком большие или маленькие → обучение нарушается

### 2️⃣ Vanishing gradients (затухающие градиенты)

**Что происходит**

В глубокой сети градиенты уменьшаются экспоненциально по мере прохождения через слои.

**Результат:** ранние слои почти не обучаются.

**Причины**

Малые веса + сигмоидная/tanh активация:

$$\sigma'(x) = \sigma(x)(1-\sigma(x)) \le 0.25$$

Множитель < 1 на каждом слое → градиент быстро стремится к 0

**Симптомы**
- Потеря способности сети обучать нижние слои
- Очень медленная сходимость
- Все веса в начале сети почти не меняются

### 3️⃣ Exploding gradients (взрывающиеся градиенты)

**Что происходит**

Градиенты очень большие, веса обновляются слишком сильно → обучение нестабильно

**Причины**
- Большие начальные веса
- Длинная сеть → многократное умножение больших матриц
- Некорректная активация (ReLU с большим входом)

**Симптомы**
- Loss становится NaN
- Веса резко прыгают

### 4️⃣ Как бороться

#### 🔹 Для vanishing gradients

**Использовать "хорошие" активации**
- ReLU, LeakyReLU, GELU → градиенты не затухают

**Инициализация весов**
- Xavier/Glorot, He

**Нормализация**
- BatchNorm → стабилизирует распределение

**Shortcut/Residual connections**
- ResNet → позволяет градиенту "протекать" через сеть

#### 🔹 Для exploding gradients

**Gradient clipping**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Меньший learning rate**

**Хорошая инициализация**

**Использовать нормализацию**
- BatchNorm, LayerNorm

### 🔹 Вывод

| Проблема | Симптом | Решение |
|----------|---------|---------|
| Vanishing | Градиенты близки к 0, ранние слои не учатся | ReLU, BatchNorm, Residual, Xavier/He |
| Exploding | Градиенты огромные, loss → NaN | Gradient clipping, уменьшить LR, нормализация |

---


# Variational Autoencoders (VAEs)

Variational Autoencoders are a type of generative model that learns to encode data into a compressed representation and then decode it back, while also being able to generate new, similar data.

## The Basic Idea

A VAE consists of two main parts:

**Encoder**: Compresses input data (like images) into a lower-dimensional latent space representation. Unlike standard autoencoders, the encoder outputs parameters of a probability distribution (typically mean and variance) rather than a single fixed point.

**Decoder**: Takes samples from this latent space and reconstructs them back into the original data format.

## What Makes Them "Variational"?

The key innovation is that VAEs don't just map inputs to fixed points in latent space. Instead, they:

1. Map inputs to **probability distributions** in latent space
2. Sample from these distributions during training
3. Ensure the latent space is continuous and well-structured

This is done by forcing the learned distributions to be similar to a standard normal distribution (mean 0, variance 1) using something called the KL divergence term in the loss function.

## The Loss Function

VAEs optimize two objectives simultaneously:

- **Reconstruction loss**: How well can the decoder recreate the original input?
- **KL divergence**: How close is the latent distribution to a standard normal distribution?

## Why This Matters

By learning smooth, continuous latent representations, VAEs can:
- Generate new data by sampling random points from the latent space
- Interpolate smoothly between different data points
- Learn meaningful features in an unsupervised way

## Common Applications

- Image generation
- Data compression
- Anomaly detection
- Drug molecule design
- Music generation

Would you like me to dive deeper into any particular aspect, like the mathematical formulation, implementation details, or specific use cases?