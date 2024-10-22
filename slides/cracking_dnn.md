# Basics

Нейросеть — это функция с миллионами параметров, которая отображает вход в выход. Параметры подбираются так, чтобы минимизировать ошибку на обучающих данных.

---

### Один нейрон (перцептрон)

```
x₁ ─┐
x₂ ─┤─ Σ(wᵢxᵢ + b) → f(z) → выход
x₃ ─┘
```

- **w** — веса (что важно)
- **b** — смещение (bias)
- **f** — функция активации (нелинейность)

Без нелинейных функций активации сеть из любого числа слоёв — это просто линейное преобразование.

---

### Прямой проход (Forward pass)

```python
# Один слой
z = W @ x + b       # линейная комбинация
a = relu(z)          # активация
```

Данные проходят слой за слоем: вход → скрытые слои → выход.

---

### Функции активации

| Функция | Формула | Где |
|---------|---------|-----|
| ReLU | max(0, x) | скрытые слои |
| Sigmoid | 1/(1+e⁻ˣ) | бинарная классификация |
| Softmax | eˣⁱ/Σeˣʲ | многоклассовая классификация |
| GELU | x·Φ(x) | трансформеры |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | RNN |

---

### Обратное распространение (Backpropagation)

1. Считаем ошибку (loss) на выходе — например, CrossEntropy
2. Берём производную loss по каждому параметру через **цепное правило**
3. Обновляем параметры в сторону уменьшения loss

**Цепное правило:**

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

**Шаг градиентного спуска:**

$$w \gets w - \eta \frac{\partial L}{\partial w}$$

где $\frac{\partial L}{\partial w}$ — градиент

**Проблемы:**
- **Затухающий градиент** — в глубоких сетях градиент умножается много раз и стремится к 0. Решение: ReLU, batch norm, residual connections.
- **Взрывной градиент** — градиент взрывает. Решение: gradient clipping.


# Exploding/Vanishing Gradients


Если градиенты слишком большие или маленькие → обучение нарушается

Vanishing gradients (затухающие градиенты)

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

Exploding gradients (взрывающиеся градиенты)

**Что происходит**

Градиенты очень большие, веса обновляются слишком сильно → обучение нестабильно

**Причины**
- Большие начальные веса
- Длинная сеть → многократное умножение больших матриц
- Некорректная активация (ReLU с большим входом)

**Симптомы**
- Loss становится NaN
- Веса резко прыгают


Как бороться

Для vanishing gradients

**Использовать "хорошие" активации**
- ReLU, LeakyReLU, GELU → градиенты не затухают

**Инициализация весов**
- Xavier/Glorot, He

**Нормализация**
- BatchNorm → стабилизирует распределение

Нормализует активации внутри батча → ускоряет обучение, снижает зависимость от инициализации.

Нормализация:

$$\hat{x} = \frac{x - \mu}{\sigma}$$

Масштабирование и сдвиг:

$$y = \gamma \cdot \hat{x} + \beta$$

где $\gamma$ и $\beta$ — обучаемые параметры.

**Shortcut/Residual connections**
- ResNet → позволяет градиенту "протекать" через сеть

Для exploding gradients

**Gradient clipping**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Меньший learning rate**

**Хорошая инициализация**

**Использовать нормализацию**
- BatchNorm, LayerNorm

Вывод

| Проблема | Симптом | Решение |
|----------|---------|---------|
| Vanishing | Градиенты близки к 0, ранние слои не учатся | ReLU, BatchNorm, Residual, Xavier/He |
| Exploding | Градиенты огромные, loss → NaN | Gradient clipping, уменьшить LR, нормализация |

---


# Variational Autoencoders (VAEs)

Variational Autoencoders are a type of generative model that learns to encode data into a compressed representation and then decode it back, while also being able to generate new, similar data.

A VAE consists of two main parts:

**Encoder**: Compresses input data (like images) into a lower-dimensional latent space representation. Unlike standard autoencoders, the encoder outputs parameters of a probability distribution (typically mean and variance) rather than a single fixed point.

**Decoder**: Takes samples from this latent space and reconstructs them back into the original data format.

What Makes Them "Variational"? The key innovation is that VAEs don't just map inputs to fixed points in latent space. Instead, they:

1. Map inputs to **probability distributions** in latent space
2. Sample from these distributions during training
3. Ensure the latent space is continuous and well-structured

This is done by forcing the learned distributions to be similar to a standard normal distribution (mean 0, variance 1) using something called the KL divergence term in the loss function.

## The Loss Function

VAEs optimize two objectives simultaneously:

- **Reconstruction loss**: How well can the decoder recreate the original input?
- **KL divergence**: How close is the latent distribution to a standard normal distribution?

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
