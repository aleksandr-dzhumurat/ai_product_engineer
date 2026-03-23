# Python programming

# Python: Итерируемые объекты, Итераторы, Генераторы

---

## Итерируемый объект (Iterable)

**Итерируемый объект** — объект, который **можно перебрать** (например, в `for` или через `next()`).

Реализует метод:

- `__iter__() -> iterator` — возвращает итератор

По сути, итерируемый объект — это объект, способный предоставить итератор.

```python
my_list = [1, 2, 3]
it = iter(my_list)   # вызывает my_list.__iter__() -> возвращает list_iterator
print(type(my_list)) # <class 'list'>          — итерируемый, но НЕ итератор
print(type(it))      # <class 'list_iterator'> — итератор
```

---

## Итератор (Iterator)

**Итератор** в Python — это объект, реализующий **оба** метода протокола итератора:

- `__iter__() -> self` — возвращает самого себя
- `__next__() -> next_value` — возвращает следующий элемент; при исчерпании бросает `StopIteration`

Итераторы являются **однопроходными**: будучи исчерпанными, они не могут быть перезапущены.

```python
it = iter([1, 2, 3])
print(next(it))  # 1
print(next(it))  # 2
print(next(it))  # 3
next(it)         # StopIteration
```

---

## Генератор (Generator)

**Генератор** (Generator) — специальный вид итераторов, **создающих значения "на лету" (lazily)** с помощью `yield` вместо `return`.

Не нужно вручную реализовывать `__iter__` и `__next__` — генераторная функция делает это автоматически. Память экономится, потому что **не строим всю последовательность заранее**.

Подходит для: потоковых данных, больших файлов и ленивой генерации чисел.

Генераторы **сохраняют своё состояние между вызовами** (Statefulness).

```python
def count_up(n):
    for i in range(n):
        yield i

gen = count_up(3)
print(next(gen))  # 0
print(next(gen))  # 1
print(next(gen))  # 2
```

Каноничный пример - чтение файла

```python
def read_large_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()  # Yield each line without loading the entire file into memory

file_path = 'large_data_file.txt'
for line in read_large_file(file_path):
    print(line)
```

Корутины (Coroutine) - расширение генераторов для **асинхронного кода и кооперативной многозадачности**, позволяют приостанавливать выполнение, возвращать значение и потом возобновлять.

```python
import asyncio

async def say_hello():
    await asyncio.sleep(1)
    print("Hello")

asyncio.run(say_hello())
```

Example mental model:

```python
[1, 2, 3, 4, 5]          # → Iterable (list) — full data in memory
iter([1, 2, 3, 4, 5])    # → Iterator — still backed by the list
(x**2 for x in range(N)) # → Generator expression — produces values lazily
```

---

## Зачем нужна iter() и ручное управление итератором

`for` действительно вызывает `iter()` под капотом автоматически — и в большинстве случаев этого достаточно. Но `iter()` напрямую нужна, когда требуется **контроль над позицией** в последовательности.

### 1. Пропустить часть элементов, затем продолжить в другом месте

Типичный пример — парсинг файла с заголовком:

```python
lines = open("data.csv")
it = iter(lines)

header = next(it)  # читаем заголовок отдельно

for line in it:    # продолжаем с первой строки данных
    process(line)
```

С обычным `for` пришлось бы добавлять счётчик или флаг `skip_first`.

### 2. Читать по несколько элементов за раз (chunking)

```python
it = iter([1, 2, 3, 4, 5, 6])

while True:
    chunk = [next(it, None) for _ in range(2)]
    chunk = [x for x in chunk if x is not None]
    if not chunk:
        break
    print(chunk)  # [1, 2], [3, 4], [5, 6]
```

### 3. Разделить один проход между несколькими потребителями

Итератор — **общий указатель**. Два `for` по одному списку обходят его с начала, а два `for` по одному итератору продолжают с места остановки:

```python
it = iter(range(6))

first_half  = [next(it) for _ in range(3)]  # [0, 1, 2]
second_half = list(it)                       # [3, 4, 5]
```

### 4. Двухаргументная форма iter(callable, sentinel)

Менее известная, но мощная: вызывает функцию повторно до тех пор, пока она не вернёт `sentinel`.

```python
import random

# Бросаем кубик, пока не выпадет 6
for roll in iter(lambda: random.randint(1, 6), 6):
    print(roll)

# Читаем файл блоками по 1024 байта до конца
with open("file.bin", "rb") as f:
    for block in iter(lambda: f.read(1024), b""):
        process(block)
```

**Когда iter() не нужна**: Если нужно просто пройтись по всем элементам — `for` достаточно. `iter()` нужна только когда важно **где** находится указатель и **кто** им управляет.

# Decorator

```python
def simple_decorator(func):
    def wrapper():
        print("Before the function call")
        func()
        print("After the function call")
    return wrapper

@simple_decorator
def say_hello():
    print("Hello, world!")

# Usage
say_hello()
```

# Data structures

## List

Тип данных `list` - динамический массив (CPython: PyListObject). Хранит указатели на объекты.

```python
typedef struct {
    PyObject_VAR_HEAD       // ob_size — текущая длина
    PyObject **ob_item;     // указатель на массив указателей
    Py_ssize_t allocated;   // выделенная ёмкость (>= ob_size)
} PyListObject;
```

При создании выделяется память **фиксированного размера**, поэтому для добавления элементов часто приходится создавать новый массив и копировать данные.
- Храним **буфер большего размера**, чем реально элементов.
- Когда буфер заполняется - увеличиваем его **геометрически**, обычно **в 1.5–2 раза** и старые элементы копируются..
- Добавление элемента в пустое место — **O(1)** амортизированное.

Почему не linked list (не надо было бы копировать)? Ответ - долгое обращение по индексу `a[i]`  - O(n) в случае linked list.

- Малые списки растут быстрее (чтобы не делать слишком много копирований)
- Большие списки растут медленнее, чтобы не расходовать лишнюю память

Примерная реализация

```python
import numpy as np

class DynamicArray:
    def __init__(self, dtype=float, capacity=4):
        self._capacity = capacity
        self._size = 0
        self._data = np.empty(self._capacity, dtype=dtype)

    def append(self, value):
        if self._size >= self._capacity:
            self._resize(2 * self._capacity)
        self._data[self._size] = value
        self._size += 1

    def _resize(self, new_capacity):
        new_data = np.empty(new_capacity, dtype=self._data.dtype)
        new_data[:self._size] = self._data[:self._size]  # копируем старые элементы
        self._data = new_data
        self._capacity = new_capacity

    def __getitem__(self, idx):
        if idx >= self._size:
            raise IndexError("Index out of bounds")
        return self._data[idx]

    def __len__(self):
        return self._size

    def __repr__(self):
        return f"DynamicArray({self._data[:self._size]!r})"
```

### Time Complexity

**O(1) - Constant:**
```python
lst.append(x)          # Add to end (amortized)
lst[i]                 # Index access
lst[-1]                # Last element
len(lst)               # Length
```

**O(n) - Linear:**
```python
lst.insert(0, x)       # Insert at beginning (shift all!)
lst.insert(i, x)       # Insert at i (shift i to end)
lst.remove(x)          # Remove by value (search + shift)
x in lst               # Membership check
lst.pop(0)             # Remove from beginning (shift)
```

**O(n log n):**
```python
lst.sort()             # Timsort
```

- [Python List Implementation - CPython Source](https://github.com/python/cpython/blob/main/Objects/listobject.c)
- [Time Complexity - Python Wiki](https://wiki.python.org/moin/TimeComplexity)

## Dict

В Python словарь (`dict`) реализован на основе **хеш-таблицы**, и алгоритм разрешения коллизий там особенный — **open addressing с perturbation и методом поиска пробой (probing)**. 

Хеш-таблица в Python

- Каждое ключ-значение хранится в **ячейке таблицы**.
- Вычисляется **хеш ключа**: `hash(key)`.
- Хеш мапится на индекс таблицы: `index = hash(key) % table_size`.
- Проблема: несколько ключей могут иметь один индекс → **коллизия**.

Хеш-таблица — это **массив фиксированного размера**.

```python
index | slot
------+-------------------------
0     | empty
1     | ("cat", 7)
2     | empty
3     | ("dog", 3)
4     | empty
5     | empty
6     | ("apple", 10)
7     | empty
```

Разрешение коллизий

Python использует **open addressing** (все элементы хранятся прямо в массиве, нет отдельных списков):

- Если индекс занят другим ключом, Python ищет **другую свободную ячейку** по специальной формуле.
- Раньше использовался линейный или двойной пробинг, сейчас — **perturbation probing**.

Perturbation probing

1. Начальный индекс: `i = hash(key) & mask` (mask = size-1)
2. Если ячейка занята:
    - Используется **perturb** = `hash(key)`
    - Затем вычисляется следующая ячейка: `i = (i * 5 + perturb + 1) & mask`
- `perturb` делится на 5 на каждой итерации, чтобы шаги поиска были “псевдослучайными”.
- Этот метод обеспечивает **хорошее распределение** и меньше кластеризации, чем простой линейный пробинг.

Итоговый процесс вставки / поиска

Вставка

1. Вычисляем `hash(key) % table_size`.
2. Проверяем ячейку:
    - Если свободна → вставляем.
    - Если ключ совпадает → обновляем значение.
    - Если занят другим ключом → идём по perturbation probing.
3. При заполнении > 2/3 → таблица **расширяется** (resize), все элементы перехешируются.

Поиск аналогично: вычисляем индекс → если не совпадает ключ, идём по probing, пока не найдём ключ или пустую ячейку.

Почему это быстро

- Средняя сложность **O(1)** для вставки и поиска (при хорошем распределении хешей).
- Плохой хеш → коллизии → O(n) в худшем случае, но Python тщательно подбирает хеш и probing для минимизации этого.

# Mutable vs immutable

mutable - можно изменить объект

```python
# Mutable (изменяемые)
lst = [1, 2, 3]
lst.append(4)      # ок, список изменился

# Immutable (неизменяемые)
t = (1, 2, 3)
t[0] = 99          # TypeError!
```

Объект hashable, если у него есть метод `__hash__()`, который возвращает одно и то же число на протяжении всей жизни объекта. Это нужно для использования в `set` и как ключ в `dict`.

Хитрый случай — tuple. Tuple сам по себе immutable, но может **содержать** mutable объекты. Tuple hashable только если **все его элементы** hashable.

# Classes

Что делает дандер `__init__(self)`? Чем отличаются атрибуты `a`, `b` и `c`?

```python
class Test:
    a = 0
    def __init__(self):
        b = 1
        self.c = 2
```

- `a = 0` — **атрибут класса**. Принадлежит самому классу, общий для всех экземпляров. При наследовании `Test.a = 99` и изменении если Child не переопределял `.a`, он видит изменение родителя
- `b = 1` — **локальная переменная** внутри `__init__`. Существует только во время выполнения метода, к экземпляру не привязана и снаружи недоступна. Не наследуется.
- `self.c = 2` — **атрибут экземпляра**. Создаётся при инициализации и принадлежит конкретному объекту. Наследуется но нужно вызвать `super().__init__()` в дочернем классе.

`__init__` — это метод-конструктор, который вызывается автоматически при создании нового объекта (`Test()`). Его задача — инициализировать атрибуты экземпляра через `self`.

## ООП: Инкапсуляция

**Инкапсуляция** — это принцип ООП, который объединяет данные и методы, работающие с ними, внутри одного объекта, и ограничивает прямой доступ к внутренним деталям извне.

Цель — скрыть реализацию и дать доступ к объекту только через контролируемый интерфейс.

---

Как реализована в Python? В отличие от Java/C++, Python не имеет жёстких модификаторов доступа. Вместо этого используются **соглашения по именованию**:

#### 1. Публичный атрибут — `name`
Доступен отовсюду, никаких ограничений.
```python
class User:
    def __init__(self):
        self.name = "Alice"  # публичный

u = User()
print(u.name)  # Alice ✅
```

---

#### 2. "Защищённый" атрибут — `_name`
Один подчёркивания — **соглашение**: "не трогай снаружи, но технически можно". Сигнал для других разработчиков.
```python
class User:
    def __init__(self):
        self._age = 30  # "защищённый"

u = User()
print(u._age)  # 30 — сработает, но это плохой тон
```

наследуется свободно

---

#### 3. "Приватный" атрибут — `__name`
Два подчёркивания включают **name mangling** — Python автоматически переименовывает атрибут в `_ClassName__name`, делая случайный доступ извне сложнее.
```python
class User:
    def __init__(self):
        self.__password = "secret"  # "приватный"

u = User()
print(u.__password)        # ❌ AttributeError
print(u._User__password)   # "secret" — обойти можно, но это явный взлом
```

Наследуется с сюпризом из-за name mangling.

---

#### 4. Контролируемый доступ через `@property`
Правильный питоновский способ инкапсуляции — геттеры и сеттеры через декоратор:
```python
class User:
    def __init__(self, age):
        self.__age = age

    @property
    def age(self):
        return self.__age  # геттер

    @age.setter
    def age(self, value):
        if value < 0:
            raise ValueError("Возраст не может быть отрицательным")
        self.__age = value  # сеттер с валидацией

u = User(25)
print(u.age)   # 25 ✅
u.age = -1     # ❌ ValueError
```

> В Python инкапсуляция строится на **доверии и соглашениях**, а не на жёстких запретах. Как говорят: *"we're all consenting adults here"*.

[Expert Python programming](https://cloud.mail.ru/public/BkKr/tEis9Thqu)
[asyncio for LLM calls](https://www.linkedin.com/posts/jimi-vaubien_ai-asyncio-llm-activity-7323270031750983681-MnYJ/)

