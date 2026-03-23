# GIL

**GIL (Global Interpreter Lock)** — это глобальная блокировка, которая гарантирует, что в каждый момент времени байткод Python выполняется только одним потоком.

Python изначально проектировался как **простой и быстрый интерпретатор**, а не как высокопараллельный runtime. С GIL:

- операции с объектами не требуют fine-grained locks
- reference counting становится **очень дешёвым**
- код интерпретатора проще и быстрее

Без GIL каждая операция с объектом потребовала бы блокировок → Python был бы медленнее в 1-потоке.

Reference counting - ключевая причина: CPython использует **reference counting** для управления памятью.

```python
Py_INCREF(obj)
Py_DECREF(obj)
```

Эти операции не атомарны

Без GIL потребовались бы:

- атомарные операции
- или mutex’ы вокруг каждого INCREF/DECREF

Это резко замедляло бы весь Python, даже без многопоточности.

Практический компромисс

- CPU-bound задачи → multiprocessing
- I/O-bound задачи → threading (GIL освобождается при io)

Для долгого времени это считалось приемлемым trade-off.

Причины удалять GIL сейчас:

1. Многоядерность стала стандартом (8–64 ядер)
2. Python всё чаще используют для CPU-heavy задач
3. Конкуренты (Java, Go, Rust) масштабируются лучше
4. Экосистема созрела для изменений

Какие плюсы отключения GIL (Python 3.14+)

- Настоящий parallelism в threading. CPU-bound код действительно масштабируется: `threading.Thread(...)`, а потоки выполняются параллельно, а не по очереди. Можно эффективно использовать все ядра без `multiprocessing` и накладных расходов на процессы.
- Проще писать и поддерживать параллельный код. Не нужны процессы, IPC и `pickle`, всё работает в общей памяти. Это снижает сложность кода и количество архитектурных костылей.
- Лучше latency и throughput. Параллельное выполнение снижает задержки и повышает пропускную способность в web-серверах, streaming-пайплайнах и in-memory обработке. Особенно важно для ML inference, data processing и real-time систем.

Минусы отключения GIL (Python 3.14+)

- **Single-threaded Python may get slower.** Removing the GIL introduces extra atomic ops, locks, and barriers even in single-thread code. Expect ~5–15% slowdown depending on workload.
- **C extensions must be updated.** Экосистема C-расширений: NumPy, pandas, PyTorch, PIL и т.д. Они **десятилетиями писались под модель с GIL.** Многие расширения предполагают, что GIL защищает их внутренние структуры и вручную освобождают GIL только в compute-heavy местах. Убрать GIL раньше = **сломать половину экосистемы**
- **The interpreter becomes more complex.** The GIL was a crude but simple design choice. Without it, the runtime is harder to reason about, implement, and maintain.

GIL — это: одна большая блокировка**,** минимальный overhead

No-GIL — это: много маленьких блокировок**,** атомарные операции, сложная memory model

Иногда одна большая блокировка быстрее, чем сотни мелких. GIL-free Python даёт реальный прирост в CPU-bound задачах с несколькими потоками и общей памятью. Он почти бесполезен для I/O-bound задач, мало влияет на NumPy-код и может быть медленнее в однопоточном сценарии.

в GIL-free Python потоки `threading.Thread` действительно могут выполняться параллельно на разных ядрах — автоматически, без `multiprocessing` но это не «магия», а смена модели исполнения интерпретатора.

```python
def work():
    for _ in range(10**8):
        pass

for _ in range(4):
    threading.Thread(target=work).start()
```

**Результат с GIL:** CPU ~100% одного ядра, время ≈ как у одного потока, потоки просто **делят GIL по тайм-слайсам** 👉 threading ≠ parallelism для CPU-bound задач

**Результат без GIL:** нет глобальной блокировки, то есть несколько потоков могут одновременно исполнять Python байткод и реально работать на разных ядрах. Тот же код: CPU ~400% на 4-ядерной машине, Время ≈ в 3–4 раза меньше, потоки действительно параллельны 👉 threading = реальный parallelism

GIL = глобальный mutex

```python
[ Thread 1 ] --\
[ Thread 2 ] --- GIL ---> Python bytecode
[ Thread 3 ] --/
```

No-GIL = fine-grained синхронизация

```python
[ Thread 1 ] ---> core 1
[ Thread 2 ] ---> core 2
[ Thread 3 ] ---> core 3
```

Основные причины использовать threading (до GIL-free): I/O-bound задачи. Ключевой момент: **threading в Python никогда не был про CPU-bound**.

```python
def download(url):
    requests.get(url)
```

Поток **освобождает GIL** во время: ожидания сети, чтения файлов, sleep

В это время другие потоки реально работают, поэтому: `threading` отлично масштабируется для I/O.

✅ Будет ускорение если:

- CPU-bound Python-код
- мало shared-mutable state
- локальные данные
- нет глобальных блокировок

❌ Может не дать прироста если:

- много shared объектов
- частые записи в dict / list
- contention на locks
- код сильно синхронизирован

Когда GIL-free дает прирост

1. **CPU-bound задачи на чистом Python**. Численные циклы, парсинг, компрессия, обработка графов и вычислительная бизнес-логика наконец начинают масштабироваться с потоками. С GIL даже 8 потоков нагружали только одно ядро на 100%, без GIL те же 8 потоков дают ~600–750% загрузки CPU. Ускорение близко к количеству ядер, за вычетом синхронизационных оверхедов - именно ради этого ключевого сценария весь No-GIL и затевался.
2. **Shared-memory параллелизм (**без `multiprocessing` ): вместо процессов, IPC, `pickle` и копирования памяти (это нужно было для GIL python) можно использовать обычный `threading` и работать с общими структурами данных. Это особенно полезно для in-memory графов, больших словарей, кешей и пайплайнов обработки. В результате снижается потребление памяти, уменьшается latency и код становится значительно проще.
3. **Высокая конкуренция потоков:** Примеры: web-серверы, очереди задач, event-driven системы. При большом числе активных потоков: исчезает глобальный bottleneck, лучше масштабирование по ядрам, меньше tail-latency
4. **Алгоритмы с fine-grained параллелизмом** характеризуются наличием множества коротких задач и интенсивным переключением контекста внутри одного процесса. Multiprocessing тут был слишком дорог → No-GIL реально выигрывает.

Когда выигрыша не будет

- I/O-bound задачи: HTTP-запросы, работа с БД, файловый ввод-вывод. Причины: GIL и так освобождался на I/O тут bottleneck — сеть или диск. asyncio / threading / event loop — всё так же эффективно.
- NumPy / PyTorch / SciPy парадоксально, но NumPy уже: написан на C, сам использует OpenMP / BLAS, освобождает GIL в тяжёлых операциях
- Малые задачи и однопоточный код**:** No-GIL → больше: атомарных операций, memory fences, locks. В этом слуйчае GIL-free может быть **медленнее** на **–5…–15%**.
- Плохо синхронизированный код. Если ты: активно используешь shared state, часто лочишь структуры, пишешь “lock-heavy” код то GIL-free может: не дать ускорения, дать деградацию, привести к contention

**Опасность GIL-free Python:** конкретный, минимальный пример, который:

- с GIL работал “нормально”
- в GIL-free Python ломает данные

❌ Небезопасный код (раньше «случайно работал»)

```python
import threading

counter = 0

def work():
    global counter
    for _ in range(1_000_000):
        counter += 1

threads = []
for _ in range(4):
    t = threading.Thread(target=work)
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print(counter)
```

Что было **с GIL:** `counter += 1` **не атомарна** но GIL гарантировал что байткод выполняется последовательно, поэтому результат почти всегда = `4_000_000`📌 **Ошибка была скрыта**, но код *не был корректным*.

Что будет **в GIL-free Python** - рандомные результаты каждый раз

```python
2_781_342
3_105_998
3_912_004
```

`counter += 1` раскладывается в:

```python
LOAD counter
LOAD_CONST 1
BINARY_ADD
STORE counter
```

В GIL-free режиме:

```python
Thread A: LOAD counter  → 100
Thread B: LOAD counter  → 100
Thread A: STORE 101
Thread B: STORE 101   ❌ потеряли инкремент
```

Получаем **classic race condition.** Правильный вариант (обязательно в GIL-free):  Используем Lock

```python
import threading

counter =0
lock = threading.Lock()

defwork():
global counter
for _inrange(1_000_000):
with lock:
     counter +=1
```

это корректно но параллелизм теряется

### list.append в GIL-free

почему `list.append` в старом Python (с GIL) считался «безопасным» и что изменилось с GIL-free.

Как работает list.append в CPython?`list` в CPython — **динамический массив** (contiguous buffer), Когда вызываешь `list.append(x)`:

1. Проверяется, хватает ли **allocated capacity**
2. Если хватает → элемент добавляется в конец массива
3. Если не хватает → создаётся новый буфер (resize), копирование старых элементов

Эта операция выглядит **атомарной на Python уровне**, потому что она **выполняется в одном байткоде**, защищённом GIL.

Роль GIL: **гарантировать, что только один поток выполняет Python-байткод одновременно, з**начит:

```python
lst = []
lst.append(1)
lst.append(2)
```

Даже если несколько потоков вызывали append одновременно: один поток завершит append до того, как другой начнёт и race condition **не проявляется**

**Вывод:** `list.append` выглядел безопасным **без явных Lock’ов**

Почему это «ложное чувство безопасности»: без GIL или при отключении GIL:

```python
lst = []
defwork():
for _inrange(100_000):
        lst.append(1)
```

- Несколько потоков могут одновременно исполнять append
- Поток А может модифицировать массив и pointer на конец, поток B тоже
- Итог: элементы **теряются** и возможен **crash при resize**, если два потока одновременно расширяют массив

Пример, который ломается в GIL-free

```python
import threading

lst = []

defwork():
for _inrange(100_000):
        lst.append(1)

threads = [threading.Thread(target=work)for _inrange(4)]
for tin threads: t.start()
for tin threads: t.join()

print(len(lst))# < 400_000
```

Thread-safe append без GIL невозможен **без lock**

Как делать правильно в GIL-free Python

Вариант 1 — Lock, Безопасно, но немного теряется параллелизм

```python
lock = threading.Lock()

defwork():
	for _inrange(100_000):
		with lock:
			lst.append(1)
```

Вариант 2 — локальные буферы: Минимизируем contention и получаем почти линейное масштабирование

```python
defwork():
    local = [1]*100_000
with lock:
        lst.extend(local)
```

Вывод

- **Старый Python (с GIL)**: `list.append` казался атомарным и thread-safe
- **Новый Python (GIL-free)**: append **не атомарен**, race condition возможен
- Любой shared mutable state теперь требует явного Lock или стратегии локальных буферов

Как `list.append` ведёт себя с GIL

- Есть список: `lst = []`
- Два потока одновременно делают: `lst.append(1)`

GIL

```python
Thread 1             Thread 2
----------           ----------
 acquires GIL
 append 1            (ждёт GIL)
 releases GIL
                     acquires GIL
                     append 1
                     releases GIL

```

Res `lst = [1, 1]` GIL сериализует байткод, каждый `append` выполняется атомарно

GIL-free Python

```python
Thread 1             Thread 2
----------           ----------
load lst pointer
                     load lst pointer
increment size pointer
                     increment size pointer  <-- конфликт!
write element
                     write element          <-- потерянный элемент

```

Race condition проявляется, список **портится**, элементы теряются

Решение

```python
lock = threading.Lock()

def safe_append(x):
    with lock:
        lst.append(x)
```

res

```python
Thread 1 acquires lock -> append -> release
Thread 2 waits -> append -> release
```

# Thread-Safe структуры в Python: GIL vs GIL-free

**Thread-safe** — это когда **операция корректно работает при параллельном выполнении потоков**, и **данные не повреждаются**.

- Для однопоточного кода это очевидно
- Для многопоточного кода нужно гарантировать, что **race condition не возникает**

Разберём thread-safe структуры данных в Python и какие гарантии они дают в старом Python с GIL и в новом GIL-free Python.

### Старый Python с GIL

Все операции с built-in структурами вроде `list.append`, `dict[key]=value`, `set.add` казались атомарными.

Почему? GIL сериализует байткод, поэтому race condition не проявлялось на уровне отдельных операций.

Примеры thread-safe операций с GIL:

| Структура | Пример безопасной операции |
| --- | --- |
| `list` | `append`, `pop` (с конца) |
| `dict` | `dict[key] = value`, `del dict[key]` |
| `set` | `add`, `remove` |
| `int` | `+=` (для маленьких значений) |

Ограничение: атомарны только одиночные операции. Сочетание нескольких операций (`if key not in dict: dict[key]=val`) уже небезопасно.

### Новый Python (GIL-free)

- GIL больше не гарантирует атомарность
- Все built-in структуры **не thread-safe**
- Любая модификация shared state может привести к race condition

Примеры потенциальных проблем:

```python
lst.append(1)      # unsafe без lock
dict[key] = value  # unsafe без lock
set.add(x)         # unsafe без lock
counter += 1       # unsafe без lock

```

Даже такие простые операции теперь требуют `Lock`, если несколько потоков используют структуру.

### 3 Как сделать thread-safe структуры

3.1. Использовать Lock или RLock

```python
lock = threading.Lock()
with lock:
    lst.append(1)

```

Гарантирует, что только один поток модифицирует структуру.

3.2. Queue (thread-safe из коробки)

Python предоставляет готовые структуры с внутренней синхронизацией:

| Структура | Thread-safe? | Комментарий |
| --- | --- | --- |
| `queue.Queue` | ✅ | FIFO очередь |
| `queue.LifoQueue` | ✅ | LIFO стек |
| `queue.PriorityQueue` | ✅ | приоритетная очередь |
| `collections.deque` | частично | `append`/`pop` с одного конца; для полного safety нужен Lock |

3.3. Atomic / локальные буферы

Для CPU-bound кода эффективнее использовать локальные массивы и потом атомарно merge.

```python
local = [1] * 100_000
with lock:
    lst.extend(local)

```

4️⃣ Итоговое правило

| Структура | Старый Python | Новый GIL-free |
| --- | --- | --- |
| `list.append` | ✅ thread-safe | ❌ unsafe |
| `dict[key]=val` | ✅ thread-safe | ❌ unsafe |
| `set.add` | ✅ thread-safe | ❌ unsafe |
| `queue.Queue` | ✅ thread-safe | ✅ thread-safe |
| `collections.deque` | частично | частично, лучше lock |

5️⃣ Короткий вывод

- В **старом Python** GIL скрывал race condition, поэтому многие операции выглядели thread-safe.
- В **новом GIL-free Python** никакие обычные структуры (`list`, `dict`, `set`) не гарантируют безопасность.
- Для безопасной многопоточности нужно использовать:
    - `Lock`, `RLock`
    - thread-safe `Queue`
    - локальные буферы + merge

## Deadlock

**Deadlock** — взаимная блокировка, когда потоки ждут друг друга бесконечно.

Пример кода

```python
lock_a = threading.Lock()
lock_b = threading.Lock()

def thread1():
    with lock_a:
        time.sleep(0.1)
        with lock_b:
            print("Thread1 done")

def thread2():
    with lock_b:
        time.sleep(0.1)
        with lock_a:
            print("Thread2 done")

```

Thread1 держит lock_a и ждёт lock_b, Thread2 держит lock_b и ждёт lock_a → deadlock.

Разница между старым и новым Python

**Старый Python (с GIL):**

- GIL ограничивал параллельное выполнение
- CPU-bound код выполнялся по очереди
- Deadlock проявлялся в основном при I/O операциях

**Новый Python (GIL-free):**

- Потоки работают реально параллельно
- CPU-bound deadlock проявляется всегда
- Требуется явный контроль race conditions

Как избежать

- Минимум shared state - Архитектура подталкивает к функциональному стилю: передал данные → получил результат. Меньше общих ресурсов = меньше рисков deadlock.
- Соблюдать единый порядок захвата locks
- Использовать RLock для повторной блокировки
- Минимизировать shared state
- Применять ThreadPoolExecutor

### Примитивы синхронизации

Примитивы синхронизации нужны для безопасной работы с shared memory между потоками.

**Mutex (Lock)** — гарантирует, что только один поток может выполнять критическую секцию кода одновременно. Используется для защиты shared переменных, dict, list. В Python это threading.Lock().

**RLock (Reentrant Lock)** — поток, который уже взял lock, может взять его снова без блокировки. Применяется в рекурсивных функциях.

**Semaphore** — разрешает N потокам одновременно заходить в критическую секцию. Используется для ограничения доступа к ресурсам, например пул соединений.

**Event** — сигнализирует потокам о каком-то событии. Один поток вызывает event.set(), другие ждут через event.wait().

**Condition** — поток может ждать определённого условия, освобождая lock. Применяется в паттерне producer-consumer, для работы с очередями.

**Barrier** — ждёт, пока N потоков достигнут точки синхронизации, потом все продолжают. Используется для синхронизации шагов в параллельных алгоритмах.

Правило выбора: один поток в критической секции — Lock, рекурсия — RLock, ограничение числа потоков — Semaphore, сигнал другим потокам — Event, ожидание условия — Condition, синхронизация N потоков — Barrier.

### Mutex (Mutual Exclusion Lock)

**Что делает:** гарантирует, что **только один поток** может выполнять критическую секцию кода одновременно.

**Применение:** защита shared переменных, dict, list.

```python
lock = threading.Lock()
with lock:
    counter +=1# только один поток одновременно
```

### RLock (Reentrant Lock)

**Что делает:** поток, который уже взял lock, может взять его снова без блокировки.

**Применение:** рекурсивные функции или повторное использование lock.

**Python:** `threading.RLock()`

### Semaphore

**Что делает:** разрешает **N потокам одновременно** заходить в критическую секцию.

**Применение:** ограничение доступа к ресурсам (например, пул соединений).

**Python:** `threading.Semaphore(value=N)`

### Event

**Что делает:** сигнализирует потокам о каком-то событии.

**Применение:** ожидание состояния, координация.

```python
event = threading.Event()
event.wait()# ждет пока другой поток вызовет event.set()
```

### Condition

**Что делает:** поток может ждать **определённого условия**, освобождая lock.

**Применение:** producer-consumer, очереди.

**Python:** `threading.Condition()`

### Barrier

**Что делает:** ждёт, пока **N потоков достигнут точки синхронизации**, потом все продолжают.

**Применение:** синхронизация шагов в параллельных алгоритмах.

**Python:** `threading.Barrier(N)`