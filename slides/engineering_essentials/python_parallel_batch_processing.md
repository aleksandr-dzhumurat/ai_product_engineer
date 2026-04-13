# Async/Await и Event Loop в Python — подробный разбор

## Содержание

1. [Контекст задачи: 1М URL](#1-контекст-задачи-1м-url)
2. [Threading vs Async — честное сравнение](#2-threading-vs-async--честное-сравнение)
3. [Event Loop — механика изнутри](#3-event-loop--механика-изнутри)
4. [Коллбэки — что это такое на самом деле](#4-коллбэки--что-это-такое-на-самом-деле)
5. [async def и корутины](#5-async-def-и-корутины)
6. [await — точка передачи управления](#6-await--точка-передачи-управления)
7. [Полная цепочка выполнения](#7-полная-цепочка-выполнения)
8. [Корнер-кейсы и ловушки](#8-корнер-кейсы-и-ловушки)

---

## 1. Контекст задачи: 1М URL

Задача: прочитать CSV с 1 000 000 URL, сделать HTTP-запрос к каждому, сохранить результат на диск.

Это классическая **I/O-bound** задача — процессор большую часть времени простаивает, ожидая ответа сети.

### Итоговое решение

```python
async def run(csv_path, out_path, concurrency=500):
    sem      = asyncio.Semaphore(concurrency)
    queue    = asyncio.Queue(maxsize=concurrency * 4)
    connector = aiohttp.TCPConnector(limit=concurrency, ttl_dns_cache=300)

    writer_task = asyncio.create_task(write_results(out_path, queue))

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [asyncio.create_task(fetch_and_enqueue(session, sem, queue, u))
                 for u in iter_urls(csv_path)]
        for coro in tqdm.as_completed(tasks, total=len(tasks)):
            await coro

    await queue.put(None)   # sentinel для writer
    await writer_task
```

**Характеристики:** ~500–2000 req/s, память < 500 МБ, 1М URL ≈ 10–30 минут.

---

## 2. Threading vs Async — честное сравнение

### Распространённое заблуждение

> "Threading хуже для I/O потому что GIL не отпускается"

Это **неверно**. GIL отпускается при любой I/O-операции — `socket.recv()`, `file.read()` и т.д. `ThreadPoolExecutor` с `requests` реально работает для этой задачи.

### Где реальная разница

| Параметр | `ThreadPoolExecutor` | `asyncio` |
|---|---|---|
| Память на 500 workers | ~4 ГБ (стек ~8 МБ × 500) | несколько МБ (корутина — объект в heap) |
| Переключение контекста | syscall ядра, 1–10 мкс | userspace `await`, нано-секунды |
| Semaphore | `threading.BoundedSemaphore` (локи + condvar) | счётчик в памяти |
| Читаемость стектрейсов | хорошая | хуже (цепочки корутин) |
| Порог рентабельности | до ~200 потоков | от ~500 корутин |

### Честный вывод

`ThreadPoolExecutor` с `requests` и `max_workers=200` **тоже справится** с 1М URL.
Async выгоднее при concurrency 500+ и когда важна latency. Выбор часто определяется тем, с чем команда лучше знакома.

### Когда threading предпочтительнее

Если внутри обработки есть CPU-heavy часть (парсинг HTML, тяжёлая обработка) — правильный выбор `ProcessPoolExecutor` для CPU + async для сети, а не чистый threading.

---

## 3. Event Loop — механика изнутри

### Метафора

Синхронный код — официант который принял заказ, ушёл на кухню, стоит и ждёт, потом несёт, и только потом идёт к следующему столику.

Event loop — умный официант: принял заказ у стола 1, отдал на кухню, сразу пошёл к столу 2, потом к столу 3. Когда кухня крикнула "готово!" — забрал и отнёс. Всё в одном потоке.

### Внутренние структуры

```python
# asyncio/base_events.py
self._ready     = deque()   # Handle-ы готовые прямо сейчас
self._scheduled = []        # TimerHandle-ы, min-heap по времени срабатывания
self._selector  = ...       # epoll (Linux) / kqueue (macOS) / IOCP (Windows)
```

### Одна итерация `_run_once()`

```python
def _run_once(self):
    # 1. Переносим из _scheduled в _ready всё что уже пора
    now = self.time()
    while self._scheduled:
        handle = self._scheduled[0]
        if handle._when > now:
            break
        heapq.heappop(self._scheduled)
        self._ready.append(handle)

    # 2. Спрашиваем ОС: какие I/O события готовы?
    #    timeout = время до следующего запланированного события
    timeout = self._scheduled[0]._when - now if self._scheduled else None
    event_list = self._selector.select(timeout)  # epoll_wait / kevent

    # 3. Для каждого готового I/O события — достаём коллбэк → в _ready
    self._process_events(event_list)

    # 4. Выполняем всё из _ready (ровно столько, сколько было на входе итерации)
    ntodo = len(self._ready)
    for _ in range(ntodo):
        handle = self._ready.popleft()
        handle._run()
```

Один `handle._run()` = один шаг корутины, от одного `await` до следующего.

---

## 4. Коллбэки — что это такое на самом деле

Коллбэк в контексте event loop — это `(функция, аргументы)`, упакованные в объект `Handle`:

```python
# asyncio/events.py
class Handle:
    def __init__(self, callback, args):
        self._callback = callback
        self._args = args

    def _run(self):
        self._callback(*self._args)
```

### Три источника коллбэков

**`call_soon` — выполни на следующей итерации:**
```python
loop.call_soon(my_func, arg1)
# → Handle(my_func, [arg1]) в deque._ready
```
`asyncio.create_task()` внутри вызывает `call_soon` чтобы поставить первый шаг корутины в очередь.

**`call_later` / `call_at` — выполни через N секунд:**
```python
loop.call_later(5.0, my_func)
# → TimerHandle в _scheduled (min-heap по времени)
```
`asyncio.sleep(5)` работает именно так: ставит коллбэк в `_scheduled`, корутина засыпает, через 5 секунд коллбэк будит её.

**`add_reader` / `add_writer` — I/O события:**
```python
loop.add_reader(sock.fileno(), my_func)
# → регистрирует fd в epoll/kqueue
# → когда сокет readable — my_func попадает в _ready
```

### Главный коллбэк: `Task.__step`

В реальности большинство коллбэков в event loop — это `Task.__step`, метод который продвигает корутину:

```python
class Task:
    def __step(self):
        try:
            # продвигаем корутину до следующего await (yield)
            result = self.__coro.send(None)
        except StopIteration as exc:
            self.set_result(exc.value)      # корутина завершилась нормально
        except Exception as exc:
            self.set_exception(exc)         # корутина упала
        else:
            # result — Future которого ждёт корутина
            result.add_done_callback(self.__step)
            # когда Future готов — __step снова попадёт в _ready
```

---

## 5. `async def` и корутины

### Отличие от обычной функции

```python
def normal():
    return 42        # выполняется сразу, возвращает 42

async def coro():
    return 42        # возвращает объект-корутину, код НЕ запущен
```

`coro()` без `await` — как создать генератор: объект создан, код не запущен. Event loop запускает его когда получает управление через `create_task` или `await`.

### Под капотом корутина — это генератор

```python
# async def / await — синтаксический сахар над генераторами
# Примерный эквивалент того что делает CPython:

async def fetch(url):
    response = await session.get(url)
    return response

# ≈ эквивалентно (упрощённо):
def fetch(url):
    response = yield session.get(url)   # yield отдаёт Future наружу
    return response                     # StopIteration(response)
```

`send(None)` продвигает генератор до следующего `yield` (= `await`). Именно это делает `Task.__step`.

---

## 6. `await` — точка передачи управления

```python
async def fetch(url):
    print("отправляем запрос")
    response = await session.get(url)   # ← здесь корутина говорит event loop:
    print("получили ответ")             #   "я жду I/O, займись другими"
    return response
```

`await` делает две вещи одновременно:
1. Приостанавливает текущую корутину (стек сохраняется)
2. Возвращает управление event loop

Корутина не блокирует поток — она "спит" пока I/O не завершится.

### Полная цепочка для одного `await session.get(url)`

```
корутина hits await
    → task.__step() заканчивает шаг, получает Future
    → регистрирует task.__step как done_callback на Future
    → event loop крутит _run_once()
    → epoll говорит "сокет readable"
    → внутренний I/O коллбэк читает данные, резолвит Future
    → Future вызывает done_callbacks → task.__step попадает в _ready
    → на следующей итерации handle._run() → task.__step()
    → корутина продолжается после await, получает response
```

---

## 7. Полная цепочка выполнения

```python
async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(fetch(session, "http://example.com")),
            asyncio.create_task(fetch(session, "http://google.com")),
            asyncio.create_task(fetch(session, "http://github.com")),
        ]
        results = await asyncio.gather(*tasks)

asyncio.run(main())
```

### Хронология

```
t=0ms    asyncio.run() создаёт event loop, запускает main()
t=1ms    create_task × 3 → call_soon × 3 → три __step в _ready
t=1ms    task1.__step: fetch отправил запрос → await → спит
t=1ms    task2.__step: fetch отправил запрос → await → спит
t=1ms    task3.__step: fetch отправил запрос → await → спит
         (три TCP-соединения летят параллельно)
t=1ms    _ready пуст → epoll_wait(timeout=...)
t=120ms  ОС: пришёл ответ для task2
t=120ms  I/O коллбэк резолвит Future → task2.__step в _ready
t=120ms  task2 дочитывает тело, завершается
t=180ms  ОС: ответ для task1 → task1 завершается
t=210ms  ОС: ответ для task3 → task3 завершается
t=210ms  gather видит все три готовы → main() продолжается
```

Всё в **одном потоке**, никаких переключений контекста ОС.

---

## 8. Корнер-кейсы и ловушки

### Блокирующий вызов убивает весь event loop

```python
async def bad():
    time.sleep(5)          # блокирует ВЕСЬ поток на 5 секунд
    # все остальные корутины стоят — event loop заморожен

async def good():
    await asyncio.sleep(5) # отпускает event loop, остальные работают
```

**Правило:** внутри `async def` нельзя вызывать ничего блокирующего синхронно.
Опасные функции: `time.sleep`, `requests.get`, синхронный `open()` на медленных дисках, любой `subprocess` без `await`.

### CPU-heavy код внутри async

```python
async def bad():
    result = heavy_computation()   # занимает 2 секунды — весь event loop стоит

async def good():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, heavy_computation)
    # выносит в ThreadPoolExecutor, event loop свободен
```

`run_in_executor` — мост между async и блокирующим кодом. Для CPU-bound правильнее `ProcessPoolExecutor`:

```python
executor = ProcessPoolExecutor(max_workers=4)
result = await loop.run_in_executor(executor, cpu_heavy_func, arg)
```

### `create_task` vs `await` — разница принципиальная

```python
# Последовательно — медленно (ждём каждый)
async def sequential():
    r1 = await fetch(url1)   # ждём
    r2 = await fetch(url2)   # ждём
    r3 = await fetch(url3)   # ждём
    # итого: t1 + t2 + t3

# Параллельно — быстро
async def parallel():
    t1 = asyncio.create_task(fetch(url1))   # запустили
    t2 = asyncio.create_task(fetch(url2))   # запустили
    t3 = asyncio.create_task(fetch(url3))   # запустили
    r1, r2, r3 = await asyncio.gather(t1, t2, t3)
    # итого: max(t1, t2, t3)
```

`create_task` немедленно ставит корутину в event loop. `await coro()` без `create_task` — последовательно.

### `gather` vs `as_completed` — разная семантика

```python
# gather: ждёт ВСЕ, возвращает в порядке задач
results = await asyncio.gather(task1, task2, task3)

# as_completed: возвращает по мере готовности
for coro in asyncio.as_completed(tasks):
    result = await coro   # первый готовый, второй готовый, ...
```

Для 1М URL `as_completed` + `tqdm` даёт живой прогресс-бар. `gather` для такого объёма хуже — держит все Future в памяти до конца.

### `gather` при ошибке

```python
# по умолчанию: первая ошибка отменяет gather, остальные таски продолжают работать (утечка!)
results = await asyncio.gather(t1, t2, t3)

# правильно для bulk-задач: собираем все результаты включая исключения
results = await asyncio.gather(t1, t2, t3, return_exceptions=True)
for r in results:
    if isinstance(r, Exception):
        log.error(r)
```

### Semaphore — не забыть `async with`

```python
sem = asyncio.Semaphore(500)

async def fetch_one(url):
    async with sem:          # acquire + release автоматически
        return await session.get(url)

# Если сделать sem.acquire() без release() при исключении — deadlock
```

### Закрытие сессии

```python
# Плохо — сессия не закрыта при исключении
session = aiohttp.ClientSession()
await do_work(session)
await session.close()

# Хорошо — context manager гарантирует закрытие
async with aiohttp.ClientSession() as session:
    await do_work(session)
```

### `asyncio.run()` нельзя вызывать внутри уже работающего loop

```python
# Ошибка в Jupyter / уже запущенном event loop:
asyncio.run(main())   # RuntimeError: cannot be called when another event loop is running

# Решение для Jupyter:
import nest_asyncio
nest_asyncio.apply()
asyncio.run(main())

# Или:
await main()   # в Jupyter можно напрямую
```

### DNS кеширование — критично для 1М URL

```python
# Без кеша: DNS-запрос на каждый хост
connector = aiohttp.TCPConnector()

# С кешем: повторные хосты резолвятся из памяти
connector = aiohttp.TCPConnector(ttl_dns_cache=300)  # 5 минут
```

Для 1М URL с повторяющимися доменами это экономит сотни тысяч DNS-запросов.

### Queue как буфер между producer и consumer

```python
queue = asyncio.Queue(maxsize=2000)  # ограничиваем размер буфера

# producer (fetcher) — кладёт результаты
await queue.put(result)

# consumer (writer) — пишет батчами на диск
buf = []
while True:
    result = await queue.get()
    if result is None:   # sentinel
        break
    buf.append(result)
    if len(buf) >= 1000:
        await flush(buf)
        buf.clear()
```

`maxsize` важен: без него при медленном диске queue растёт до исчерпания памяти.

---

## Итоговая схема

```
asyncio.run(main())
    └── создаёт EventLoop
            ├── _ready: deque[Handle]          ← call_soon, завершённые I/O
            ├── _scheduled: heap[TimerHandle]  ← call_later, asyncio.sleep
            └── _selector (epoll/kqueue)       ← ожидание I/O от ОС

Handle._run()
    └── Task.__step(coro)
            ├── coro.send(None) → продвигает до await
            ├── получает Future
            └── future.add_done_callback(Task.__step)
                    └── когда I/O готов → __step снова в _ready
```

**Три примитива, на которых строится всё:**
- `call_soon` — запланировать шаг
- `call_later` — запланировать по времени
- `add_reader/add_writer` — запланировать по I/O событию