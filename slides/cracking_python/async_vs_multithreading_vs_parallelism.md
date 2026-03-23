**Параллелизм (parallelism)** — выполнение нескольких задач **одновременно** на нескольких ядрах процессора.

**Конкурентность (concurrency)** — управление несколькими задачами так, как будто они выполняются одновременно, даже если по факту процессор переключается между ними (например, при работе с одним ядром).

# Параллелизм (Multiprocessing)

Параллелизм требует IPC (Inter-Process Communication), механизмы общения **между процессами**:

- pipe
- socket
- shared memory
- message queue

Для **threading** не нужен IPC: потоки: **не изолированы**  и живут внутри одного процесса, для мультипроцессинга  требуется IPC + pickle и возникают накладные расходы на создание процессов.

**Изоляция сбоев:** Сбой в одном потоке может привести к завершению всего процесса, в то время как сбой в одном процессе не влияет на работу других.

Использование нескольких процессов (каждый со своей памятью и GIL) для выполнения задач одновременно на нескольких ядрах.

Модуль: `multiprocessing`.

- Для CPU-bound задач, таких как вычисления, обработка изображений, ML-модели - данные не надо шерить.
- Когда нужно использовать все доступные ядра процессора.

```python
from multiprocessing import Process

def task():
    print("Process running")

p = Process(target=task)
p.start()
```

Сколько запускать процессов: `import os; print(os.cpu_count())` (количество логических ядер). Всегда можно проверить по htop что загружены все ядра ≈ 100%

Сравнение

| Характеристика | Multithreading | Async/await | Multiprocessing |
| --- | --- | --- | --- |
| Использует GIL | Да | Да | Нет |
| Подходит для | I/O-bound задач | I/O-bound задач | CPU-bound задач |
| Параллельное выполнение | Нет (GIL) | Нет (один поток) | Да |
| Использует ядра | Одно | Одно | Много |
| Простота отладки | Средняя | Сложная | Средняя |
| Обмен данными | Легкий | Через event loop | Сложный (IPC) |

# Многопоточность (Multithreading)

Многопоточность позволяет запускать несколько потоков (threads) в одном процессе. В Python используется стандартный модуль `threading`.

Все потоки:

- живут **в одном процессе**
- видят **одну и ту же память (heap)**

🚫 Ограничения: **GIL (Global Interpreter Lock)** — основной "тормоз" многопоточности в Python. Он позволяет исполняться только одному потоку Python байткода в один момент времени. Потоки **OS-managed:** ОС переключает потоки (по очереди), но они **по очереди** держат GIL → реального параллелизма нет, потоки **не ускоряют** CPU-bound задачи. 

```python
import threading

def worker(id):
    print(f"Thread {id} starting")
# Do some work
    print(f"Thread {id} finished")

threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```

Python изначально проектировался как **простой и быстрый интерпретатор**, а не как высокопараллельный runtime: c GIL reference counting становится **очень дешёвым, код интерпретатора проще и быстрее, операции с объектами не требуют fine-grained locks.**

Почему GIL начали убирать именно сейчас: Многоядерность стала стандартом.

Пример кода который не работает из-за GIL

```python
def work():
    for _ in range(10**8):
        pass

for _ in range(4):
    threading.Thread(target=work).start()
```

Это **CPU-bound код**, а GIL делает `threading` бесполезным для такого случая.. Результат: 1 ядро загружено на ~100%. Общее время выполнения: время выполнения ≈ **время одного `work()` × 4** (даже хуже из-за расходов на переключение контекста).

Если отключить GIL - увидим загрузку всех ядер.

Основные причины использовать threading (до GIL-free), что освобождает GIL

- вызов C-кода, библиотеки NumPy, PyTorch, OpenCV задействуют все ядра.
- Когда нужно **перекрыть ожидание** — например, загрузку данных с сети, ввод/вывод.
- При **I/O-bound** задачах, таких как HTTP-запросы, файловые операции, ожидание базы данных.
- sleep()

Задачи которые **ускорятся** в GIL-free

- CPU-bound Python-код
- если мало shared-mutable state `global_sum` - плохо, лучше посчитать несколько `local_sum`
- локальные данные, нет глобальных блокировок

**Примеры** таких задач

- поиск простых чисел в диапазонах
- grid-search гиперпараметров
- feature engineering (map-only, без reduce - преобразования строго одного столбца)
- логические симуляции (monte carlo)

Задачи которые **не ускорятся**

- много shared объектов, код сильно синхронизирован чтобы не портились данные
- частые записи в dict / list: в GIL-free реализации возникают Locks чтобы не портились данные (с чтением проблем нет)
- contention (спор) на locks - воркер отработал быстро, много переключений контекстаю Если lock держится дольше, чем полезная работаю

# Асинхронность (Async / Await)

Один поток, неблокирующий ввод-вывод.

Асинхронность — это способ писать неблокирующий код, используя ключевые слова `async` и `await`. Управление задачами выполняется внутри одного потока с помощью **событийного цикла** (`event loop`).

Модуль: `asyncio`.

- При большом количестве **I/O-bound** операций.
- Когда важно **масштабировать** обработку запросов без создания лишних потоков или процессов.
- Идеально подходит для высоконагруженных API, парсинга, чат-ботов, веб-серверов (например, FastAPI).

Ограничения:

- Не подходит для **CPU-bound** задач.
- Требует, чтобы весь стек вызовов был асинхронным.

```python
import asyncio

async def task():
    print("Async task running")

asyncio.run(task())
```

Какую модель выбрать?

- **много мелких IO** → async
- **мало, но долгие IO** → threads
- **чистые вычисления** → multiprocessing
- **I/O-bound задачи** (запросы, файлы, БД) → Лучше использовать **async/await** или **multithreading**.
- **CPU-bound задачи** (вычисления, аналитика) → Используй **multiprocessing**.
- **Простая параллельная загрузка файлов/веб-страниц**:
    
    → `concurrent.futures.ThreadPoolExecutor`
    
- **Параллельные вычисления**:
    
    → `concurrent.futures.ProcessPoolExecutor` 
    

---

## Типичные вопросы на собеседованиях

- Что такое GIL и как он влияет на многопоточность?
- Когда стоит использовать `asyncio`, а когда `multiprocessing`?
- Как работает event loop?
- Как параллелизировать задачу, чтобы использовать все ядра?
- Как обрабатывать ошибки в асинхронных функциях?
- Чем отличаются потоки от процессов?

These three concepts are related to concurrent programming but have important distinctions. Let me break them down:

## Asynchronous Programming

**Definition:** A programming paradigm that allows operations to run without blocking the execution flow, using a single thread with an event loop.

Imagine that your program is making an HTTP request. The processor is simply waiting for a response—this is called an **I/O-bound operation**. Synchronous code blocks the entire thread:

```python
# Синхронно — каждый запрос ждёт предыдущего
result1 = requests.get("https://api.example.com/1")  # ждём 1 сек
result2 = requests.get("https://api.example.com/2")  # ждём ещё 1 сек
# Итого: ~2 секунды
```

Event Loop — это бесконечный цикл, который:

1. Смотрит на очередь задач (корутин)
2. Запускает задачу до первого `await`
3. Пока задача **ждёт** (сеть, диск) — переключается на другую
4. Возвращается, когда данные готовы

```python
┌─────────────────────────────────────┐
│             EVENT LOOP              │
│                                     │
│  [Task A] → await → приостановлен   │
│  [Task B] → await → приостановлен   │
│  [Task C] → выполняется...          │
│       ↓                             │
│  Task A получил данные → продолжает │
└─────────────────────────────────────┘
```

Вызов `fetch_data(url)` **не запускает** функцию — он создаёт объект корутины. Чтобы запустить — нужен `await` или `asyncio.run()`.

```python
async def fetch_data(url):
    # Это корутина — функция, которую можно приостановить
    ...
```

await — точка приостановки

```python
import asyncio
import aiohttp

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()  # ← здесь корутина приостанавливается
                                      #   и event loop берёт следующую задачу

async def main():
    async with aiohttp.ClientSession() as session:
        # Запускаем ОБА запроса почти одновременно
        task1 = asyncio.create_task(fetch(session, "https://api.example.com/1"))
        task2 = asyncio.create_task(fetch(session, "https://api.example.com/2"))
        
        result1, result2 = await asyncio.gather(task1, task2)
        # Итого: ~1 секунда вместо 2!

asyncio.run(main())  # запускает event loop
```

Python Asyncio

| Концепция | Что делает |
| --- | --- |
| `async def` | Объявляет корутину |
| `await` | Приостанавливает корутину, отдаёт управление event loop |
| `asyncio.create_task()` | Планирует корутину для выполнения |
| `asyncio.gather()` | Запускает несколько задач параллельно |
| `asyncio.run()` | Создаёт event loop и запускает корутину |

**async/await — это НЕ многопоточность.** Всё выполняется в одном потоке. Это **кооперативная многозадачность** — задачи сами уступают управление через `await`.

**Key Characteristics:**

- **Event Loop Based:** Operations yield control when waiting, allowing other operations to run
- **Non-blocking:** Functions return control to the caller before completing
- **Cooperative:** Tasks voluntarily yield control (no preemption)

**Core mental model:** Async is *cooperative* single-threaded concurrency. A single event loop runs on one thread. When a coroutine hits an `await`, it *yields control* back to the event loop, which can then run another coroutine. No threads, no GIL contention.

**Key definitions:**

- **Event loop** — The central scheduler. It maintains a queue of ready tasks and runs them one at a time. `asyncio.run()` creates and runs the event loop.
- **Coroutine** — A function defined with `async def`. Calling it returns a coroutine *object*; it does not execute until awaited or scheduled.
- **`await`** — Suspends the current coroutine until the awaited future/coroutine completes, giving the event loop a chance to run other tasks.

**When async helps:** I/O-bound concurrency where you're making many network calls or waiting on external services. A single async server can handle thousands of in-flight requests with minimal memory overhead compared to a thread-per-request model.

**Typical pitfalls:**

1. Accidentally calling a blocking function (e.g., `requests.get()`) inside an `async` function — this blocks the entire event loop. Use `aiohttp` instead, or wrap blocking calls in `asyncio.to_thread()`. Use asyncio.sleep() instead of time.sleep()
2. Forgetting to `await` a coroutine — it silently does nothing and Python emits a warning.
3. Mixing sync and async code incorrectly — e.g., calling `asyncio.run()` inside an already-running event loop (common in Jupyter notebooks; use `nest_asyncio` or restructure).
4. CPU-bound work inside coroutines — async does not help here; offload to a process pool via `loop.run_in_executor()`.

**Best For:**

- High-concurrency I/O operations (network servers, web scraping)
- Maintaining responsiveness in interactive applications
- Scenarios where thread overhead would be problematic

**Challenges:**

- Different programming model (callbacks, promises, coroutines)
- Cannot utilize multiple CPU cores on a single event loop
- All tasks must cooperate to prevent blocking the loop

Example (Python):

```python
import asyncio

async def worker(id):
    print(f"Task {id} starting")
    await asyncio.sleep(1)# Non-blocking sleep
    print(f"Task {id} finished")

async def main():
    tasks = [worker(i) for i in range(5)]
    await asyncio.gather(*tasks)

asyncio.run(main())
```

Consider the following Python code:

```python
import asyncio

async def task1():
    print("Task 1 started")
    await asyncio.sleep(2)
    print("Task 1 finished")
    return "Result from Task 1"

async def task2():
    print("Task 2 started")
    await asyncio.sleep(1)
    print("Task 2 finished")
    return "Result from Task 2"

async def main():
    print("Main started")

    # Option 1: Sequential execution
    result1 = await task1()
    result2 = await task2()

    print("Results from sequential execution:")
    print(result1, result2)

    # Option 2: Concurrent execution
    task1_coro = task1()
    task2_coro = task2()
    result1, result2 = await asyncio.gather(task1_coro, task2_coro)

    print("Results from concurrent execution:")
    print(result1, result2)

    print("Main finished")

# Run the main function
asyncio.run(main())
```

1. What will be the output of the code above?
2. Explain the difference between the sequential execution and concurrent execution parts of the code.
3. What is the purpose of `asyncio.gather()` in the context of asynchronous programming in Python?
4. How would the behavior of the code change if `await asyncio.gather(task1(), task2())` was replaced with two separate `await` statements for each task?
5. How does the `asyncio.run()` function work, and why is it necessary in this code?

**Expected Answers:**

The output of the code will be:

```
Main started
Task 1 started
Task 1 finished
Task 2 started
Task 2 finished
Results from sequential execution:
Result from Task 1 Result from Task 2
Task 1 started
Task 2 started
Task 2 finished
Task 1 finished
Results from concurrent execution:
Result from Task 1 Result from Task 2
Main finished

```

1. Sequential Execution: In this part of the code, `task1()` is awaited before `task2()` starts. This means `task2()` does not begin until `task1()` has completed, resulting in a total wait time equal to the sum of the two task durations (3 seconds).
2. Concurrent Execution: Here, `task1()` and `task2()` are started nearly simultaneously using `asyncio.gather()`. This allows both tasks to run concurrently, resulting in a shorter total wait time, which is the duration of the longer task (2 seconds).
3. `asyncio.gather()` is used to run multiple asynchronous operations concurrently. It takes in multiple coroutine objects and returns a single coroutine that gathers the results of all the coroutines. This allows the tasks to run in parallel, leading to more efficient execution when tasks involve waiting (e.g., I/O operations).
4. If `await asyncio.gather(task1(), task2())` is replaced with two separate `await` statements (`await task1(); await task2();`), the tasks will be executed sequentially, just like in the first part of the code. This would negate the concurrent execution and result in a longer total wait time.
5. `asyncio.run()` is a high-level function that is used to execute an asynchronous function from the top level of a Python program. It sets up an event loop, runs the given coroutine, and then closes the loop. It is necessary in this code to start and manage the asynchronous execution of the `main()` function.

This question tests the candidate's understanding of asynchronous programming in Python, particularly the differences between sequential and concurrent execution, the use of `asyncio.gather()`, and the role of the event loop in managing asynchronous tasks.

### **Question:** Concurrency and Asynchronous Programming

Consider the following Python code snippet that uses the `asyncio` library to handle asynchronous tasks:

```python
import asyncio

async def fetch_data(id):
    print(f"Fetching data for id: {id}")
    await asyncio.sleep(1)  # Simulate an I/O operation
    print(f"Data fetched for id: {id}")
    return f"Data for {id}"

async def process_data(id):
    print(f"Processing data for id: {id}")
    await asyncio.sleep(2)  # Simulate processing time
    print(f"Data processed for id: {id}")

async def main():
    ids = [1, 2, 3]
    tasks = []

    for id in ids:
        data = await fetch_data(id)
        task = asyncio.create_task(process_data(id))
        tasks.append(task)

    await asyncio.gather(*tasks)

asyncio.run(main())
```

1. What is the purpose of `asyncio.sleep()` in this code? How does it impact the behavior of the `fetch_data` and `process_data` functions?
2. Explain how the `asyncio.create_task()` function is used in this code. What does it do, and why is it important?
3. What will be the output of this code when executed? Describe the order of print statements.
4. What changes would you make to the `main()` function if you want to ensure that all `fetch_data` operations are completed before any `process_data` tasks are started? Provide a revised version of the `main()` function.
5. How would you modify this code to handle exceptions that might be raised during the execution of `fetch_data` or `process_data`? Illustrate your changes with code.

**Expected Answers:**

1. `asyncio.sleep()` is used to simulate asynchronous I/O operations and delays. It allows other tasks to run while the current task is waiting. In `fetch_data`, it simulates fetching data, while in `process_data`, it simulates processing time.
2. `asyncio.create_task()` schedules the `process_data` function to run concurrently with other tasks. It creates a Task object that runs the specified coroutine in the background, allowing other tasks to be scheduled and executed concurrently. This is crucial for running multiple tasks concurrently without blocking.
3. The output will show `Fetching data for id` messages for each `id`, followed by `Data fetched for id` messages, and then `Processing data for id` messages, and finally `Data processed for id` messages. The exact order may vary, but `Fetching data` and `Data fetched` will appear for all ids before any `Processing data` and `Data processed`.
    
    ```
    Fetching data for id: 1
    Fetching data for id: 2
    Fetching data for id: 3
    Data fetched for id: 1
    Data fetched for id: 2
    Data fetched for id: 3
    Processing data for id: 1
    Processing data for id: 2
    Processing data for id: 3
    Data processed for id: 1
    Data processed for id: 2
    Data processed for id: 3
    
    ```
    

Revised `main()` Function:

```python
async def main():
    ids = [1, 2, 3]
    fetch_tasks = [fetch_data(id) for id in ids]
    data_results = await asyncio.gather(*fetch_tasks)

    tasks = [asyncio.create_task(process_data(id)) for id in ids]
    await asyncio.gather(*tasks)

```

- In this revised function, `fetch_data` tasks are awaited and completed before any `process_data` tasks are started.

Handling Exceptions:

```python
async def fetch_data(id):
    try:
        print(f"Fetching data for id: {id}")
        await asyncio.sleep(1)
        print(f"Data fetched for id: {id}")
        return f"Data for {id}"
    except Exception as e:
        print(f"Error fetching data for id: {id}: {e}")
        return None

async def process_data(id):
    try:
        print(f"Processing data for id: {id}")
        await asyncio.sleep(2)
        print(f"Data processed for id: {id}")
    except Exception as e:
        print(f"Error processing data for id: {id}: {e}")

async def main():
    ids = [1, 2, 3]
    fetch_tasks = [fetch_data(id) for id in ids]
    data_results = await asyncio.gather(*fetch_tasks)

    tasks = [asyncio.create_task(process_data(id)) for id in ids]
    await asyncio.gather(*tasks)

```

- In this updated code, exceptions in `fetch_data` and `process_data` are caught and logged, ensuring that the program continues to run even if errors occur.

## Parallelism

**Definition:** Executing multiple computations simultaneously using multiple processing units (CPUs/cores).

Key Characteristics:

- **True Simultaneous Execution:** Tasks actually run at the same time
- **Separate Memory Spaces:** Often using separate processes
- **CPU-Intensive:** Makes full use of multiple CPU cores

Best For:

- CPU-bound tasks (data processing, simulations, rendering)
- Computationally expensive operations
- Maximizing hardware utilization

Challenges:

- Process creation overhead
- Inter-process communication complexity
- Memory usage (separate memory spaces)

Example (Python):

```python
from multiprocessing import Pool

def worker(id):
    print(f"Process {id} working")
    result = sum(i*i for i in range(10_000_000))
    return result

if __name__ == "__main__":
    with Pool(4) as p:
        results = p.map(worker, range(5))
    print("Results:", results)
```

Key Differences Summarized

| Aspect | Multithreading | Asynchronous | Parallelism |
| --- | --- | --- | --- |
| **Execution Model** | Multiple threads | Single thread with event loop | Multiple processes |
| **Best For** | Mixed I/O and CPU tasks | I/O-bound tasks | CPU-bound tasks |
| **Utilizes Multiple Cores** | Partially (with GIL limitations in Python) | No (single thread) | Yes (fully) |
| **Concurrency Control** | Locks, semaphores | Awaitable objects, coroutines | Message passing |
| **Memory Model** | Shared memory | Shared memory | Separate memory |
| **Context Switching** | OS-managed (preemptive) | Developer-managed (cooperative) | OS-managed (between processes) |

## Real-World Examples

1. **Web Server:**
    - **Multithreading:** Each request handled by a separate thread
    - **Asynchronous:** One thread handles multiple requests via async/await
    - **Parallelism:** Multiple server processes, each handling requests
2. **Image Processing Application:**
    - **Multithreading:** UI thread + background processing thread
    - **Asynchronous:** Loading images asynchronously without blocking UI
    - **Parallelism:** Processing different images on different CPU cores
3. **Database System:**
    - **Multithreading:** Connection pooling and request handling
    - **Asynchronous:** Non-blocking I/O for disk operations
    - **Parallelism:** Partitioning data across multiple processes/nodes

Understanding the appropriate use case for each approach is crucial for designing efficient concurrent systems.

# Message Passing in Parallelism

Message passing is a communication paradigm used in parallel computing where processes exchange data by sending and receiving messages rather than sharing memory. It's a fundamental concept for coordinating work between multiple processes that typically operate in separate memory spaces.

Key Concepts of Message Passing

- **Isolated Memory Spaces**: Each process has its own memory that other processes cannot directly access
- **Explicit Communication**: Processes must explicitly send/receive data via messages
- **Coordination**: Synchronization between processes is achieved through messages

2. Message Types

- **Data Messages**: Transfer data between processes
- **Control Messages**: Coordinate activities (start/stop/synchronize)
- **Task Messages**: Distribute work units to processes

3. Communication Patterns

- **Point-to-Point**: Direct communication between two processes
- **Broadcast**: One process sends to all other processes
- **Scatter/Gather**: Distribute data to many processes, then collect results
- **Master-Worker**: Central process distributes tasks and collects results

Implementation Examples

### Python with Multiprocessing

```python
from multiprocessing import Process, Queue

def worker(input_queue, output_queue, worker_id):
    while True:
# Receive a message (task)
        task = input_queue.get()
        if task == "STOP":
            break

# Process the task
        result = f"Worker {worker_id} processed: {task * 2}"

# Send result message back
        output_queue.put(result)

# Create communication channels
task_queue = Queue()
result_queue = Queue()

# Create worker processes
workers = [
    Process(target=worker, args=(task_queue, result_queue, i))
    for i in range(4)
]

# Start workers
for w in workers:
    w.start()

# Send task messages
for i in range(10):
    task_queue.put(i)

# Send termination messages
for _ in range(len(workers)):
    task_queue.put("STOP")

# Collect result messages
results = []
for _ in range(10):
    results.append(result_queue.get())

# Wait for workers to finish
for w in workers:
    w.join()

print(results)

```

### MPI (Message Passing Interface)

MPI is the standard protocol for message passing in high-performance computing:

```python

from mpi4py import MPI

# Get process info
comm = MPI.COMM_WORLD
rank = comm.Get_rank()# ID of this process
size = comm.Get_size()# Total number of processes# Root process (rank 0) sends data to others
if rank == 0:
# Send different data to each worker
    for i in range(1, size):
        data = {'task': i, 'parameters': [i*10, i*20]}
        comm.send(data, dest=i)

# Collect results
    results = []
    for i in range(1, size):
        result = comm.recv(source=i)
        results.append(result)
    print("Results:", results)

# Worker processes
else:
# Receive task from root
    data = comm.recv(source=0)

# Process data
    task_id = data['task']
    params = data['parameters']
    result = sum(params) * task_id

# Send result back to root
    comm.send(result, dest=0)

```

Advantages of Message Passing

1. **Scalability**: Can scale across multiple computers/clusters
2. **Clarity**: Makes data dependencies explicit
3. **Fault Isolation**: Failures in one process don't directly corrupt others
4. **Reduced Race Conditions**: No shared memory means fewer synchronization issues
5. **Distributed Computing**: Natural model for cluster and cloud environments

Challenges of Message Passing

1. **Overhead**: Message passing has higher overhead than shared memory
2. **Complexity**: Can be more complex to implement than shared-memory models
3. **Data Duplication**: Data may need to be copied between processes
4. **Bandwidth Limitations**: Communication channels can become bottlenecks
5. **Deadlocks**: Incorrect message handling can lead to processes waiting indefinitely

Real-World Application

- **Distributed Systems**: Microservices communicating via message queues
- **High-Performance Computing**: Scientific simulations using MPI
- **Big Data Processing**: MapReduce and Spark use message passing for task distribution
- **Actor Systems**: Erlang/Elixir and Akka implement concurrency via message passing
- **GPU Computing**: Communication between CPU and multiple GPU processes

Message passing represents a powerful paradigm that enables truly parallel computation across multiple processors, machines, or even globally distributed systems, while maintaining clean separation between execution units.

# Awaitable Objects and Coroutines

Awaitable objects and coroutines are core concepts in asynchronous programming that enable non-blocking code execution. They're particularly prominent in modern languages like Python, JavaScript, C#, and Rust. Let's explore these concepts in depth:

## Coroutines

A coroutine is a function that can pause its execution, yield control back to the caller, and later resume from where it left off. Unlike regular functions that run to completion once called, coroutines can be suspended and resumed multiple times.

Key Characteristics

1. **Cooperative Multitasking**: Coroutines voluntarily yield control, allowing other code to run
2. **Preserved State**: When resumed, a coroutine continues with all its local variables intact
3. **Single-threaded**: Typically run on a single thread, eliminating many concurrency issues
4. **Sequential Appearance**: Allows asynchronous code to look like synchronous cod

```python
async def example_coroutine():
    print("Starting coroutine")
# Pause and yield control
    await asyncio.sleep(1)
    print("Resumed after 1 second")
# Pause again
    await asyncio.sleep(2)
    print("Completed after another 2 seconds")
    return "Result"

```

In this example:

- The `async` keyword declares a coroutine function
- The function can pause execution at `await` points
- While paused, other coroutines can run
- The event loop manages when to resume the coroutine

## Awaitable Objects

An awaitable object is any object that can be used with the `await` syntax. It represents an asynchronous operation that can be paused and resumed later.

Types of Awaitable Objects

1. **Coroutine Objects**: Created when calling an `async def` function
2. **Tasks**: High-level awaitable wrappers around coroutines (manages execution)
3. **Futures/Promises**: Placeholders for results that will be available later
4. **Custom Awaitables**: Objects implementing `__await__` method

How Awaitables Work

1. When you `await` an awaitable object:
    - The current coroutine is suspended
    - Control returns to the event loop
    - The event loop runs other coroutines or handles I/O
    - When the awaited operation completes, the coroutine resumes
2. The completion mechanism depends on the awaitable type:
    - I/O operations complete when data is available
    - Sleep operations complete after the specified time
    - Other coroutines complete when they return a value

Python Example with Different Awaitables

```python
import asyncio

async def fetch_data(url):
# Create a coroutine for an I/O operation
    print(f"Fetching {url}...")
    await asyncio.sleep(2)# Simulating network request
    return f"Data from {url}"

async def main():
# Awaiting a coroutine directly
    result1 = await fetch_data("example.com/api")
    print(result1)

# Awaiting a Task (wrapping a coroutine)
    task = asyncio.create_task(fetch_data("example.com/api2"))
    result2 = await task
    print(result2)

# Awaiting a Future
    future = asyncio.Future()
# Simulate setting the result after 1 second
    asyncio.create_task(set_future_result(future))
    result3 = await future
    print(result3)

async def set_future_result(future):
    await asyncio.sleep(1)
    future.set_result("Future result")

asyncio.run(main())

```

The Relationship Between Coroutines and Awaitables

- Every coroutine function (defined with `async def`) returns a coroutine object when called
- This coroutine object is awaitable
- Inside coroutines, you can only `await` awaitable objects
- The `await` expression extracts the result of the awaitable when it completes

### Event Loop and Execution Model

The event loop is the orchestrator that:

1. Maintains a queue of tasks and coroutines
2. Runs them one by one until they yield control (at `await` points)
3. Handles I/O and timers while coroutines are paused
4. Resumes coroutines when their awaited operations complete

```

┌─────────────────────┐
│     Event Loop      │
└─────────┬───────────┘
          │
          │ manages
          ▼
┌─────────────────────┐      awaits      ┌─────────────────┐
│  Active Coroutine   │─────────────────▶│ Awaitable Object│
└─────────────────────┘                  └─────────┬───────┘
          ▲                                        │
          │                                        │
          │ resumes when                           │ completes
          │ awaitable completes                    │
          │                                        │
┌─────────┴───────────┐                  ┌─────────▼───────┐
│ Suspended Coroutine │◀─────────────────│   I/O or Timer  │
└─────────────────────┘                  └─────────────────┘

```

Real-World Examples and Use Cases

1. **Web Servers**:
    
    ```python
    
    async def handle_request(request):
        user_data = await database.fetch_user(request.user_id)
        if user_data:
            return await template.render("profile.html", user=user_data)
        else:
            return "User not found"
    
    ```
    
2. **API Clients**:
    
    ```python
    async def get_user_with_posts(user_id):
    # These requests can happen concurrently
        user = await api.fetch_user(user_id)
        posts = await api.fetch_posts(user_id)
        return {"user": user, "posts": posts}
    
    ```
    
3. **Asynchronous Context Managers**:
    
    ```python
    async with aiohttp.ClientSession() as session:
        async with session.get('https://api.example.com/data') as response:
            data = await response.json()
    
    ```
    

## Advantages of Coroutines and Awaitables

1. **Readability**: Asynchronous code looks similar to synchronous code
2. **Error Handling**: Try/except works naturally with async/await
3. **Resource Efficiency**: Many concurrent operations without multiple threads
4. **Composability**: Awaitable operations can be easily combined

Unlike callbacks or Promise chains, the coroutine model lets you write asynchronous code that reads top-to-bottom like synchronous code, while maintaining all the efficiency benefits of non-blocking operations.

# Implementing Shared Memory in Python

Python offers several mechanisms for implementing shared memory between processes. Here's a comprehensive guide to the different approaches:

### 1. Using `multiprocessing.shared_memory` (Python 3.8+)

The most direct way to implement shared memory in modern Python is using the built-in `shared_memory` module.

```python
import numpy as np
from multiprocessing import Process, shared_memory

def producer():
# Create a shared memory block
    shm = shared_memory.SharedMemory(create=True, size=28)

# Create a NumPy array using the shared memory buffer
    shared_array = np.ndarray((7,), dtype=np.int64, buffer=shm.buf)
    shared_array[:] = np.arange(7)# Set initial values

    print(f"Producer set: {shared_array}")
    print(f"Shared memory name: {shm.name}")

# Keep the shared memory alive (in a real app, use proper synchronization)
    input("Press Enter to release shared memory...")

# Clean up from this process
    shm.close()
    shm.unlink()# Free and remove the shared memory block

def consumer(shm_name):
# Attach to the existing shared memory block
    existing_shm = shared_memory.SharedMemory(name=shm_name)

# Create a NumPy array using the shared memory buffer
    shared_array = np.ndarray((7,), dtype=np.int64, buffer=existing_shm.buf)
    print(f"Consumer got: {shared_array}")

# Modify the array - changes will be visible to all processes
    shared_array[0] = 888
    print(f"Consumer modified: {shared_array}")

# Clean up from this process
    existing_shm.close()

if __name__ == "__main__":
# Start the producer process
    prod_process = Process(target=producer)
    prod_process.start()

# Wait for the producer to create the shared memory
    shm_name = input("Enter the shared memory name: ")

# Start the consumer process
    cons_process = Process(target=consumer, args=(shm_name,))
    cons_process.start()

# Join processes
    cons_process.join()
    prod_process.join()

```

### 2. Using `multiprocessing.Array` and `multiprocessing.Value`

For simpler use cases, Python's multiprocessing module provides `Array` and `Value` objects that implement shared memory:

```python
from multiprocessing import Process, Array, Value

def worker(shared_array, shared_value):
    print(f"Initial values in worker: array={list(shared_array)}, value={shared_value.value}")

# Modify the shared data
    for i in range(len(shared_array)):
        shared_array[i] = shared_array[i] * 2

    shared_value.value = 42

    print(f"Modified in worker: array={list(shared_array)}, value={shared_value.value}")

if __name__ == "__main__":
# 'd' is the typecode for double precision float
    shared_array = Array('d', [1.0, 2.0, 3.0, 4.0, 5.0])

# 'i' is the typecode for signed integer
    shared_value = Value('i', 0)

    print(f"Initial values in main: array={list(shared_array)}, value={shared_value.value}")

# Start the worker process
    p = Process(target=worker, args=(shared_array, shared_value))
    p.start()
    p.join()

# Access the modified shared data
    print(f"After worker: array={list(shared_array)}, value={shared_value.value}")

```

### 3. Using Memory-Mapped Files with `mmap`

You can use memory-mapped files for shared memory across processes:

```python
import mmap
import os
from multiprocessing import Process

def writer():
# Create a file and memory-map it
    with open("shared_memory.dat", "wb") as f:
# Create a 1 MB file
        f.write(b"\x00" * 1024 * 1024)

    with open("shared_memory.dat", "r+b") as f:
# Memory-map the file
        mm = mmap.mmap(f.fileno(), 0)

# Write a string to the beginning of the file
        mm.write(b"Hello, shared memory!")

        print("Data written to shared memory")
        mm.close()

def reader():
# Wait a moment to ensure writer has created the file
    import time
    time.sleep(1)

    with open("shared_memory.dat", "r+b") as f:
# Memory-map the file
        mm = mmap.mmap(f.fileno(), 0)

# Read data from the memory-mapped file
        mm.seek(0)
        data = mm.read(20).decode()
        print(f"Read from shared memory: {data}")

# Write a response
        mm.seek(20)
        mm.write(b"Response from reader")

        mm.close()

if __name__ == "__main__":
# Start processes
    writer_process = Process(target=writer)
    reader_process = Process(target=reader)

    writer_process.start()
    reader_process.start()

    writer_process.join()
    reader_process.join()

# Clean up the file
    os.unlink("shared_memory.dat")

```

### 4. Using Shared Memory with Manager

For more complex data structures, you can use a Manager:

```python
from multiprocessing import Process, Manager

def modify_dict(shared_dict):
    shared_dict["key1"] = "Modified value"
    shared_dict["key2"] = 100
    shared_dict["new_key"] = "Added by child process"
    print(f"Dict in child process: {shared_dict}")

if __name__ == "__main__":
    with Manager() as manager:
# Create a shared dictionary
        shared_dict = manager.dict()

# Initialize the dictionary
        shared_dict["key1"] = "Initial value"
        shared_dict["key2"] = 42

        print(f"Initial dict: {shared_dict}")

# Start a process that modifies the dictionary
        p = Process(target=modify_dict, args=(shared_dict,))
        p.start()
        p.join()

# The changes are visible in the parent process
        print(f"Dict after modification: {shared_dict}")

```

### 5. Using Shared Memory with NumPy and `multiprocessing.Array`

For numerical computing, combining NumPy with shared memory is common:

```python
import numpy as np
from multiprocessing import Process, Array

def process_data(shared_data):
# Create a NumPy array that uses the shared memory buffer
    arr = np.frombuffer(shared_data.get_obj())

# Reshape if needed (for multidimensional arrays)
    arr = arr.reshape((5, 5))

# Process the data
    arr += 10
    print(f"Array in child process:\n{arr}")

if __name__ == "__main__":
# Create a shared array of 25 floats
    shared_array = Array('d', 25)

# Create a NumPy array that uses the shared memory buffer
    arr = np.frombuffer(shared_array.get_obj())
    arr = arr.reshape((5, 5))

# Initialize the array
    for i in range(5):
        for j in range(5):
            arr[i, j] = i * 5 + j

    print(f"Initial array:\n{arr}")

# Start a process to modify the data
    p = Process(target=process_data, args=(shared_array,))
    p.start()
    p.join()

# The changes are visible in the parent process
    print(f"Array after processing:\n{arr}")

```

## 6. Synchronization for Shared Memory

When using shared memory, you'll often need synchronization mechanisms:

```python

python
from multiprocessing import Process, Array, Value, Lock
import time
import random

def worker(id, counter, array, lock):
    for _ in range(5):
# Simulate some work
        time.sleep(random.random() * 0.2)

# Safely update shared memory
        with lock:
            counter.value += 1
            array[id] = counter.value
            print(f"Worker {id}: counter = {counter.value}, array = {list(array)}")

if __name__ == "__main__":
# Create shared memory
    counter = Value('i', 0)
    array = Array('i', [0] * 4)

# Create a lock for synchronization
    lock = Lock()

# Create processes
    processes = []
    for i in range(4):
        p = Process(target=worker, args=(i, counter, array, lock))
        processes.append(p)
        p.start()

# Wait for all processes to finish
    for p in processes:
        p.join()

    print(f"Final counter value: {counter.value}")
    print(f"Final array: {list(array)}")

```

Best Practices for Shared Memory in Python

1. **Always use synchronization** when multiple processes modify shared memory
2. **Close and unlink shared memory** when you're done with it
3. **Be cautious with typecodes** in Array and Value (they must match your data)
4. **Use shared_memory for large data** and Array/Value for simpler cases
5. **Consider pickling/unpickling limitations** when using Manager objects
6. **Plan your memory layout** carefully for efficient processing

Performance Considerations

- Direct shared memory (shared_memory, Array, Value) is faster than Manager-based solutions
- Memory-mapped files are efficient for very large datasets
- Avoid unnecessary copying between shared and local memory
- Batch operations when possible to minimize synchronization overhead

Shared memory is powerful but comes with complexity - always ensure proper synchronization to avoid race conditions and memory corruption in your multi-process applications.

# OS-Managed (Preemptive) Context Switching

Context switching is a fundamental operating system mechanism where the CPU switches from executing one process or thread to another. When this switching is managed by the operating system and can occur without the running process's cooperation, it's called **preemptive context switching**.

Preemptive context switching allows the operating system to interrupt a running process at any time, save its current state, and switch to another process. This happens without the explicit consent or cooperation of the currently running process.

The Context Switching Process

1. **Interrupt Occurs**: A hardware interrupt, system call, or timer expiration signals the OS
2. **Save Current Context**: The OS saves the current process's execution context (registers, program counter, stack pointer)
3. **Select Next Process**: The scheduler selects which process to run next
4. **Load New Context**: The OS loads the context of the new process
5. **Resume Execution**: The CPU continues execution with the new process

```
Process A running     Context Switch        Process B running
┌─────────────────┐   ┌─────────────┐       ┌─────────────────┐
│ - CPU registers │   │ 1. Save A's │       │ - CPU registers │
│ - Program       │──▶│   context   │──────▶│ - Program       │
│   counter       │   │ 2. Load B's │       │   counter       │
│ - Stack pointer │   │   context   │       │ - Stack pointer │
└─────────────────┘   └─────────────┘       └─────────────────┘
                            ▲
                            │
                      ┌─────┴─────┐
                      │ Operating │
                      │  System   │
                      │ Scheduler │
                      └───────────┘

```

Key Components of Preemptive Context Switching

1. Hardware Support

- **Interrupt Mechanism**: Hardware interrupts allow external events to pause execution
- **Timer Interrupts**: Hardware timers generate interrupts at specific intervals
- **Memory Management Unit (MMU)**: Manages process memory spaces during switches
- **Context Save/Restore Instructions**: CPU instructions to efficiently save/restore register states

2. OS Scheduler

- **Scheduler Algorithm**: Determines which process runs next (Round Robin, Priority-based, etc.)
- **Time Slices**: Allocates CPU time for each process before preemption
- **Priority Levels**: Determines which processes get preferential treatment
- **Ready Queue**: Maintains list of processes ready to execute

3. Process Control Block (PCB)

The PCB is a data structure maintained by the OS that contains all the information needed to save and restore a process context:

- Process ID
- CPU register values
- Program counter
- Memory management information
- I/O status information
- Accounting information
- Scheduling information

Causes of Preemptive Context Switches

1. **Time Slice Expiration**: Process used its allocated CPU time (quantum)
2. **Higher Priority Process**: A higher priority process becomes ready to run
3. **I/O Request**: Process requests I/O operation that would block
4. **Page Fault**: Process accesses memory not currently in physical RAM
5. **System Calls**: Process makes a call to the operating system
6. **Hardware Interrupts**: External device requires attention
7. **Synchronization Events**: Process waits for a lock or semaphore

Advantages of Preemptive Context Switching

1. **Fairness**: Prevents any single process from monopolizing the CPU
2. **Responsiveness**: Allows high-priority tasks to execute promptly
3. **System Reliability**: Prevents infinite loops or CPU-bound tasks from freezing the system
4. **Multitasking**: Enables many processes to appear to run simultaneously
5. **Real-time Support**: Essential for real-time systems with strict timing requirements

Disadvantages and Overhead

1. **Context Switch Cost**: Time spent saving/restoring context isn't doing useful work
2. **Cache Invalidation**: Process switch often invalidates CPU cache, causing slowdowns
3. **TLB Flushes**: Translation Lookaside Buffer may need to be flushed during switches
4. **Scheduler Overhead**: CPU time spent deciding which process runs next
5. **Unpredictability**: Makes execution timing less predictable for individual processes

Context Switch Metrics

- **Frequency**: Typical systems perform hundreds to thousands per second
- **Duration**: Modern systems take 1-10 microseconds per context switch
- **Overhead Percentage**: Can consume 5-50% of CPU time in busy multitasking systems

Examples in Different Operating Systems

Linux

- Uses the Completely Fair Scheduler (CFS) for preemptive multitasking
- Dynamic time slices based on process "niceness" and system load
- Supports real-time scheduling policies with fixed priorities

Windows

- Implements preemptive multitasking with thread priorities
- Uses multilevel feedback queue scheduler
- Windows quantum typically ranges from 20-120ms

Real-Time Operating Systems (RTOS)

- Even stricter preemption rules for time-critical applications
- Fixed priority scheduling with immediate preemption
- Deterministic context switch times for predictable performance

Practical Implications for Developers

1. **Thread Safety**: Preemptive switching means thread synchronization is essential
2. **Critical Sections**: Code sections that must not be interrupted need proper locks
3. **Lock Granularity**: Too many locks increase context switch frequency
4. **Thread Affinity**: Can reduce context switches by keeping threads on specific CPUs
5. **Batch Processing**: Grouping operations can reduce switch frequency

Preemptive context switching is a cornerstone of modern multitasking operating systems, allowing them to maintain responsiveness, fairness, and system stability while running many processes simultaneously.