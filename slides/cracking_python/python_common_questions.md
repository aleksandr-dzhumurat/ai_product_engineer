# Python Common Questions

## Advanced Python: LRU cache

**Question:** Consider the following Python code:

```python
from functools import lru_cache

class Fibonacci:
    def __init__(self):
        self.memo = {}

    def fib(self, n):
        if n in self.memo:
            return self.memo[n]
        if n <= 1:
            return n
        self.memo[n] = self.fib(n-1) + self.fib(n-2)
        return self.memo[n]

@lru_cache(maxsize=None)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    fib_instance = Fibonacci()
    for i in range(10):
        print(f"Fibonacci (class method) for {i}: {fib_instance.fib(i)}")
        print(f"Fibonacci (lru_cache) for {i}: {fibonacci(i)}")

if __name__ == "__main__":
    main()

```

1. Explain how the `Fibonacci` class computes Fibonacci numbers and the role of `self.memo`.
2. Describe the purpose and functionality of the `@lru_cache` decorator in the `fibonacci` function. How does it optimize the computation?
3. What are the key differences between using the `Fibonacci` class with memoization and the `fibonacci` function with `lru_cache`?
4. Discuss the time complexity of both the class-based and decorator-based approaches to computing Fibonacci numbers.
5. What will be the output of the provided code snippet, and why does the output of both methods (class and decorator) produce the same results?

**Expected Answers:**

**LRU** stands for **Least Recently Used**. You can implement in using OrderedDict

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()  # To store the cache
        self.capacity = capacity    # Maximum size of the cache

    def get(self, key: int) -> int:
        """Retrieve an item from the cache."""
        if key in self.cache:
            # Move the accessed key to the end to mark it as recently used
            self.cache.move_to_end(key)
            return self.cache[key]
        return -1  # Return -1 if the key is not in the cache

    def put(self, key: int, value: int) -> None:
        """Add or update an item in the cache."""
        if key in self.cache:
            # Update the value of the key and move it to the end
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            # Evict the least recently used item
            self.cache.popitem(last=False)  # Remove the first (least recently used) item
        self.cache[key] = value  # Add the new key-value pair

# Example usage
lru = LRUCache(3)
lru.put(1, 100)
lru.put(2, 200)
lru.put(3, 300)
print(lru.get(2))  # Output: 200 (Accessing key 2 moves it to the most recently used position)
lru.put(4, 400)    # Evicts key 1 (the least recently used key)
print(lru.get(1))  # Output: -1 (Key 1 has been evicted)
print(lru.get(3))  # Output: 300
print(lru.get(4))  # Output: 400

```

1. **`Fibonacci` Class Explanation:**
    - The `Fibonacci` class computes Fibonacci numbers using a memoization technique.
    - `self.memo` is a dictionary that stores previously computed Fibonacci values to avoid redundant calculations.
    - The `fib` method checks if the result for `n` is already in `self.memo`. If not, it recursively computes the Fibonacci number and stores the result in `self.memo`.
2. **`@lru_cache` Decorator Purpose and Functionality:**
    - The `@lru_cache` decorator is used to cache the results of the `fibonacci` function to optimize performance.
    - `maxsize=None` means the cache size is unlimited.
    - It stores the results of expensive function calls and returns the cached result when the same inputs occur again, reducing the number of computations and improving efficiency.
3. **Differences Between Class-Based and Decorator-Based Approaches:**
    - **Class-Based Approach (`Fibonacci` class)**: Uses an instance variable (`self.memo`) to store computed values and manages the cache manually. Suitable for situations where different instances may need to maintain separate caches.
    - **Decorator-Based Approach (`lru_cache`)**: Automatically manages caching and cache size. It is a more concise and less error-prone way to achieve memoization without modifying the function code. The cache is global to the function and shared across all calls.
4. **Time Complexity Discussion:**
    - **Class-Based Approach**: The time complexity is O(n) due to memoization. Without memoization, it would be O(2^n) because of the exponential number of recursive calls.
    - **Decorator-Based Approach**: Similarly, with `lru_cache`, the time complexity is O(n). It optimizes recursive calls by caching results, reducing redundant computations.
5. **Output and Result:**
    - The output of the provided code will display Fibonacci numbers from 0 to 9 computed using both methods. Both methods produce the same results due to their correct implementation of the Fibonacci sequence.
    - Both methods achieve the same results, demonstrating that both memoization techniques are effective for optimizing Fibonacci number calculations.

This question assesses the candidate's understanding of advanced Python features, including class-based memoization and function decorators, and their ability to compare and analyze different optimization techniques.

## Decorators and Caching

Ref: [Decorator](https://www.notion.so/Decorator-17e9c76f79e880819e5fd5427a3ae353?pvs=21) 

**Question:**

Consider the following Python code that defines a custom decorator for caching function results:

```python
import time

def custom_cache(max_size=100):
    def decorator(func):
        cache = {}
        def wrapper(*args):
            if args in cache:
                print(f"Cache hit for args: {args}")
                return cache[args]
            result = func(*args)
            if len(cache) >= max_size:
                print("Cache is full. Removing the oldest item.")
                cache.pop(next(iter(cache)))
            cache[args] = result
            print(f"Cache miss for args: {args}")
            return result
        return wrapper
    return decorator

@custom_cache(max_size=3)
def slow_function(x):
    time.sleep(2)  # Simulate a time-consuming computation
    return x * x

def main():
    print(slow_function(1))  # Takes time, then caches the result
    print(slow_function(2))  # Takes time, then caches the result
    print(slow_function(1))  # Should hit the cache
    print(slow_function(3))  # Takes time, then caches the result
    print(slow_function(4))  # Takes time, oldest item (1) should be removed from cache
    print(slow_function(2))  # Should hit the cache

if __name__ == "__main__":
    main()

```

1. Explain the purpose of the `custom_cache` decorator and how it functions. How does it handle the caching of function results?
2. What are the roles of the `cache` dictionary and the `wrapper` function within the decorator? How does the decorator ensure that the cache does not exceed the specified `max_size`?
3. Describe the output of the provided code when executed. Why do some calls to `slow_function` result in a cache hit while others result in a cache miss?
4. How does the custom decorator compare to Python’s built-in `functools.lru_cache` in terms of functionality and implementation?
5. If you were to optimize the `custom_cache` decorator further, what improvements would you consider implementing?

**Expected Answers:**

1. **Purpose and Functionality of `custom_cache`:**
    - The `custom_cache` decorator is designed to cache the results of a function to avoid redundant computations for previously encountered arguments.
    - It uses an inner `cache` dictionary to store function results. If the function is called with the same arguments, the cached result is returned instead of recomputing.
    - The decorator has a `max_size` parameter to limit the number of cached items, ensuring that the cache does not grow indefinitely.
2. **Roles of `cache` and `wrapper` Function:**
    - The `cache` dictionary stores function results, where the keys are function arguments and values are the corresponding results.
    - The `wrapper` function handles checking the cache for existing results, computing results if not cached, and managing the cache size by removing the oldest item when the maximum size is exceeded.
    - When `max_size` is reached, the oldest cached item is removed using `pop(next(iter(cache)))`.
3. **Output Description:**
    - The output will include messages indicating cache hits and misses as well as the results of function calls.
    - Initial calls to `slow_function` with new arguments will result in cache misses, and the results will be computed and cached.
    - Repeated calls with the same arguments will hit the cache.
    - When the cache reaches its maximum size, adding new items will cause the oldest item to be removed, as demonstrated by the output when `slow_function(4)` removes the oldest cached result.
4. **Comparison with `functools.lru_cache`:**
    - **Functionality**: Both `custom_cache` and `lru_cache` provide caching, but `lru_cache` implements an LRU (Least Recently Used) eviction policy automatically.
    - **Implementation**: `lru_cache` is more efficient and easier to use, as it handles cache management and eviction internally without manual code. The custom decorator requires explicit handling of cache size and eviction logic.
5. **Potential Improvements:**
    - Implementing a more efficient eviction strategy, such as LRU, to manage the cache.
    - Using an `OrderedDict` to simplify cache management and eviction.
    - Adding thread safety to handle concurrent access if the function is used in a multi-threaded environment.
    - Improving performance and reducing memory usage by optimizing cache storage and access patterns.

This question assesses the candidate’s understanding of decorators, custom caching mechanisms, and comparisons with built-in caching solutions. It also tests their ability to analyze and improve caching implementations.