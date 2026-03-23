[datasketch](https://github.com/ekzhu/datasketch): MinHash, LSH, LSH Forest, Weighted MinHash, HyperLogLog, HyperLogLog++, LSH Ensemble and HNSW

# 2. Algorithms & Python Concepts (15 min)

## 2.1 Algorithm Complexity

| Algorithm / Structure | Average Case | Worst Case | Notes |
|---|---|---|---|
| **Sorting — QuickSort** | O(n log n) | O(n²) | Worst case on already-sorted input; mitigated by randomized pivot |
| **Sorting — MergeSort** | O(n log n) | O(n log n) | Stable sort; extra O(n) space |
| **Sorting — TimSort** | O(n log n) | O(n log n) | Python's built-in `sorted()` and `.sort()` use TimSort; optimal on partially sorted data |
| **Searching — Binary Search** | O(log n) | O(log n) | Requires sorted input |
| **Searching — Linear Search** | O(n) | O(n) | No precondition |
| **Hash Table — lookup/insert** | O(1) | O(n) | Worst case when all keys collide; good hash function + load-factor management keeps it amortized O(1) |
| **Hash Table — Hash Map vs Hash Set** | O(1) | O(n) | Set stores keys only; Map stores key-value pairs |

**Trade-offs & optimization points to mention:**

QuickSort is generally faster in practice than MergeSort due to cache locality (operates in-place), but its worst case is dangerous in adversarial settings. Python avoids this entirely by using TimSort. For nearly-sorted data, insertion sort (O(n) best case) or TimSort's natural-run detection is optimal.

For hashing, the key trade-offs are: (a) collision resolution strategy — separate chaining vs open addressing; (b) load factor threshold triggering a resize; (c) hash function quality. Python's `dict` uses open addressing with a compact table design.
