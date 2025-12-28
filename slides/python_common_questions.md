
Python Performance: list internals



## Python Performance Bonus

### List Implementation in Python

**Internal Structure:**
- Dynamic array (CPython: PyListObject)
- Stores **pointers** to objects, not objects themselves
- Preallocates memory (capacity > size)
- ~12.5% extra capacity typically

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

### Why Insert at Beginning is O(n)

```python
lst = [1, 2, 3]
lst.insert(0, 0)  # O(n) - shift ALL elements right

# Internal steps:
# [1, 2, 3, _]
# [_, 1, 2, 3]  ← All elements shifted
# [0, 1, 2, 3]  ← Insert 0
```

### Amortized O(1) for Append

**When capacity reached:**
1. Allocate new larger array (new_capacity ≈ old_capacity × 1.125)
2. Copy all elements → O(n)
3. Free old array
4. Append element

**Amortization:** Infrequent resizing amortizes to O(1) per operation

**References:**
- [Python List Implementation - CPython Source](https://github.com/python/cpython/blob/main/Objects/listobject.c)
- [Time Complexity - Python Wiki](https://wiki.python.org/moin/TimeComplexity)



### Почему append O(1), а insert(0) O(n)?

**List = динамический массив** (не linked list!)
```
Внутреннее устройство:
lst = [1, 2, 3]
Память: [1][2][3][_][_][_]  ← Есть запас (capacity > size)

append(4):  O(1)
[1][2][3][4][_][_]  ← Просто записываем в следующую ячейку

insert(0, 0): O(n)
Нужно сдвинуть ВСЕ элементы вправо:
[1][2][3] → [_][1][2][3]
            Shift!
```

**Амортизированная O(1) для append:**
```
Capacity заполнен → нужно расширение

Старый массив: [1,2,3,4] capacity=4
Новый массив: [1,2,3,4,_,_,_,_,_] capacity=9

Копирование: O(n)
Но происходит редко!

В среднем: O(1) амортизированная