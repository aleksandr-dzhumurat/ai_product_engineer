For system design interview: 
[excalidraw](https://excalidraw.com/)

# Q1

```python
"""

Imagine we have a website, and we monitor the pages customers view.

Each time a person visits the site, we record the  (CustomerId, PageId) in a log file. 

By the end of the day, we accumulate a substantial log file with numerous entries in this format. 

This process is repeated daily, resulting in a new log file for each day.

Now, if we have two log files (one from day 1 and the other from day 2)


------------------------------------------------------------------------------
QUESTION
we aim to identify 'loyal customers' who satisfy two conditions:
(a) they visited the website on both days, and 
(b) they explored at least two different pages.

------------------------------------------------------------------------------
"""



def print_loyal_customer(f1: list, f2: list):
    print(len(f1), len(f2))
    
day_1_log = [
    (1, 1),
    (2, 4),
    (3, 7),
    (3, 8),
    (1, 1),
    (1, 1),
    (3, 9),
    (4, 10),
    (4, 11),
    (4, 12),
    (5, 13),
    (5, 14),
    (5, 15),
    (2, 5),
    (2, 6),
]

day_2_log = [
    (1, 1),
    (17, 17),
    (18, 18),
    (21, 19),
    (2, 3),
    (3, 9),
    (3, 9),
    (3, 8),
    (5, 1),
    (5, 28),
    (6, 29),
    (6, 30),
    (7, 31),
    (7, 32),
    (8, 33),
    (8, 34),
]

print_loyal_customer(day_1_log, day_2_log)
```

# Q2

```python
# Please review this code.
# Describe its idea and answer the questions in the comments.

from functools import lru_cache
from typing import Sequence

import numpy as np


class Reco:
    def __init__(self, embeddings: dict[int, Sequence[float]]):
        # let's rename a and b, which names will be more descriptive? Class instantiation below may help you.
        self.a = {i: j for i, j in enumerate(embeddings.keys())} # self.index, key:(0,....,len(embeddigs)-1), value: int (key from embeddings )
        self.b = {j: i for i, j in self.a.items()} # self.inverted_index

        self.embeddings = np.array([e for e in embeddings.values()]) # matrix(num_embeds x size(Sequence[float]))
        self.embeddings /= np.linalg.norm(self.embeddings, axis=0)  # what we do here? *why we may want to do this? implementing cosine similarity

    @lru_cache # what's that, what "@" syntax mean?
    def recommend(self, item_id: int, n: int = 10) -> list[int]:
        item_emb = self.embeddings[self.b[item_id]]
        scores = self.embeddings.dot(item_emb)
        recomemnded_ix = scores.argsort()[::-1]  # what [::-1] do?

        return [
                   self.a[ix]
                   for ix in recomemnded_ix
               ][1: n+1]  # *why we drop first element here?


if __name__ == '__main__':
    item_count = 10 ** 4
    max_item_id = 10 ** 5
    embedding_size = 100
    item_ids = np.random.randint(max_item_id, size=item_count)

    r = Reco({
        item_id: np.random.random(embedding_size)
        for item_id in item_ids
    })

    r.recommend(item_ids[0])
    r.recommend(item_ids[1])
    r.recommend(item_ids[0]) #  Cached value
```
