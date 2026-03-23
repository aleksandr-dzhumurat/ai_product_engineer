For system design interview: 
[excalidraw](https://excalidraw.com/)

* Write a function that finds the **longest sequence of the same character repeated consecutively** in a given string. The function should return the character and the length of the longest sequence. (example_stirng = "aabaaaccqcp a ccpppppppqwppp")
* Write a function that receives a list of logs like user_id, timestamp, event_type, and returns users who performed a "purchase" within 7 days of their first "view".
* Python function that splits a dataset by **user** — 70% of users go to the **test set**, 30% to **train**, based on user IDs [ {'user': 1}, {'user': 2}, {'user': 1}, {'user': 4},]
* Remove rare categories (occurring less than 2 times) ['a', 'a', 'b', 'c', 'c', 'd']
* Convert categorical values to one-hot encoding one_hot_encode(['cat', 'dog', 'cat'])) → [{'cat': 1, 'dog': 0}, {'cat': 0, 'dog': 1}, {'cat': 1, 'dog': 0}]

# Q1

explain code below
```python
def file_cache(filepath):
    def decorator(func):
        def wrapper(*args, **kwargs):
            overwrite = kwargs.get('overwrite', False)
            if os.path.exists(filepath) and not overwrite:
                print(f"Loading data from {filepath}")
                return pd.read_csv(filepath)
            else:
                print(f"Processing and saving data to {filepath}")
                result = func(*args, **kwargs)
                result.to_csv(filepath, index=False)
                print(f'Preprocessed impressions: num rows {result.shape[0]} in {filepath}')
                return result
        return wrapper
    return decorator
```

Это **декоратор** `file_cache` — он оборачивает любую функцию, которая возвращает DataFrame, и добавляет ей кэширование через CSV-файл.

usage example
```python
@file_cache('data/processed.csv')
def process_data(df):
    # тяжёлые вычисления...
    return df
```

# Q2


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
```python
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

# Task: hotel ranking

At [Booking.com](http://booking.com/) we want to recognize k performing hotels. We plan to identify these by analyzing their user reviews and calculating a review score for each of the hotel.
To calculate the score, we have:
a list of user reviews for each hotel, a list of positive keywords and a list of negative keywords.
Positive keywords weigh 3 points each and negative keywords weigh -1 each.
For example, given the input below:

```shell
positive keywords: "breakfast beach citycenter location metro view staff price" negative keywords: "not",
number of hotels: m = 5,
array of hotel ids: [1,2,1,1,2], number of reviews: n=5, array of reviews: [
"This hotel has a nice view of the citycenter. The location is perfect.",
"The breakfast is ok. Regarding location, it is quite far from citycenter but price is cheap so it is worth."
"Location is excellent, 5 minutes from citycenter.
There is also a metro station very close to the hotel."
"They said I couldn't take my dog and there were other guests with dogs! That is not fair.",
"Very friendly staff and good cost-benefit ratio. Its location is a bit far from citycenter."
],
number of hotels we want to award: k = 2
```

then top k Hotels will be 2, 1.

```python
def awardTopKHotels (positiveKeywords, negativeKeywords, hotelIds, reviews, k):
	def remove_puntuation(s):
		translation_table = str.maketrans ('", "", punctuation)
		cleaned_string = s.translate (translation_table)
		return cleaned_string
	
	scores = dict.fromkeys (set (hotelIds), 0)
	positive_tokens = positiveKeywords.split(' ")
	negative_tokens = negativeKeywords.split(' ")
	for i, review in enumerate (reviews):
	review tokens = set(remove punctuation (review).split())
	positive_score = sum(i in review_tokens for i in positive_tokens) * 3
	negative_score = sum(i in review_tokens for i in negative_tokens) * -1
	scores [hotelIds[i]] += positive_score + negative_score
	top_k = list (i[0] for i in sorted(scores.items (), key=lambda item: item[1], reverse=True)) [: k]
	return top k
```

## Pure python analysis

Analyze user behavior events to identify user_ids for premium upgrade:
- 80%+ utilization from any usage_report event
- Zero support_issue events

```python
def identify_upgrade_candidates(events) -> list:
    """
    Returns:
    list: [user_ids]
    """
    # Your code here
    pass

# Test data - Event stream from 3 users (11 total events)
test_events_1 = [
    # User 101 events (account created 220 days ago)
    {'dt': '2025-04-01', 'days_since_account_created': 0, 'user_id': 101, 'event_type': 'account_created', 'data': {'plan': 'basic'}},
    {'dt': '2025-06-01', 'days_since_account_created': 215, 'user_id': 101, 'event_type': 'usage_report', 'data': {'usage_percentage': 85}},
    {'dt': '2025-10-01', 'days_since_account_created': 175, 'user_id': 101, 'event_type': 'checkout', 'data': {'item': 'addon_storage', 'amount': 9.99}},
    {'dt': '2025-12-01', 'days_since_account_created': 200, 'user_id': 101, 'event_type': 'checkout', 'data': {'item': 'premium_support', 'amount': 19.99}},
    
    # User 102 events (account created 200 days ago)
    {'dt': '2025-02-01', 'days_since_account_created': 0, 'user_id': 102, 'event_type': 'account_created', 'data': {'plan': 'basic'}},
    {'dt': '2025-05-01', 'days_since_account_created': 135, 'user_id': 102, 'event_type': 'plan_change', 'data': {'from_plan': 'basic', 'to_plan': 'premium'}},
    {'dt': '2025-08-01', 'days_since_account_created': 185, 'user_id': 102, 'event_type': 'usage_report', 'data': {'usage_percentage': 90}},
    {'dt': '2025-11-01','days_since_account_created': 160, 'user_id': 102, 'event_type': 'checkout', 'data': {'item': 'api_credits', 'amount': 29.99}},
    
    # User 103 events (account created 50 days ago)
    {'dt': '2025-01-01','days_since_account_created': 0, 'user_id': 103, 'event_type': 'account_created', 'data': {'plan': 'standard'}},
    {'dt': '2025-09-01','days_since_account_created': 45, 'user_id': 103, 'event_type': 'usage_report', 'data': {'usage_percentage': 82}},
    {'dt': '2025-12-01', 'days_since_account_created': 30, 'user_id': 103, 'event_type': 'support_issue', 'data': {'issue': 'payment_failed'}}
]

upgrade_candidates = identify_upgrade_candidates(test_events_1)
print(f"Upgrade candidates: {upgrade_candidates}")
```