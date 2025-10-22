from functools import partial
from multiprocessing import Pool

import numpy as np

NUM_PROCESSES = 2

N = 50


def top_n_recommends_random(watch_history, all_content, n=N):
    """
    Check if random top-N recommendations hit the validation set.

    Args:
        watch_history: Tuple of (train_items, valid_items)
        all_content: Array of all available content IDs
        n: Number of recommendations to generate

    Returns:
        1 if there's a hit, 0 otherwise
    """
    top_n_result = np.random.choice(all_content, size=n, replace=True)
    hit = 0
    if len(watch_history[1]) > 0 and np.intersect1d(watch_history[1], top_n_result).size > 0:
        hit = 1
    return hit


def top_n_recommends_popular(watch_history, top_items, n=N):
    """
    Check if top-N popular recommendations hit the validation set.

    Args:
        watch_history: Tuple of (train_items, valid_items)
        top_items: Array of top popular items
        n: Number of recommendations to generate

    Returns:
        1 if there's a hit, 0 otherwise
    """
    top_n_result = top_items[:n]
    hit = 0
    if len(watch_history[1]) > 0 and np.intersect1d(watch_history[1], top_n_result).size > 0:
        hit = 1
    return hit


def randomized(all_content, train_valid_pairs):
    # Create a partial function with all_content pre-filled
    recommend_func = partial(top_n_recommends_random, all_content=all_content, n=N)

    with Pool(NUM_PROCESSES) as p:
        hits = p.map(recommend_func, train_valid_pairs)
    res = 'Num hits %.4f from %d' % (sum(hits)/len(hits), len(hits))
    return res



def top_popular(top_100_popular_items, train_valid_pairs):
    # Create a partial function with top_100_popular_items pre-filled
    recommend_func = partial(top_n_recommends_popular, top_items=top_100_popular_items, n=N)

    with Pool(NUM_PROCESSES) as p:
        hits = p.map(recommend_func, train_valid_pairs)

    res = 'Num hits %.4f from %d' % (sum(hits)/len(hits), len(hits))
    return res