import os
import pickle
from multiprocessing import Pool
from typing import Dict

import numpy as np
import pandas as pd

NUM_PROCESSES = 2
root_data_dir = os.environ['DATA_DIR']
user_item_views_df = pd.read_csv(os.path.join(root_data_dir, 'user_item_views.zip'), compression='zip')
unique_items = user_item_views_df.item_id.unique()
item_to_id = {j: i for i, j in enumerate(unique_items)}


with open(os.path.join(root_data_dir, 'ground_truth_dataset.pkl'), 'rb') as f:
    ground_truth_dataset = pickle.load(f)
with open(os.path.join(root_data_dir, 'test_dataset.pkl'), 'rb') as f:
    test_dataset = pickle.load(f)
print(
    'test dataset users %d, valid dataset users %d' %
    (len(test_dataset), len(ground_truth_dataset))
)

def get_als_action_history_vector(item_to_id: Dict[int, int], action_history, binary=True) -> np.ndarray:
    """Получить историю действий для ALS

    :param item_to_id: справочник контента ALS
    :return:
    """
    als_action_history_vector = np.zeros(len(item_to_id), dtype=int)
    for iid, item_attr in action_history.items():
        if iid in item_to_id.keys():
            if binary:
                als_action_history_vector[item_to_id[iid]] = 1
            else:
                als_action_history_vector[item_to_id[iid]] = item_attr
    return als_action_history_vector

def vectorize_action_history(action_history):
    res = get_als_action_history_vector(item_to_id, action_history)
    return res

def save_embeds(embeds_list, dataset_path):
    passage_embeddings = np.array([embedding for embedding in embeds_list]).astype("float32")
    with open(dataset_path, 'wb') as f:
        np.save(f, passage_embeddings)
    print(f'Data saved: {passage_embeddings.shape} to {dataset_path}')

if __name__ == '__main__':
    print('data preparing started...')
    with Pool(NUM_PROCESSES) as p:
        # персональная история просмотров
        test_dataset_vectors = p.map(vectorize_action_history, test_dataset)
        # валидация
        ground_truth_dataset_vectors = p.map(vectorize_action_history, ground_truth_dataset)
    print('test dataset length: %d, shape: %d' % (len(test_dataset_vectors), test_dataset_vectors[0].size))
    test_dataset_path = os.path.join(root_data_dir, 'pipelines-data', 'test_embeds.npy')
    save_embeds(test_dataset_vectors, test_dataset_path)
    train_dataset_path = os.path.join(root_data_dir, 'pipelines-data', 'train_embeds.npy')
    save_embeds(ground_truth_dataset_vectors, train_dataset_path)