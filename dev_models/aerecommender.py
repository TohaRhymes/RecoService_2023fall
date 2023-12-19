import warnings
from typing import List

import numpy as np
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")
tqdm.pandas()


class AERecommender:
    MODEL_NAME = "Autoencoder"

    def __init__(self, X_preds, X_train_and_val, X_test, user_id2uid, popular_list):
        self.X_preds = X_preds.cpu().detach().numpy()
        self.X_train_and_val = X_train_and_val
        self.X_test = X_test
        self.user_id2uid = user_id2uid
        self.popular_list = popular_list

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_select_idx, topn=10, verbose=False) -> List:
        user_preds = self.X_preds[user_id][items_to_select_idx]
        items_idx = items_to_select_idx[np.argsort(-user_preds)[:topn]]

        # Recommend the highest predicted rating movies
        # that the user hasn't seen yet.
        return items_idx

    def recommend(self, user_id, k_recos=10, verbose=False):
        if user_id in self.user_id2uid:
            user_id = self.user_id2uid[user_id]
        else:
            return self.popular_list[:k_recos]
        X_total = self.X_train_and_val + self.X_test
        all_nonzero = np.argwhere(X_total[user_id] > 0).ravel()
        select_from = np.setdiff1d(np.arange(X_total.shape[1]), all_nonzero)
        preds = self.recommend_items(user_id, select_from, topn=k_recos)
        return preds

    def evaluate(self, size=100):
        X_total = self.X_train_and_val + self.X_test

        true_5 = []
        true_10 = []

        for user_id in range(len(self.X_test)):
            non_zero = np.argwhere(self.X_test[user_id] > 0).ravel()
            all_nonzero = np.argwhere(X_total[user_id] > 0).ravel()
            select_from = np.setdiff1d(np.arange(X_total.shape[1]), all_nonzero)

            for non_zero_idx in non_zero:
                random_non_interacted_100_items = np.random.choice(select_from, size=20, replace=False)
                preds = self.recommend_items(user_id, np.append(random_non_interacted_100_items, non_zero_idx), topn=10)
                true_5.append(non_zero_idx in preds[:5])
                true_10.append(non_zero_idx in preds)

        return {"recall@5": np.mean(true_5), "recall@10": np.mean(true_10)}
