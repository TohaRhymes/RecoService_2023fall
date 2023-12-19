# считаем расстояние между вектором юзера и вектором айтема
from typing import List

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as ED


class DSSMModel:
    def __init__(
        self,
        items_vecs,
        users_dssm,
        users_meta_feats,
        user_id_to_uid,
        interactions_vec,
        u2v,
        iid_to_item_id,
        popular_list,
    ):
        # Ensure that none of the arguments are None
        if any(
            arg is None
            for arg in [
                items_vecs,
                users_dssm,
                users_meta_feats,
                user_id_to_uid,
                interactions_vec,
                u2v,
                iid_to_item_id,
                popular_list,
            ]
        ):
            raise ValueError("None of the arguments can be None")

        self.items_vecs = items_vecs
        self.users_dssm = users_dssm
        self.users_meta_feats = users_meta_feats
        self.user_id_to_uid = user_id_to_uid
        self.interactions_vec = interactions_vec
        self.u2v = u2v
        self.iid_to_item_id = iid_to_item_id
        self.popular_list = popular_list

    def find_indices_of_smallest(self, arr, k_recos=10) -> np.ndarray:
        k_recos = min(len(arr), k_recos)
        return np.argsort(arr)[:k_recos]

    def recommend_dssm(self, uid, k_recos=10) -> List[int]:
        user_meta_feats = self.users_meta_feats.iloc[uid]
        user_interaction_vec = self.interactions_vec[uid]
        user_vec = self.u2v.predict(
            [np.array(user_meta_feats).reshape(1, -1), np.array(user_interaction_vec).reshape(1, -1)],
            verbose=False,
        )
        dists = ED(user_vec, self.items_vecs)
        topk_iids = self.find_indices_of_smallest(dists[0], k_recos=k_recos)
        topk_iids_item = [self.iid_to_item_id[iid] for iid in topk_iids.reshape(-1)]
        return topk_iids_item

    def recommend(self, user_id, k_recos=10):
        if user_id in self.user_id_to_uid:
            uid = self.user_id_to_uid[user_id]
            return self.recommend_dssm(uid, k_recos=k_recos)
        else:
            return self.popular_list[:k_recos]
