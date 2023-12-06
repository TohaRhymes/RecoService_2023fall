import os
import pickle
from typing import List

from dotenv import load_dotenv
from rectools.dataset import Dataset

from dev_models.dev_eval import read_kion_dataset
from models.CustomUnpickler import CustomUnpickler

load_dotenv(".test_env")
POPULAR_NAME = os.getenv("POPULAR_NAME")
LFM_NAME = os.getenv("LFM_NAME")
DATA_DIR = os.getenv("DATA_DIR")


class LFM(pickle.Unpickler):
    def __init__(self):
        # load data and watched by users
        kion_data = read_kion_dataset(fast_check=1, data_dir=DATA_DIR)
        interactions = kion_data["interactions"]
        data_for_predict = Dataset.construct(interactions.df)
        self.watched = dict(
            interactions.df[["user_id", "item_id"]].groupby("user_id")[
                "item_id"].agg(list))

        # save max number of films
        max_k = len(kion_data["items"]["item_id"].unique())

        # == extract popular ===
        # Load the popular model
        if os.path.exists(POPULAR_NAME):
            with open(POPULAR_NAME, "rb") as file:
                self.loaded_popular = CustomUnpickler(file).load()
        else:
            self.loaded_popular = None

        # get popular list (all items, but ranked)
        sample_popular_user = data_for_predict.user_id_map.external_ids[0]
        self.popular_list = list(
            self.loaded_popular.recommend(dataset=data_for_predict,
                                          users=[sample_popular_user, ],
                                          k=max_k, filter_viewed=False)[
                "item_id"]
        )

        # == load the real LFM model ===
        if os.path.exists(LFM_NAME):
            with open(LFM_NAME, "rb") as file:
                self.loaded_lfm = CustomUnpickler(file).load()
        else:
            self.loaded_lfm = None

    def predict(self, user_id: int, k: int = 10) -> List[int]:
        # Assuming data_for_predict is available as a class attribute
        # or passed as an argument

        final_prediction = []
        if user_id in self.watched:
            cur_watched = self.watched[user_id]
            final_prediction = self.loaded_lfm.get_item_list_for_user(user_id,
                                                                      top_n=k).tolist()
            # check watched
            final_prediction = [film for film in final_prediction if
                                film not in cur_watched]
            # append popular, if not enough
            for item in self.popular_list:
                if len(final_prediction) >= k:
                    break
                if item not in cur_watched and item not in final_prediction:
                    final_prediction.append(item)
        else:
            final_prediction = self.popular_list[:k]

        return final_prediction
