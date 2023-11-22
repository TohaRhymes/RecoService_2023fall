import os
import zipfile as zf
from copy import deepcopy
from time import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from IPython.display import display
from rectools import Columns
from rectools.dataset import Dataset, Interactions
from rectools.metrics import calc_metrics
from rectools.model_selection import Splitter
from rectools.models.base import ModelBase
from tqdm.auto import tqdm

REPORT_PART = [
    """----------------------------------------------------------
User: {cur_user}
Already watched films amount: {len_interactions}
""",
    """Recomended films amount: {len_recos}
(Amount of all films: {total_len})""",
]


def read_kion_dataset() -> Tuple[Interactions, pd.DataFrame, pd.DataFrame]:
    # dir and files constants
    DATA_DIR = "../data"
    KION_DIR = os.path.join(DATA_DIR, "data_original")
    INTERACTIONS_DATA = os.path.join(KION_DIR, "interactions.csv")
    USERS_DATA = os.path.join(KION_DIR, "users.csv")
    ITEMS_DATA = os.path.join(KION_DIR, "items.csv")
    # download dataset if it is not loaded
    if not os.path.isdir(KION_DIR):
        url = "https://github.com/irsafilo/KION_DATASET/raw/f69775be31fa5779907cf0a92ddedb70037fb5ae/data_original.zip"

        req = requests.get(url, stream=True, timeout=100)
        ZIP_FILE = os.path.join(DATA_DIR, "kion.zip")
        with open(ZIP_FILE, "wb") as fd:
            total_size_in_bytes = int(req.headers.get("Content-Length", 0))
            progress_bar = tqdm(desc="kion dataset download",
                                total=total_size_in_bytes, unit="iB",
                                unit_scale=True)
            for chunk in req.iter_content(chunk_size=2 ** 20):
                progress_bar.update(len(chunk))
                fd.write(chunk)

        with zf.ZipFile(ZIP_FILE, "r") as files:
            files.extractall(DATA_DIR)
        os.remove(ZIP_FILE)

    INTERACTIONS = Interactions(
        pd.read_csv(INTERACTIONS_DATA, parse_dates=["last_watch_dt"]).rename(
            columns={"last_watch_dt": Columns.Datetime,
                     "total_dur": Columns.Weight}
        )
    )
    USERS = pd.read_csv(USERS_DATA)
    ITEMS = pd.read_csv(ITEMS_DATA)
    return INTERACTIONS, USERS, ITEMS


def calculate_metrics(models: Dict[str, ModelBase], metrics: Dict,
                      cv: Splitter, K_RECOS: int) -> pd.DataFrame:
    """
    Reference notebook:
    github.com/MobileTeleSystems/RecTools/blob/main/examples/2_cross_validation.ipynb

    :param models: Dict: str(name of model) -> RecTools' Model
    :param metrics: Dict: str(name of metric) -> RecTools' Metric
    :param cv: RecTools' Splitter
    :param K_RECOS: Amount of recommendations
    :return: pd.DataFrame, that shows metrics for all the models
    """

    INTERACTIONS, _, _ = read_kion_dataset()

    results = []
    fold_iterator = cv.split(INTERACTIONS, collect_fold_stats=True)

    for train_ids, test_ids, fold_info in tqdm((fold_iterator),
                                               total=cv.n_splits):
        df_train = INTERACTIONS.df.iloc[train_ids]
        dataset = Dataset.construct(df_train)

        df_test = INTERACTIONS.df.iloc[test_ids][Columns.UserItem]
        test_users = np.unique(df_test[Columns.User])

        catalog = df_train[Columns.Item].unique()

        for model_name, model in models.items():
            cur_model = deepcopy(model)
            print(
                f"======================================================="
                f"|| Model: {model_name} | Fold: {fold_info['i_split']}"
                f"======================================================="
            )

            last_time = time()
            cur_model.fit(dataset)
            print(f"Fit time: {round(time() - last_time, 2)} sec.")

            last_time = time()
            recos = cur_model.recommend(
                users=test_users,
                dataset=dataset,
                k=K_RECOS,
                filter_viewed=True,
            )
            print(f"Recommend time: {round(time() - last_time, 2)} sec.")

            last_time = time()
            metric_values = calc_metrics(
                metrics,
                reco=recos,
                interactions=df_test,
                prev_interactions=df_train,
                catalog=catalog,
            )
            print(f"Metrics time: {round(time() - last_time, 2)} sec.")

            res = {"fold": fold_info["i_split"], "model": model_name}
            res.update(metric_values)
            results.append(res)
    return pd.DataFrame(results).groupby("model").agg(
        ["mean", "std"]).drop(["fold"], axis=1)


def visualize(
    model: ModelBase,
    dataset: Tuple[Interactions, pd.DataFrame, pd.DataFrame],
    # Interactions, Users, Items
    user_list: List[int],
    item_data: List[str],
    K_RECOS: int = 10,
) -> None:
    interactions = dataset[0]
    items = dataset[2]

    dataset_for_train = Dataset.construct(interactions.df)
    recos = model.recommend(
        users=user_list,
        dataset=dataset_for_train,
        k=K_RECOS,
        filter_viewed=True,
    )

    items_counter = interactions.df.item_id.value_counts()
    items_counter = pd.merge(items[["item_id"] + item_data], items_counter,
                             left_on="item_id", right_index=True).rename(
        columns={"count": "views"}
    )

    print("Visual report")
    for cur_user in user_list:
        cur_user_interactions = interactions.df[
            interactions.df.user_id == cur_user]
        cur_user_recos = recos[recos.user_id == cur_user]

        report_dict = {
            "cur_user": cur_user,
            "len_interactions": len(cur_user_interactions),
            "len_recos": len(cur_user_recos),
            "total_len": interactions.df.item_id.unique().shape[0],
        }

        print(REPORT_PART[0].format(**report_dict))
        display(
            pd.merge(cur_user_interactions, items_counter, on="item_id").drop(
                ["user_id"], axis=1))
        print(REPORT_PART[1].format(**report_dict))
        display(pd.merge(cur_user_recos, items_counter, on="item_id").drop(
            ["user_id"], axis=1))
