import os
import zipfile as zf
from copy import deepcopy
from time import time
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
import requests
import seaborn as sns
from rectools import Columns
from rectools.dataset import Dataset, Interactions
from rectools.metrics import calc_metrics
from rectools.model_selection import Splitter
from rectools.models.base import ModelBase
from tqdm.auto import tqdm

from dev_models.userknn import UserKnn

REPORT_PART = [
    """----------------------------------------------------------
User: {cur_user}
Already watched films amount: {len_interactions}
Display last {k_display} watched:
""",
    """
Recommended films amount: {len_recos}
(Amount of all films: {total_len})
Display first {k_display} recommendations:""",
]
DATA_DIR = "../data"
CMAP = sns.diverging_palette(10, 133, as_cmap=True)


def print_verbose(s: str, verbose: int = 0):
    if verbose > 0:
        print(s)


def read_kion_dataset(fast_check: float = 1, data_dir: str = DATA_DIR) -> Dict[str, Union[Interactions, pd.DataFrame]]:
    # dir and files constants
    kion_dir = os.path.join(data_dir, "data_original")
    intersections_data = os.path.join(kion_dir, "interactions.csv")
    users_data = os.path.join(kion_dir, "users.csv")
    items_data = os.path.join(kion_dir, "items.csv")
    zip_file = os.path.join(data_dir, "kion.zip")
    # download dataset if it is not loaded
    if not os.path.isdir(kion_dir):
        url = "https://github.com/irsafilo/KION_DATASET/raw/f69775be31fa5779907cf0a92ddedb70037fb5ae/data_original.zip"

        req = requests.get(url, stream=True, timeout=100)
        with open(zip_file, "wb") as fd:
            total_size_in_bytes = int(req.headers.get("Content-Length", 0))
            progress_bar = tqdm(desc="kion dataset download", total=total_size_in_bytes, unit="iB", unit_scale=True)
            for chunk in req.iter_content(chunk_size=2**20):
                progress_bar.update(len(chunk))
                fd.write(chunk)

        with zf.ZipFile(zip_file, "r") as files:
            files.extractall(data_dir)
        os.remove(zip_file)
    interactions_df = pd.read_csv(intersections_data, parse_dates=["last_watch_dt"]).rename(
        columns={"last_watch_dt": Columns.Datetime, "total_dur": Columns.Weight}
    )
    if fast_check < 1:
        interactions = Interactions(interactions_df.sample(frac=fast_check))
    else:
        interactions = Interactions(interactions_df)
    # also read users and items
    users = pd.read_csv(users_data)
    items = pd.read_csv(items_data)
    return {"interactions": interactions, "users": users, "items": items}


def group_by_and_beautify(results: List[Dict], custom_order: List, style: bool = False) -> pd.DataFrame:
    # groupby model
    resulting_data = pd.DataFrame(results).groupby("model").agg(["mean", "std"]).drop(["fold"], axis=1)
    #  order by input metrics
    new_columns = pd.MultiIndex.from_product(
        [custom_order, resulting_data.columns.levels[1]], names=resulting_data.columns.names
    )
    resulting_data = resulting_data.reindex(columns=new_columns)
    # separate metric_name and k by `@`
    new_columns = []
    for col in resulting_data.columns:
        first_level, second_level = col[0].split("@")
        new_columns.append((first_level, int(second_level), col[1]))
    resulting_data.columns = pd.MultiIndex.from_tuples(new_columns, names=["Metric", "At", "Stat"])
    # return styled/not styled
    if style:
        mean_metric_subset = [(metric, at, agg) for metric, at, agg in resulting_data.columns if agg == "mean"]
        return resulting_data.style.background_gradient(subset=mean_metric_subset, axis=0, cmap=CMAP)
    return resulting_data


def calculate_metrics(
    models: Dict[str, Union[ModelBase, UserKnn]],
    dataset: Dict[str, Union[Interactions, pd.DataFrame]],
    metrics: Dict,
    cv: Splitter,
    k_recos: int,
    style: bool = False,
    verbose: int = 0,
) -> pd.DataFrame:
    """
    Reference notebook:
    github.com/MobileTeleSystems/RecTools/blob/main/examples
    /2_cross_validation.ipynb

    :param models: Dict: str(name of model) -> RecTools' Model
    :param dataset: Dict: str(name of model) -> Interactions,users,items
    :param metrics: Dict: str(name of metric) -> RecTools' Metric
    :param cv: RecTools' Splitter
    :param k_recos: Amount of recommendations
    :param style: bool: whether to style output table or not
    :return: pd.DataFrame, that shows metrics for all the models
    """

    interactions = dataset["interactions"]

    results = []
    cv_iterator = cv.split(interactions, collect_fold_stats=True)
    cv_total = cv.n_splits

    for train_ids, test_ids, fold_info in tqdm(cv_iterator, total=cv_total):
        df_train = interactions.df.iloc[train_ids].copy()
        dataset_train = Dataset.construct(df_train)

        df_test = interactions.df.iloc[test_ids][Columns.UserItem].copy()
        test_users = np.unique(df_test[Columns.User])

        catalog = df_train[Columns.Item].unique()

        for model_name, model in models.items():
            cur_model = deepcopy(model)
            print_verbose(
                f"======================================================="
                f"|| Model: {model_name} | Fold: {fold_info['i_split']}"
                f"=======================================================",
                verbose=verbose,
            )

            last_time = time()
            cur_model.fit(dataset_train)
            print_verbose(f"Fit time: {round(time() - last_time, 2)} sec.", verbose=verbose)

            last_time = time()

            recos = cur_model.recommend(users=test_users, dataset=dataset_train, k=k_recos, filter_viewed=False)
            print_verbose(f"Recommend time: {round(time() - last_time, 2)} sec.", verbose=verbose)

            last_time = time()
            metric_values = calc_metrics(
                metrics,
                reco=recos,
                interactions=df_test,
                prev_interactions=df_train,
                catalog=catalog,
            )
            print_verbose(f"Metrics time: {round(time() - last_time, 2)} sec.", verbose=verbose)

            res = {"fold": fold_info["i_split"], "model": model_name}
            res.update(metric_values)
            results.append(res)
    return group_by_and_beautify(results, list(metrics.keys()), style)


def visualize(
    model: ModelBase,
    dataset: Dict[str, Union[Interactions, pd.DataFrame]],
    # Interactions, Users, Items
    user_list: List[int],
    item_data: List[str],
    k_recos: int = 10,
    k_display: int = 10,
    display: Callable = print,
) -> None:
    interactions = dataset["interactions"]
    items = dataset["items"]

    dataset_for_train = Dataset.construct(interactions.df)

    recos = model.recommend(
        users=user_list,
        dataset=dataset_for_train,
        k=k_recos,
        filter_viewed=True,
    )

    items_counter = interactions.df.item_id.value_counts()
    items_counter = pd.merge(items[["item_id"] + item_data], items_counter, left_on="item_id", right_index=True).rename(
        columns={"count": "views"}
    )

    print("Visual report")
    for cur_user in user_list:
        cur_user_interactions = interactions.df[interactions.df.user_id == cur_user]
        cur_user_recos = recos[recos.user_id == cur_user]

        report_dict = {
            "cur_user": cur_user,
            "len_interactions": len(cur_user_interactions),
            "len_recos": len(cur_user_recos),
            "total_len": interactions.df.item_id.unique().shape[0],
            "k_display": k_display,
        }
        # display last (most recent) k_display items, that user interacted with
        print(REPORT_PART[0].format(**report_dict))
        display(
            pd.merge(cur_user_interactions, items_counter, on="item_id")
            .drop(["user_id"], axis=1)
            .sort_values(by="datetime", ascending=False)
            .head(k_display)
        )
        print(REPORT_PART[1].format(**report_dict))
        display(
            pd.merge(cur_user_recos, items_counter, on="item_id", how="left").drop(["user_id"], axis=1).head(k_display)
        )
