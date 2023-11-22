# Standard library imports
import os
import zipfile as zf
from copy import deepcopy
from time import time
from typing import Dict, List

# Third-party imports
import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

# Local imports
from rectools import Columns
from rectools.dataset import Interactions, Dataset
from rectools.metrics import calc_metrics
from rectools.model_selection import Splitter
from rectools.models.base import ModelBase

# dir and files constants
DATA_DIR = '../data'
KION_DIR = os.path.join(DATA_DIR, 'data_original')
INTERACTIONS_DATA = os.path.join(KION_DIR, 'interactions.csv')
USERS_DATA = os.path.join(KION_DIR, 'users.csv')
ITEMS_DATA = os.path.join(KION_DIR, 'items.csv')
# download dataset if it is not loaded
if not os.path.isdir(KION_DIR):
    url = 'https://github.com/irsafilo/KION_DATASET/raw/f69775be31fa5779907cf0a92ddedb70037fb5ae/data_original.zip'

    req = requests.get(url, stream=True)
    ZIP_FILE = os.path.join(DATA_DIR, 'kion.zip')
    with open(ZIP_FILE, 'wb') as fd:
        total_size_in_bytes = int(req.headers.get('Content-Length', 0))
        progress_bar = tqdm(desc='kion dataset download',
                            total=total_size_in_bytes, unit='iB',
                            unit_scale=True)
        for chunk in req.iter_content(chunk_size=2 ** 20):
            progress_bar.update(len(chunk))
            fd.write(chunk)

    files = zf.ZipFile(ZIP_FILE, 'r')
    files.extractall(DATA_DIR)
    files.close()
    os.remove(ZIP_FILE)

INTERACTIONS = Interactions(pd.read_csv(INTERACTIONS_DATA,
                                        parse_dates=[
                                            "last_watch_dt"]).rename(
    columns={'last_watch_dt': Columns.Datetime,
             'total_dur': Columns.Weight
             }
))
USERS = pd.read_csv(USERS_DATA)
ITEMS = pd.read_csv(ITEMS_DATA)


def calculate_metrics(models: Dict[str, ModelBase],
                      metrics: Dict,
                      cv: Splitter,
                      K_RECOS: int) -> pd.DataFrame:
    results = []
    fold_iterator = cv.split(INTERACTIONS, collect_fold_stats=True)

    n_splits = cv.n_splits

    for train_ids, test_ids, fold_info in tqdm((fold_iterator),
                                               total=n_splits):

        df_train = INTERACTIONS.df.iloc[train_ids]
        dataset = Dataset.construct(df_train)

        df_test = INTERACTIONS.df.iloc[test_ids][Columns.UserItem]
        test_users = np.unique(df_test[Columns.User])

        catalog = df_train[Columns.Item].unique()

        for model_name, model in models.items():
            cur_model = deepcopy(model)
            print(
                f"===== Model: {model_name} | Fold: {fold_info['i_split']} =====")

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
    return pd.DataFrame(results).groupby('model').mean().drop(['fold'], axis=1)


def visualize(model: ModelBase,
              dataset: List[Interactions, pd.DataFrame],
              # Interactions, Users, Items
              user_list: List[int],
              item_data: pd.DataFrame) -> None:
    pass
