import os
import pickle

from dotenv import load_dotenv
from rectools.dataset import Dataset

from dev_models.dev_eval import read_kion_dataset
from dev_models.userknn import UserKnn

load_dotenv(".test_env")
USERKNNCOS70_NAME = os.getenv("USERKNNCOS70_NAME")
DATA_DIR = os.getenv("DATA_DIR")


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "userknn":
            return UserKnn
        return super().find_class(module, name)


class UserKnnCos70(pickle.Unpickler):
    def __init__(self):
        # Load the model when initializing the class
        if os.path.exists(USERKNNCOS70_NAME):
            with open(USERKNNCOS70_NAME, "rb") as file:
                self.model = CustomUnpickler(file).load()
            self.data_for_predict = Dataset.construct(
                read_kion_dataset(fast_check=1, data_dir=DATA_DIR)["interactions"].df
            )
        else:
            self.model = None
            self.data_for_predict = None

    def predict(self, user_id):
        # Assuming data_for_predict is available as a class attribute
        # or passed as an argument
        return self.model.recommend_for_user(dataset=self.data_for_predict, user_id=user_id)
