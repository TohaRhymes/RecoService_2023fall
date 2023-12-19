import os
import pickle
from typing import List

from dotenv import load_dotenv

from models.CustomUnpickler import CustomUnpickler

load_dotenv(".test_env")
DSSM_NAME = os.getenv("DSSM_NAME")
DATA_DIR = os.getenv("DATA_DIR")


class DSSMModelProd(pickle.Unpickler):
    def __init__(self):
        # Load the model when initializing the class
        if os.path.exists(DSSM_NAME):
            with open(DSSM_NAME, "rb") as file:
                self.model = CustomUnpickler(file).load()
        else:
            self.model = None

    def predict(self, user_id, k: int = 10) -> List[int]:
        return self.model.recommend(user_id, k_recos=k)
