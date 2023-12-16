import pickle

from dev_models.aerecommender import AERecommender
from dev_models.dssmmodel import DSSMModel
from dev_models.userknn import UserKnn


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "userknn":
            return UserKnn
        if module == "dssmmodel":
            return DSSMModel
        if module == "ae" or name == "AERecommender":
            return AERecommender
        return super().find_class(module, name)
