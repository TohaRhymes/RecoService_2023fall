import pickle

from dev_models.aerecommender import AERecommender
from dev_models.dssmmodel import DSSMModel
from dev_models.ranker_lfm_popular import RankerLFMPopular
from dev_models.recbole_model import Recbole
from dev_models.userknn import UserKnn


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "userknn":
            return UserKnn
        if module == "dssmmodel":
            return DSSMModel
        if module == "ae" or name == "AERecommender":
            return AERecommender
        if name == "Recbole":
            return Recbole
        if name == "RankerLFMPopular":
            return RankerLFMPopular
        return super().find_class(module, name)
