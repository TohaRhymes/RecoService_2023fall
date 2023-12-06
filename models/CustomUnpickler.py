import pickle

from dev_models.userknn import UserKnn


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "userknn":
            return UserKnn
        return super().find_class(module, name)
