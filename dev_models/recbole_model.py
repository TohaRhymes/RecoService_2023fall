class Recbole:
    def __init__(
        self,
        id_2_reco,
        popular_list,
    ):
        if any(
            arg is None
            for arg in [
                id_2_reco,
                popular_list,
            ]
        ):
            raise ValueError("None of the arguments can be None")

        self.id_2_reco = id_2_reco
        self.popular_list = popular_list

    def recommend(self, user_id, k_recos=10):
        if str(user_id) in self.id_2_reco:
            return self.id_2_reco[str(user_id)]
        else:
            return self.popular_list[:k_recos]
