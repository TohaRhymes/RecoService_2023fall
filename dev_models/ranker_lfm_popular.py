class RankerLFMPopular:
    def __init__(self, recos, watched, popular_list):
        if any(arg is None for arg in [recos, watched, popular_list]):
            raise ValueError("None of the arguments can be None")

        self.recos = recos
        self.watched = watched
        self.popular_list = popular_list

    def recommend(self, user_id, k_recos=10):
        final_prediction = []
        if user_id in self.recos:
            final_prediction = self.recos[user_id]
            # check watched
            if user_id in self.watched:
                cur_watched = self.watched[user_id]
                final_prediction = [film for film in final_prediction if film not in cur_watched]
            # append popular, if not enough
            for item in self.popular_list:
                if len(final_prediction) >= k_recos:
                    break
                if item not in cur_watched and item not in final_prediction:
                    final_prediction.append(item)
        else:
            final_prediction = self.popular_list[:k_recos]
        return final_prediction
