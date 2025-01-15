"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Marco Valentini'
__email__ = 'm.valentini7@phd.poliba.it'

import pickle
import time

from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.knn.user_knn_fairness.user_knn_similarity import Similarity
from elliot.recommender.base_recommender_model import init_charger


class UserKNNfairness(RecMixin, BaseRecommenderModel):
    r"""
    GroupLens: An Open Architecture for Collaborative Filtering of Netnews

    For further details, please refer to the `paper https://www.sciencedirect.com/science/article/pii/S0306457321001369?ref=pdf_download&fr=RR-2&rr=8f05cb5e1b595252
    https://proceedings.mlr.press/v81/ekstrand18b/ekstrand18b.pdf`_

    Args:
        neighbors: Number of item neighbors
        similarity: Similarity function
        implementation: Implementation type ('aiolli', 'classical')
        pre_post_processing: ('value', 'parity', 'interactions', 'users') # in this way we can apply either a
        preprocessing or a post processing strategy in the experiment, they haven't been tought to be combined


    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        UserKNNfairness:
          meta:
            save_recs: True
          neighbors: 40
          similarity: cosine
          pre_post_processing: parity/interactions

    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_num_neighbors", "neighbors", "nn", 40, int, None),
            ("_similarity", "similarity", "sim", "cosine", None, None),
            ("_implicit", "implicit", "bin", False, None, None),
            ("_pre_post_processing", "pre_post_processing", "preposp", None, None, None),
            ("_asymmetric_alpha", "asymmetric_alpha", "asymalpha", False, None, lambda x: x if x else ""),
            ("_tversky_alpha", "tversky_alpha", "tvalpha", False, None, lambda x: x if x else ""),
            ("_tversky_beta", "tversky_beta", "tvbeta", False, None, lambda x: x if x else "")
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict

        self._model = Similarity(data=self._data, num_neighbors=self._num_neighbors, similarity=self._similarity, implicit=self._implicit, pre_post_processing=self._pre_post_processing, alpha=self._asymmetric_alpha,
                                     tversky_alpha=self._tversky_alpha, tversky_beta=self._tversky_beta)

    def get_single_recommendation(self, mask, k, *args):
        return {u: self._model.get_user_recs(u, mask, k) for u in self._ratings.keys()}

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    @property
    def name(self):
        return f"UserKNNfairness_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        start = time.time()
        self._model.initialize()
        end = time.time()
        print(f"The similarity computation has taken: {end - start}")

        print(f"Transactions: {self._data.transactions}")
        self._ratings = self._model._ratings

        self.evaluate()

        self._model.restore_items_users()
        self._ratings = self._model._data._old_ratings