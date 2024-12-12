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
from elliot.recommender.knn.item_knn_fairness.item_knn_similarity import Similarity
from elliot.recommender.base_recommender_model import init_charger


class ItemKNNfairness(RecMixin, BaseRecommenderModel):
    r"""
    Amazon.com recommendations: item-to-item collaborative filtering

    For further details, please refer to the `paper https://www.sciencedirect.com/science/article/pii/S0306457321001369?ref=pdf_download&fr=RR-2&rr=8f05cb5e1b595252`_

    Args:
        neighbors: Number of item neighbors
        similarity: Similarity function
        implementation: Implementation type ('aiolli', 'classical')
        post_processing: ('value', 'parity')
        pre_processing: ('interactions', 'items')


    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        ItemKNN:
          meta:
            save_recs: True
          neighbors: 40
          similarity: cosine
          post_processing: parity
          pre_processing: interactions
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_num_neighbors", "neighbors", "nn", 40, int, None),
            ("_similarity", "similarity", "sim", "cosine", None, None),
            ("_implicit", "implicit", "bin", False, None, None),
            ("_post_processing", "post_processing", "posp", None, None, None),
            ("_pre_processing", "pre_processing", "prep", None, None, None)
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict

        self._model = Similarity(data=self._data, num_neighbors=self._num_neighbors, similarity=self._similarity, implicit=self._implicit,  post_processing=self._post_processing, pre_processing=self._pre_processing)

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
        return f"ItemKNNfairness_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        start = time.time()
        self._model.initialize()
        end = time.time()
        print(f"The similarity computation has taken: {end - start}")

        print(f"Transactions: {self._data.transactions}")

        self.evaluate()