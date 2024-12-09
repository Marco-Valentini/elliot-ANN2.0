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
from elliot.recommender.ann.item_ann_faiss_lsh.item_ann_faiss_lsh_similarity import LSHfaissSimilarity
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger



class ItemANNfaissLSH(RecMixin, BaseRecommenderModel):
    r"""
    FAISS IndexLSH-based ANN recommendations: item-to-item collaborative filtering, with approximated neighbors search

    For further details, please refer to the `paper MISSING`_

    Args:
        neighbors: Number of item neighbors
        similarity: Similarity function ('cosine')
        nbits: number of bits used for the encoding (2**nbits total buckets)


    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        ItemANNfaissLSH:
          meta:
            verbose: True
            save_recs: True
          neighbors: 40
          similarity: cosine
          nbits: 5
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        # Define Parameters as tuples: (variable_name, public_name, shortcut, default, reading_function, printing_function)

        self._params_list = [
            ("_num_neighbors", "neighbors", "nn", 40, int, None),
            ("_similarity", "similarity", "sim", "cosine", None, None),
            ("_implicit", "implicit", "bin", False, None, None),
            ("_nbits", "nbits", "nb", 1, int, None)
        ]
        self.autoset_params()
        # Here we have a dictionary containing dictionaries UserID: {ItemID:Rating}
        self._ratings = self._data.train_dict
        # initialize the LSH Similarity model
        self._model = LSHfaissSimilarity(data=self._data, num_neighbors=self._num_neighbors, similarity=self._similarity,
                                    implicit=self._implicit, nbits=self._nbits)

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
        return f"ItemANNfaissLSH_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        start = time.time()
        self._model.initialize()
        end = time.time()
        print(f"The similarity computation has taken: {end - start}")

        print(f"Transactions: {self._data.transactions}")

        self.evaluate()