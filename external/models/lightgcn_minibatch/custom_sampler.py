"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np


class Sampler:
    def __init__(self, indexed_ratings, epochs, steps, seed=42):
        np.random.seed(seed)
        self._indexed_ratings = indexed_ratings
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}
        self._freq_users = dict.fromkeys(self._users,
                                         dict.fromkeys(list(range(epochs)), dict.fromkeys(list(range(steps)), 0)))
        self._freq_items = dict.fromkeys(self._items,
                                         dict.fromkeys(list(range(epochs)), dict.fromkeys(list(range(steps)), 0)))

    def step(self, events: int, batch_size: int, current_epoch):
        r_int = np.random.randint
        n_users = self._nusers
        n_items = self._nitems
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict

        def sample(current_batch):
            u = r_int(n_users)
            self._freq_users[u][current_epoch][current_batch] += 1
            ui = ui_dict[u]
            lui = lui_dict[u]
            if lui == n_items:
                self._freq_users[u][current_epoch][current_batch] -= 1
                sample(current_batch)
            i = ui[r_int(lui)]
            self._freq_items[i][current_epoch][current_batch] += 1

            j = r_int(n_items)
            while j in ui:
                j = r_int(n_items)
            self._freq_items[j][current_epoch][current_batch] += 1
            return u, i, j

        for batch_start in range(0, events, batch_size):
            bui, bii, bij = map(np.array,
                                zip(*[sample(idx) for idx in
                                      range(batch_start, min(batch_start + batch_size, events))]))
            yield bui[:, None], bii[:, None], bij[:, None]
