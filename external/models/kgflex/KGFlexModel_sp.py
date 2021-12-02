"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import time
import pickle
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix

_RANDOM_SEED = 42


class KGFlexModel():
    def __init__(self,
                 data,
                 learning_rate,
                 n_features,
                 feature_key_mapping,
                 item_features_mapper,
                 embedding_size,
                 users_features,
                 **kwargs):


        self._data = data
        self._learning_rate = learning_rate
        self._n_users = data.num_users
        self._n_items = data.num_items
        self._users = data.private_users.keys()

        self._n_features = n_features
        self._embedding_size = embedding_size

        # Global features embeddings
        self.Gf = np.random.randn(n_features, embedding_size) / 10
        self.Gb = np.random.randn(n_features) / 10

        # Personal features embeddings
        # users features mask
        self.Mu = np.zeros(shape=(self._n_users, n_features))
        for u, f in users_features.items():
            self.Mu[u][list(map(feature_key_mapping.get, f))] = 1
        self.Mu = self.Mu.astype(bool)

        # users features mask along embedding axis
        self.Me = np.repeat(self.Mu[:, :, np.newaxis], embedding_size, axis=2)


        # initialize users features vectors
        self.P = np.zeros((self._n_users, n_features, embedding_size))
        self.P[self.Me] = np.random.randn(*self.P[self.Me].shape) / 10

        # users constant weights
        self.K = np.zeros(shape=(self._n_users, n_features))
        for u in self._users:
            for feature in users_features[u].keys():
                self.K[u][feature_key_mapping[feature]] = users_features[u][feature]


        self.P_sp = dict()
        self.K_sp = dict()
        for u in tqdm(range(self._n_users), desc='Sparse', total=self._n_users):
            self.P_sp[u] = np.zeros((n_features, embedding_size))
            self.P_sp[u][self.Me[u]] = np.random.randn(*self.P_sp[u][self.Me[u]].shape) / 10
            self.P_sp[u] = csr_matrix(self.P_sp[u])

            self.K_sp[u] = np.zeros(shape=n_features)
            for feature, info_gain in users_features[u].items():
                self.K_sp[u][feature_key_mapping[feature]] = info_gain
            self.K_sp[u] = csr_matrix(self.K_sp[u])

        # item features mask
        self.Mi = np.zeros(shape=(self._n_items, n_features))
        for item, features in item_features_mapper.items():
            for f in features:
                key = feature_key_mapping.get(f)
                if key:
                    self.Mi[item][key] = 1
        self.Mi = self.Mi.astype(bool)

        # user item features
        self.user_item_features = dict()
        self._n_features_range = np.array(range(self._n_features))
        for u in tqdm(range(self._n_users), desc='user-item features'):
            u_i = dict()
            for i in range(self._n_items):
                u_i[i] = self._n_features_range[np.multiply(self.Mu[u], self.Mi[i])]
            self.user_item_features[u] = u_i

    def __call__(self, *inputs):

        user, item = inputs
        user, item = np.array(user), np.array(item)

        #p = self.P_sp[u][self.user_item_features[u][i]]

        return np.array(
            [np.sum(
                (np.sum(
                    np.multiply(
                        self.P_sp[u][self.user_item_features[u][i]].A, self.Gf[self.user_item_features[u][i], :]),
                    axis=1) + self.Gb[self.user_item_features[u][i]]
                 )
            ) for u, i in zip(user, item)])

        # HYBRID SPARSE
        # return np.array(
        #     [np.sum(
        #         (np.sum(
        #             np.multiply(
        #                 self.P_sp[u][self.user_item_features[u][i]].A, self.Gf[self.user_item_features[u][i], :]),
        #             axis=1) + self.Gb[self.user_item_features[u][i]]
        #          ) * self.K_sp[u].T[self.user_item_features[u][i]].A
        #     ) for u, i in zip(user, item)])

        # PURE SPARSE
        # return np.array(
        #     [np.sum((np.sum(csr_matrix.multiply(self.P_sp[u][self.user_item_features[u][i]],
        #                                         self.Gf[self.user_item_features[u][i]]), axis=1).A1 + self.Gb[
        #                  self.user_item_features[u][i]]) * self.K[u][self.user_item_features[u][i]]) for u, i in
        #      zip(user, item)])
        # DENSE
        # return np.array(
        #     [np.sum(
        #         (np.sum(
        #             np.multiply(
        #                 self.P[u, self.user_item_features[u][i], :], self.Gf[self.user_item_features[u][i], :]),
        #             axis=1) + self.Gb[self.user_item_features[u][i]]
        #          ) * self.K[u][self.user_item_features[u][i]]
        #     ) for u, i in zip(user, item)])

    def train_step(self, batch):

        loss = 0.0
        user, pos, neg = batch
        user = user[:, 0]
        pos = pos[:, 0]
        neg = neg[:, 0]
        call1 = time.perf_counter()
        x_p = self(user, pos)
        x_n = self(user, neg)
        print(f'Calls: {time.perf_counter() - call1}')
        x_pn = np.subtract(x_p, x_n)
        d_loss = (1 / (1 + np.exp(x_pn)))

        m_p = np.multiply(self.Mu[user], self.Mi[pos])
        m_n = np.multiply(self.Mu[user], self.Mi[neg])

        indexes_p = np.array(range(self._n_features))
        indexes_n = np.array(range(self._n_features))

        for us_, mask_pos, mask_neg, d_loss_ in zip(user, m_p, m_n, d_loss):
            f_p = indexes_p[mask_pos]
            f_n = indexes_n[mask_neg]

            # updates
            #DENSE
            # self.P[us_, f_p] += self._learning_rate * self.Gf[f_p] * self.K[us_][f_p][:, np.newaxis] * d_loss_
            # self.P[us_, f_n] += self._learning_rate * self.Gf[f_n] * -self.K[us_][f_n][:, np.newaxis] * d_loss_
            # self.Gf[f_p] += self._learning_rate * self.P[us_, f_p] * self.K[us_][f_p][:, np.newaxis] * d_loss_
            # self.Gf[f_n] += self._learning_rate * self.P[us_, f_n] * -self.K[us_][f_n][:, np.newaxis] * d_loss_
            # self.Gb[f_p] += self._learning_rate * self.K[us_][f_p] * d_loss_
            # self.Gb[f_n] += self._learning_rate * -self.K[us_][f_n] * d_loss_


            # SPARSE
            # k_f_p = self.K_sp[us_].T[f_p] * d_loss_ * self._learning_rate
            # k_f_n = self.K_sp[us_].T[f_n] * d_loss_ * self._learning_rate
            # self.P_sp[us_][f_p] += k_f_p.multiply(self.Gf[f_p])
            # self.P_sp[us_][f_n] += k_f_n.multiply(self.Gf[f_n])
            # self.Gf[f_p] += k_f_p.multiply(self.P_sp[us_][f_p])
            # self.Gf[f_n] += k_f_n.multiply(self.P_sp[us_][f_n])
            # self.Gb[f_p] += k_f_p.todense().A1
            # self.Gb[f_n] += k_f_n.todense().A1

            # self.P_sp[us_][f_p] += (self.K_sp[us_].T[f_p] * d_loss_).multiply(self._learning_rate * self.Gf[f_p])
            # self.P_sp[us_][f_n] += (self.K_sp[us_].T[f_n] * d_loss_).multiply(self._learning_rate * self.Gf[f_n])
            # self.Gf[f_p] += (self.K_sp[us_].T[f_p] * d_loss_).multiply(self._learning_rate * self.P_sp[us_][f_p])
            # self.Gf[f_n] += (self.K_sp[us_].T[f_n] * d_loss_).multiply(self._learning_rate * self.P_sp[us_][f_n])
            # self.Gb[f_p] += (self.K_sp[us_].T[f_p] * d_loss_).todense().A1
            # self.Gb[f_n] += (self.K_sp[us_].T[f_n] * d_loss_).todense().A1

            #MIXED
            k_f_p = self.K_sp[us_].todense().A1[f_p] * d_loss_ * self._learning_rate
            k_f_n = self.K_sp[us_].todense().A1[f_n] * d_loss_ * self._learning_rate
            p_u = self.P_sp[us_].todense().A
            self.P_sp[us_][f_p] += self.Gf[f_p] * k_f_p[:, np.newaxis]
            self.P_sp[us_][f_n] += self.Gf[f_n] * k_f_n[:, np.newaxis]
            self.Gf[f_p] += p_u[f_p] * k_f_p[:, np.newaxis]
            self.Gf[f_n] += p_u[f_n] * k_f_n[:, np.newaxis]
            self.Gb[f_p] += k_f_p
            self.Gb[f_n] += k_f_n

            loss += d_loss_
        return loss

    def predict(self, user):

        eval_items = list(range(self._n_items))

        selfPu = self.P[user]
        selfGf = self.Gf
        selfGb = self.Gb
        selfKu = self.K[user]
        selfuser_item_featuresu = self.user_item_features[user]
        return np.array(
            [np.sum(
                (np.sum(
                    np.multiply(
                        selfPu[ui], selfGf[ui]),
                    axis=1) + selfGb[ui]
                 ) * selfKu[ui]
            ) for ui in [selfuser_item_featuresu[i] for i in eval_items]])

    def get_user_recs(self, u, mask, k):
        user_id = self._data.public_users.get(u)
        user_recs = self.predict(user_id)
        user_recs_mask = mask[user_id]
        user_recs[~user_recs_mask] = -np.inf
        indices, values = zip(*[(self._data.private_items.get(u_list[0]), u_list[1])
                                for u_list in enumerate(user_recs)])
        indices = np.array(indices)
        values = np.array(values)
        local_k = min(k, len(values))
        partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
        real_values = values[partially_ordered_preds_indices]
        real_indices = indices[partially_ordered_preds_indices]
        local_top_k = real_values.argsort()[::-1]
        return [(real_indices[item], real_values[item]) for item in local_top_k]

    def get_config(self):
        raise NotImplementedError

    def get_model_state(self):
        saving_dict = {
            '_global_features': self.Gf,
            '_global_bias': self.Gb,
            '_user_feature_mask': self.Mu,
            'user_feature_mask_along_embedding': self.Me,
            '_user_feature_embeddings': self.P,
            '_user_feature_weights': self.K,
            '_item_feature_mask': self.Mi}
        return saving_dict

    def set_model_state(self, saving_dict):
        self.Gf = saving_dict['_global_features']
        self.Gb = saving_dict['_global_bias']
        self.Mu = saving_dict['_user_feature_mask']
        self.Me = saving_dict['user_feature_mask_along_embedding']
        self.P = saving_dict['_user_feature_embeddings']
        self.K = saving_dict['_user_feature_weights']
        self.Mi = saving_dict['_item_feature_mask']

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)
