from types import SimpleNamespace
import pandas as pd
import typing as t
import json

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class ItemPopularityUserTolerance(AbstractLoader):
    '''
    This dataloader has been added to include in the models the information about the clustering of the items based on their
    popularity, and the clustering of the users based on their activity on the platform
    '''

    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.user_file = getattr(ns, "user_file", None)
        self.item_file = getattr(ns, "item_file", None)
        self.user_group_file = getattr(ns, "user_group_file", None)
        self.item_group_file = getattr(ns, "item_group_file", None)
        self.users = users
        self.items = items
        self.map_items = self.load_attribute_file(
            self.item_file)  # contains a mapping from public Item ID to Popularity cluster
        self.map_users = self.load_attribute_file(
            self.user_file)  # contains a mapping from public User ID to Activity cluster
        with open(self.user_group_file) as file:
            self.map_users_group = json.load(file)  # contains a mapping from group ID to list of User IDs
        with open(self.item_group_file) as file:
            self.map_items_group = json.load(file)  # contains a mapping from group ID to list of Item IDs
        self.items = self.items & set(self.map_items.keys())
        self.users = self.users & set(self.map_users.keys())

    def get_mapped(self):
        return self.users, self.items

    def filter(self, users, items):
        self.users = self.users & users
        self.items = self.items & items

    def create_namespace(self):
        ns = SimpleNamespace()
        ns.__name__ = "ItemPopularityUserTolerance"
        ns.object = self
        ns.feature_map_items = self.map_items
        ns.feature_map_users = self.map_users
        ns.user_group_map = self.map_users_group
        ns.item_group_map = self.map_items_group
        ns.features_item = list({f for i in self.items for f in ns.feature_map_items[i]})
        ns.features_user = list({f for i in self.users for f in ns.feature_map_users[i]})
        ns.nfeatures_item = len(ns.features_item)
        ns.nfeatures_user = len(ns.features_user)
        ns.private_features_item = {p: f for p, f in enumerate(ns.features_item)}
        ns.public_features_item = {v: k for k, v in ns.private_features_item.items()}
        ns.private_features_user = {p: f for p, f in enumerate(ns.features_user)}
        ns.public_features_user = {v: k for k, v in ns.private_features_user.items()}
        return ns

    def load_attribute_file(self, attribute_file, separator='\t'):
        map_ = {}
        with open(attribute_file) as file:
            for line in file:
                line = line.split(separator)
                int_list = [int(i) for i in line[1:]]
                map_[int(line[0])] = list(set(int_list))
        return map_
