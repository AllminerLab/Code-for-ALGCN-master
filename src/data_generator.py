from torch.utils.data import Dataset, DataLoader
import pandas as pd
import logging
import numpy as np
from collections import defaultdict
import scipy.sparse as sp
import random
from tqdm import tqdm
import scipy

def get_user2items_dict(data_dict):
    max_user = 0
    max_item = 0
    user2items_dict = defaultdict(list)
    for u_id, i_id in zip(data_dict["user_id"],data_dict["item_id"]):
        user2items_dict[u_id].append(i_id)
        if u_id>max_user:
            max_user = u_id
        if i_id>max_item:
            max_item = i_id
    print("max_user: ", max_user)
    print("max_item: ", max_item)
    return user2items_dict

def get_user2items_group_dict(data_dict, num_group=10):
    user2items_group_dict = [defaultdict(list) for i in range(num_group)]
    for u_id, i_id, g_id in zip(data_dict["user_id"],data_dict["item_id"],data_dict["group_id"]):
        user2items_group_dict[int(g_id)][u_id].append(i_id)
    return user2items_group_dict

def get_item2users_dict(data_dict):
    item2users_dict = defaultdict(list)
    for u_id, i_id in zip(data_dict["user_id"],data_dict["item_id"]):
        item2users_dict[i_id].append(u_id)
    return item2users_dict

def load_data(data_path, sep=",", usecols=None):
    logging.info("Reading file: " + data_path)
    ddf = pd.read_csv(data_path, sep=sep, usecols=usecols)
    data_dict = dict()
    for feature in usecols:
        data_dict[feature] = ddf.loc[:, feature].values
    num_samples = len(list(data_dict.values())[0])
    return data_dict, num_samples



class TrainDataset(Dataset):
    def __init__(self,data_path):
        self.data_dict, self.num_samples = load_data(data_path, usecols=["user_id","item_id","label"])
        self.labels = self.data_dict["label"]
        self.users = self.data_dict["user_id"]
        self.items = self.data_dict["item_id"]
        self.pos_users = self.data_dict["user_id"]
        self.pos_items = self.data_dict["item_id"]
        self.neg_items = None
        self.all_items = None
        self.num_items = len(set(self.data_dict["item_id"]))
        self.num_users = len(set(self.data_dict["user_id"]))
        self.max_user = np.max(self.users)
        self.max_item = np.max(self.items)



    def __getitem__(self, index):
        return self.pos_users[index], self.pos_items[index], self.neg_items[index]

    def __len__(self):
        return self.num_samples

class TrainGenerator(DataLoader):
    def __init__(self, data_path, batch_size=32, shuffle=True,
                 user_sample_rate=None, item_sample_rate=None, item_freq=None, user_freq=None,
                 num_neg=None,num_pos=None, num_pos_user=None, tau=None, **kwargs):

        self.dataset = TrainDataset(data_path)
        super(TrainGenerator, self).__init__(dataset=self.dataset, batch_size=batch_size,
                                             shuffle=shuffle)
        self.set_item_ids = set(self.dataset.items)
        self.user2items_dict = get_user2items_dict(self.dataset.data_dict)
        self.item2users_dict = get_item2users_dict(self.dataset.data_dict)
        self.num_samples = self.dataset.num_samples
        self.num_batches = int(np.ceil(self.num_samples * 1.0 / batch_size))

        self.num_items = self.dataset.num_items
        self.num_users = self.dataset.num_users
        self.max_user = self.dataset.max_user
        self.max_item = self.dataset.max_item
        self.item_sample_rate = item_sample_rate
        self.user_sample_rate = user_sample_rate
        self.item_freq = item_freq
        self.user_freq = user_freq

        self.num_pos = num_pos
        self.num_neg = num_neg
        self.num_pos_user = num_pos_user
        self.tau = tau
        print("self.pos_num: ",self.num_pos)
        print("self.num_negs: ", self.num_neg)

        if self.item_sample_rate:
            self.get_user2items_sample_rate_dict()
        self.adj_mat = self.build_sparse_graph()


    def get_user2items_sample_rate_dict(self):
        self.user2items_sam_rate_dict = defaultdict(list)
        for u in self.user2items_dict.keys():
            items = self.user2items_dict[u]
            for i in items:
                self.user2items_sam_rate_dict[u].append(self.item_sample_rate[i])
            if len(items)>0:
                S = sum(self.user2items_sam_rate_dict[u])
                self.user2items_sam_rate_dict[u] = [i/S for i in self.user2items_sam_rate_dict[u]]


    def __iter__(self):
        if self.num_pos>1:
            self.rns_item_sampling()
        if self.num_pos_user>1:
            self.rns_user_sampling()
        self.negative_sampling()
        iter = super(TrainGenerator, self).__iter__()
        while True:
            try:
                yield next(iter) # a batch iterator
            except StopIteration:
                break

    def __len__(self):
        return self.num_batches

    def negative_sampling(self):
        if self.num_neg > 0:
            neg_item_indexes = np.random.choice(self.num_items,
                                                size=(self.num_samples, self.num_neg),
                                                replace=True)
            self.dataset.neg_items = neg_item_indexes
            #self.dataset.all_items = np.hstack([self.dataset.items.reshape(-1, 1),neg_item_indexes])
        tqdm.write("negative sampling.")


    def uniform_resampling(self,):
        if self.num_pos!=-1:
            # sample users:
            train_users_set = list(set(self.dataset.data_dict["user_id"]))
            if self.user_sample_rate:
                users_index = np.random.choice(len(train_users_set), self.dataset.num_samples, p=self.user_sample_rate)
            else:
                users_index = np.random.choice(len(train_users_set), self.dataset.num_samples)
            self.dataset.pos_users = np.array(train_users_set)[users_index]

            # sample items:
            items = []
            if self.item_sample_rate:
                for k,u in enumerate(self.dataset.pos_users):
                    pos_items = self.user2items_dict[u]
                    pos_items_rate = self.user2items_sam_rate_dict[u]
                    items.append(np.random.choice(pos_items,size=(self.num_pos),p=pos_items_rate))
            else:
                for k,u in enumerate(self.dataset.pos_users):
                    pos_items = self.user2items_dict[u]
                    items.append(np.random.choice(pos_items,size=(self.num_pos)))
            self.dataset.pos_items = np.array(items)

        if self.num_pos_user>1:
            users = []
            if self.num_pos < 2:
                for u,i in zip(self.dataset.pos_users, self.dataset.pos_items):
                    pos_users = self.item2users_dict[int(i)]
                    users.append([u, np.random.choice(pos_users).tolist()])
            else:
                for u,i in zip(self.dataset.pos_users, self.dataset.pos_items):
                    pos_users = self.item2users_dict[i[0]]
                    users.append([u, int(np.random.choice(pos_users))])
            self.dataset.pos_users = np.array(users)

    def rns_item_sampling(self):
        items = []
        for u,i in zip(self.dataset.users, self.dataset.items):
            pos_items = self.user2items_dict[u]
            #items.append([i, int(np.random.choice(pos_items))])
            items.append(np.append([i], np.random.choice(pos_items, self.num_pos-1)))
        self.dataset.pos_items = np.array(items)
        tqdm.write("rns item sampling.")



    def rns_user_sampling(self):
        users = []
        for u,i in zip(self.dataset.users, self.dataset.items):
            pos_users = self.item2users_dict[i]
            #users.append([u, int(np.random.choice(pos_users))])
            users.append(np.append([u], np.random.choice(pos_users, self.num_pos_user-1)))
        tqdm.write("rns user sampling.")
        self.dataset.pos_users = np.array(users)


    def update(self, delta = 1):
        if self.user_freq:
            print("delta: ", delta)
            def f(x, delta):
                if x > 0:
                    x = x ** delta
                return x

            self.user_sample_rate = [f(i,delta) for i in self.user_freq]
            sum_freq = sum(self.user_sample_rate)
            self.user_sample_rate = [i/sum_freq for i in self.user_sample_rate]

    def build_sparse_graph(self):
        a = np.array([self.dataset.users])
        b = np.array([self.dataset.items])

        data_cf = np.append(a.T, b.T, axis=1)
        n_users = self.max_user+1
        n_items = self.max_item+1

        def _bi_norm_lap(adj):
            rowsum = np.array(adj.sum(1))

            d_inv_sqrt = np.power(rowsum, -1).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)


            tau = self.tau
            print("tau:", tau)
            d_inv_log = np.power(rowsum, tau).flatten()
            d_inv_log = sp.diags(d_inv_log)


            bi_lap = d_mat_inv_sqrt.dot(adj)
            bi_lap = d_inv_log.dot(bi_lap)
            #bi_lap = bi_lap.dot(d_inv_log)

            return bi_lap.tocoo()


        cf = data_cf.copy()
        cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
        cf_ = cf.copy()
        cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

        cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]

        vals = [1.] * len(cf_)
        mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(n_users + n_items, n_users + n_items))
        logging.info("create adj_mat successfully.")
        return _bi_norm_lap(mat)



class TestDataset(Dataset):
    def __init__(self, data_path, col_name, usecols):
        self.col_name = col_name
        self.data_dict, self.num_samples = load_data(data_path, usecols=usecols)

    def __getitem__(self, index):
        return self.data_dict[self.col_name][index]

    def __len__(self):
        return self.num_samples




class TestGenerator(object):
    def __init__(self, data_path, item_corpus_path, batch_size=256, shuffle=False
                 ):
        user_dataset = TestDataset(data_path, col_name="user_id", usecols=["user_id","item_id","label"])
        self.user2items_dict = get_user2items_dict(user_dataset.data_dict)

        # group test dataset
        #user_group_dataset = TestDataset(data_path, col_name="user_id", usecols=["user_id", "item_id", "label", "group_id"])
        #self.user2items_group_dict = get_user2items_group_dict(user_group_dataset.data_dict)
        self.user2items_group_dict = None

        # pick users of unique query_index
        self.test_users, _ = np.unique(user_dataset.data_dict["user_id"],
                                                    return_index=True)
        user_dataset.num_samples = len(self.test_users)
        self.num_samples = len(user_dataset)
        user_dataset.data_dict["user_id"] = self.test_users

        item_dataset = TestDataset(item_corpus_path, "item_id", usecols=["item_id"])
        self.user_loader = DataLoader(dataset=user_dataset, batch_size=batch_size,
                                      shuffle=shuffle, num_workers=1)
        self.item_loader = DataLoader(dataset=item_dataset, batch_size=batch_size,
                                      shuffle=shuffle, num_workers=1)

def data_generator(train_data_path, valid_data_path, item_corpus_path, test_data_path=None, batch_size=256, num_negs=None, **kwargs):
    test_gen = None
    train_gen = TrainGenerator(data_path=train_data_path, batch_size= batch_size, num_negs=num_negs, **kwargs)
    valid_gen = TestGenerator(data_path=valid_data_path, item_corpus_path=item_corpus_path, batch_size= batch_size)
    if test_data_path:
        test_gen = TestGenerator(data_path=test_data_path, item_corpus_path=item_corpus_path, batch_size=batch_size)

    return train_gen, valid_gen, test_gen

def train_data_generator(train_data_path, item_sample_rate=None, batch_size=None, num_negs=None, **kwargs):
    print("num_negs: ", num_negs)
    train_gen = TrainGenerator(data_path=train_data_path, batch_size= batch_size, item_sample_rate=item_sample_rate, num_negs=num_negs, **kwargs)

    return train_gen
