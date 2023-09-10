'''
Created on October 1, 2020
@author: Tinglin Huang (huangtinglin@outlook.com)
'''
import random

import numpy as np
import torch
import torch.nn as nn
from src.models.base_model import BaseModel
import torch.nn.functional as F
from src.util.spmm import SpecialSpmm, CHUNK_SIZE_FOR_SPMM
import scipy.sparse as sp
import time

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, n_hops, n_users, adj_mat,
                 edge_dropout_rate=0.5, mess_dropout_rate=0.1, device=None):
        super(GraphConv, self).__init__()
        self.device = device
        self.adj_mat = adj_mat
        self.mat = self._init_graph(adj_mat)
        #self.one_coo = self.gen_one_mat(self.adj_mat)

        #self.mat = self.sp2tensor(interact_mat)

        self.n_users = n_users
        self.n_items = self.mat.shape[0]-n_users
        self.n_hops = n_hops
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate



        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _init_graph(self, adj_mat):
        return self._convert_sp_mat_to_sp_tensor(adj_mat).to(self.device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))





    def forward(self, user_embed, item_embed,
                mess_dropout=False, edge_dropout=False):
        # user_embed: [n_users, channel]
        # item_embed: [n_items, channel]
        # all_embed: [n_users+n_items, channel]
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        #if mess_dropout:
            #all_embed = self.dropout(all_embed, self.training)
        #    all_embed = F.dropout(all_embed, self.mess_dropout_rate, self.training)

        agg_embed = all_embed
        embs = [all_embed]

        interact_mat = self.mat
        for hop in range(self.n_hops):
            agg_embed = SpecialSpmm()(interact_mat, agg_embed)


            # agg_embed = self.dropout(agg_embed, self.training)
            embs.append(agg_embed)
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]

        return embs[:self.n_users, :], embs[self.n_users:, :],# new_u_embed, new_i_embed




class ALGCN(BaseModel):
    def __init__(self,
                 model_name=None,
                 device=None,
                 adj_mat=None,

                 embedding_dim=64,
                 train_gen=None,
                 max_user_id=None,
                 max_item_id=None,
                 n_layers=0,
                 num_negs=0,
                 emb_lambda=0,
                 is_pos_item_mixing=False,
                 is_pos_user_mixing=False,
                 is_hard_negative_sampling=False,
                 is_hard_neg_mixing=False,
                 mixing_weight=None,
                 **param):
        super(ALGCN, self).__init__(device=device, **param)
        self.model_name = model_name
        self.n_users = max_user_id+1
        self.n_items = max_item_id+1
        self.adj_mat = adj_mat
        self.is_pos_item_mixing = is_pos_item_mixing
        self.is_pos_user_mixing = is_pos_user_mixing
        self.is_hard_negative_sampling = is_hard_negative_sampling
        self.is_hard_neg_mixing = is_hard_neg_mixing
        self.mixing_weight = mixing_weight

        self.decay = emb_lambda
        self.emb_size = embedding_dim
        self.context_hops = n_layers
        print("n layer:", n_layers)
        self.mess_dropout = False
        self.mess_dropout_rate = 0
        self.edge_dropout = False
        self.edge_dropout_rate = 0
        self.pool = "mean"
        self.n_negs = num_negs
        self.K = 1

        self.pretrain = False
        if self.pretrain:
            print("loading pretrained embeddings")
            path_u = "./data/lightgcn_AmazonBooks_user_embs_4.pt"
            path_i = "./data/lightgcn_AmazonBooks_item_embs_4.pt"
            user_emb = torch.load(path_u)
            item_emb = torch.load(path_i)
            self.user_embed = torch.nn.Embedding.from_pretrained(user_emb[:,0,:], freeze = True)
            self.item_embed = torch.nn.Embedding.from_pretrained(item_emb[:,0,:], freeze = True)
        else:
            self.user_embed = torch.nn.Embedding(self.n_users, self.emb_size)
            self.item_embed = torch.nn.Embedding(self.n_items, self.emb_size)

            self.user_weight = torch.nn.Embedding(self.n_users, 1)
            self.item_weight = torch.nn.Embedding(self.n_items, 1)

            self._init_weight()
        #self._init_graph()
        #self.user_embed = nn.Parameter(self.user_embed)
        #self.item_embed = nn.Parameter(self.item_embed)

        self.user_embedding = None
        self.item_embedding = None

        self.gcn = self._init_model()
        self.to(device=self.device)


    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        initializer(self.user_embed.weight)
        initializer(self.item_embed.weight)

        # [n_users+n_items, n_users+n_items]
    def _init_graph(self):
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         adj_mat=self.adj_mat,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate,
                         device=self.device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, user, pos_item, neg_item):
        if user.ndim==1:
            user = user.unsqueeze(1)
        if pos_item.ndim==1:
            pos_item = pos_item.unsqueeze(1)
        if neg_item.ndim==1:
            neg_item = neg_item.unsqueeze(1)
        #user, pos_item, neg_item = inputs[:3]
        # user: [batch size, user_num]
        # pos_item: [batch size, pos_num]
        # pos_item: [batch size, neg_num]

        # user_all_embeddings: [n_users, channel]
        # item_all_embeddings: [n_items, channel]
        user_gcn_embs, item_gcn_embs = self.gcn(self.user_embed.weight,
                                              self.item_embed.weight,
                                              edge_dropout=False,
                                              mess_dropout=0.2)



        self.user_gcn_embs = user_gcn_embs
        self.item_gcn_embs = item_gcn_embs
        self.user_embedding, self.item_embedding = self.pooling(user_gcn_embs), self.pooling(item_gcn_embs)
        #self.user_embedding, self.item_embedding = user_gcn_embs, item_gcn_embs

        #user_gcn_emb = user_gcn_embs[user]
        #pos_gcn_emb = item_gcn_embs[pos_item]
        #neg_gcn_emb = item_gcn_embs[neg_item]

        #self.batch_embeds = [user_gcn_emb[:,:,0,:], pos_gcn_emb[:,:,0,:], neg_gcn_emb[:,:,0,:]]
        self.batch_embeds = [self.user_embed(user), self.item_embed(pos_item), self.item_embed(neg_item)]
        weight = None

        u_e = self.user_tower(user)
        pos_e = self.item_tower(pos_item)
        neg_e = self.item_tower(neg_item)


        return {"embeds": self.batch_embeds, "weight": weight,
                "user_vec": u_e, "pos_item_vec": pos_e, "neg_item_vec": neg_e}



    def user_tower(self, input):
        user_vec = self.user_embedding[input]
        return user_vec

    def item_tower(self, input):
        item_vec = self.item_embedding[input]
        return item_vec

    def user_towers(self, input):
        return self.user_gcn_embs[input]

    def item_towers(self, input):
        return self.item_gcn_embs[input]

    def get_user_item_embedding(self):
        return self.user_embedding, self.item_embedding

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=-2)
        elif self.pool == 'sum':
            return embeddings.sum(dim=-2)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)


    def preprocess(self):
        print("preprocess...")
        user_gcn_embs, item_gcn_embs,  = self.gcn(self.user_embed.weight,
                                                    self.item_embed.weight,
                                                    edge_dropout=False,
                                                    mess_dropout=self.mess_dropout)
        self.user_embedding, self.item_embedding = self.pooling(user_gcn_embs), self.pooling(item_gcn_embs)
        #self.user_embedding, self.item_embedding = user_gcn_embs, item_gcn_embs

    def save_gcn_embeds(self, name="None"):

        torch.save(self.user_gcn_embs.detach().cpu(), "./data/"+name+"_user_embs_" + str(self.context_hops+1) + ".pt")
        torch.save(self.item_gcn_embs.detach().cpu(), "./data/"+name+"_item_embs_" + str(self.context_hops+1) + ".pt")
        print("saved gcn embeddings")

    def save_embed(self, k=0):
        torch.save(self.user_embed.weight.detach().cpu(), "./data/Yelp18_user_emb.pt")
        torch.save(self.item_embed.weight.detach().cpu(), "./data/Yelp18_item_emb.pt")
        print("saved embeddings")


