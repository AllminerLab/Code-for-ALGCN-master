import logging

from torch import nn
import torch.nn.functional as F
import torch
from src.models.base_model import BaseModel

class MF(BaseModel):
    def __init__(self,
                 embedding_dim=64,
                 max_user_id=None,
                 max_item_id=None,
                 embedding_dropout=0.1,
                 is_hard_negative_sampling=False,
                 emb_lambda=0,
                 is_pretrained=False,
                 device=None,
                 is_proto=False,
                 **param
                 ):
        self.num_users = max_user_id + 1
        self.num_items = max_item_id + 1
        super().__init__(device)

        self.model_name = "mf"
        self.embedding_dim = embedding_dim
        self.is_hard_negative_sampling = is_hard_negative_sampling
        self.is_pretrained = is_pretrained

        self.decay = emb_lambda
        #self.user_embedding = nn.Embedding(self.num_users, embedding_dim,  )
        #self.item_embedding = nn.Embedding(self.num_items, embedding_dim,  )
        self.user_embedding = nn.Embedding(self.num_users, embedding_dim, sparse=True)
        self.item_embedding = nn.Embedding(self.num_items, embedding_dim, sparse=True)

        if self.is_pretrained:
            self.load_embedding(self.save_model_path)
        self.dropout = nn.Dropout(embedding_dropout)

        # init weights:
        self.apply(self.init_weights)

        self.batch_embeds = None




    def forward(self, user_id, pos_item_id, neg_item_id):
        user_vec = self.user_tower(user_id)  # [batch_size, embed_dim]
        pos_item_vec = self.item_tower(pos_item_id)  # [batch_size, num_neg+1, embed_dim]
        neg_item_vec = self.item_tower(neg_item_id)

        if pos_item_vec.dim()==2:
            pos_item_vec = pos_item_vec.unsqueeze(1)
        if neg_item_vec.dim()==2:
            neg_item_vec = neg_item_vec.unsqueeze(1)
        if user_vec.dim()==2:
            user_vec = user_vec.unsqueeze(1)

        # mixing
        #mix_item_vec,weight = self.mixing(pos_item_vec[:,0,:], neg_item_vec[:,0,:])
        #pos_item_vec = mix_item_vec

        pos_y_pred = torch.bmm(pos_item_vec, user_vec.permute(0,2,1)).squeeze(-1)  # [batch_size, num_pos+1]
        neg_y_pred = torch.bmm(neg_item_vec, user_vec.permute(0,2,1)).squeeze(-1)  # [batch_size, num_neg+1]


        self.batch_embeds = [user_vec, pos_item_vec, neg_item_vec]
        weight = None
        return {"pos_y_pred": pos_y_pred,"neg_y_pred":neg_y_pred, "embeds": self.batch_embeds, "weight":weight,
                "user_vec": user_vec, "pos_item_vec": pos_item_vec, "neg_item_vec": neg_item_vec}



    def user_tower(self, input):
        user_vec = self.user_embedding(input)
        return user_vec

    def item_tower(self, input):
        item_vec = self.item_embedding(input)
        return item_vec

    def get_user_item_embedding(self):
        return self.user_embedding.weight, self.item_embedding.weight
    """
    @torch.no_grad()
    def inference(self, users, items):
        user_vec = self.user_embedding(users)
        item_vec = self.item_embedding(items)
        return (user_vec * item_vec).sum(-1)
    """


