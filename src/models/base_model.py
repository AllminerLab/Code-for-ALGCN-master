import torch.nn as nn
import logging
import torch
from scipy.cluster.vq import kmeans2
import numpy as np
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, device):
        super(BaseModel, self).__init__()
        self.device = device

    @torch.no_grad()
    def inference(self, users, items):
        user_vec = self.user_tower(users)
        item_vec = self.item_tower(items)
        return (user_vec * item_vec).sum(-1)

    def load_embedding(self, path):
        user_embedding = torch.load(path + "user_embedding.pt")
        item_embedding = torch.load(path + "item_embedding.pt")
        self.user_embedding = torch.nn.Embedding.from_pretrained(user_embedding, freeze=False)
        self.item_embedding = torch.nn.Embedding.from_pretrained(item_embedding, freeze=False)
        logging.info("load embedding weight from {} successfully".format(path))

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
            logging.info("successfully init Linear.")
        elif type(m) == nn.Embedding and not self.is_pretrained:
            nn.init.xavier_uniform_(m.weight)
            print("successfully init embedding.")

    @torch.no_grad()
    def hard_negative_sample(self, user_id, neg_item_id):
        """
            cand ids -> tower -> embs -> selected id
            return neg_item_id
        """

        user_vec = self.user_tower(user_id)  # [batch_size, embed_dim]
        # [batch_size, num_neg+1, embed_dim]
        neg_item_vec = self.item_tower(neg_item_id)
        if neg_item_vec.dim()==2:
            neg_item_vec = neg_item_vec.unsqueeze(1)

        neg_y_pred = torch.bmm(neg_item_vec, user_vec.unsqueeze(-1)).squeeze(-1)  # [batch_size, num_neg+1]
        neg_index = torch.max(neg_y_pred, dim=1)[1].view(-1, 1)  # [batch_size, 1]

        select_neg_item_id = torch.gather(neg_item_id, 1, neg_index)
        return select_neg_item_id

    def mixing(self, emb1, emb2, a=0.1):
        # (batch_size, embed_dim)
        Beta = torch.distributions.beta.Beta(torch.tensor([a]), torch.tensor([a]))
        seed = Beta.sample((emb1.shape[0],1)).squeeze(1).to(self.device)
        seed = torch.max(seed, 1.0-seed)
        mixing_vec = seed * emb1 + (1 - seed) * emb2  # mixing

        return mixing_vec.unsqueeze(1), seed

    @torch.no_grad()
    def user_cluster(self, k=4, is_normalize=False):
        user_embedding,_ = self.get_user_item_embedding()
        if is_normalize:
            user_emb = np.array(F.normalize(user_embedding, dim=-1).cpu().numpy())
        else:
            user_emb = np.array((user_embedding).cpu().numpy())
        user_centroids, user_labels = kmeans2(user_emb, k)
        weight = torch.FloatTensor(user_labels).unsqueeze(1)
        self.user_labels = torch.nn.Embedding.from_pretrained(weight, freeze=True).to(self.device)
        user_centroids = torch.FloatTensor(user_centroids).squeeze()
        self.user_centroids = torch.nn.Embedding.from_pretrained(user_centroids, freeze=True).to(self.device)

    @torch.no_grad()
    def item_cluster(self, k=4, is_normalize=False):
        '''
        called per epoch
        '''
        _, item_embedding = self.get_user_item_embedding()
        if is_normalize:
            item_emb = np.array(F.normalize(item_embedding, dim=-1).cpu().numpy())
        else:
            item_emb = np.array((item_embedding).cpu().numpy())
        item_centroids, item_labels = kmeans2(item_emb, k)
        weight = torch.FloatTensor(item_labels).unsqueeze(1)
        self.item_labels = torch.nn.Embedding.from_pretrained(weight, freeze=True).to(self.device)
        item_centroids = torch.FloatTensor(item_centroids).squeeze()
        self.item_centroids = torch.nn.Embedding.from_pretrained(item_centroids, freeze=True).to(self.device)

    @torch.no_grad()
    def get_user_centroids(self, id):
        label = self.user_labels(id).squeeze(-1)
        user_centro = self.user_centroids(label.long())
        #if user_centro.dim()!=3:
        #    user_centro = user_centro.unsqueeze(1)
        #print(user_centro.shape)
        return user_centro

    @torch.no_grad()
    def get_item_centroids(self, id):
        label = self.item_labels(id).squeeze(-1)
        item_centro = self.item_centroids(label.long())
        #if item_centro.dim()!=3:
        #    item_centro = item_centro.unsqueeze(1)
        #print(item_centro.shape)
        return item_centro

    @torch.no_grad()
    def hard_negative_sample_with_Kmeans(self, user_id, pos_item_id, neg_item_ids):
        batch_size, neg_num = neg_item_ids.shape
        b_pos_item_label = self.item_labels(pos_item_id.unsqueeze(1))
        neg_item_label = self.item_labels(neg_item_ids)
        neg_item_mask = b_pos_item_label.repeat(1, neg_num, 1).eq(neg_item_label)  # True should be eliminated.
        neg_item_mask.squeeze_(-1)

        user_vec = self.user_tower(user_id)  # [batch_size, embed_dim]
        # [batch_size, num_neg+1, embed_dim]
        neg_item_vec = self.item_tower(neg_item_ids)
        if neg_item_vec.dim() == 2:
            neg_item_vec = neg_item_vec.unsqueeze(1)

        neg_y_pred = torch.bmm(neg_item_vec, user_vec.unsqueeze(-1)).squeeze(-1)  # [batch_size, num_neg+1]
        neg_y_pred = torch.where(neg_item_mask, torch.tensor(1.0, dtype=neg_y_pred.dtype).to(neg_y_pred.device),
                                 neg_y_pred/1e4)

        neg_index = torch.multinomial(F.softmax(neg_y_pred, dim=-1), 1, replacement=True).detach()
        #neg_index = torch.max(neg_y_pred, dim=1)[1].detach().view(-1, 1)  # [batch_size, 1]

        select_neg_item_id = torch.gather(neg_item_ids, 1, neg_index)

        return select_neg_item_id

    @torch.no_grad()
    def user_item_cluster(self, k=4):
        '''
        called per epoch
        '''
        embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        # user_item_emb = np.array(F.normalize(embedding, dim=-1).cpu().numpy())
        user_item_emb = np.array(embedding.cpu().numpy())
        self.centroids, labels = kmeans2(user_item_emb, k)
        num_user = self.user_embedding.weight.shape[0]
        user_labels = torch.FloatTensor(labels[ :num_user]).unsqueeze(1)
        self.user_labels = torch.nn.Embedding.from_pretrained(user_labels, freeze=True).to(self.device)
        item_labels = torch.FloatTensor(labels[num_user:]).unsqueeze(1)
        self.item_labels = torch.nn.Embedding.from_pretrained(item_labels, freeze=True).to(self.device)

    @torch.no_grad()
    def hard_negative_sample_with_ui_Kmeans(self, user_id, pos_item_id, neg_item_ids):
        '''
        '''
        batch_size, neg_num = neg_item_ids.shape
        #b_pos_item_label = self.item_labels(pos_item_id.unsqueeze(1))
        user_label = self.user_labels(user_id.unsqueeze(1))
        neg_item_label = self.item_labels(neg_item_ids)
        #neg_item_mask = b_pos_item_label.repeat(1, neg_num, 1).eq(neg_item_label)  # True should be eliminated.
        neg_item_mask = user_label.repeat(1, neg_num, 1).eq(neg_item_label)
        neg_item_mask.squeeze_(-1)

        user_vec = self.user_tower(user_id)  # [batch_size, embed_dim]
        # [batch_size, num_neg+1, embed_dim]
        neg_item_vec = self.item_tower(neg_item_ids)
        if neg_item_vec.dim() == 2:
            neg_item_vec = neg_item_vec.unsqueeze(1)

        neg_y_pred = torch.bmm(neg_item_vec, user_vec.unsqueeze(-1)).squeeze(-1)  # [batch_size, num_neg+1]
        neg_y_pred = torch.where(neg_item_mask, torch.tensor(-1e5, dtype=neg_y_pred.dtype).to(neg_y_pred.device),
                                 neg_y_pred)
        #neg_index = torch.max(neg_y_pred, dim=1)[1].detach().view(-1, 1)  # [batch_size, 1]
        neg_index = torch.multinomial(F.softmax(neg_y_pred, dim=-1), 1, replacement=True).detach()
        select_neg_item_id = torch.gather(neg_item_ids, 1, neg_index)

        return select_neg_item_id

    def preprocess(self):
        pass

    def get_user_item_embedding(self):
        pass