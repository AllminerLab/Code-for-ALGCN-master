import torch
import torch.nn as nn
import torch.nn.functional as F



class base_sampler(nn.Module):
    """
    Uniform sampler
    """
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super(base_sampler, self).__init__()
        self.num_items = num_items
        self.num_neg = num_neg
        self.device = device

    def update_pool(self, model, **kwargs):
        pass

    def forward(self, user_id, **kwargs):
        batch_size = user_id.shape[0]
        return torch.randint(0, self.num_items, size=(batch_size, self.num_neg), device=self.device), -torch.log(
            self.num_items * torch.ones(batch_size, self.num_neg, device=self.device))


class two_pass(base_sampler):
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super().__init__(num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs)
        self.num_users = num_users
        self.sample_size = sample_size  # importance sampling
        self.pool_size = pool_size  # resample
        self.pool = torch.zeros(num_users, pool_size, device=device, dtype=torch.long)

    def update_pool(self, model, batch_size=2048, cover_flag=False, **kwargs):
        print("update pool.")
        num_batch = (self.num_users // batch_size) + 1
        for ii in range(num_batch):
            start_idx = ii * batch_size
            end_idx = min(start_idx + batch_size, self.num_users)
            user_batch = torch.arange(start_idx, end_idx, device=self.device)

            neg_items, neg_q = self.sample_Q(user_batch)
            tmp_pool, tmp_score = self.re_sample(user_batch, model, neg_items, neg_q)
            self.__update_pool__(user_batch, tmp_pool, tmp_score, cover_flag=cover_flag)

    def sample_Q(self, user_batch):
        batch_size = user_batch.shape[0]
        return torch.randint(0, self.num_items, size=(batch_size, self.sample_size), device=self.device), -torch.log(
            self.num_items * torch.ones(batch_size, self.sample_size, device=self.device))

    def re_sample(self, user_batch, model, neg_items, log_neg_q):
        ratings = model.inference(user_batch.repeat(self.sample_size, 1).T, neg_items)
        pred = ratings - log_neg_q
        sample_weight = F.softmax(pred, dim=-1)
        idices = torch.multinomial(sample_weight, self.pool_size, replacement=True)
        return torch.gather(neg_items, 1, idices), torch.gather(sample_weight, 1, idices)

    def __update_pool__(self, user_batch, tmp_pool, tmp_score, cover_flag):
        if cover_flag is True:
            self.pool[user_batch] = tmp_pool
            return

        idx = self.pool[user_batch].sum(-1) < 1

        user_init = user_batch[idx]
        self.pool[user_init] = tmp_pool[idx]

        user_update = user_batch[~idx]
        num_user_update = user_update.shape[0]
        idx_k = torch.randint(0, 2 * self.pool_size, size=(num_user_update, self.pool_size), device=self.device)
        candidate = torch.cat([self.pool[user_update], tmp_pool[~idx]], dim=1)
        self.pool[user_update] = torch.gather(candidate, 1, idx_k)
        return

    # @profile
    def forward(self, user_id, **kwargs):
        batch_size = user_id.shape[0]
        candidates = self.pool[user_id]
        idx_k = torch.randint(0, self.pool_size, size=(batch_size, self.num_neg), device=self.device)
        return torch.gather(candidates, 1, idx_k), -torch.log(
            self.pool_size * torch.ones(batch_size, self.num_neg, device=self.device))


class two_pass_weight(two_pass):
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super(two_pass_weight, self).__init__(num_users, num_items, sample_size, pool_size, num_neg, device)
        self.pool_weight = torch.zeros(num_users, pool_size, device=device)

    def __update_pool__(self, user_batch, tmp_pool, tmp_score, cover_flag=False):
        if cover_flag is True:
            self.pool[user_batch] = tmp_pool
            self.pool_weight[user_batch] = tmp_score.detach()
            return

        idx = self.pool[user_batch].sum(-1) < 1

        user_init = user_batch[idx]
        if len(user_init) > 0:
            self.pool[user_init] = tmp_pool[idx]
            self.pool_weight[user_init] = tmp_score[idx]

        user_update = user_batch[~idx]
        num_user_update = user_update.shape[0]
        if num_user_update > 0:
            idx_k = torch.randint(0, 2 * self.pool_size, size=(num_user_update, self.pool_size), device=self.device)
            candidate = torch.cat([self.pool[user_update], tmp_pool[~idx]], dim=1)
            candidate_weight = torch.cat([self.pool_weight[user_update], tmp_score[~idx]], dim=1)
            self.pool[user_update] = torch.gather(candidate, 1, idx_k)
            self.pool_weight[user_update] = torch.gather(candidate_weight, 1, idx_k).detach()

    def forward(self, user_id, **kwargs):
        batch_size = user_id.shape[0]
        candidates = self.pool[user_id]
        candidates_weight = self.pool_weight[user_id]
        idx_k = torch.randint(0, self.pool_size, size=(batch_size, self.num_neg), device=self.device)
        return torch.gather(candidates, 1, idx_k), torch.log(torch.gather(candidates_weight, 1, idx_k))




class tapast(base_sampler):
    """
    The dynamic sampler
    """
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super().__init__(num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs)
        self.pool_size = pool_size
        self.num_users = num_users

    def forward(self, user_id, model=None, **kwargs):
        batch_size = user_id.shape[0]
        pool = torch.randint(0, self.num_items, size=(batch_size, self.num_neg, self.pool_size), device=self.device)

        rats = model.inference(user_id.repeat(1, 1, 1).T, pool)

        #weight_temp = 0.1
        #weight = F.softmax(rats.reshape(-1,rats.shape[-1])/ weight_temp, dim=-1)
        #r_idx = torch.multinomial(weight, 1, replacement=True)    #(batch_size*num_neg, 1)
        #r_idx = r_idx.reshape(batch_size, -1, 1)

        r_v, r_idx = rats.max(dim=-1)
        r_idx = r_idx.unsqueeze(-1)
        return torch.gather(pool, 2, r_idx).squeeze(-1), None #torch.exp(r_v)




class gain_sampler(base_sampler):
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super().__init__(num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs)
        self.num_users = num_users
        self.sample_size = sample_size  # importance sampling
        self.pool_size = pool_size  # resample

        # each user store num_neg negative items
        self.G = torch.zeros(num_users, num_items, device=device, dtype=torch.float)

        self.pool_id = torch.zeros(num_users, num_neg, pool_size, device=device, dtype=torch.long)
        self.pool_score = torch.zeros(num_users, num_neg, pool_size, device=device, dtype=torch.float)

    def forward(self, user_id, model=None, **kwargs):
        batch_size = user_id.shape[0]

        idx_item = torch.randint(0, self.num_items, size=(batch_size, self.pool_size), device=self.device)
        score_item = self.G[user_id, :].gather(dim=1, index=idx_item)   # (batch_size, num_neg, pool_size)

        rats = model.inference(user_id.repeat(1, 1).T, idx_item)    # (batch_size, pool_size)

        #r_v, r_idx = ((score_item - rats)/rats).max(dim=-1)         # (batch_size, num_neg)
        r_v, r_idx = (score_item - rats).topk(dim=-1, k=self.num_neg)  # (batch_size, num_neg)



        # update:
        self.G[user_id, :] = self.G[user_id, :].scatter(1, r_idx, r_v)
        #self.G[user_id, :][:,idx_item] = rats.squeeze(1)

        return torch.gather(idx_item.repeat(batch_size, 1, 1), 2, r_idx.unsqueeze(-1)).squeeze(-1), torch.exp(r_v)
    """
    def forward(self, user_id, model=None, **kwargs):
        batch_size = user_id.shape[0]
        idx_item = self.pool_id[user_id]
        score_item =  self.pool_score[user_id]

        #print((user_id.repeat(1, 1, 1).T).shape)
        #print(idx_item.shape)
        rats = model.inference(user_id.repeat(1, 1, 1).T, idx_item)
        #r_v, r_idx = (F.softmax(score_item - rats, dim=-1)*F.softmax(rats, dim=-1)).max(dim=-1)
        r_v, r_idx = (torch.sigmoid(score_item - rats) * torch.sigmoid(rats)).max(dim=-1)
        #r_v, r_idx = ((torch.sigmoid(score_item - rats))/(torch.sigmoid(rats)+1.0e-6)).max(dim=-1)

        # update:
        self.pool_id[user_id] = torch.randint(0, self.num_items, size=(batch_size, self.num_neg, self.pool_size), device=self.device)
        self.pool_score[user_id] = model.inference(user_id.repeat(1, 1, 1).T, self.pool_id[user_id])

        return torch.gather(idx_item, 2, r_idx.unsqueeze(-1)).squeeze(-1), torch.exp(r_v)
    """
