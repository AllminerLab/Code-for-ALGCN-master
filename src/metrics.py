import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numba

def evaluate_metrics(train_user2items,
                     valid_user2items,
                     query_indexes,
                     metrics,
                     user_embs=None,
                     item_embs=None,
                     valid_user2items_group=None,
                     parallel=False):
    logging.info("Evaluating metrics for {:d} users...".format(len(user_embs)))
    metric_callers = []
    max_topk = 0
    print(metrics)
    for metric in metrics:
        metric_callers.append(eval(metric))
        max_topk = max(max_topk, int(metric.split("k=")[-1].strip(")")))
        """
        try:
            metric_callers.append(eval(metric))
            max_topk = max(max_topk, int(metric.split("k=")[-1].strip(")")))
        except:
            raise NotImplementedError('metrics={} not implemented.'.format(metric))
        """


    #without parallel:
    results = []
    num_chunk = 100
    chunk_size = int(np.ceil(len(user_embs) / float(num_chunk)))
    with tqdm(total=num_chunk) as pbar:


        for idx in range(0, len(user_embs), chunk_size):
            chunk_user_embs = user_embs[idx: (idx + chunk_size), :]
            chunk_query_indexes = query_indexes[idx: (idx + chunk_size)]

            mask_matrix = np.zeros(shape=(chunk_user_embs.shape[0],item_embs.shape[0]))
            for i, query_index in enumerate(chunk_query_indexes):
                train_items = train_user2items[query_index]
                mask_matrix[i, train_items] = -np.inf  # remove clicked items in train data

            topk_items_chunk = evaluate_block(max_topk, chunk_user_embs=chunk_user_embs, item_embs=item_embs,
                                              sim_matrix=mask_matrix)
            true_items_chunk = [valid_user2items[query_index] for query_index in chunk_query_indexes]
            if valid_user2items_group != None:
                group_true_items_chunk = [valid_user2items_group[query_index] for query_index in chunk_query_indexes]
                result_chunk = [[fn(topk_items, group_true_items, true_items) for fn in [eval("Recall_group(k=20)")]] \
                                for topk_items, group_true_items, true_items in zip(topk_items_chunk, group_true_items_chunk, true_items_chunk)]
            else:
                result_chunk = [[fn(topk_items, true_items) for fn in metric_callers] \
                                 for topk_items, true_items in zip(topk_items_chunk, true_items_chunk)]


            results.extend(result_chunk)
            pbar.update(1)


    average_result = np.average(np.array(results), axis=0).tolist()
    return_dict = dict(zip(metrics, average_result))
    print('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in zip(metrics, average_result)))
    return return_dict


#@numba.jit(nopython=True)
def evaluate_block(max_topk, chunk_user_embs=None, item_embs=None, sim_matrix=None):
    sim_matrix += np.dot(chunk_user_embs, item_embs.T)


    item_indexes = np.argpartition(-sim_matrix, max_topk)[:, 0:max_topk]
    sim_matrix = sim_matrix[np.arange(item_indexes.shape[0])[:, None], item_indexes]
    sorted_idxs = np.argsort(-sim_matrix, axis=1)
    topk_items_chunk = item_indexes[np.arange(sorted_idxs.shape[0])[:, None], sorted_idxs]



    return topk_items_chunk




def draw_negative(train_user2items,
                     valid_user2items,
                     query_indexes,
                     metrics,
                     user_embs=None,
                     item_embs=None,
                     epoch=None,
                     ):
    #without parallel:
    true_neg_score = np.array([])
    false_neg_score = np.array([])
    num_chunk = 100
    chunk_size = int(np.ceil(len(user_embs) / float(num_chunk)))
    with tqdm(total=num_chunk) as pbar:
        for idx in range(0, len(user_embs), chunk_size):
            chunk_user_embs = user_embs[idx: (idx + chunk_size), :]
            chunk_query_indexes = query_indexes[idx: (idx + chunk_size)]
            true_neg_score_, false_neg_score_ = draw_block(chunk_query_indexes, train_user2items,
                                                           valid_user2items, chunk_user_embs=chunk_user_embs,
                                                           item_embs=item_embs)
            true_neg_score = np.append(true_neg_score, true_neg_score_)
            false_neg_score = np.append(false_neg_score, false_neg_score_)
            pbar.update(1)



    draw_hist(true_neg_score, false_neg_score, epoch)
    return

def draw_hist(true_neg, false_neg, epoch):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    df_false = pd.DataFrame({"score": false_neg})
    df_true = pd.DataFrame({"score": true_neg})

    sns.histplot(data=df_false, x="score", color="skyblue", label="false_neg", kde=True,)
    sns.histplot(data=df_true, x="score", color="red", label="true_neg", kde=True)
    plt.suptitle('epoch '+str(epoch))
    plt.show()

    df_false = df_false[df_false["score"] > 0.8]
    df_true = df_true[df_true["score"] > 0.8]
    sns.histplot(data=df_false, x="score", color="skyblue", label="false_neg", kde=True)
    sns.histplot(data=df_true, x="score", color="red", label="true_neg", kde=True)
    plt.suptitle('epoch '+str(epoch))
    plt.show()

def draw_block(chunk_query_indexes, train_user2items,
                   valid_user2items, chunk_user_embs=None, item_embs=None):
    sim_matrix = np.dot(chunk_user_embs, item_embs.T)
    sim_matrix = (sim_matrix.max() - sim_matrix) / (sim_matrix.max() - sim_matrix.min())

    true_neg_score = np.array([])
    false_neg_score = np.array([])
    id_items = np.arange(0, len(item_embs))
    for i, query_index in enumerate(chunk_query_indexes):
        train_item = train_user2items[query_index]
        test_item = valid_user2items[query_index]
        false_neg_score = np.append(false_neg_score, sim_matrix[i, test_item])

        sim_matrix[i, train_item] = -np.inf  # remove clicked items in train data
        sim_matrix[i, test_item] = -np.inf
        true_neg = np.setdiff1d(id_items, np.union1d(train_item, test_item))
        true_neg = np.random.choice(true_neg, 100)
        true_neg_score = np.append(true_neg_score, sim_matrix[i, true_neg])

    return true_neg_score, false_neg_score



class Recall(object):
    """Recall metric."""

    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(true_items) & set(topk_items)
        recall = len(hit_items) / (len(true_items) + 1e-12)
        return recall

class Recall_group(object):
    """Recall metric."""

    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, group_true_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(group_true_items) & set(topk_items)
        recall = len(hit_items) / (len(true_items) + 1e-12)
        return recall


class NormalizedRecall(object):
    """Recall metric normalized to max 1."""

    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(true_items) & set(topk_items)
        recall = len(hit_items) / min(self.topk, len(true_items) + 1e-12)
        return recall


class Precision(object):
    """Precision metric."""

    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(true_items) & set(topk_items)
        precision = len(hit_items) / (self.topk + 1e-12)
        return precision


class F1(object):
    def __init__(self, k=1):
        self.precision_k = Precision(k)
        self.recall_k = Recall(k)

    def __call__(self, topk_items, true_items):
        p = self.precision_k(topk_items, true_items)
        r = self.recall_k(topk_items, true_items)
        f1 = 2 * p * r / (p + r + 1e-12)
        return f1


class DCG(object):
    """ Calculate discounted cumulative gain
    """

    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        true_items = set(true_items)
        dcg = 0
        for i, item in enumerate(topk_items):
            if item in true_items:
                dcg += 1 / np.log(2 + i)
        return dcg


class NDCG(object):
    """Normalized discounted cumulative gain metric."""

    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        dcg_fn = DCG(k=self.topk)
        idcg = dcg_fn(true_items[:self.topk], true_items)
        dcg = dcg_fn(topk_items, true_items)
        return dcg / (idcg + 1e-12)


class MRR(object):
    """MRR metric"""

    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        true_items = set(true_items)
        mrr = 0
        for i, item in enumerate(topk_items):
            if item in true_items:
                mrr += 1 / (i + 1.0)
        return mrr


class HitRate(object):
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(true_items) & set(topk_items)
        hit_rate = 1 if len(hit_items) > 0 else 0
        return hit_rate


class MAP(object):
    """
    Calculate mean average precision.
    """

    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        true_items = set(true_items)
        pos = 0
        precision = 0
        for i, item in enumerate(topk_items):
            if item in true_items:
                pos += 1
                precision += pos / (i + 1.0)
        return precision / (pos + 1e-12)




