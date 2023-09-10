import pandas as pd
from collections import defaultdict

max_user = 0
max_item = 0

def gen_zero():
    return 0
item_freq = defaultdict(gen_zero)
user_freq = defaultdict(gen_zero)

user_history_dict = dict()
train_data = []
item_corpus = []
corpus_index = dict()
with open("train.txt", "r") as fid:
    for line in fid:
        splits = line.strip().split()
        user_id = splits[0]

        if int(user_id)>max_user:
            max_user = int(user_id)

        items = splits[1:]
        user_history_dict[user_id] = items
        user_freq[user_id] = len(items)
        for item in items:

            if item not in corpus_index:
                corpus_index[item] = len(corpus_index)
                item_corpus.append([corpus_index[item], item])


            item_freq[corpus_index[item]]+=1

            train_data.append([user_id, corpus_index[item], 1, user_freq[user_id]])
train = pd.DataFrame(train_data, columns=["user_id", "item_id", "label", "user_freq"])
print("train samples:", len(train))
train.to_csv("train.csv", index=False)



test_data = []
with open("test.txt", "r") as fid:
    for line in fid:
        splits = line.strip().split()
        user_id = splits[0]

        if int(user_id)>max_user:
            max_user = int(user_id)

        items = splits[1:]
        for item in items:

            if item not in corpus_index:
                corpus_index[item] = len(corpus_index)
                item_corpus.append([corpus_index[item], item])

            test_data.append([user_id, corpus_index[item], 1])
test = pd.DataFrame(test_data, columns=["user_id", "item_id", "label"])
print("test samples:", len(test))
test.to_csv("valid.csv", index=False)


corpus = pd.DataFrame(item_corpus, columns=["item_id", "_"])
del corpus["_"]
print("max item id:", len(corpus)-1)
print("max user id:", max_user)
corpus.to_csv("item_corpus.csv", index=False)



data_user_freq = pd.DataFrame({"user_id":user_freq.keys(), "user_freq":user_freq.values()})
data_user_freq.to_csv("user_freq.csv", index=False)
data_item_freq = pd.DataFrame({"item_id":item_freq.keys(), "item_freq":item_freq.values()})
data_item_freq.to_csv("item_freq.csv", index=False)
print(data_user_freq)
print(data_item_freq)

