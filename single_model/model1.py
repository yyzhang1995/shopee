import torch
import re
from common.load_data import load_data
from common.funcs import cos_similarity
import random
import numpy as np
'''
针对商品的文字进行处理
'''

def title_clean(title):
    """
    对文本进行清晰，去除掉标点符号、数字
    :param title:
    :return:
    """
    def clean(s):
        res = re.sub(r"[^a-z]", " ", s)
        res = re.sub(r" +", " ", res).strip()
        return res

    for i in range(len(title)):
        title[i] = clean(title[i])
    return title


def generate_matrix(title, min_word_len=2):
    """
    生成单词向量
    :param title:
    :param min_word_len:
    :return:
    """
    # 先切分,并且去除掉长度小于2的单词
    n = len(title) # n是样本的数目
    title_split = []
    ensemble = set()
    word_order = {}
    for t in title:
        t_split = [tij for tij in t.split(" ") if len(tij) >= min_word_len]
        title_split.append(t_split)
        ensemble = ensemble | set(t_split)
    w = len(ensemble)
    i = 0
    for word in ensemble:
        word_order[word] = i
        i += 1
    print(title_split[0:20])
    word_matrix = torch.zeros(size=(n, w))
    for i in range(n):
        for word in title_split[i]:
            word_matrix[i][word_order[word]] += 1

    print(word_matrix[0][word_order['paper']])
    return word_matrix


def train_test_split(word_matrix, label):
    """
    用于切分训练集和验证集,验证集的占比为10%
    :param word_matrix:
    :param label:
    :return:
    """
    # print(word_matrix.shape, label.shape)
    dataset = torch.cat((word_matrix, label.view(-1, 1)), dim=1)
    # print(dataset.shape)
    n = dataset.shape[0]
    chosen_id = np.random.choice(np.asarray(range(n)), size=n // 10, replace=False)
    valid_data = dataset[torch.tensor(chosen_id, dtype=torch.long)]
    train_id_bool = np.ones(n)
    train_id_bool[chosen_id] = 0
    train_data = dataset[torch.tensor(train_id_bool ,dtype=torch.bool)]
    # print(valid_data.shape, train_data.shape)
    valid_features = valid_data[:, 0:-1]
    valid_labels = valid_data[:, -1]
    train_features = train_data[:, 0:-1]
    train_labels = train_data[:, -1]
    # print(valid_features.shape)
    return train_features, train_labels, valid_features, valid_labels

title, label = load_data()
n = len(title)
title_cleaned = title_clean(title)
print(title_cleaned[0:20])
word_matrix = generate_matrix(title_cleaned)
train_features, train_labels, valid_features, valid_labels = train_test_split(word_matrix, label)

valid_n,train_n = valid_features.shape[0], train_features.shape[0]
acc_num = 0
for i in range(100):
    print(i, acc_num)
    # max_sim = -float("inf")
    sim = torch.cosine_similarity(valid_features[i].view(1, -1), train_features, dim=1)
    # for j in range(train_n):
    #     sim_ij = cos_similarity(train_features[j], valid_features[i])
    #     if sim_ij > max_sim:
    #         max_sim = sim_ij
    #         max_sim_train_id = j
    # print(train_labels[sim.argmax()].item(), valid_labels[i].item())
    if train_labels[sim.argmax()].item() == valid_labels[i].item():
        acc_num += 1
print(acc_num / n * 100)


# 观察有哪些字符需要去掉
# ensemble = set()
# for t in title:
#     ensemble = ensemble | set(list(t))
# l = list(ensemble)
# l.sort()
# print(l)
'''
[' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']
'''