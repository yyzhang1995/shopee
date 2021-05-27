import torch
import numpy as np
import random
import csv
import re
from common.utils import trans_label_group_to_label

# for KNN --------------------------------------------------------------------------------
def load_title(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        data = [s for s in reader]
        data[0:1] = []
        title = [di[3].lower() for di in data]
        label_group = [int(di[4]) for di in data]
        label_group = torch.tensor(label_group)
    return title, label_group


def load_data_title():
    train_csv = "..\\preliminary\\mini_split_train.csv"
    valid_csv = "..\\preliminary\\mini_split_valid.csv"
    train_title, train_label_group = load_title(train_csv)
    valid_title, valid_label_group = load_title(valid_csv)
    return train_title, train_label_group, valid_title, valid_label_group

# ------------------------------------------------------------------------------------------


def data_iter_random(title_indices, labels, batch_size, device=None):
    """
    返回包含了文本向量和标签的生成器
    :param corpus_indices: 所有标题的词向量，统一到最长的长度，如果长度不够则在前侧补空格
    :param batch_size:
    :param num_steps:
    :param device:
    :return:
    """
    n = len(labels)
    indices = list(range(n))
    random.shuffle(indices)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(0, n, batch_size):
        batch_indices = torch.LongTensor(indices[i: min(i + batch_size, n)])
        yield title_indices.index_select(0, batch_indices), labels.index_select(0, batch_indices)


def _load_text(file_route):
    with open(file_route, 'r') as f:
        reader = csv.reader(f)
        data = [s for s in reader]
        title = [di[3].lower() for di in data]
        label_group = [int(di[-1]) for di in data]
    return title, label_group


def load_text(min_word_len=2, tfidf=True):
    csv_train = "..\\preliminary\\mini_split_train.csv"
    csv_valid = "..\\preliminary\\mini_split_valid.csv"

    train_titles, train_label_group = _load_text(csv_train)
    valid_titles, valid_label_group = _load_text(csv_valid)
    train_titles = text_clean(train_titles)
    valid_titles = text_clean(valid_titles)
    train_titles_splitted = [[w for w in title.split() if len(w) > min_word_len] for title in train_titles]
    valid_titles_splitted = [[w for w in title.split() if len(w) > min_word_len] for title in valid_titles]

    # 获得单词与序号的一一对应关系idx_to_char和char_to_idx
    word_dict = set(' ')
    max_len = 0
    for titles_splitted in [train_titles_splitted, valid_titles_splitted]:
        for title in titles_splitted:
            word_dict = word_dict | set(title)
            max_len = max(max_len, len(title))
    idx_to_char = list(word_dict)
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(idx_to_char)

    # 将每个title转化为用字典索引表示的向量，向量大小为max_len维
    train_titles_indices = [[char_to_idx[w] for w in title_split] for title_split in train_titles_splitted]
    valid_titles_indices = [[char_to_idx[w] for w in title_split] for title_split in valid_titles_splitted]

    space = char_to_idx[' ']
    train_titles_indices = [[space] * (max_len - len(title_split)) + title_split
                            for title_split in train_titles_indices]
    valid_titles_indices = [[space] * (max_len - len(title_split)) + title_split
                            for title_split in valid_titles_indices]
    train_titles_indices = torch.tensor(train_titles_indices)
    valid_titles_indices = torch.tensor(valid_titles_indices)

    # 将label_group也映射维0-n的整数
    label, label_to_label_group, label_num= trans_label_group_to_label(train_label_group + valid_label_group)
    train_labels, valid_labels = label[:len(train_label_group)], label[len(train_label_group):]

    return train_titles_indices, train_labels, valid_titles_indices, valid_labels, \
           vocab_size, idx_to_char, char_to_idx, label_to_label_group, max_len


def bootstrap(X, label, subsample_rate=0.7):
    n = X.shape[0]
    subsample_indices = np.random.choice(n, int(n * subsample_rate))
    return X[subsample_indices, :], label[subsample_indices]


def to_tf_idf(word_vector):
    doc = word_vector.shape[0]
    tf = word_vector / word_vector.sum(dim=1, keepdim=True)
    word_exist = (word_vector > 0).int()
    idf = torch.log10(doc / (word_exist.sum(dim=0, keepdim=True) + 1))
    return tf * idf


# 清洗文本
def _clean(s):
    res = re.sub(r"[^a-z]", " ", s)
    res = re.sub(r" +", " ", res).strip()
    return res


def text_clean(text):
    """
    对文本进行清晰，去除掉标点符号、数字
    :param text:
    :return:
    """
    for i in range(len(text)):
        text[i] = _clean(text[i])
    return text


def generate_matrix(title, min_word_len=2, tfidf=True):
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
    word_matrix = torch.zeros(size=(n, w))
    for i in range(n):
        for word in title_split[i]:
            word_matrix[i][word_order[word]] += 1
    if tfidf:
        word_matrix = to_tf_idf(word_matrix)
    return word_matrix





if __name__ == '__main__':
    X = torch.tensor([[1, 0, 0, 2, 0],
                      [0, 1, 1, 0, 0],
                      [2, 0, 1, 3, 0],
                      [0, 1, 2, 0, 1],
                      [0, 1, 0, 1, 1],
                      [1, 0, 0, 0, 0]])
    to_tf_idf(X)