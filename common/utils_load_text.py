import torch
import random
import csv
from common.utils import text_clean, trans_label_group_to_label


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


def load_text(min_word_len=2):
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


if __name__ == '__main__':
    train_iter, _, _, _, _, _ = load_text(500)
    for X, y in train_iter:
        print(X.shape)
    print()
    for X, y in train_iter:
        print(X.shape)
    print()
    for X, y in train_iter:
        print(X.shape)

    g = gen()
    for e in range(3):
        for i in g:
            print(i)
