import torch
import re
from common.load_data import load_data_title
from common.funcs import cos_similarity
from common.utils import text_clean, generate_matrix
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
针对商品的文字进行处理
'''


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
    train_data = dataset[torch.tensor(train_id_bool, dtype=torch.bool)]
    # print(valid_data.shape, train_data.shape)
    valid_features = valid_data[:, 0:-1]
    valid_labels = valid_data[:, -1].long()
    train_features = train_data[:, 0:-1]
    train_labels = train_data[:, -1].long()
    # print(valid_features.shape)
    return train_features, train_labels, valid_features, valid_labels

title, label = load_data_title()
n = len(title)
title_cleaned = text_clean(title)
print(title_cleaned[0:20])
word_matrix = generate_matrix(title_cleaned)
train_features, train_labels, valid_features, valid_labels = train_test_split(word_matrix, label)

valid_n, train_n = valid_features.shape[0], train_features.shape[0]
acc_num = 0
train_features = train_features.to(device)
valid_features = valid_features.to(device)
pred_label = torch.zeros_like(valid_labels)
for i in range(valid_n):
    print(i, acc_num)
    sim = torch.cosine_similarity(valid_features[i].view(1, -1), train_features, dim=1)
    pred_label[i] = train_labels[sim.argmax()].cpu().item()
    if train_labels[sim.argmax()].cpu().item() == valid_labels[i].item():
        acc_num += 1
accuracy = acc_num / valid_n * 100

precision = precision_score(y_true=valid_labels, y_pred=pred_label, average='macro')
recall = recall_score(y_true=valid_labels, y_pred=pred_label, average='macro')
f1 = f1_score(y_true=valid_labels, y_pred=pred_label, average='macro')
print("accuracy: %.3f, precision: %.3f, recall: %.3f, f1: %.3f" %
      (accuracy, precision, recall, f1))

with open("result_model1.txt", 'a') as f:
    f.write(time.asctime(time.localtime()) + '\n')
    f.write("valid set number: %d \n" % (valid_n))
    f.write("accuracy: %.3f \n" % (accuracy))
    f.write("precision rate: %.3f \n" % (precision))
    f.write("recall rate: %.3f \n" % (recall))
    f.write("f1 score: %.3f \n" % (f1))
    f.write('\n')

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