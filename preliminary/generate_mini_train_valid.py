import torch
import numpy as np
import random
import csv

"""
需要将数据集划分成训练集和验证集
划分的方式为对于给定的数据集和数据子集，每一类商品随机挑选一个放入验证集，其余作为训练集
"""

def train_valid_split(data):
    data = sorted(data, key=lambda x: x[-1])
    n = len(data)
    valid_data = [False] * n
    data_valid = []
    i = 0
    while i < n:
        label = data[i][-1]
        count = 0
        while i + count < n and data[i + count][-1] == label:
            count += 1
        v = random.choice(list(range(count))) + i
        data_valid.append(data[v])
        valid_data[v] = True
        i += count
    data_train = [data[i] for i in range(n) if not valid_data[i]]
    return data_train, data_valid


def data_split(train_file_route, output_file_prefix):
    with open(train_file_route, 'r') as f:
        reader = csv.reader(f)
        data = [s for s in reader]
        data_title = data[0]
        data[0:1] = []
    for di in data:
        di[-1] = int(di[-1])
    data_train, data_valid = train_valid_split(data)
    output_train_csv = output_file_prefix + "_train.csv"
    output_valid_csv = output_file_prefix + "_valid.csv"
    with open(output_train_csv, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data_train)

    with open(output_valid_csv, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data_valid)


if __name__ == '__main__':
    # 进行mini数据集的分割
    train_file_route = "mini_train.csv"
    output_file_prefix = "mini_split"
    data_split(train_file_route, output_file_prefix)
