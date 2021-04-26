import torch
import numpy as np
import csv
from common.utils import trans_label_group_to_label


def generate_mini_dataset(mini_class_num):
    file_route = r"E:\资料\模式识别\作业\大作业\shopee-product-matching"
    train_csv = file_route + "\\train.csv"
    with open(train_csv, 'r') as f:
        reader = csv.reader(f)
        data = [s for s in reader]
        data_title = data[0]
        data[0:1] = []
        label_group = [int(di[4]) for di in data]
    num_sample = len(label_group)

    label, num_total_class = trans_label_group_to_label(label_group)
    sub_indices = torch.tensor(np.random.choice(range(num_total_class), mini_class_num, replace=False), dtype=torch.long)
    check = torch.zeros(num_total_class)
    check[sub_indices] = 1

    # 挑选子集作为训练集
    mini_data = [data[i] for i in range(num_sample) if check[label[i]] == 1]
    mini_data.insert(0, data_title)
    return mini_data


def main():
    mini_data = generate_mini_dataset(2000)
    with open("mini_train.csv", 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(mini_data)


if __name__ == '__main__':
    main()