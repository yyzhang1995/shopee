import torch
import csv

# 注意title中会出现逗号！需要用csv.read
def load_title(fileroute):
    train_csv = fileroute + "\\train.csv"
    with open(train_csv, 'r') as f:
        reader = csv.reader(f)
        data = [s for s in reader]
        data[0:1] = []
        title = [di[3].lower() for di in data]
        label_group = [int(di[4]) for di in data]
        label_group = torch.tensor(label_group)
    return title, label_group


def load_data():
    fileroute = r"E:\资料\模式识别\作业\大作业\shopee-product-matching"
    title, label_group = load_title(fileroute)
    return title, label_group

if __name__ == '__main__':
    load_data()