import torch


def cos_similarity(x, y):
    # print((x * x).sum())
    return torch.dot(x, y) / ((x * x).sum().pow(0.5) * torch.sqrt((y * y).sum()))

#
# x = torch.tensor([1.0, 2.0])
# y = torch.tensor([4.0, 0.0])
# print(cos_similarity(x, y))