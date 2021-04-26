import torch
import csv
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import time
import numpy as np
from common.utils import generate_matrix


def load_title(fileroute):
    train_csv = "..\\preliminary\\mini_train.csv"
    with open(train_csv, 'r') as f:
        reader = csv.reader(f)
        data = [s for s in reader]
        data[0:1] = []
        title = [di[3].lower() for di in data]
        label_group = [int(di[4]) for di in data]
        label_group = torch.tensor(label_group)
    return title, label_group


def _load_data_image(file_route):
    """
    用于读取图片
    :param fileroute:
    :return:
    """
    # 先读取train.csv当中的图片和label_group,label_group是其中标注的组别
    train_csv = "..\\preliminary\\mini_train.csv"
    image_file = file_route + "\\train_images"
    with open(train_csv, 'r') as f:
        reader = csv.reader(f)
        data = [s for s in reader]
        data[0:1] = []
        label_group = [int(di[4]) for di in data]
        img_name = [di[1] for di in data]
    num_samples = len(img_name)
    features = torch.zeros((num_samples, 3, 224, 224))
    trans = [transforms.ToTensor(), transforms.Resize((224, 224))]
    transform = transforms.Compose(trans)
    for i in range(len(img_name)):
        if (i + 1) % 1000 == 0: print("loading %d images" % (i + 1))
        img_route = image_file + "\\" + img_name[i]
        img = plt.imread(img_route).copy()
        img = transform(img)
        features[i] = img
    return features, label_group


def trans_label_group_to_label(label_group):
    """
    把label_group转化成label
    :param label_group:
    :return:
    """
    labels_without_replic = list(set(label_group))
    label_dict = {}
    i = 0
    for label_group_num in labels_without_replic:
        label_dict[label_group_num] = i
        i += 1
    label = [label_dict[label_group_num] for label_group_num in label_group]
    number_of_labels = len(labels_without_replic)
    return torch.tensor(label), number_of_labels


def load_data_image(batch_size):
    """

    :return:
    """
    fileroute = r"E:\资料\模式识别\作业\大作业\shopee-product-matching"
    features, label_group = _load_data_image(fileroute)
    label, label_num = trans_label_group_to_label(label_group)
    data = torch.utils.data.TensorDataset(features, label)
    train_iter = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return train_iter


def load_image_with_subclass(fileroute, subclasses):
    # 先读取train.csv当中的图片和label_group,label_group是其中标注的组别
    train_csv = fileroute + "\\train.csv"
    image_file = fileroute + "\\train_images"
    with open(train_csv, 'r') as f:
        reader = csv.reader(f)
        data = [s for s in reader]
        data[0:1] = []
        label_group = [int(di[4]) for di in data]
        img_name = [di[1] for di in data]
    num_sample = len(label_group)
    label, num_total_class = trans_label_group_to_label(label_group)
    sub_indices = torch.tensor(np.random.choice(range(num_total_class), subclasses, replace=False), dtype=torch.long)
    check = torch.zeros(num_total_class)
    check[sub_indices] = 1

    # 挑选子集作为训练集
    sub_group_label = [label_group[i] for i in range(num_sample) if check[label[i]] == 1]
    sub_img_name = [img_name[i] for i in range(num_sample) if check[label[i]] == 1]
    sub_label, num_subclass = trans_label_group_to_label(sub_group_label)
    print("number of subclass = ", num_subclass)
    num_sub_samples = len(sub_group_label)
    features = torch.zeros((num_sub_samples, 3, 224, 224))
    trans = [transforms.ToTensor(), transforms.Resize((224, 224))]
    transform = transforms.Compose(trans)
    for i in range(num_sub_samples):
        img_route = image_file + "\\" + img_name[i]
        img = plt.imread(img_route).copy()
        img = transform(img)
        features[i] = img
    return features, sub_label

def load_data_image_with_subclass(batch_size, sub_classes):
    """

    :param batch_size:
    :param sub_classes:
    :return:
    """
    fileroute = r"E:\资料\模式识别\作业\大作业\shopee-product-matching"
    features, labels = load_image_with_subclass(fileroute, sub_classes)
    data = torch.utils.data.TensorDataset(features, labels)
    train_iter =torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return train_iter


def load_data_title():
    fileroute = r"E:\资料\模式识别\作业\大作业\shopee-product-matching"
    title, label_group = load_title(fileroute)
    return title, label_group


def _load_data_text_img(file_route):
    train_csv = "..\\preliminary\\mini_train.csv"
    image_file = file_route + "\\train_images"
    with open(train_csv, 'r') as f:
        reader = csv.reader(f)
        data = [s for s in reader]
        data[0:1] = []
        label_group = [int(di[4]) for di in data]
        img_name = [di[1] for di in data]
        title = [di[3].lower() for di in data]
    # # 直接在此处进行文本清洗
    # word_matrix = generate_matrix(title, min_word_len=2)
    # 读取图像
    trans = [transforms.ToTensor(), transforms.Normalize(mean=0, std=1), transforms.Resize((100, 100))]
    transform = transforms.Compose(trans)
    num_samples = len(img_name)
    features = torch.zeros((num_samples, 100 * 100 * 3))
    for i in range(num_samples):
        if (i + 1) % 1000 == 0: print("loaded %d image " % (i + 1))
        img_route = image_file + "\\" + img_name[i]
        img = plt.imread(img_route).copy()
        img = transform(img)
        features[i] = img.view(1, -1)
    return features, title, label_group


def load_data_text_img():
    """
    用于读取图像以及文本信息,用于多重检索或者跨模态检索
    :return:
    """
    file_route = r"E:\资料\模式识别\作业\大作业\shopee-product-matching"
    return _load_data_text_img(file_route)

def _load_image_flatten(file_route):
    """
    用于KNN的训练，会将维数压缩到100*100
    :param file_route:
    :return:
    """
    train_csv = "..\\preliminary\\mini_train.csv"
    image_file = file_route + "\\train_images"
    with open(train_csv, 'r') as f:
        reader = csv.reader(f)
        data = [s for s in reader]
        data[0:1] = []
        label_group = [int(di[4]) for di in data]
        img_name = [di[1] for di in data]
    trans = [transforms.ToTensor(), transforms.Normalize(mean=0, std=1), transforms.Resize((100, 100))]
    transform = transforms.Compose(trans)
    num_samples = len(img_name)
    features = torch.zeros((num_samples, 100 * 100 * 3))
    for i in range(num_samples):
        if (i + 1) % 1000 == 0: print("loaded %d image " % (i + 1))
        img_route = image_file + "\\" + img_name[i]
        img = plt.imread(img_route).copy()
        img = transform(img)
        features[i] = img.view(1, -1)
    return features, label_group


def load_image_flatten():
    """
    用于进行数据切分
    :return:
    """
    file_route = r"E:\资料\模式识别\作业\大作业\shopee-product-matching"
    features, label_group = _load_image_flatten(file_route)
    return features, torch.tensor(label_group, dtype=np.long)


if __name__ == '__main__':
    start = time.time()
    # train_iter = load_data_image(64)
    train_iter = load_data_image_with_subclass(10, 10)
    print(time.time() - start)

    # trans = transforms.ToPILImage()
    # for X, y in train_iter:
    #     print(y)
    #     plt.imshow(trans(X[0]))
    #     plt.show()
    #     break
