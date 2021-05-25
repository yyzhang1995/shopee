import torch
import csv
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import time
import numpy as np
from common.utils import generate_matrix, trans_label_group_to_label, ShopeeData
import sys


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


def load_data_title():
    file_route = r"E:\资料\模式识别\作业\大作业\shopee-product-matching"
    title, label_group = load_title(file_route)
    return title, label_group


# -------------------------------------------------------------------------------------- #
def _load_data_image(file_route, resize, csv_file=None):
    """
    用于读取图片
    :param file_route:
    :return:
    """
    # 先读取train.csv当中的图片和label_group,label_group是其中标注的组别
    if csv_file is None:
        csv_file = "..\\preliminary\\mini_train.csv"
    image_file = file_route + "\\train_images"
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        data = [s for s in reader]
        data[0:1] = []
        label_group = [int(di[4]) for di in data]
        img_name = [di[1] for di in data]
    num_samples = len(img_name)
    features = torch.zeros((num_samples, 3, resize, resize))
    trans = [transforms.ToTensor(), transforms.Normalize(0, 1), transforms.Resize((resize, resize))]
    transform = transforms.Compose(trans)
    for i in range(len(img_name)):
        if (i + 1) % 1000 == 0: print("loading %d images" % (i + 1))
        img_route = image_file + "\\" + img_name[i]
        img = plt.imread(img_route).copy()
        img = transform(img)
        features[i] = img
    return features, label_group


def _load_imgs_text_annotations(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        data = [s for s in reader]
        label_group = [int(di[4]) for di in data]
        text = [di[3] for di in data]
        img_name = [di[1] for di in data]
    return img_name, text, label_group


def load_data_image(batch_size, resize=224, train_aug=None, test_aug=None):
    """
    仅仅读取图片
    @:param batch_size:
    @:param resize:
    :return:
    """

    csv_train = "..\\preliminary\\mini_split_train.csv"
    csv_valid = "..\\preliminary\\mini_split_valid.csv"
    img_path = r"E:\资料\模式识别\作业\大作业\shopee-product-matching\train_images"
    train_img_name,_,  train_label_group = _load_imgs_text_annotations(csv_train)
    valid_img_name,_,  valid_label_group = _load_imgs_text_annotations(csv_valid)
    # 把标签进行相应的转换
    label, label_to_label_group, label_num = trans_label_group_to_label(train_label_group + valid_label_group)
    train_label, valid_label = label[:len(train_label_group)], label[len(train_label_group):]

    train_dataset = ShopeeData(img_path, train_img_name, train_label, resize=resize,
                               aug=train_aug)
    valid_dataset = ShopeeData(img_path, valid_img_name, valid_label, resize=resize,
                               aug=test_aug)

    num_workers = 0 if sys.platform.startswith('win32') else 4
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_iter, valid_iter, label_to_label_group


# -----------------------------------------------------------------------------------------------#
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

# --------------------------------------------------------------------------------------- #
def show_image(X, y):
    max_photo = min(10, X.shape[0])
    trans = transforms.ToPILImage()
    photo = 0
    for i in range(max_photo):
        photo += 1
        plt.subplot(1, max_photo, photo)
        plt.title(y[i])
        plt.imshow(trans(X[i]))
    plt.show()

if __name__ == '__main__':
    start = time.time()
    train_aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])
    test_aug = transforms.ToTensor()
    train_iter, valid_iter = load_data_image(10, train_aug=train_aug, train_test_split=True)
    print(time.time() - start)

    for X, y in train_iter:
        print("Xshape", X.shape)
        show_image(X, y)
        break

    plt.show()
