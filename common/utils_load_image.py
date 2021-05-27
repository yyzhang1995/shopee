from torch.utils.data import DataLoader
import csv
import torch
from torchvision import transforms
from common.utils import trans_label_group_to_label, ShopeeData
import sys
import matplotlib.pyplot as plt

file_route = r"E:\资料\模式识别\作业\大作业\shopee-product-matching"


# KNN ----------------------------------------------------------------------------------
def _load_image_flatten(csv_file, resize=(100, 100), show_image=False):
    """
    用于KNN的训练，会将维数压缩到100*100
    :param file_route:
    :return:
    """
    image_file = file_route + "\\train_images"
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        data = [s for s in reader]
        label_group = [int(di[4]) for di in data]
        img_name = [di[1] for di in data]
    label_group = torch.tensor(label_group)
    trans = [transforms.ToTensor(),
             transforms.Normalize(mean=0, std=1),
             transforms.Grayscale(),
             transforms.Resize(resize)]
    transform = transforms.Compose(trans)
    num_samples = len(img_name)
    features = torch.zeros((num_samples, resize[0] * resize[1]))
    photo = 0
    for i in range(num_samples):
        if (i + 1) % 1000 == 0: print("loaded %d image " % (i + 1))

        img_route = image_file + "\\" + img_name[i]
        img = plt.imread(img_route).copy()
        if i < 5 and show_image:
            photo += 1
            plt.subplot(5, 2, photo)
            plt.imshow(img)
        img = transform(img)
        features[i] = img.view(1, -1)
        if i < 5 and show_image:
            photo += 1
            plt.subplot(5, 2, photo)
            plt.imshow(transforms.ToPILImage()(img))
    if show_image:
        plt.show()
    return features, label_group


def load_image_flatten(resize=(100, 100)):
    """
    用于进行数据切分
    :return:
    """
    train_csv = "..\\preliminary\\mini_split_train.csv"
    valid_csv = "..\\preliminary\\mini_split_valid.csv"
    train_features, train_label = _load_image_flatten(train_csv, resize, show_image=True)
    valid_features, valid_label = _load_image_flatten(valid_csv, resize)
    return train_features, train_label, valid_features, valid_label

# KNN ----------------------------------------------------------------------------------

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
