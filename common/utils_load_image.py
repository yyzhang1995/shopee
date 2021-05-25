from torch.utils.data import DataLoader
import csv
import torch
from common.utils import trans_label_group_to_label, ShopeeData
import sys


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