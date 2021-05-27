import torch, torchvision
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


class ShopeeData(Dataset):
    def __init__(self, img_path, imgs, labels, resize, transforms=None, aug=None):
        self.img_path = img_path
        self.imgs = imgs
        self.labels = labels
        self.aug = aug
        self.resize = resize
        self.transforms = transforms
        # features = torch.zeros(size=(len(labels), 3, resize, resize))
        # if pre_trans is None:
        #     pre_trans = transforms.Compose([
        #         transforms.Resize((resize, resize)),
        #         transforms.ToTensor()
        #     ])
        # for i in range(len(imgs)):
        #     if (i + 1) % 1000 == 0: print("loaded " + str(i + 1) + " photo")
        #     img = Image.open(img_path + "\\" + imgs[i])
        #     if pre_trans:
        #         features[i] = pre_trans(img)
        #     else:
        #         features[i] = transforms.ToTensor()(img)
        # self.features = features

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        label = self.labels[index]
        img_id = self.imgs[index]
        img = plt.imread(self.img_path + "\\" + img_id)
        if self.transforms:
            return self.transforms(img), label
        else:
            return img, label


def evaluate_accuracy(data_iter, net, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        if isinstance(net, torch.nn.Module):
            net.eval()
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()
        else:
            try:
                acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
            except TypeError:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def trans_label_group_to_label(label_group):
    """
    把label_group转化成label
    :param label_group:
    :return:
    """
    labels_to_label_group = list(set(label_group))
    label_dict = {}
    i = 0
    for label_group_num in labels_to_label_group:
        label_dict[label_group_num] = i
        i += 1
    label = [label_dict[label_group_num] for label_group_num in label_group]
    number_of_labels = len(labels_to_label_group)
    return torch.tensor(label), labels_to_label_group, number_of_labels

