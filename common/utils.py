import torch
import re

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
    labels_without_replic = list(set(label_group))
    label_dict = {}
    i = 0
    for label_group_num in labels_without_replic:
        label_dict[label_group_num] = i
        i += 1
    label = [label_dict[label_group_num] for label_group_num in label_group]
    number_of_labels = len(labels_without_replic)
    return torch.tensor(label), number_of_labels

# def data_iter(batch_size):


# 清洗文本

def _clean(s):
    res = re.sub(r"[^a-z]", " ", s)
    res = re.sub(r" +", " ", res).strip()
    return res

def text_clean(text):
    """
    对文本进行清晰，去除掉标点符号、数字
    :param text:
    :return:
    """
    for i in range(len(text)):
        text[i] = _clean(text[i])
    return text


def generate_matrix(title, min_word_len=2):
    """
    生成单词向量
    :param title:
    :param min_word_len:
    :return:
    """
    # 先切分,并且去除掉长度小于2的单词
    n = len(title) # n是样本的数目
    title_split = []
    ensemble = set()
    word_order = {}
    for t in title:
        t_split = [tij for tij in t.split(" ") if len(tij) >= min_word_len]
        title_split.append(t_split)
        ensemble = ensemble | set(t_split)
    w = len(ensemble)
    i = 0
    for word in ensemble:
        word_order[word] = i
        i += 1
    word_matrix = torch.zeros(size=(n, w))
    for i in range(n):
        for word in title_split[i]:
            word_matrix[i][word_order[word]] += 1
    return word_matrix