import torch
import csv

def eval_model(pred_res, label, top):
    """
    对模型的预测结果进行评估，采用top-n准确率，精确率，召回率和F值进行评估,对每一条结果取平均值
    :param pred_res:
    :param label:
    :param top:
    :return:
    """
    n = len(label)
    acc_num = (pred_res[:, :top] == label.view(-1, 1)).sum()
    top_n_acc = acc_num.item() / n
    # 构建混淆矩阵
    TP = []
    FP = []
    FN = []
    for i in range(n):
        TP_i, FP_i, FN_i = 0, 0, 0
        for j in range(n):
            if j ==i:
                if label[i] == pred_res[j, 0]: TP_i += 1
                else: FN_i += 1
            else:
                if label[i] == pred_res[j, 0]: FP_i += 1
        TP.append(TP_i)
        FP.append(FP_i)
        FN.append(FN_i)
    precision = []
    recall = []
    F_score = []
    for i in range(n):
        precision.append(TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) > 0 else 0)
        recall.append(TP[i] / (TP[i] + FN[i]))
        try:
            F_score.append(2 * recall[i] * precision[i] / (recall[i] + precision[i]))
        except ZeroDivisionError:
            F_score.append(0)
    return top_n_acc, sum(precision) / n, sum(recall) / n, sum(F_score) / n


def load_res(res_file):
    with open(res_file, 'r') as f:
        res = [l.strip().split(',') for l in f.readlines()]
    label = [int(l[0]) for l in res]
    data = [l[1:] for l in res]
    for l in data:
        for i in range(len(l)):
            l[i] = int(l[i])
    return torch.tensor(label), torch.tensor(data)


def check_model(train_csv, res_model, timestamp):
    """
    对结果进行深入的分析，比如分析那些分类分对了，哪些没有分对
    :param train_csv:
    :param res_model:
    :param timestamp:
    :return:
    """
    res_top_n_class = res_model + ("_top_n_class_%s" % timestamp.replace(':', "_")) + ".txt"
    label, data = load_res(res_top_n_class)
    # with open(train_csv, 'r') as f:
    #     csv_reader = csv.reader(f)
    #     input_data = [l for l in csv_reader]

    label_to_label_group = []
    with open(res_model + ".txt", 'r') as f:
        while True:
            l = f.readline()
            if l.find(timestamp) != -1:
                while l.find("label to label group") == -1:
                    l = f.readline()
                label_to_label_group = f.readline()
                break
            if not l:
                break
    label_to_label_group = list(map(int, label_to_label_group.split(',')))

    # 随机挑选出错误分类的样例
    n = len(label)
    wrong_class = (data[:, 0] != label)
    wrong_class_num = 0
    wrong_class_title = []
    wrong_class_fig = []
    i = 0
    while wrong_class_num < 5:
        while not wrong_class[i]:
            i += 1

        wrong_class_num += 1
        i += 1


if __name__ == '__main__':
    label, data = load_res("../task1_text/result_model2_top_n_class.txt")
    top = 1
    # print(eval_model(torch.tensor(data), torch.tensor(label), top))
    check_model("", "../task1_text/result_model2", 'Sat May  8 23:03:56 2021')