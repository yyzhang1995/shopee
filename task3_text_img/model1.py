import torch
import numpy as np
from common.load_data import load_data_text_img
from common.utils import generate_matrix, trans_label_group_to_label
from sklearn.metrics import f1_score, precision_score, recall_score
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_test_split(features, word_matrix, label):
    """
    用于切分训练集和验证集,验证集的占比为10%
    :param word_matrix:
    :param label:
    :return:
    """
    # print(word_matrix.shape, label.shape)
    num_features = features.shape[1]
    dataset = torch.cat((features,word_matrix, label.view(-1, 1)), dim=1)
    # print(dataset.shape)
    n = dataset.shape[0]
    chosen_id = np.random.choice(np.asarray(range(n)), size=n // 10, replace=False)
    valid_data = dataset[torch.tensor(chosen_id, dtype=torch.long)]
    train_id_bool = np.ones(n)
    train_id_bool[chosen_id] = 0
    train_data = dataset[torch.tensor(train_id_bool, dtype=torch.bool)]
    # print(valid_data.shape, train_data.shape)

    valid_features = valid_data[:, 0:-1]
    valid_labels = valid_data[:, -1].long()

    train_features = train_data[:, 0:-1]
    train_labels = train_data[:, -1].long()
    # print(valid_features.shape)

    return train_features, train_labels, valid_features, valid_labels


features, text, label_group = load_data_text_img()
label, _ = trans_label_group_to_label(label_group)
word_matrix = generate_matrix(text)
train_features, train_labels, valid_features, valid_labels = train_test_split(features, word_matrix, label)
dim_features = features.shape[1]

valid_n, train_n = valid_features.shape[0], train_features.shape[0]
train_image_features = train_features[:, :dim_features].to(device)
train_text_features = train_features[:, dim_features:].to(device)
valid_image_features = valid_features[:, :dim_features].to(device)
valid_text_features = valid_features[:, dim_features:].to(device)
# train_features = train_features.to(device)
# valid_features = valid_features.to(device)
image_pred_label, text_pred_label = torch.zeros_like(valid_labels), torch.zeros_like(valid_labels)
pred_label = torch.zeros_like(valid_labels)

# 先给出文字和图片各自的准确率
acc_num_image = 0
acc_num_text = 0
for i in range(valid_n):
    sim_image = torch.cosine_similarity(valid_image_features[i].view(1, -1), train_image_features, dim=1)
    sim_text = torch.cosine_similarity(valid_text_features[i].view(1, -1), train_text_features, dim=1)
    image_pred_label[i] = train_labels[sim_image.argmax()].cpu().item()
    text_pred_label[i] = train_labels[sim_text.argmax()].cpu().item()
    if image_pred_label[i] == valid_labels[i].item():
        acc_num_image += 1
    if text_pred_label[i] == valid_labels[i].item():
        acc_num_text += 1

acc_image = acc_num_image / valid_n
acc_text = acc_num_text / valid_n

# 开始正式的训练
acc_num = 0
for i in range(valid_n):
    print(i, acc_num)
    sim_image = torch.cosine_similarity(valid_image_features[i].view(1, -1), train_image_features, dim=1) * acc_image
    sim_text = torch.cosine_similarity(valid_text_features[i].view(1, -1), train_text_features, dim=1) * acc_text
    sim = sim_image + sim_text
    pred_label[i] = train_labels[sim.argmax()].cpu().item()
    if pred_label[i] == valid_labels[i].item():
        acc_num += 1
accuracy = acc_num / valid_n * 100

precision = precision_score(y_true=valid_labels, y_pred=pred_label, average='macro')
recall = recall_score(y_true=valid_labels, y_pred=pred_label, average='macro')
f1 = f1_score(y_true=valid_labels, y_pred=pred_label, average='macro')
print("accuracy: %.3f, precision: %.3f, recall: %.3f, f1: %.3f" %
      (accuracy, precision, recall, f1))

with open("result_model1.txt", 'a') as f:
    f.write(time.asctime(time.localtime()) + '\n')
    f.write("valid set number: %d \n" % (valid_n))
    f.write("accuracy: %.3f \n" % (accuracy))
    f.write("precision rate: %.3f \n" % (precision))
    f.write("recall rate: %.3f \n" % (recall))
    f.write("f1 score: %.3f \n" % (f1))
    f.write('\n')

# 直接用把所有的特征合并进行分类，效果不佳
# 直接取并集（即只要有其中一个预测正确就算正确）