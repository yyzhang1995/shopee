import torch
from common.utils_load_text import load_data_title, text_clean, generate_matrix
import numpy as np
import time
from common.utils_eval import eval_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
针对商品的文字进行处理
'''
tf_idf = False
train_title, train_labels, valid_title, valid_labels = load_data_title()
title = train_title + valid_title
title_cleaned = text_clean(title)
word_matrix = generate_matrix(title_cleaned, tf_idf)

n, n_train = len(title), len(train_title)
print(title_cleaned[0:5])
train_features, valid_features, = word_matrix[:n_train,:], word_matrix[n_train:,:]
# train_features, train_labels, valid_features, valid_labels = train_test_split(word_matrix, label)

valid_n, train_n = valid_features.shape[0], train_features.shape[0]
acc_num = 0
train_features = train_features.to(device)
valid_features = valid_features.to(device)
pred_label = torch.zeros_like(valid_labels)
for i in range(valid_n):
    sim = torch.cosine_similarity(valid_features[i].view(1, -1), train_features, dim=1)
    pred_label[i] = train_labels[sim.argmax()].cpu().item()
    if train_labels[sim.argmax()].cpu().item() == valid_labels[i].item():
        acc_num += 1
accuracy = acc_num / valid_n * 100

# 评估模型
_, precision, recall, f1 = eval_model(pred_res=pred_label.view(-1, 1), label=valid_labels, top=1)
print("accuracy: %.3f, precision: %.3f, recall: %.3f, f1: %.3f" %
      (accuracy, precision, recall, f1))

timestamp = time.asctime(time.localtime())
with open(".\\results\\result_model1.txt", 'a') as f:
    f.write(timestamp + '\n')
    f.write("valid set number: %d \n" % (valid_n))
    f.write("tfidf: %d\n" % (tf_idf))
    f.write("accuracy: %.3f \n" % (accuracy))
    f.write("precision rate: %.3f \n" % (precision))
    f.write("recall rate: %.3f \n" % (recall))
    f.write("f1 score: %.3f \n" % (f1))
    f.write('\n')


np.savetxt('.\\results\\result_model1_prediction_%s.txt' % timestamp.replace(':','_'),
           torch.cat((valid_labels.view(-1, 1), pred_label.view(-1, 1)), dim=1).numpy(), fmt='%d', delimiter=',')
