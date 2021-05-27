import torch
import numpy as np
from common.utils_load_image import load_image_flatten
from common.utils_eval import eval_model
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resize = (200, 200)
train_features, train_labels, valid_features, valid_labels = load_image_flatten(resize)

valid_n, train_n = valid_features.shape[0], train_features.shape[0]
acc_num = 0
pred_label = torch.zeros_like(valid_labels)

train_features = train_features.to(device)
valid_features = valid_features.to(device)

for i in range(valid_n):
    print(i, acc_num)
    sim = torch.cosine_similarity(valid_features[i].view(1, -1), train_features, dim=1)
    pred_label[i] = train_labels[sim.argmax()].cpu().item()
    if train_labels[sim.argmax()].cpu().item() == valid_labels[i].item():
        acc_num += 1
accuracy = acc_num / valid_n * 100

_, precision, recall, f1 = eval_model(pred_res=pred_label.view(-1, 1), label=valid_labels, top=1)
print("accuracy: %.3f, precision: %.3f, recall: %.3f, f1: %.3f" %
      (accuracy, precision, recall, f1))

timestamp = time.asctime(time.localtime())
with open(".\\results\\result_model1.txt", 'a') as f:
    f.write(timestamp + '\n')
    f.write("valid set number: %d \n" % (valid_n + 1))
    f.write("resize: %d, %d \n" % resize)
    f.write("accuracy: %.3f \n" % (accuracy))
    f.write("precision rate: %.3f \n" % (precision))
    f.write("recall rate: %.3f \n" % (recall))
    f.write("f1 score: %.3f \n" % (f1))
    f.write('\n')

np.savetxt('.\\results\\result_model1_prediction_%s.txt' % timestamp.replace(':','_'),
           torch.cat((valid_labels.view(-1, 1), pred_label.view(-1, 1)), dim=1).numpy(), fmt='%d', delimiter=',')
