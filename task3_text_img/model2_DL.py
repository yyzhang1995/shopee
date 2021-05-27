import torch
from torch import nn, optim
from common.utils_text import to_onehot, grad_clipping, evaluate_text_accuracy, top_n_accuracy
from common.utils_load_text import load_text, data_iter_random
import time
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from common.layers import GlobalAvgPool2d, FlattenLayer
from common.utils import evaluate_accuracy
from common.utils_load_image import load_data_image
from common.utils_image import top_n_accuracy_image


class MixModel(nn.Module):
    def __init__(self, rnn, resnet):
        super(MixModel, self).__init__()
        self.rnn = rnn
        self.resnet = resnet
        self.dense = nn.Linear(2000, 2000)

    def forward(self, X_img, X_text):
        X1 = self.resnet(X_img)
        X2 = self.rnn(X_text, state=None)[0]
        output = self.dense(torch.cat((X1, X2)))
        return output


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, n_class):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.drop = nn.Dropout()
        self.dense = nn.Linear(self.hidden_size, n_class)
        self.state = None

    def forward(self, inputs, state):
        X = to_onehot(inputs, self.vocab_size)
        # X的维度是num_stpes, batch_size, vocal_size
        Y, self.state = self.rnn(torch.stack(X), state)
        # Y的维度对应是num_steps, batch_size, hidden_size
        # 接下来对Y做一个池化
        Y = Y.mean(dim=0)
        output = self.dense(self.drop(Y.view(-1, Y.shape[-1])))
        return output, self.state


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(X + Y)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


res_net = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

res_net.add_module('resnet_block1', resnet_block(64, 64, 2, first_block=True))
res_net.add_module('resnet_block2', resnet_block(64, 128, 2))
res_net.add_module('resnet_block3', resnet_block(128, 256, 2))
res_net.add_module('resnet_block4', resnet_block(256, 512, 2))
res_net.add_module('global_avg_pool', GlobalAvgPool2d())
res_net.add_module('fc',
               nn.Sequential(
                   FlattenLayer(),
                   nn.Linear(512, 1000)
               ))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size, lr = 128, 0.001
train_title_indices, train_labels, valid_title_indices, valid_labels,\
vocab_size, idx_to_char, char_to_idx, label_to_label_group, max_len = load_text()

num_hiddens = 256
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
rnn_net = RNNModel(rnn_layer, n_class=1000)

mixed_net = MixModel(rnn_net, res_net)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(mixed_net.parameters(), lr=lr)

clipping_theta = 1e-2
model = mixed_net.to(device)
num_epochs = 250
state = None
train_loss_list, train_acc_list, test_acc_list = [], [], []
for epoch in range(num_epochs):
    train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
    train_iter = data_iter_random(train_title_indices, train_labels, batch_size=batch_size)
    test_iter = data_iter_random(valid_title_indices, valid_labels, batch_size=batch_size)
    for X, y in train_iter:
        X = X.to(device)
        y = y.to(device)
        y_hat, _ = model(X, None)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        grad_clipping(model.parameters(), clipping_theta, device)
        optimizer.step()
        train_loss_sum += l.cpu().item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
        n += y.shape[0]
    test_acc = evaluate_text_accuracy(test_iter, model)
    train_loss_list.append(train_loss_sum / n)
    train_acc_list.append(train_acc_sum / n)
    test_acc_list.append(test_acc)
    print("epoch: %d, train loss: %.4f, train acc: %.3f, test acc:%.3f" %
          (epoch + 1, train_loss_sum / n, train_acc_sum / n, test_acc))

# 需要存储每一步的acc（train、test）和loss
# 存储top10的预测confidence score

timestamp = time.asctime(time.localtime())
with open("result_model2.txt", 'a') as f:
    f.write(timestamp + "\n")
    # 输出超参数
    f.write("batch_size : " + str(batch_size) + "\n")
    f.write("lr : " + str(lr) + "\n")
    f.write("num hiddens : " + str(num_hiddens) + "\n")
    f.write("clipping theta : " + str(clipping_theta) + "\n")
    f.write("label to label group : \n")
    f.write(','.join(map(str, label_to_label_group)))
    f.write('\n')
    # 输出训练过程
    f.write("train loss : \n")
    f.write(','.join(map(str, train_loss_list)))
    f.write('\n')
    f.write("train acc : \n")
    f.write(','.join(map(str, train_acc_list)))
    f.write('\n')
    f.write("test acc : \n")
    f.write(','.join(map(str, test_acc_list)))
    f.write('\n\n')

# 输出top_n accuracy以及对应的置信度
test_iter = data_iter_random(valid_title_indices, valid_labels, batch_size=batch_size)
top_n_matrix, top_n_confidence, y_true = top_n_accuracy(test_iter, model, 10, len(valid_labels))
top_n_matrix = torch.cat([y_true.view(-1, 1), top_n_matrix], dim=1)
np.savetxt('result_model2_top_n_class_%s.txt' % timestamp.replace(':','_'), top_n_matrix.detach().numpy(), fmt='%d', delimiter=',')
np.savetxt('result_model2_top_n_indices_%s.txt' % timestamp.replace(':','_'), top_n_confidence.detach().numpy(), fmt='%.10f', delimiter=',')


if __name__ == '__main__':
    pass