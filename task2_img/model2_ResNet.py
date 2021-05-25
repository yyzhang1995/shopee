import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
from common.layers import GlobalAvgPool2d, FlattenLayer
from common.utils import evaluate_accuracy
from common.utils_load_image import load_data_image
from common.utils_image import top_n_accuracy_image
import time
import numpy as np


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


net = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

net.add_module('resnet_block1', resnet_block(64, 64, 2, first_block=True))
net.add_module('resnet_block2', resnet_block(64, 128, 2))
net.add_module('resnet_block3', resnet_block(128, 256, 2))
net.add_module('resnet_block4', resnet_block(256, 512, 2))
net.add_module('global_avg_pool', GlobalAvgPool2d())
net.add_module('fc',
               nn.Sequential(
                   FlattenLayer(),
                   nn.Linear(512, 2000)
               ))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size, lr = 64, 0.001
train_iter, test_iter, label_to_label_group = load_data_image(batch_size)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

num_epochs = 10

net = net.to(device)
train_loss_list, train_acc_list, test_acc_list = [], [], []
for epoch in range(num_epochs):
    train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_loss_sum += l.cpu().item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
        n += y.shape[0]
    test_acc = evaluate_accuracy(test_iter, net)
    train_loss_list.append(train_loss_sum / n)
    train_acc_list.append(train_acc_sum / n)
    test_acc_list.append(test_acc)
    print("epoch: %d, train loss: %.4f, train acc: %.3f, test acc: %.3f" %
          (epoch + 1, train_loss_sum / n, train_acc_sum / n, test_acc))

timestamp = time.asctime(time.localtime())

# 输出top_n accuracy以及对应的置信度
top_n_matrix, top_n_confidence, y_true = top_n_accuracy_image(test_iter, net, 10, 350)
top_n_matrix = torch.cat([y_true.view(-1, 1), top_n_matrix], dim=1)
np.savetxt('result_model2_top_n_class_%s.txt' % timestamp.replace(':','_'),
           top_n_matrix.detach().numpy(), fmt='%d', delimiter=',')
np.savetxt('result_model2_top_n_indices_%s.txt' % timestamp.replace(':','_'),
           top_n_confidence.detach().numpy(), fmt='%.10f', delimiter=',')

# 保存模型
torch.save(net, 'model2_ResNet18_%s.pkl' % timestamp.replace(':', '_'))

with open("result_model2.txt", 'a') as f:
    f.write(timestamp + "\n")
    # 输出超参数
    f.write("batch_size : " + str(batch_size) + "\n")
    f.write("lr : " + str(lr) + "\n")
    f.write('resize : ' + str(224) + "\n")
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


