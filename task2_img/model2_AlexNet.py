import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from common.utils import evaluate_accuracy
from sklearn.metrics import f1_score, precision_score, recall_score
from common.load_data import load_data_image, load_data_image_with_subclass
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

seed = 2020311188
# sub_class = 1000
# sub_class 越多, 学习率越低可能效果会越好！

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# setup_seed(seed)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续三个卷积层
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2000)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

net = AlexNet()
print(net)

# 定义网络参数
batch_size = 192
lr = 1e-2
epoch_num = 1000

# 读取数据
train_iter = load_data_image(batch_size)

# 定义损失函数
loss = torch.nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.Adam(net.parameters(), lr=lr)

# 训练模型
net = net.to(device)
for epoch in range(epoch_num):
    train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
    y_total, y_hat_total = None, None
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
        if y_total is None:
            y_total = y.clone().cpu()
            y_hat_total = y_hat.argmax(dim=1).cpu()
        else:
            y_total = torch.cat((y_total, y.clone().cpu()), dim=0)
            y_hat_total = torch.cat((y_hat_total, y_hat.argmax(dim=1).cpu()), dim=0)
    # test_acc = evaluate_accuracy(test_iter, net)
    print("epoch: %d, train loss: %.4f, train acc: %.3f" %
          (epoch + 1, train_loss_sum / n, train_acc_sum / n))

print(y_total, y_hat_total)
precision = precision_score(y_total, y_hat_total, average='macro')
recall = recall_score(y_total, y_hat_total, average='macro')
f1 = f1_score(y_total, y_hat_total, average='macro')
print(precision, recall, f1)