import torch
from torch import nn, optim
import torch.nn.functional as F
from common.layers import GlobalAvgPool2d, FlattenLayer
from common.utils import evaluate_accuracy
from common.load_data import load_data_image
from torchvision import transforms, models


# class Residual(nn.Module):
#     def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
#         super(Residual, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         if use_1x1conv:
#             self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
#         else:
#             self.conv3 = None
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, X):
#         Y = F.relu(self.bn1(self.conv1(X)))
#         Y = self.bn2(self.conv2(Y))
#         if self.conv3:
#             X = self.conv3(X)
#         return F.relu(X + Y)
#
#
#
# def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
#     if first_block:
#         assert in_channels == out_channels
#     blk = []
#     for i in range(num_residuals):
#         if i == 0 and not first_block:
#             blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
#         else:
#             blk.append(Residual(out_channels, out_channels))
#     return nn.Sequential(*blk)
#
#
# net = nn.Sequential(
#     nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
#     nn.BatchNorm2d(64),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
# )
#
# net.add_module('resnet_block1', resnet_block(64, 64, 2, first_block=True))
# net.add_module('resnet_block2', resnet_block(64, 128, 2))
# net.add_module('resnet_block3', resnet_block(128, 256, 2))
# net.add_module('resnet_block4', resnet_block(256, 512, 2))
# net.add_module('global_avg_pool', GlobalAvgPool2d())
# net.add_module('fc',
#                nn.Sequential(
#                    FlattenLayer(),
#                    nn.Linear(512, 2000)
#                ))

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# batch_size, lr = 64, 0.001
# train_iter, test_iter = load_data_image(batch_size, train_test_split=True)
#
# loss = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=lr)
#
# num_epochs = 60
#
# net = net.to(device)
# for epoch in range(num_epochs):
#     train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
#     for X, y in train_iter:
#         X = X.to(device)
#         y = y.to(device)
#         y_hat = net(X)
#         l = loss(y_hat, y)
#         optimizer.zero_grad()
#         l.backward()
#         optimizer.step()
#         train_loss_sum += l.cpu().item()
#         train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
#         n += y.shape[0]
#     test_acc = evaluate_accuracy(test_iter, net)
#     print("epoch: %d, train loss: %.4f, train acc: %.3f, test acc: %.3f" %
#           (epoch + 1, train_loss_sum / n, train_acc_sum / n, test_acc))

# 定义图片规范化方法
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_augs = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
test_augs = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    normalize
])

# 定义预训练网络

pretrained_net = models.resnet18(pretrained=True)
print(pretrained_net.fc)

pretrained_net.fc = nn.Linear(in_features=512, out_features=2000, bias=True)

lr = 0.001
output_params = list(map(id, pretrained_net.fc.parameters()))
feature_params = filter(lambda p:id(p) not in output_params, pretrained_net.parameters())

optimizer = optim.Adam([{'params': feature_params},
                       {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}], lr=lr)


def train_models(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
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
        print("epoch: %d, train loss: %.4f, train acc: %.3f, test acc: %.3f" %
              (epoch + 1, train_loss_sum / n, train_acc_sum / n, test_acc))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size, num_epochs = 96, 50
train_iter, test_iter = load_data_image(batch_size, train_aug=train_augs, test_aug=test_augs)
loss = nn.CrossEntropyLoss()

train_models(train_iter, test_iter, pretrained_net, loss, optimizer, device, num_epochs)
