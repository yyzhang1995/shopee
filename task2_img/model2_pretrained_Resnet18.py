import torch
from torch import nn, optim
import torch.nn.functional as F
from common.layers import GlobalAvgPool2d, FlattenLayer
from common.utils import evaluate_accuracy
from common.load_data import load_data_image
from torchvision import transforms, models

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
