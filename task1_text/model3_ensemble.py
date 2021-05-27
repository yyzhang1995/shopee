import torch
from torch import nn, optim
from common.utils_text import to_onehot, grad_clipping, evaluate_text_accuracy, voting, evaluate_text_accuracy_bagging
from common.utils_load_text import load_text, data_iter_random, bootstrap
import time
import numpy as np


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


# 基于多个深度神经网络的bagging方法,可以研究bagging数量的变化与分类准确率的关系
num_predictors = 40
num_epochs = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size, lr = 128, 0.001
train_title_indices, train_labels, valid_title_indices, valid_labels,\
vocab_size, idx_to_char, char_to_idx, label_to_label_group, max_len = load_text()

num_hiddens = 256
loss = nn.CrossEntropyLoss()

# 对model和optimizer进行打包
models = []
optimizers = []
for i in range(num_predictors):
    rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
    model = RNNModel(rnn_layer, n_class=2000)
    models.append(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizers.append(optimizer)

clipping_theta = 1e-2
subsample_rate = 0.8 # bagging的子采样率
train_loss_list, train_acc_list, test_acc_list = [], [], []
for i in range(num_predictors):
    # 先进行子采样
    train_title_indices_sub, train_labels_sub = bootstrap(train_title_indices, train_labels)
    test_acc_for_one_predictor = []
    state = None
    models[i] = models[i].to(device)
    optimizer = optimizers[i]

    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        train_iter = data_iter_random(train_title_indices_sub, train_labels_sub, batch_size=batch_size)
        test_iter = data_iter_random(valid_title_indices, valid_labels, batch_size=batch_size)
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat, _ = models[i](X, None)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            grad_clipping(models[i].parameters(), clipping_theta, device)
            optimizer.step()
            train_loss_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
        test_acc = evaluate_text_accuracy(test_iter, models[i])
        test_acc_for_one_predictor.append(test_acc)
        print("epoch: %d, train loss: %.4f, train acc: %.3f, test acc: %.3f" %
              (epoch + 1, train_loss_sum / n, train_acc_sum / n, test_acc))

    # 训练完之后计算在测试集上的准确率
    test_iter = data_iter_random(valid_title_indices, valid_labels, batch_size=batch_size)
    test_acc = evaluate_text_accuracy_bagging(test_iter, models, i + 1)
    test_acc_list.append(test_acc)
    train_iter = data_iter_random(train_title_indices_sub, train_labels_sub, batch_size=batch_size)
    train_acc = evaluate_text_accuracy_bagging(train_iter, models, i + 1)
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss_sum / n)
    print("test acc: %.3f" % test_acc)
    models[i] = models[i].cpu()


timestamp = time.asctime(time.localtime())
with open("result_model3.txt", 'a') as f:
    f.write(timestamp + "\n")
    # 输出超参数
    f.write("batch_size : " + str(batch_size) + "\n")
    f.write("lr : " + str(lr) + "\n")
    f.write("num hiddens : " + str(num_hiddens) + "\n")
    f.write("clipping theta : " + str(clipping_theta) + "\n")
    f.write("subsample rate : " + str(subsample_rate) + "\n")
    f.write("number predictors : " + str(num_predictors) + "\n")
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
    f.write('\n')
    f.write("test acc for one predictor: \n")
    f.write(','.join(map(str, test_acc_for_one_predictor)))
    f.write('\n\n')

# 输出所有分类器的投票
test_iter = data_iter_random(valid_title_indices, valid_labels, batch_size=batch_size)
voting_matrix, y_true = voting(test_iter, models, len(valid_labels))
voting_matrix = torch.cat([y_true.view(-1, 1), voting_matrix], dim=1)
np.savetxt('result_model3_voting_matrix_%s.txt' % timestamp.replace(':','_'), voting_matrix.detach().numpy(), fmt='%d', delimiter=',')
