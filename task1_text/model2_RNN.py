import torch
from torch import nn, optim
from common.utils_text import to_onehot, grad_clipping, evaluate_text_accuracy, top_n_accuracy
from common.utils_load_text import load_text, data_iter_random
import time
import numpy as np
# 可以尝试增加dropout

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, n_class):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, n_class)
        self.state = None

    def forward(self, inputs, state):
        X = to_onehot(inputs, self.vocab_size)
        # X的维度是num_stpes, batch_size, vocal_size
        Y, self.state = self.rnn(torch.stack(X), state)
        # Y的维度对应是num_steps, batch_size, hidden_size
        # 接下来对Y做一个池化
        Y = Y.mean(dim=0)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size, lr = 128, 0.001
train_title_indices, train_labels, valid_title_indices, valid_labels,\
vocab_size, idx_to_char, char_to_idx, label_to_label_group, max_len = load_text()

num_hiddens = 256
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
model = RNNModel(rnn_layer, n_class=2000)

# 确保模型可以正常工作
# for X, y in train_iter:
#     X1, _ = model(X, None)
#     print(X1.shape)
#     break

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

clipping_theta = 1e-2
model = model.to(device)
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


def test():
    num_hiddens = 256
    rnn_layer = nn.RNN(input_size=500, hidden_size=num_hiddens)
    model = RNNModel(rnn_layer, 500)
    X = torch.tensor([[1, 2, 3, 4, 5],
                      [5, 6, 7, 8, 9]])
    Y, state = model(X, None)
    print(Y.shape)


if __name__ == '__main__':
    pass