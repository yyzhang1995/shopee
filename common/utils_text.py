import torch

__all__ = ['to_onehot', 'grad_clipping', 'evaluate_text_accuracy', 'top_n_accuracy']


def one_hot(x, n_class, dtype=torch.float32):
    x = x.long()
    res = torch.zeros(size=(x.shape[0], n_class), dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


def to_onehot(X, n_class):
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


def evaluate_text_accuracy(data_iter, net, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        if isinstance(net, torch.nn.Module):
            net.eval()
            acc_sum += (net(X.to(device), state=None)[0].argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()
        else:
            try:
                acc_sum += (net(X, state=None, is_training=False).argmax(dim=1) == y).float().sum().item()
            except TypeError:
                acc_sum += (net(X, None).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def top_n_accuracy(data_iter, net, n, num_samples, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    top_n_matrix = torch.zeros(size=(num_samples, n))
    top_n_confidence = torch.zeros(size=(num_samples, n))
    y_true = torch.zeros(num_samples)
    pos = 0
    for X, y in data_iter:
        net.eval()
        y_hat = net(X.to(device), None)[0]
        y_hat_values, y_hat_indices = torch.softmax(y_hat, dim=1).sort(dim=1, descending=True)
        top_n_matrix[pos:(pos + y.shape[0]), :] = y_hat_indices[:, :n]
        top_n_confidence[pos:(pos + y.shape[0]), :] = y_hat_values[:, :n]
        y_true[pos:(pos + y.shape[0])] = y
        pos += y.shape[0]
        net.train()
    return top_n_matrix, top_n_confidence, y_true