import torch


def top_n_accuracy_image(data_iter, net, n, num_samples,
                         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    top_n_matrix = torch.zeros(size=(num_samples, n))
    top_n_confidence = torch.zeros(size=(num_samples, n))
    y_true = torch.zeros(num_samples)
    pos = 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()
            y_hat = net(X.to(device))
            y_hat_values, y_hat_indices = torch.softmax(y_hat, dim=1).sort(dim=1, descending=True)
            top_n_matrix[pos:(pos + y.shape[0]), :] = y_hat_indices[:, :n]
            top_n_confidence[pos:(pos + y.shape[0]), :] = y_hat_values[:, :n]
            y_true[pos:(pos + y.shape[0])] = y
            pos += y.shape[0]
            net.train()
    return top_n_matrix, top_n_confidence, y_true
