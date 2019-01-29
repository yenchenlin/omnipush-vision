import torch
import torch.utils.data as utils
import torch.optim as optim
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N, D_in, H, D_out = 64, 3, 10, 3
max_epochs = 51


X_train = np.load('./dataset/X_train.npy')
Y_train = np.load('./dataset/Y_train.npy')
X_test = np.load('./dataset/X_test.npy')
Y_test = np.load('./dataset/Y_test.npy')
X_train = [x['action'] for x in X_train]
X_test = [x['action'] for x in X_test]
Y_train = [y['displacement'] for y in Y_train]
Y_test = [y['displacement'] for y in Y_test]
X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
Y_train, Y_test = torch.Tensor(Y_train), torch.Tensor(Y_test)

# Scale up x, y to match the scale of angle
Y_train[:, 0:2] *= 10
Y_test[:, 0:2] *= 10
print("Train input size: {}, Train output size: {}".format(X_train.size(),
                                                           Y_train.size()))
print("Test input size: {}, Test output size: {}".format(X_test.size(),
                                                         Y_test.size()))


params = {'batch_size': 64,
          'shuffle': True,
          'pin_memory': True}
dataset_train = utils.TensorDataset(X_train, Y_train)
dataloader_train = utils.DataLoader(dataset_train, **params)
dataset_test = utils.TensorDataset(X_test, Y_test)
dataloader_test = utils.DataLoader(dataset_test, **params)


model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, D_out),
        ).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)


n_batch_train = X_train.size()[0] // N
for epoch in range(max_epochs):
    running_error = torch.tensor([0.0, 0.0, 0.0]).to(device)
    running_loss = 0.0
    model.train()
    for i, (x, y) in enumerate(dataloader_train):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        error = torch.mean((y_pred - y).abs_(), 0)
        loss = torch.mean((y_pred - y) ** 2)
        loss.backward()
        optimizer.step()
        running_error += error.detach()
        running_loss += loss.detach()

    if epoch % 5 == 0:
        print("Epoch {}, Loss: {}\nError: {}".format(
            epoch, running_loss/n_batch_train, running_error/n_batch_train))

    if epoch % 5 == 0:
        eval_error = torch.tensor([0.0, 0.0, 0.0]).to(device)
        model.eval()
        for i, (x, y) in enumerate(dataloader_test):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            error = torch.sum((y_pred - y).abs_(), 0).detach()
            eval_error += error

        print("Eval: Epoch {}, Error: {}".format(
          epoch, eval_error/Y_test.size()[0]))
