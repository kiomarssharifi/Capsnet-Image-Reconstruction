import os

import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
import torch.nn.functional as f
from ray import tune
from ray.tune.suggest import BasicVariantGenerator
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorchtools import EarlyStopping

os.system("pkill -9 'ray'")
os.system("rm -rf ~/ray_results/")

device = torch.device('cuda')


class Net(nn.Module):
    def __init__(self, input_size, config):
        super(Net, self).__init__()
        # self.sq = config["sq"]
        # self.do = config["do"]
        # self.bn = config["bn"]
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.fc1 = nn.Linear(input_size, self.n1)
        self.bn1 = nn.BatchNorm1d(self.n1)
        self.fc2 = nn.Linear(self.n1, self.n2)
        self.fc3 = nn.Linear(self.n2, self.n3)
        self.fc4 = nn.Linear(self.n3, 128)


    def forward(self, x):
        x = self.fc1(x)
        x = f.relu(x)
        # x = f.dropout(x, p=self.do, training=self.training)
        # if self.bn == 1:
        #     x = self.bn1(x)
        x = self.fc2(x)
        x = f.relu(x)
        # x = f.dropout(x, p=self.do, training=self.training)
        x = self.fc3(x)
        x = f.relu(x)
        # x = f.dropout(x, p=self.do, training=self.training)
        x = self.fc4(x)
        # if self.sq == 1:
        #     x = x.view(-1, 8, 16)
        #     x = self.squash(x)
        #     x = x.view(-1, 8 * 16)
        return x

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)


def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm)


def train(model, optimizer, criterion, train_loader, wd):
    # Set model to training mode
    model.train()

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)
        # Zero gradient buffers
        optimizer.zero_grad()
        # Pass data through the network
        output = model(data)
        # Calculate loss
        loss = criterion(output, target)

        l2_reg = None
        for name, param in model.named_parameters():
            if not (name == 'fc3.bias' or name == 'fc3.weight'):
                if l2_reg is None:
                    l2_reg = 0.5 * param.norm(2)**2
                else:
                    l2_reg += 0.5 * param.norm(2)**2
        loss += l2_reg * wd
        # Backpropagate
        loss.backward()
        # Update weights
        optimizer.step()

    return loss.data.item()


def validation(model, criterion, validation_loader):
    model.eval()
    val_loss = 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()

    val_loss /= len(validation_loader)

    return val_loss


def train_fmri(config, reporter):
    x = np.genfromtxt(dir + 'probe/prob.' + sub + '.' + ses + '.' + roi + '.txt', delimiter=',')
    y = np.load(dir + 'digitcaps.npy')
    y = y.reshape((128, 8*16))
    input_size = x.shape[1]
    rnd = np.random.permutation(len(x))
    x = x[rnd, :]
    y = y[rnd, :]
    x = x - 100
    x_train = torch.from_numpy(x[:98, :])
    y_train = torch.from_numpy(y[:98, :])
    x_valid = torch.from_numpy(x[98:, :])
    y_valid = torch.from_numpy(y[98:, :])
    train_dataset = TensorDataset(x_train.float(), y_train.float())
    validation_dataset = TensorDataset(x_valid.float(), y_valid.float())
    train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=30, shuffle=False)
    model = Net(input_size, config=config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=100, verbose=False, filename='checkpoint.pt')
    for epoch in range(1, epochs + 1):
        train(model=model, optimizer=optimizer, criterion=criterion, train_loader=train_loader, wd=config["wd"])
        val_loss = validation(model=model, criterion=criterion, validation_loader=validation_loader)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        reporter(training_iteration=epoch, mean_loss=val_loss)


sub = 'sub03'
ses = 'ses01'
roi = 'v1.v2.v3'
dir = '/media/hdd2/users/sharifi/capsnet/'
epochs = 10000
bs = 10
n1 = 256
n2 = 512
n3 = 256
lr = 1e-4

ray.init()

searcher = BasicVariantGenerator()

analysis = tune.run(
    train_fmri,
    name="ax",
    num_samples=10000,
    config={
            "wd": tune.sample_from(lambda spec: np.random.uniform(1e-6, 1e-2)),
        },
    resources_per_trial={
        "cpu": 1,
        "gpu": 1 / 8
    })

print("Best config: ", analysis.get_best_config(metric="mean_loss"))

df = analysis.dataframe()

plt.plot(df['config/wd'], df['mean_loss'], '.')
plt.show()
