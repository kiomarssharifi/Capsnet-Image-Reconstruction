import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

n_epochs = 10000
batch_size = 5
learning_rate = 1e-4
momentum = 0.5
log_interval = 100
dropout_p = .0

random_seed = 1
torch.manual_seed(random_seed)
roi = 'v1.v2.v3'
label = 'all class ' + roi
device = torch.device('cuda')


class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=dropout_p, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=dropout_p, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            writer.add_scalar('loss/train', loss.item(), epoch)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(epoch):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    writer.add_scalar('loss/valid', test_loss, epoch)
    writer.add_scalar('acc/valid', 100. * correct / len(test_loader.dataset), epoch)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


subs = ['sub01', 'sub02', 'sub03', 'sub04']
sess = ['ses01', 'ses02']
x = []
for iSub in range(len(subs)):
    for iSes in range(len(sess)):
        sub = subs[iSub]
        ses = sess[iSes]
        x.append(np.genfromtxt('./probe/prob.' + sub + '.' + ses + '.' + roi + '.txt', delimiter=','))
x = np.asarray(x)
x = x.reshape((-1, 100))
input_size = x.shape[1]

y0 = [[0] * 16, [1] * 16, [2] * 16, [3] * 16, [4] * 16, [5] * 16, [6] * 16, [7] * 16]
y0 = np.asarray(y0).reshape(-1, )
y = y0
for _ in range(7):
    y = np.concatenate([y, y0], axis=0)

writer = SummaryWriter(comment=label)
rnd = np.random.permutation(len(x))
x = x[rnd, :]
y = y[rnd]

x = x - 100
train_num = int(len(x) * 0.7)
valid_num = len(x) - train_num
x_train = torch.from_numpy(x[:train_num, :])
y_train = torch.from_numpy(y[:train_num])
x_valid = torch.from_numpy(x[train_num:, :])
y_valid = torch.from_numpy(y[train_num:])

train_dataset = TensorDataset(x_train.float(), y_train.long())
test_dataset = TensorDataset(x_valid.float(), y_valid.long())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=valid_num, shuffle=False)

network = Net(input_size).to(device)
optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=1e-1)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

test(0)
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test(epoch)
