import glob

import numpy as np
import torch
import torch.nn.functional as f
import torch.nn.functional as func
from PIL import Image
from skimage.measure import compare_ssim
from tensorboardX import SummaryWriter
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid

from pytorchtools import EarlyStopping




batch_size = 32
epochs = 10000
lr = 1e-4
interval = 100
dropout_p = .0
NUM_CLASSES = 8
roi = 'v1.v2.v3'
label = 'all class ' + roi
device = torch.device('cuda')


class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)

    def forward(self, x):
        x = self.fc1(x)
        x = f.relu(x)
        # x = f.dropout(x, p=dropout_p, training=self.training)
        # x = self.bn1(x)
        x = self.fc2(x)
        x = f.relu(x)
        # x = f.dropout(x, p=dropout_p, training=self.training)
        # x = self.bn2(x)
        x = self.fc3(x)
        x = f.relu(x)
        # x = f.dropout(x, p=dropout_p, training=self.training)
        # x = self.bn3(x)
        x = self.fc4(x)
        # output = f.log_softmax(x, dim=1)
        return x


def train(epoch):
    # Set model to training mode
    model.train()
    correct = 0
    total = 0
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
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        print((predicted == target).float().size(), 'type:', (predicted == target).float().type())
        correct += (predicted == target).float().sum()
        # Backpropagate
        loss.backward()
        # Update weights
        optimizer.step()

    if epoch % (interval/10) == 1:
        writer.add_scalar('loss/train', loss.data.item(), epoch)
        print('Epoch: {} ,Train Loss: {:.6f} ,Train acc: {:.2f}'.format(epoch, loss.data.item(), 100*correct/total))
        writer.add_scalar('acc/valid', 100*correct/total, epoch)



def validation(epoch):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    _, predicted = torch.max(output, 1)
    if epoch % (interval/10) == 1:
        writer.add_scalar('loss/valid', val_loss, epoch)
        writer.add_scalar('acc/valid', 100*correct/total, epoch)
        print('Epoch: {} ,Valid Loss: {:.6f} ,Valid acc: {:.2f}'.format(epoch, val_loss, 100*correct/total))

    return output, val_loss


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
input_size = 100

y0 = []
y0.append([1] * 16)
y0.append([2] * 16)
y0.append([3] * 16)
y0.append([4] * 16)
y0.append([5] * 16)
y0.append([6] * 16)
y0.append([7] * 16)
y0.append([8] * 16)
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
validation_dataset = TensorDataset(x_valid.float(), y_valid.long())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=valid_num, shuffle=False)

for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

model = Net(input_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

print(model)


early_stopping = EarlyStopping(patience=500, verbose=False, filename='checkpoint.pt')

for epoch in range(1, epochs + 1):
    train(epoch)
    outValid, val_loss = validation(epoch)

    # early_stopping(val_loss, model)
    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     break
