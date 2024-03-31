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

import pytorch_ssim

sub = 'sub04'
ses = 'ses01'
roi = 'v1.v2.v3'

label = ': ens ' + sub + ' ' + ses + ' ' + roi
batch_size = 5
epochs = 10000
lr = 1e-4
wd = 0e-4
interval = 100
dropout_p = .0
NUM_CLASSES = 8
NUM_ROUTING_ITERATIONS = 3

device = torch.device('cuda')


class FCNet(nn.Module):
    def __init__(self, input_size):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)

    def forward(self, x):
        x = self.fc1(x)
        x = f.relu(x)
        x = self.fc2(x)
        x = f.relu(x)
        x = self.fc3(x)
        x = f.relu(x)
        x = self.fc4(x)
        return x


def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm)


def train(epoch):
    # Set model to training mode
    ensemble.train()
    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)
        # Zero gradient buffers
        optimizer.zero_grad()
        # Pass data through the network
        classes, reconstructions = ensemble(data)
        # Calculate loss
        reconstructions = reconstructions.view(reconstructions.size()[0], 1, 28, 28)
        target = target.view(target.size()[0], 1, 28, 28)
        loss = -ssim_loss(target, reconstructions)

        l2_reg = None
        for name, param in ensemble.named_parameters():
            if name == 'fcnet.fc1.bias' or name == 'fcnet.fc1.weight' or \
                    name == 'fcnet.fc2.bias' or name == 'fcnet.fc2.weight' or \
                    name == 'fcnet.fc3.bias' or name == 'fcnet.fc3.weight':
                if l2_reg is None:
                    l2_reg = 0.5 * param.norm(2) ** 2
                else:
                    l2_reg += 0.5 * param.norm(2) ** 2
        loss += l2_reg * wd
        # Backpropagate
        loss.backward()
        # Update weights
        optimizer.step()

    if epoch % (interval / 10) == 1:
        writer.add_scalar('loss/train', -loss.data.item(), epoch)
        print('Epoch: {} Train Loss: {:.6f}'.format(epoch, -loss.data.item()))


def validation(epoch):
    ensemble.eval()
    val_loss = 0

    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        classes, reconstructions = ensemble(data)
        # Calculate loss
        target = target.view(target.size()[0], 1, 28, 28)
        reconstructions = reconstructions.view(reconstructions.size()[0], 1, 28, 28)
        val_loss -= ssim_loss(target, reconstructions).data.item()

    val_loss /= len(validation_loader)

    if epoch % (interval / 10) == 1:
        writer.add_scalar('loss/valid', -val_loss, epoch)
        print('Epoch: {} Valid Loss: {:.6f}'.format(epoch, -val_loss))
    return reconstructions


def validation_on_train(epoch):
    ensemble.eval()
    val_loss = 0
    for data, target in train_for_validation_loader:
        data = data.to(device)
        target = target.to(device)
        classes, reconstructions = ensemble(data)

    return reconstructions


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = func.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size())).to(device)
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)
        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        # x = func.relu(self.conv1(x), inplace=True)
        # x = self.primary_capsules(x)
        # x = self.digit_capsules(x).squeeze().transpose(0, 1)
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = func.softmax(classes, dim=-1)
        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(NUM_CLASSES)).to(device).index_select(dim=0, index=max_length_indices.data)
        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        return classes, reconstructions


def get_iterator():
    images = []
    labels = []
    files = glob.glob('stim/*')
    files = np.sort(files)
    for i in range(0, len(files)):
        im = Image.open(files[i])
        im.load()
        images.append(np.asarray(im))
        labels.append(i)
    images = np.asarray(images)
    images = images[:, 2:30, 2:30]
    labels = np.asarray(labels)
    images = np.expand_dims(images, axis=1)
    return images


class Ensemble(nn.Module):
    def __init__(self, fcnet, capsnet):
        super(Ensemble, self).__init__()
        self.fcnet = fcnet
        self.capsnet = capsnet

    def forward(self, x):
        digitcaps = self.fcnet(x)
        digitcaps = digitcaps.reshape((-1, 8, 16))
        classes, reconstructions = self.capsnet(squash(digitcaps))

        return classes, reconstructions


x = np.genfromtxt('./probe/prob.' + sub + '.' + ses + '.' + roi + '.txt', delimiter=',')
y = np.load('digitcaps.npy')
y = y.reshape((128, 8 * 16))
input_size = x.shape[1]

y_im = torch.tensor((get_iterator()) / 255.0)

writer = SummaryWriter(comment=label)
rnd = np.random.permutation(len(x))
x = x[rnd, :]
y = y[rnd, :]
y_im = y_im[rnd, :, :, :]

x = x - 100
x_train = torch.from_numpy(x[:98, :])
y_train = torch.from_numpy(y[:98, :])
x_valid = torch.from_numpy(x[98:, :])
y_valid = torch.from_numpy(y[98:, :])

y_im_train = y_im[:98, :, :, :]
y_im_valid = y_im[98:, :, :, :]

train_dataset = TensorDataset(x_train.float(), y_im_train.float())
validation_dataset = TensorDataset(x_valid.float(), y_im_valid.float())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
train_for_validation_loader = DataLoader(dataset=train_dataset, batch_size=98, shuffle=False)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=30, shuffle=False)

for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

fcnet = FCNet(input_size).to(device)
optimizer = torch.optim.Adam(fcnet.parameters(), lr=lr)
criterion = nn.MSELoss()
ssim_loss = pytorch_ssim.SSIM(window_size=5)

capsnet = CapsuleNet().to(device)
capsnet.load_state_dict(torch.load('stage1/epochs/epoch_500.pt'))
for param in capsnet.parameters():
    param.requires_grad = False

ensemble = Ensemble(fcnet=fcnet, capsnet=capsnet)

rndTrainArg = np.argsort(rnd[:30])
rndValidArg = np.argsort(rnd[98:])

ground_truth_valid = y_im_valid.reshape((30, 1, 28, 28)).cpu().detach().double()
ground_truth_train = y_im_train.reshape((98, 1, 28, 28)).cpu().detach().double()

for epoch in range(1, epochs + 1):
    train(epoch)
    reconValid = validation(epoch)
    reconTrain = validation_on_train(epoch)

    if epoch % interval == 1:
        reconValid = reconValid.reshape((30, 1, 28, 28)).cpu().detach().double()

        grid = make_grid(torch.cat((ground_truth_valid[rndValidArg, :, :, :],
                                    reconValid[rndValidArg, :, :, :]), 0),
                         nrow=30, normalize=True, range=(0, 1)).numpy()
        writer.add_image('Valid', grid, epoch)

        # SSIM
        A = ground_truth_valid.float().cpu().detach().numpy()
        B = reconValid.float().cpu().detach().numpy()
        ssim = []
        for i in range(len(ground_truth_valid)):
            ssim.append(compare_ssim(A[i, 0, :, :], B[i, 0, :, :]))
        ssimArgSort = np.argsort(ssim)
        writer.add_scalar('ssim/valid', np.mean(ssim), epoch)
        writer.add_scalar('ssimMax/valid', np.max(ssim), epoch)

        grid = make_grid(torch.cat((ground_truth_valid[rndValidArg, :, :],
                                    reconValid[rndValidArg, :, :]), 0),
                         nrow=30, normalize=True, range=(0, 1)).numpy()
        writer.add_image('ssimValid', grid, epoch)

        reconTrain = reconTrain.reshape((98, 1, 28, 28))
        reconTrain = reconTrain[:30, :, :, :].cpu().detach().double()
        grid = make_grid(torch.cat((ground_truth_train[rndTrainArg, :, :, :],
                                    reconTrain[rndTrainArg, :, :, :]), 0),
                         nrow=30, normalize=True, range=(0, 1)).numpy()
        writer.add_image('Train', grid, epoch)

        # SSIM
        A = ground_truth_train[:30, :, :, :].float().cpu().detach().numpy()
        B = reconTrain.float().cpu().detach().numpy()
        ssim = []
        for i in range(len(A)):
            ssim.append(compare_ssim(A[i, 0, :, :], B[i, 0, :, :]))
        ssimArgSort = np.argsort(ssim)
        # ssimArgSort = ssimArgSort[::-1]
        writer.add_scalar('ssim/train', np.mean(ssim), epoch)
        writer.add_scalar('ssimMax/train', np.max(ssim), epoch)

        grid = make_grid(torch.cat((ground_truth_train[ssimArgSort, :, :, :].double(),
                                    reconTrain[ssimArgSort, :, :, :].double()), 0),
                         nrow=30, normalize=True, range=(0, 1)).numpy()
        writer.add_image('ssimTrain', grid, epoch)
