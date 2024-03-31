import glob
import numpy as np
import torch
import torch.nn.functional as f
import torch.nn.functional as func
from PIL import Image
from skimage.metrics import structural_similarity
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from pytorchtools import EarlyStopping
import scipy.io

result = []

epochs = 10000
interval = 100
dropout_p = .0
NUM_CLASSES = 8
NUM_ROUTING_ITERATIONS = 3
device = torch.device('cuda')
# random_seed = 1
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# np.random.seed(random_seed)

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)

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
        # x = x.view(-1, 8, 16)
        # x = self.squash(x)
        # x = x.view(-1, 8 * 16)
        return x

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)


def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm)


def train(epoch):
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

    if epoch % (interval/1000) == 1:
        print('Epoch: {} Train Loss: {:.6f}'.format(epoch, loss.data.item()))


def validation(epoch):
    model.eval()
    val_loss = 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()

    val_loss /= len(validation_loader)

    if epoch % (interval/1000) == 1:
        print('Epoch: {} Valid Loss: {:.6f}'.format(epoch, val_loss))
    return output, val_loss


def validation_on_train(epoch):
    model.eval()
    val_loss = 0
    for data, target in train_for_validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)

    return output

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
    data0 = []
    labels = []
    char_labels = []
    files = glob.glob('stim/*')
    files = np.sort(files)
    for i in range(0, len(files)):
        im = Image.open(files[i])
        im.load()
        data0.append(np.asarray(im))
        labels.append(i)
        name = files[i]
        char_labels.append(int(name[5]))
    data0 = np.asarray(data0)
    data0 = data0[:, 2:30, 2:30]
    labels = np.asarray(labels)
    data0 = np.expand_dims(data0, axis=1)
    data = data0
    for _ in range(7):
        data = np.concatenate([data, data0], axis=0)
    return data, char_labels


fileIndex = 6
lr = 1e-5
cntAll = 3*4*2*4*5
cnt = 0
for bs in [8, 16, 32, 64, 96]:
    for wd in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0]:
        for roi in ['v1', 'v1.v2', 'v1.v2.v3']:
            for subs in [['sub01'], ['sub02'], ['sub03'], ['sub04']]:
                for sess in [['ses01'], ['ses02']]:
                    cnt = cnt + 1
                    print(int(cnt/cntAll*100))
                    x = []
                    for iSub in range(len(subs)):
                        for iSes in range(len(sess)):
                            sub = subs[iSub]
                            ses = sess[iSes]
                            x.append(np.genfromtxt('./probe/prob.' + sub + '.' + ses + '.' + roi + '.txt', delimiter=','))
                    x = np.asarray(x)
                    x = x.reshape((-1, 100))

                    y0 = np.load('digitcaps.npy')
                    y0 = y0.reshape((128, 8*16))
                    y = y0
                    for i in range(7):
                        y = np.concatenate([y, y0], axis=0)
                    input_size = x.shape[1]

                    rnd = np.random.permutation(len(x))
                    x = x[rnd, :]
                    y = y[rnd, :]

                    x = x - 100
                    train_num = int(len(x) * 0.75)
                    valid_num = len(x) - train_num
                    x_train = torch.from_numpy(x[:train_num, :])
                    y_train = torch.from_numpy(y[:train_num, :])
                    x_valid = torch.from_numpy(x[train_num:, :])
                    y_valid = torch.from_numpy(y[train_num:, :])

                    train_dataset = TensorDataset(x_train.float(), y_train.float())
                    validation_dataset = TensorDataset(x_valid.float(), y_valid.float())

                    train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
                    train_for_validation_loader = DataLoader(dataset=train_dataset, batch_size=train_num, shuffle=False)
                    validation_loader = DataLoader(dataset=validation_dataset, batch_size=valid_num, shuffle=False)

                    model = Net(input_size).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    criterion = nn.MSELoss()

                    capsnet = CapsuleNet().to(device)
                    capsnet.load_state_dict(torch.load('stage1/epochs/epoch_500.pt'))

                    rndTrainArg = np.argsort(rnd[:train_num])
                    rndValidArg = np.argsort(rnd[-valid_num:])
                    rndTrain = rnd[:train_num]
                    rndValid = rnd[-valid_num:]
                    data, char_labels = get_iterator()
                    char_labels = np.asarray(char_labels)
                    ground_truth_train = torch.tensor((data[rnd[:train_num], :, :]) / 255.0)
                    ground_truth_valid = torch.tensor((data[rnd[-valid_num:], :, :]) / 255.0)
                    char_labels_train = char_labels[rnd[:train_num]]
                    char_labels_valid = char_labels[rnd[-valid_num:]]

                    early_stopping = EarlyStopping(patience=500, verbose=False, filename='checkpoint.pt')

                    for epoch in range(1, epochs + 1):
                        train(epoch)

                        # early_stopping(val_loss, model)
                        # if early_stopping.early_stop:
                        #     print("Early stopping")
                        #     break

                        if epoch == epochs:
                            outValid, val_loss = validation(epoch)
                            outTrain = validation_on_train(epoch)

                            outValid = outValid.reshape((valid_num, 8, 16))
                            reconstructions = capsnet.forward(torch.tensor(squash(outValid)).to(device))[1]
                            reconstruction_valid = reconstructions.cpu().view_as(ground_truth_valid).data

                            # SSIM
                            ground_truth_valid_numpy = ground_truth_valid.float().cpu().detach().numpy()
                            reconstruction_valid_numpy = reconstruction_valid.float().cpu().detach().numpy()
                            ssim = []
                            for i in range(len(ground_truth_valid)):
                                ssim.append(structural_similarity(ground_truth_valid_numpy[i, 0, :, :], reconstruction_valid_numpy[i, 0, :, :]))

                            for i in range(len(ground_truth_valid_numpy)):
                                A = ground_truth_valid_numpy[i, 0, :, :]
                                B = reconstruction_valid_numpy[i, 0, :, :]
                                result.append({'ssim': ssim[i],
                                               'mse': ((A - B)**2).mean(axis=None),
                                               'pcc': np.corrcoef(A.flat, B.flat)[0, 1],
                                               'index': rndValid[i],
                                               'char_index': char_labels_valid[i],
                                               'kind_index': int(rndValid[i] % 16) + 1,
                                               'type': 'valid',
                                               'roi': roi,
                                               'sub': subs,
                                               'sess': sess,
                                               'value': 'good' if char_labels_valid[i] <= 4 else 'bad',
                                               'original': A,
                                               'reconstruction': B,
                                               'wd': wd,
                                               'lr': lr,
                                               'bs': bs
                                               })

                            outTrain = outTrain.reshape((train_num, 8, 16))
                            reconstructions = capsnet.forward(torch.tensor(squash(outTrain)).to(device))[1]
                            reconstruction_train = reconstructions.cpu().view_as(ground_truth_train).data

                            # SSIM
                            ground_truth_train_numpy = ground_truth_train.float().cpu().detach().numpy()
                            reconstruction_train_numpy = reconstruction_train.float().cpu().detach().numpy()
                            ssim = []
                            for i in range(len(ground_truth_train)):
                                ssim.append(structural_similarity(ground_truth_train_numpy[i, 0, :, :], reconstruction_train_numpy[i, 0, :, :]))

                            for i in range(len(ground_truth_train_numpy)):
                                A = ground_truth_train_numpy[i, 0, :, :]
                                B = reconstruction_train_numpy[i, 0, :, :]
                                result.append({'ssim': ssim[i],
                                               'mse': ((A - B) ** 2).mean(axis=None),
                                               'pcc': np.corrcoef(A.flat, B.flat)[0, 1],
                                               'index': rndTrain[i],
                                               'char_index': char_labels_train[i],
                                               'kind_index': int(rndTrain[i] % 16) + 1,
                                               'type': 'train',
                                               'roi': roi,
                                               'sub': subs,
                                               'sess': sess,
                                               'value': 'good' if char_labels_train[i] <= 4 else 'bad',
                                               'original': A,
                                               'reconstruction': B,
                                               'wd': wd,
                                               'lr': lr,
                                               'bs': bs
                                               })

scipy.io.savemat('result' + str(fileIndex) + '.mat', mdict={'result': result})
