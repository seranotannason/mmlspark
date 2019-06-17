# Modifications copyright (C) Microsoft Corporation
# Licensed under the BSD license
# Adapted from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py

'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd

from petastorm.pytorch import DataLoader
from petastorm import make_reader

# Get the Azure ML run object
from azureml.core.run import Run
run = Run.get_context()

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--input_data', type=str, help='training data')
parser.add_argument('--output_dir', type=str, help='output directory')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
# TODO: Add these transforms to Petastorm flow
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# ================== WIP ======================


# Get Dataset from DataStore, utilizing --input_data argument

def get_dataset(filename):
    print("Filename: ", filename)
    # Download the file if it's not present in the datastore
    if not os.path.exists(filename):
        print("Downloading the data from torchvision.datasets...")
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    else:
        print("Use the data from {}".format(filename))
        
        # Get dataframe from Parquet datastore
        with DataLoader(make_reader('file://' + filename), batch_size=128) as train_loader:
            i = 0
            for batch_idx, row in enumerate(train_loader):
                data, target = row['image'], row['label']
                print("data shape: ", data.shape)
                print("label shape: ", target.shape)
                i += 1
            print(i)
            
        # df_dataset = spark.read.parquet(filename).toPandas()        
        # df_dataset = pd.read_parquet(filename)
        
        df_dataset.columns = ['features', 'targets']

        # Split dataframe into train and test
        df_trainset = df_dataset.sample(frac=0.75, random_state=200)
        df_testset = df_dataset.drop(df_trainset.index)

        # Create train and test tensors from Pandas dataframes
        trainset_features = torch.tensor(df_trainset['features'].values)
        trainset_targets = torch.tensor(df_trainset['targets'].values)        
        trainset = torch.utils.data.TensorDataset(trainset_features, trainset_targets)
        
        testset_features = torch.tensor(df_testset['features'].values)
        testset_targets = torch.tensor(df_testset['targets'].values)        
        testset = torch.utils.data.TensorDataset(testset_features, testset_targets)
        
    print("Trainset: ", trainset)
    print("Testset: ", testset)
    return trainset, testset

trainset, testset = get_dataset(args.input_data)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


# ================== WIP ======================

# Create DataLoaders from train and test Datasets
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = LeNet()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # ======================= WIP ====================
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()))
            
            # log the loss to the Azure ML run
            run.log('loss', loss.item())
            
        # ======================= WIP ====================
        

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # ======================= WIP ====================
    test_loss /= len(testloader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    # ======================= WIP ====================
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(net, os.path.join(args.output_dir, 'model.pt'))        
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)