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
from petastorm import make_reader, TransformSpec
from petastorm.predicates import in_pseudorandom_split

# Get the Azure ML run object
from azureml.core.run import Run
run = Run.get_context()

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--train_batch_size', default=128, type=int, help='batch size for training')
parser.add_argument('--test_batch_size', default=100, type=int, help='batch size for testing')
parser.add_argument('--input_data', type=str, help='training data')
parser.add_argument('--output_dir', type=str, help='output directory')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# ================== WIP ======================

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


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = LeNet()
net = net.float()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    total = 0
    for batch_idx, sample_batched in enumerate(trainloader):
        inputs = sample_batched['image'].float().to(device)
        print("inputs shape: ", inputs.shape)
        targets = sample_batched['label'].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # ======================= WIP ====================
        total += targets.size(0)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), loss.item()))
            
            # log the loss to the Azure ML run
            run.log('loss', loss.item())

    print("Total samples in train set: {}".format(total))
            
        # ======================= WIP ====================
        

def test(epoch, testloader):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(testloader):
            inputs = sample_batched['image'].float().to(device)
            targets = sample_batched['label'].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.shape[0]
            correct += predicted.eq(targets).sum().item()

    # ======================= WIP ====================
    test_loss /= total

    print("Total samples in testset: {}".format(total))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))
    # ======================= WIP ====================
    
    # Save checkpoint.
    acc = 100. * correct/total
    if acc > best_acc:
        print('Saving..')
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(net, os.path.join(args.output_dir, 'model.pt'))        
        best_acc = acc


# ========================= WIP ==========================

# Data
print('==> Preparing data..')

def _transform_row_train(cifar_row):
    print("cifar row type: ", type(cifar_row))
    print("cifar row: ", cifar_row.keys())

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    result_row = {
        'image': transform_train(transforms.ToPILImage()(cifar_row['image'])),
        'label': cifar_row['label']
    }
    return result_row

def _transform_row_test(cifar_row):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    result_row = {
        'image': transform_test(transforms.ToPILImage()(cifar_row['image'])),
        'label': cifar_row['label']
    }  
    return result_row

transform_spec_train = TransformSpec(_transform_row_train, removed_fields=['filename'])
transform_spec_test = TransformSpec(_transform_row_test, removed_fields=['filename'])

filename = args.input_data
loop_epochs = 10
for epoch in range(loop_epochs):
    with DataLoader(make_reader('file://' + filename, predicate=in_pseudorandom_split([0.75, 0.25], 0, 'filename'), 
                                transform_spec=transform_spec_train), 
                    batch_size=args.train_batch_size) as trainloader:
        train(epoch, trainloader)

    with DataLoader(make_reader('file://' + filename, predicate=in_pseudorandom_split([0.75, 0.25], 1, 'filename'), 
                                transform_spec=transform_spec_test), 
                    batch_size=args.test_batch_size) as testloader:
        test(epoch, testloader)


# ========================= WIP ==========================