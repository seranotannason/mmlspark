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
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--loop_epochs', default=5, type=float, help='number of epochs')
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
'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

# ================== WIP ======================


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()
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
    running_loss = 0.0
    for batch_idx, sample_batched in enumerate(trainloader):
        inputs = sample_batched['image'].float().to(device)
        targets = sample_batched['label'].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # ======================= WIP ====================
        total += targets.size(0)
        running_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            # log the loss to the Azure ML run
            run.log('loss', running_loss/(batch_idx + 1))
 
            print('Train Epoch: {} [{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), loss.item()))
            

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

transform_spec_train = TransformSpec(_transform_row_train)
transform_spec_test = TransformSpec(_transform_row_test)

for epoch in range(args.loop_epochs):
    with DataLoader(make_reader('file://' + args.input_data, predicate=in_pseudorandom_split([0.75, 0.25], 0, 'image'), 
                                transform_spec=transform_spec_train), 
                    batch_size=args.train_batch_size) as trainloader:
        train(epoch, trainloader)

    with DataLoader(make_reader('file://' + args.input_data, predicate=in_pseudorandom_split([0.75, 0.25], 1, 'image'), 
                                transform_spec=transform_spec_test), 
                    batch_size=args.test_batch_size) as testloader:
        test(epoch, testloader)


# ========================= WIP ==========================