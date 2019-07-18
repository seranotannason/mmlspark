# Modifications copyright (C) Microsoft Corporation
# Licensed under the BSD license
# Adapted from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py

'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import sys
import argparse
import cv2
import numpy as np
import pandas as pd

from petastorm.pytorch import DataLoader
from petastorm import make_reader, TransformSpec
from petastorm.predicates import in_pseudorandom_split

import mlflow
import mlflow.pytorch
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle

from azureml.core.run import Run

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

'''Train CIFAR10 with PyTorch.'''

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--output_dir', type=str, help='output directory')
parser.add_argument('--loop_epochs', default=2, type=int, help='number of epochs')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

aml_run = Run.get_context()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
best_model = None

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
        
        total += targets.size(0)
        running_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            # log the loss to the Azure ML run
            aml_run.log('loss', running_loss/(batch_idx + 1))
 
            print('Train Epoch: {} [{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), loss.item()))

    print("Total samples in train set: {}".format(total))
        

def test(epoch, testloader):
    global best_acc
    global best_model
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for _, sample_batched in enumerate(testloader):
            inputs = sample_batched['image'].float().to(device)
            targets = sample_batched['label'].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.shape[0]
            correct += predicted.eq(targets).sum().item()

    test_loss /= total
    print("Total samples in testset: {}\n".format(total))
    print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, total,
        100. * correct / total))
    
    # Save checkpoint.
    acc = 100. * correct/total
    if acc > best_acc:
        best_acc = acc
        
        pytorch_index = "https://download.pytorch.org/whl/"
        pytorch_version = "cpu/torch-1.1.0-cp36-cp36m-linux_x86_64.whl"
        deps = [
            "cloudpickle=={}".format(cloudpickle.__version__),
            pytorch_index + pytorch_version,
            "torchvision=={}".format(torchvision.__version__),
            "Pillow=={}".format("6.0.0")
        ]

        model_env = _mlflow_conda_env(additional_pip_deps=deps)
        mlflow.pytorch.save_model(net, args.output_dir + str(epoch), conda_env=model_env)
        best_model = net
        
def get_best_model():
    return best_model    

if __name__ == "__main__":
    with mlflow.start_run() as mlflow_run:
        for epoch in range(args.loop_epochs):
            train(epoch, trainloader)
            trainloader.reader.reset()

            test(epoch, testloader)
            testloader.reader.reset()

        mlflow.pytorch.log_model(best_model, args.output_dir)
        