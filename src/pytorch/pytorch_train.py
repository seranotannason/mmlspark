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

from pytorch_net import ResNet18

import mlflow
import mlflow.pytorch
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle

import cv2
import numpy as np

# Get the Azure ML run object
from azureml.core.run import Run
aml_run = Run.get_context()

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--loop_epochs', default=2, type=float, help='number of epochs')
parser.add_argument('--train_batch_size', default=128, type=int, help='batch size for training')
parser.add_argument('--test_batch_size', default=100, type=int, help='batch size for testing')
parser.add_argument('--input_data', type=str, help='training data')
parser.add_argument('--output_dir', type=str, help='output directory')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
best_model_path = ''

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
    global best_model_path
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

    test_loss /= total
    print("Total samples in testset: {}".format(total))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))
    
    # Save checkpoint.
    acc = 100. * correct/total
    if acc > best_acc:
        best_acc = acc
        # print('Saving..')
        # os.makedirs(args.output_dir, exist_ok=True)
        # torch.save(net.state_dict(), os.path.join(args.output_dir, 'model.pt'))

        # ==================== WIP: Try to save codeless PyTorch model ======================
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
        best_model_path = args.output_dir + str(epoch)
        # ==================== WIP: Try to save codeless PyTorch model ======================

print('==> Preparing data..')

def _transform_row_train(cifar_row):
    image = cv2.imdecode(np.frombuffer(cifar_row['image'], dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    result_row = {
        'image': transform_train(image),
        'label': cifar_row['label']
    }
    return result_row

def _transform_row_test(cifar_row):
    image = cv2.imdecode(np.frombuffer(cifar_row['image'], dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    result_row = {
        'image': transform_test(image),
        'label': cifar_row['label']
    }  
    return result_row

transform_spec_train = TransformSpec(_transform_row_train)
transform_spec_test = TransformSpec(_transform_row_test)

# ==================== WIP: Try to save codeless PyTorch model ======================
with mlflow.start_run() as mlflow_run:
# ==================== WIP: Try to save codeless PyTorch model ======================
    for epoch in range(args.loop_epochs):
        with DataLoader(make_reader('file://' + args.input_data, predicate=in_pseudorandom_split([0.75, 0.25], 0, 'image'), 
                                    transform_spec=transform_spec_train), 
                        batch_size=args.train_batch_size) as trainloader:
            train(epoch, trainloader)

        with DataLoader(make_reader('file://' + args.input_data, predicate=in_pseudorandom_split([0.75, 0.25], 1, 'image'), 
                                    transform_spec=transform_spec_test), 
                        batch_size=args.test_batch_size) as testloader:
            test(epoch, testloader)

    print('==> Saving best model...')
    aml_run.upload_folder(args.output_dir, best_model_path)
