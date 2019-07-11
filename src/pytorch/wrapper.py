import runpy
import sys
import numpy as np
import cv2
import argparse
from torchvision import transforms
from petastorm.pytorch import DataLoader
from petastorm import make_reader, TransformSpec
from petastorm.predicates import in_pseudorandom_split

print("Entering wrapper.py...\n")

parser = argparse.ArgumentParser(description='Wrapper for AML Training Script')
parser.add_argument('--input_data', type=str, help='training data')
parser.add_argument('--train_data_percentage', default=0.75, type=float, help='proportion of training data in the dataset')
parser.add_argument('--train_batch_size', default=128, type=int, help='batch size for training')
parser.add_argument('--test_batch_size', default=100, type=int, help='batch size for testing')
parser.add_argument('--feature_column', type=str, help='name of feature column')
parser.add_argument('--training_script', type=str, help='name of training script')
parser.add_argument('--loop_epochs', default=2, type=int, help='number of epochs')
parser.add_argument('--is_managed', type=bool, help='flag indicating script boilerplate is managed by AML')
args, remaining_args = parser.parse_known_args()

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

# TODO: check that 0 <= train_ratio <= 1
train_ratio = args.train_data_percentage
test_ratio = 1.0 - train_ratio

print("Entering wrapper.py (2)...\n")
# TODO: generalize transform
# TODO: generic predicate
# TODO: add seed
trainloader = DataLoader(make_reader('file://' + args.input_data, predicate=in_pseudorandom_split([train_ratio, test_ratio], 0, args.feature_column), 
                                    transform_spec=transform_spec_train), 
                        batch_size=args.train_batch_size)

testloader =  DataLoader(make_reader('file://' + args.input_data, predicate=in_pseudorandom_split([train_ratio, test_ratio], 1, args.feature_column), 
                                    transform_spec=transform_spec_test), 
                        batch_size=args.test_batch_size)
    
sys.argv = sys.argv[:1] + remaining_args

# TODO: improve this
if not args.is_managed:
    print("Entering custom code...\n")
    runpy.run_path(args.training_script, globals(), run_name="__main__")
else:
    print("Entering managed boilerplate...\n")
    import pytorch_train
    from pytorch_train import train, test
    import mlflow

    with mlflow.start_run() as mlflow_run:
        for epoch in range(args.loop_epochs):
            train(epoch, trainloader)
            trainloader.reader.reset()

            test(epoch, testloader)
            testloader.reader.reset()

    print('==> Saving best model to {}...'.format(pytorch_train.best_model_path))
    pytorch_train.aml_run.upload_folder(pytorch_train.args.output_dir, pytorch_train.best_model_path)
