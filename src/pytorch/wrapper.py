import runpy
import sys
import numpy as np
import cv2
import argparse
import pickle
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
parser.add_argument('--train_preprocessor_filename', default='train_preprocessor', type=str, help='file that contains pickled preprocessing function for training data')
parser.add_argument('--val_preprocessor_filename', default='val_preprocessor', type=str, help='file that contains pickled preprocessing function for validation data')
args, remaining_args = parser.parse_known_args()

with open(args.train_preprocessor_filename, mode='rb') as train_preprocessor_file:
    train_preprocessor = pickle.loads(train_preprocessor_file.read())

with open(args.val_preprocessor_filename, mode='rb') as val_preprocessor_file:
    val_preprocessor = pickle.loads(val_preprocessor_file.read())

transform_spec_train = TransformSpec(train_preprocessor)
transform_spec_val = TransformSpec(val_preprocessor)

# TODO: check that 0 <= train_ratio <= 1
train_ratio = args.train_data_percentage
test_ratio = 1.0 - train_ratio

print("Entering wrapper.py (2)...\n")
# TODO: generic predicate
# TODO: add seed
# TODO: add ID column, change feature_column here
trainloader = DataLoader(make_reader('file://' + args.input_data, predicate=in_pseudorandom_split([train_ratio, test_ratio], 0, args.feature_column), 
                                    transform_spec=transform_spec_train), 
                        batch_size=args.train_batch_size)

testloader =  DataLoader(make_reader('file://' + args.input_data, predicate=in_pseudorandom_split([train_ratio, test_ratio], 1, args.feature_column), 
                                    transform_spec=transform_spec_val), 
                        batch_size=args.test_batch_size)
    
sys.argv = sys.argv[:1] + remaining_args

# TODO: check with AK and improve this
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
