# TODO: have this be autogenerated by PyTorchEstimator.py
import pyarrow
import runpy
import sys
import numpy as np
import cv2
import argparse
import pickle
import importlib
import mlflow
from torchvision import transforms
from petastorm.pytorch import DataLoader
from petastorm import make_reader, TransformSpec
from petastorm.predicates import in_pseudorandom_split

print("Entering wrapper.py...\n")

parser = argparse.ArgumentParser(description='Wrapper for AML Training Script')
parser.add_argument('--input_data', type=str, help='training data')
parser.add_argument('--output_dir', type=str, help='output directory')
parser.add_argument('--train_data_percentage', default=0.75, type=float, help='proportion of training data in the dataset')
parser.add_argument('--train_batch_size', default=128, type=int, help='batch size for training')
parser.add_argument('--test_batch_size', default=100, type=int, help='batch size for testing')
parser.add_argument('--feature_column', type=str, help='name of feature column')
parser.add_argument('--training_script_name', type=str, help='name of training script')
parser.add_argument('--loop_epochs', default=2, type=int, help='number of epochs')
parser.add_argument('--is_managed', default=True, type=lambda x: (str(x).lower()=='true'), help='flag indicating script boilerplate is managed by AML')
parser.add_argument('--train_preprocessor_filename', default='train_preprocessor', type=str, help='file that contains pickled preprocessing function for training data')
parser.add_argument('--val_preprocessor_filename', default='val_preprocessor', type=str, help='file that contains pickled preprocessing function for validation data')
args, remaining_args = parser.parse_known_args()

with open(args.train_preprocessor_filename, mode='rb') as train_preprocessor_file:
    train_preprocessor = pickle.loads(train_preprocessor_file.read())

with open(args.val_preprocessor_filename, mode='rb') as val_preprocessor_file:
    val_preprocessor = pickle.loads(val_preprocessor_file.read())

transform_spec_train = TransformSpec(train_preprocessor)
transform_spec_val = TransformSpec(val_preprocessor)

if args.train_data_percentage < 0 or args.train_data_percentage > 1:
    raise Exception("Invalid train data percentage")
train_ratio = args.train_data_percentage
test_ratio = 1.0 - train_ratio

print("Entering wrapper.py (2)...\n")
# TODO: generic predicate
trainloader = DataLoader(make_reader('file://' + args.input_data, predicate=in_pseudorandom_split([train_ratio, test_ratio], 0, args.feature_column), 
                                    transform_spec=transform_spec_train), 
                        batch_size=args.train_batch_size)

testloader =  DataLoader(make_reader('file://' + args.input_data, predicate=in_pseudorandom_split([train_ratio, test_ratio], 1, args.feature_column), 
                                    transform_spec=transform_spec_val), 
                        batch_size=args.test_batch_size)
    
sys.argv = sys.argv[:1] + remaining_args
sys.argv.extend(['--output_dir', args.output_dir, '--loop_epochs', str(args.loop_epochs)])

if __name__ == "__main__":
    if not args.is_managed:
        print("Entering custom code...\n")
        runpy.run_path(args.training_script_name + '.py', globals(), run_name="__main__")
    else:
        print("Entering managed boilerplate...\n")
        globals_dict = runpy.run_path(args.training_script_name + '.py', globals())
        train = globals_dict['train']
        test = globals_dict['test']
        get_best_model = globals_dict['get_best_model']

        with mlflow.start_run() as mlflow_run:
            for epoch in range(args.loop_epochs):
                train(epoch, trainloader)
                trainloader.reader.reset()

                test(epoch, testloader)
                testloader.reader.reset()

            best_model = get_best_model()
            mlflow.pytorch.log_model(best_model, args.output_dir)
