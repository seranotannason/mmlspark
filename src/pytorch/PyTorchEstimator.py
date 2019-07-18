# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

# TODO: Make variable naming convention consistent
# TODO: Encapsulation, i.e. getters and setters

import sys
if sys.version >= '3':
    basestring = str

from PyTorchModel import PyTorchModel

import os
import shutil
import ntpath
import importlib.util
import uuid
import pyarrow
import cloudpickle
from functools import reduce
from petastorm.etl.dataset_metadata import materialize_dataset, infer_or_load_unischema

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.conda_dependencies import CondaDependencies
from azureml.data.azure_storage_datastore import AzureFileDatastore, AzureBlobDatastore
from azureml.train.dnn import PyTorch
from azureml.core.workspace import Workspace
from azureml.core import Experiment
from azureml.core import ScriptRunConfig
from azureml.core.runconfig import DataReferenceConfiguration
from azureml.core.runconfig import MpiConfiguration
from azureml.core.model import Model
from azureml.widgets import RunDetails

from pyspark.ml import Estimator
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession


sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
TRAINING_SCRIPT_NAME = "pytorch_train"

class PyTorchEstimator(Estimator):
    """
    ``PyTorchEstimator`` trains a model on a dataset. The
    result is a ``PyTorchModel``.

    Args:

        workspace (azureml.core.workspace.Workspace): Azure ML Workspace object
        clusterName (str): Name of Azure ML Compute Target in the above workspace
        scriptsFolder (str): Full path to folder containing all user scripts
        trainingScriptName (str): Name of training script without extension
        nodeCount (int): Number of nodes to train on
        outputPath (str): Cloud path that the model will be saved to, relative to /outputs dir
        experimentName (str): Name of current experiment
        trainPreprocessor (function): Preprocessing function for training data
        valPreprocessor (function): Preprocessing function for validation data
        testPreprocessor (function): Preprocessing function for test data
        environment (azureml.core.Environment): Azure ML Environment object
        trainDataPercentage (float): Percentage of data dedicated for training
        trainBatchSize (int): Size of training data batches
        testBatchSize (int): Size of test data batches
        loopEpochs (int): Number of epochs to train
        featureColumn (str): Name of column containing the features
        userDefinedArgs ()
        

    """
    def __init__(self, workspace, clusterName, scriptsFolder, trainingScriptName, nodeCount, outputPath, experimentName, train_preprocessor, val_preprocessor, 
                    test_preprocessor, environment, train_data_percentage, train_batch_size, test_batch_size, loop_epochs, feature_column, 
                    user_defined_args, is_managed, allow_default_env_config):
        self.workspace = workspace
        self.clusterName = clusterName
        self.scripts_folder = scriptsFolder
        self.training_script_name = trainingScriptName
        self.nodeCount = nodeCount
        self.outputPath = outputPath
        self.experimentName = experimentName
        self.train_preprocessor = train_preprocessor
        self.val_preprocessor = val_preprocessor
        self.test_preprocessor = test_preprocessor
        self.environment = environment
        self.train_data_percentage = train_data_percentage
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.loop_epochs = loop_epochs
        self.feature_column = feature_column
        self.user_defined_args = list(reduce(lambda x, y: x + y, user_defined_args.items()))
        self.is_managed = is_managed
        self.allow_default_env_config = allow_default_env_config

    def _create_project_directory(self, train_preprocessor_filename, val_preprocessor_filename):
        training_script_path = os.path.join(self.scripts_folder, self.training_script_name + '.py')
        if not os.path.isfile(training_script_path):
            raise Exception("Training script does not exist in the scripts folder.")

        working_folder = '/tmp/pytorch_scripts'
        if os.path.isdir(working_folder):
            shutil.rmtree(working_folder)
        shutil.copytree(self.scripts_folder, working_folder)
        os.rename(os.path.join(working_folder, self.training_script_name + '.py'), 
                  os.path.join(working_folder, TRAINING_SCRIPT_NAME + '.py'))
        shutil.copy('./wrapper.py', working_folder)

        pickled_train_preprocessor = cloudpickle.dumps(self.train_preprocessor)
        with open(os.path.join(working_folder, train_preprocessor_filename), 'wb+') as train_preprocessor_file:
            train_preprocessor_file.write(pickled_train_preprocessor)

        pickled_val_preprocessor = cloudpickle.dumps(self.val_preprocessor)
        with open(os.path.join(working_folder, val_preprocessor_filename), 'wb+') as val_preprocessor_file:
            val_preprocessor_file.write(pickled_val_preprocessor)

        return working_folder
    
    def _infer_unischema(self, dataset):
        pandas_dataset = dataset.toPandas()
        arrow_schema = pyarrow.Schema.from_pandas(pandas_dataset)

        import numpy as np
        from petastorm.codecs import ScalarCodec
        from pyspark.sql.types import StringType, ShortType, LongType, IntegerType, BooleanType, DoubleType, \
            ByteType, \
            FloatType, DecimalType, DateType, TimestampType
        from decimal import Decimal
        from pyspark.sql.types import StructField, StructType
        def _numpy_and_codec_from_arrow_type(field_type):
            from pyarrow import types

            if types.is_int8(field_type):
                np_type = np.int8
                codec = ScalarCodec(ByteType())
            elif types.is_int16(field_type):
                np_type = np.int16
                codec = ScalarCodec(ShortType())
            elif types.is_int32(field_type):
                np_type = np.int32
                codec = ScalarCodec(IntegerType())
            elif types.is_int64(field_type):
                np_type = np.int64
                codec = ScalarCodec(LongType())
            elif types.is_string(field_type):
                np_type = np.unicode_
                codec = ScalarCodec(StringType())
            elif types.is_boolean(field_type):
                np_type = np.bool_
                codec = ScalarCodec(BooleanType())
            elif types.is_float32(field_type):
                np_type = np.float32
                codec = ScalarCodec(FloatType())
            elif types.is_float64(field_type):
                np_type = np.float64
                codec = ScalarCodec(DoubleType())
            elif types.is_decimal(field_type):
                np_type = Decimal
                codec = ScalarCodec(DecimalType(field_type.precision, field_type.scale))
            elif types.is_binary(field_type):
                codec = ScalarCodec(StringType())
                np_type = np.string_
            elif types.is_fixed_size_binary(field_type):
                codec = ScalarCodec(StringType())
                np_type = np.string_
            elif types.is_date(field_type):
                np_type = np.datetime64
                codec = ScalarCodec(DateType())
            elif types.is_timestamp(field_type):
                np_type = np.datetime64
                codec = ScalarCodec(TimestampType())
            elif types.is_list(field_type):
                _, np_type = _numpy_and_codec_from_arrow_type(field_type.value_type)
                codec = None
            else:
                raise ValueError('Cannot auto-create unischema due to unsupported column type {}'.format(field_type))
            return codec, np_type
        
        # Create Unischema
        from petastorm.unischema import Unischema, UnischemaField    
        unischema_fields = []
        for column_name in arrow_schema.names:
            arrow_field = arrow_schema.field_by_name(column_name)
            field_type = arrow_field.type
            codec, np_type = _numpy_and_codec_from_arrow_type(field_type)
            unischema_fields.append(UnischemaField(column_name, np_type, (), codec, arrow_field.nullable))
        
        return Unischema('inferred_schema', unischema_fields)

    def _get_or_create_compute_target(self):
        try:
            compute_target = ComputeTarget(workspace=self.workspace, name=self.clusterName)
            print('Found existing compute target.')
        except ComputeTargetException:
            print('Creating a new compute target...')
            compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6', 
                                                                max_nodes=4)

            compute_target = ComputeTarget.create(self.workspace, self.clusterName, compute_config)
            compute_target.wait_for_completion(show_output=True)

        return compute_target

    def _upload_dataset_to_datastore(self, dataset, datastore, unischema):
        local_path = 'file:///tmp/dataset'
        datastore_path = 'data/dataset' + str(uuid.uuid4())
        # TODO: wasb 
        with materialize_dataset(spark, local_path, unischema):
            dataset.coalesce(10) \
                .write \
                .mode('overwrite') \
                .parquet(local_path)
        datastore.upload(src_dir='/tmp/dataset', target_path=datastore_path, overwrite=True, show_progress=True)

        # Get dataset path on datastore
        ds_data = datastore.path(datastore_path)
        return ds_data

    def _create_script_params(self, ds_data, train_preprocessor_filename, val_preprocessor_filename):
        script_params = [
            '--input_data', str(ds_data),
            '--output_dir', self.outputPath,
            '--train_data_percentage', self.train_data_percentage,
            '--train_batch_size', self.train_batch_size,
            '--test_batch_size', self.test_batch_size,
            '--loop_epochs', self.loop_epochs,
            '--feature_column', self.feature_column,
            '--training_script_name', TRAINING_SCRIPT_NAME,
            '--is_managed', self.is_managed,
            '--train_preprocessor_filename', train_preprocessor_filename,
            '--val_preprocessor_filename', val_preprocessor_filename,
        ]
        script_params.extend(self.user_defined_args)

        return script_params

    def _create_script_run_config(self, ds_data, working_folder, script_params):
        script_run_config = ScriptRunConfig(source_directory=working_folder, script='wrapper.py', arguments=script_params)
        script_run_config.run_config.target = self.clusterName
        script_run_config.run_config.environment = self.environment

        petastorm_pkg = CondaDependencies._register_private_pip_wheel_to_blob(self.workspace, '/home/azureuser/serano-petastorm/dist/petastorm-0.9.0.dev0-py2.py3-none-any.whl')
        pip_packages = [
            "pandas", 
            "opencv-python-headless", 
            petastorm_pkg, 
            "azureml-mlflow", 
            "Pillow==6.0.0", 
            "torch==1.1", 
            "torchvision==0.2.1",
            "horovod==0.16.1",
            "pyarrow==0.12.1"
        ]
        for pip_package in pip_packages:
            script_run_config.run_config.environment.python.conda_dependencies.add_pip_package(pip_package)
        conda_packages = [
            "opencv"
        ]
        for conda_package in conda_packages:
            script_run_config.run_config.environment.python.conda_dependencies.add_conda_package(conda_package)

        script_run_config.run_config.node_count = self.nodeCount
        script_run_config.run_config.mpi = MpiConfiguration()
        script_run_config.run_config.communicator = "Mpi"
        script_run_config.run_config.data_references = {
            ds_data.data_reference_name: DataReferenceConfiguration(datastore_name=ds_data.datastore.name, 
                mode='mount', path_on_datastore=ds_data.path_on_datastore, 
                path_on_compute=ds_data.path_on_compute, overwrite=ds_data.overwrite) 
        }
        script_run_config.run_config.environment.docker.enabled = True
        script_run_config.run_config.environment.docker.gpu_support = True
        if self.allow_default_env_config:
            script_run_config.run_config.environment.docker.base_image = "mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.0-cudnn7-ubuntu16.04"
            script_run_config.run_config.environment.environment_variables.update({
                "NCCL_IB_DISABLE": "1",
                "NCCL_SOCKET_IFNAME": "eth0",
                "NCCL_TREE_THRESHOLD": "0",
                "PYTHONFAULTHANDLER": "enabled",
            })
        return script_run_config

    def _fit(self, dataset):
        """
        Fits a model to the input dataset. This is called by the default implementation of fit.
        :param dataset: input dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`
        :returns: fitted model
        """
        train_preprocessor_filename = 'train_preprocessor'
        val_preprocessor_filename = 'val_preprocessor'

        working_folder = self._create_project_directory(train_preprocessor_filename, val_preprocessor_filename)
        unischema = self._infer_unischema(dataset)
        print("Inferred unischema: ", unischema)
        datastore = self.workspace.get_default_datastore()
        compute_target = self._get_or_create_compute_target()
        ds_data = self._upload_dataset_to_datastore(dataset, datastore, unischema)

        script_params = self._create_script_params(ds_data, train_preprocessor_filename, val_preprocessor_filename)
        runconfig = self._create_script_run_config(ds_data, working_folder, script_params)
        experiment = Experiment(self.workspace, name=self.experimentName)
        run = experiment.submit(config=runconfig)
        print("Job submitted!")

        fittedModel = PyTorchModel(run.id, experiment, self.workspace, self.outputPath, self.test_preprocessor)
        return fittedModel
