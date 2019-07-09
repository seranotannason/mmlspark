# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

# TODO: Make variable naming convention consistent
# TODO: Encapsulation, i.e. getters and setters

import sys
if sys.version >= '3':
    basestring = str

import os
import shutil
import ntpath
import importlib.util
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
from petastorm.etl.dataset_metadata import materialize_dataset, infer_or_load_unischema
from pyspark.ml import Estimator
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from PyTorchModel import PyTorchModel
import uuid
import pyarrow

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)


class PyTorchEstimator(Estimator):
    """
    ``PyTorchEstimator`` trains a model on a dataset. The
    result is a ``PyTorchModel``.

    Args:

        workspace (Workspace): Azure ML Workspace object
        clusterName (str): Name of Azure ML Compute Target in the above workspace
        trainingScript (str): Full path to training script
        nodeCount (int): Number of nodes to train on
        modelPath (str): Cloud path that the model will be saved to, relative to /outputs dir
        experimentName (str): Name of current experiment

    """

    def __init__(self, workspace, clusterName, trainingScript, modelScript, nodeCount, modelPath, experimentName, preprocessor, environment):
        self.workspace = workspace
        self.clusterName = clusterName
        self.trainingScript = trainingScript
        self.modelScript = modelScript
        self.nodeCount = nodeCount
        self.modelPath = modelPath
        self.experimentName = experimentName
        self.preprocessor = preprocessor
        self.environment = environment

    def _fit(self, dataset):
        """
        Fits a model to the input dataset. This is called by the default implementation of fit.
        :param dataset: input dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`
        :returns: fitted model
        """
        # ============================ WIP: GET UNISCHEMA FROM DATASET ==================================

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
        self.unischema = Unischema('inferred_schema', unischema_fields)
        print(self.unischema)


        # ============================ WIP: GET UNISCHEMA FROM DATASET ==================================

        # Get datastore and compute target from workspace
        datastore = self.workspace.get_default_datastore()
        try:
            compute_target = ComputeTarget(workspace=self.workspace, name=self.clusterName)
            print('Found existing compute target.')
        except ComputeTargetException:
            print('Creating a new compute target...')
            compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6', 
                                                                max_nodes=4)

            # create the cluster
            compute_target = ComputeTarget.create(self.workspace, self.clusterName, compute_config)
            compute_target.wait_for_completion(show_output=True)

        # Upload dataset to datastore
        local_path = 'file:///tmp/dataset'
        datastore_path = 'data/dataset' + str(uuid.uuid4())
        # TODO: wasb 
        with materialize_dataset(spark, local_path, self.unischema):
            dataset.coalesce(10) \
                .write \
                .mode('overwrite') \
                .parquet(local_path)
        datastore.upload(src_dir='/tmp/dataset', target_path=datastore_path, overwrite=True, show_progress=True)

        # Get dataset path on datastore
        # path_on_datastore = 'tmp/data/dataset.parquet'
        ds_data = datastore.path(datastore_path)

        # Arguments to training script
        script_params = {
            '--input_data': ds_data,
            '--output_dir': self.modelPath
        }

        # Create a project directory
        project_folder = './pytorch-train'
        os.makedirs(project_folder, exist_ok=True)
        shutil.copy(self.trainingScript, project_folder)
        shutil.copy(self.modelScript, project_folder)

        # Extract names of scripts from full path to pass as argument to PyTorch
        training_script_name = ntpath.basename(self.trainingScript)

        # Create an experiment
        experiment = Experiment(self.workspace, name=self.experimentName)

        # ================================= WIP: Replacing PyTorch with RunConfig ===================================
        # Arguments to training script
        script_params = [
            '--input_data', str(ds_data),
            '--output_dir', self.modelPath
        ]

        runconfig = ScriptRunConfig(source_directory=project_folder, script=training_script_name, arguments=script_params)
        runconfig.run_config.target = self.clusterName
        runconfig.run_config.environment = self.environment
        runconfig.run_config.environment.docker.base_image = "mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.0-cudnn7-ubuntu16.04"
        runconfig.run_config.environment.docker.enabled = True
        runconfig.run_config.environment.docker.gpu_support = True
        runconfig.run_config.environment.environment_variables = {
            "EXAMPLE_ENV_VAR": "EXAMPLE_VALUE",
            "NCCL_IB_DISABLE": "1",
            "NCCL_SOCKET_IFNAME": "eth0",
            "NCCL_TREE_THRESHOLD": "0",
        }
        runconfig.run_config.node_count = self.nodeCount
        runconfig.run_config.mpi = MpiConfiguration()
        runconfig.run_config.communicator = "Mpi"
        runconfig.run_config.data_references = {
            ds_data.data_reference_name: DataReferenceConfiguration(datastore_name=ds_data.datastore.name, 
                mode='mount', path_on_datastore=ds_data.path_on_datastore, 
                path_on_compute=ds_data.path_on_compute, overwrite=ds_data.overwrite) 
        }
        run = experiment.submit(config=runconfig)

        # # Create a PyTorch estimator
        # # TODO: Lighten the burden on user to manually specify pip packages
        # petastorm_pkg = CondaDependencies._register_private_pip_wheel_to_blob(self.workspace, '/home/azureuser/serano-petastorm/dist/petastorm-0.9.0.dev0-py2.py3-none-any.whl')
        # estimator = PyTorch(source_directory=project_folder,
        #                     compute_target=compute_target,
        #                     entry_script=training_script_name,
        #                     script_params=script_params,
        #                     node_count=self.nodeCount,
        #                     distributed_training=MpiConfiguration(),
        #                     use_gpu=True,
        #                     pip_packages=['pandas', 'opencv-python-headless', petastorm_pkg, "azureml-mlflow", "Pillow==6.0.0"],
        #                     conda_packages=['opencv'])

        # # Submit job
        # run = experiment.submit(estimator)

        print("Job submitted!")
        # ================================= WIP: Replacing PyTorch with RunConfig ===================================

        fittedModel = PyTorchModel(run.id, experiment, self.workspace, self.modelPath, self.preprocessor)
        return fittedModel
