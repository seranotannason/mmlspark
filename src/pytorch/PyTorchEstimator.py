# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

# TODO: Make variable naming convention consistent
# TODO: Encapsulation, i.e. getters and setters

import sys
if sys.version >= '3':
    basestring = str

import os
import shutil
import torch
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.data.azure_storage_datastore import AzureFileDatastore, AzureBlobDatastore
from azureml.train.dnn import PyTorch
from azureml.core.workspace import Workspace
from azureml.core import Experiment
from azureml.core.runconfig import MpiConfiguration
from azureml.core.model import Model
from azureml.widgets import RunDetails
from petastorm.etl.dataset_metadata import materialize_dataset
from pyspark.ml import Estimator
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from PyTorchModel import PyTorchModel

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)


class PyTorchEstimator(Estimator):
    """
    ``PyTorchEstimator`` trains a model on a dataset. The
    result is a ``PyTorchModel``.

    Args:

        workspace (Workspace): Azure ML Workspace object
        clusterName (str): Name of Azure ML Compute Target in the above workspace
        trainingScript (str): Name of training script
        nodeCount (int): Number of nodes to train on
        modelPath (str): Cloud path that the model will be saved to, relative to /outputs dir
        experimentName (str): Name of current experiment
        unischema (Unischema): Unischema of Petastorm dataset

    """

    def __init__(self, workspace, clusterName, trainingScript, nodeCount, modelPath, experimentName, unischema):
        self.workspace = workspace
        self.clusterName = clusterName
        self.trainingScript = trainingScript
        self.nodeCount = nodeCount
        self.modelPath = modelPath
        self.experimentName = experimentName
        self.unischema = unischema

    def _fit(self, dataset):
        """
        Fits a model to the input dataset. This is called by the default implementation of fit.
        :param dataset: input dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`
        :returns: fitted model
        """
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
        # dataset.write.mode('overwrite').parquet('data/dataset.parquet')
        output_dir = 'file:///tmp/data'
        output_filename = 'dataset.parquet'
        output_url = os.path.join(output_dir, output_filename) 
        with materialize_dataset(spark, output_url, self.unischema):
            dataset.coalesce(10) \
                .write \
                .mode('overwrite') \
                .parquet(output_url)
        datastore.upload(src_dir='/tmp/data', target_path='data', overwrite=True, show_progress=True)

        # Get dataset path on datastore
        # path_on_datastore = 'tmp/data/dataset.parquet'
        ds_data = datastore.path(output_url)

        # Train on remote compute
        script_params = {
            '--input_data': ds_data,
            '--output_dir': self.modelPath
        }

        # Create a project directory
        project_folder = './pytorch-train'
        os.makedirs(project_folder, exist_ok=True)
        shutil.copy(self.trainingScript, project_folder)

        # Create an experiment
        experiment = Experiment(self.workspace, name=self.experimentName)

        # Create a PyTorch estimator
        # TODO: Lighten the burden on user to manually specify pip packages
        estimator = PyTorch(source_directory=project_folder,
                            compute_target=compute_target,
                            entry_script=self.trainingScript,
                            script_params=script_params,
                            node_count=self.nodeCount,
                            distributed_training=MpiConfiguration(),
                            use_gpu=True,
                            pip_packages=['pandas', 'pyarrow', 'pyspark'])

        # Submit job
        run = experiment.submit(estimator)
        print("Job submitted!")
        RunDetails(run).show()
        run.wait_for_completion(show_output=True)

        # Download PyTorch model from completed job
        localPath = os.path.join(os.getcwd(), 'model.pt')
        run.download_file(self.modelPath, localPath)
        torchModel = torch.load(localPath)
        torchModel.eval()

        # Create PyTorchModel object from trained PyTorch model
        fittedModel = PyTorchModel(torchModel)

        return fittedModel
