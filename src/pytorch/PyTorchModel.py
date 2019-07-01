# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import sys
import torch
import os, importlib
from pyspark.ml import Transformer
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
from azureml.core import Run

from mlflow.pytorch import load_model

if sys.version >= '3':
    basestring = str

class PreTransformer:
    def __init__(self, preprocessor, torchModel):
        self.preprocessor = preprocessor
        self.torchModel = torchModel

        
    def transformData(self, data):
        print("transforming data...")
        
        dataTensor = self.preprocessor(data)
        outputTensor = self.torchModel(dataTensor)

        # Convert tensor back to vector
        outputArray = outputTensor.detach().numpy().reshape(-1)
        outputVector = Vectors.dense(outputArray)
        return outputVector

class PyTorchModel(Transformer):
    """
    ``PyTorchModel`` transforms one dataset into another.

    Args:

        runId (str): Azure ML Run ID
        unischema (Unischema): Unischema of Petastorm dataset

    """

    def __init__(self, runId, experiment, workspace, modelPath, unischema, preprocessor):
        self.runId = runId
        self.experiment = experiment
        self.workspace = workspace
        self.modelPath = modelPath
        self.unischema = unischema
        # self.preprocessor = preprocessor
        

        # ======================= MIGRATED HERE ============================
        # Download PyTorch model from completed job
        run = Run(self.experiment, self.runId)
        if run.get_status() != 'Completed':
            raise ValueError('Run not completed.')

        cloudPath = self.modelPath
        localPath = os.path.join(os.getcwd(), self.modelPath)
        run.download_files(cloudPath, os.getcwd())

        torchModel = load_model(localPath)
        torchModel.eval()

        print("Trained model loaded.")
        # ======================= MIGRATED HERE ============================

        self.pretransformer = PreTransformer(preprocessor, torchModel)

    def _transform(self, dataset):
        """
        Transforms the input dataset.
        :param dataset: input dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`
        :returns: transformed dataset
        """

        # Transform input col

        transformDataUDF = udf(self.pretransformer.transformData, VectorUDT())
        dataset = dataset.withColumn(self.outputCol, transformDataUDF(self.inputCol))
        return dataset

    def setInputCol(self, inputCol):
        self.inputCol = inputCol
        return self

    def setOutputCol(self, outputCol):
        self.outputCol = outputCol
        return self

    def setPreprocessor(self, preprocessor):
        self.preprocessor = preprocessor

