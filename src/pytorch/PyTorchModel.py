# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import sys
import os
import importlib
import numpy as np

from pyspark.ml import Transformer
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT

from azureml.core import Run

from mlflow.pytorch import load_model

if sys.version >= '3':
    basestring = str

class PreTransformer:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

        
    def transformData(self, data):
        print("transforming data...")
        
        # Preprocess using user's function then convert to a one-sized batch
        dataTensor = self.preprocessor(data).unsqueeze(0) 
        outputTensor = self.torchModel(dataTensor)

        # Convert tensor back to vector
        outputArray = outputTensor.detach().numpy().reshape(-1)
        outputVector = Vectors.dense(outputArray)
        return outputVector

    def setTorchModel(self, torchModel):
        self.torchModel = torchModel

class PyTorchModel(Transformer):
    """
    ``PyTorchModel`` transforms one dataset into another.

    Args:

        runId (str): Azure ML Run ID

    """

    def __init__(self, runId, experiment, workspace, outputPath, preprocessor):
        self.runId = runId
        self.experiment = experiment
        self.workspace = workspace
        self.outputPath = outputPath
        self.pretransformer = PreTransformer(preprocessor)

    def _transform(self, dataset):
        """
        Transforms the input dataset.
        :param dataset: input dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`
        :returns: transformed dataset
        """
        # Download PyTorch model from completed job
        run = Run(self.experiment, self.runId)
        if run.get_status() != 'Completed':
            raise Exception('Run not completed.')

        cloudPath = self.outputPath
        print('cloudPath: {}'.format(cloudPath))
        localPath = os.path.join(os.getcwd(), self.outputPath)
        print('localPath: {}'.format(cloudPath))
        run.download_files(cloudPath, os.getcwd())

        torchModel = load_model(localPath)
        print("torchModel: ", torchModel)
        print("type of torchModel: ", type(torchModel))
        torchModel.eval()

        print("Trained model loaded.")
        self.pretransformer.setTorchModel(torchModel)
        
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

