# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import sys
import torch
from pyspark.ml import Transformer
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT

if sys.version >= '3':
    basestring = str

class PyTorchModel(Transformer):
    """
    ``PyTorchModel`` transforms one dataset into another.

    Args:

        fittedModel (torch.nn.Module): trained PyTorch model

    """

    def __init__(self, fittedModel):
        self.fittedModel = fittedModel

    def _transform(self, dataset):
        """
        Transforms the input dataset.
        :param dataset: input dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`
        :returns: transformed dataset
        """
        def transformData(data):
            # Convert PySpark vector to NumPy array, then to Torch tensor
            dataTensor = torch.from_numpy(data.toArray())

            # Transform the tensor
            outputTensor = self.fittedModel(dataTensor)

            # Convert tensor back to vector
            outputVector = Vectors.dense(outputTensor.numpy())
            return outputVector

        transformDataUDF = udf(transformData, VectorUDT())

        # Transform input col
        dataset = dataset.withColumn(self.outputCol, transformDataUDF(self.inputCol))
        return dataset

    # TODO: probably change these
    def setInputCol(self, inputCol):
        self.inputCol = inputCol

    def setOutputCol(self, outputCol):
        self.outputCol = outputCol

