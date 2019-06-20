
# coding: utf-8


# In[3]:


from PyTorchEstimator import PyTorchEstimator
from azureml.core.workspace import Workspace
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)


# In[4]:


# Get the CIFAR10 dataset as Python dictionary
import os, tarfile, pickle
import urllib.request
cdnURL = "https://amldockerdatasets.azureedge.net"
# Please note that this is a copy of the CIFAR10 dataset originally found here:
# http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
dataFile = "cifar-10-python.tar.gz"
dataURL = cdnURL + "/CIFAR10/" + dataFile
if not os.path.isfile(dataFile):
    urllib.request.urlretrieve(dataURL, dataFile)
with tarfile.open(dataFile, "r:gz") as f:
    test_dict = pickle.load(f.extractfile("cifar-10-batches-py/test_batch"),
                            encoding="latin1")


# In[5]:


from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField
from petastorm.codecs import ScalarCodec, NdarrayCodec, CompressedImageCodec
from pyspark.sql.types import *
import numpy as np

# Generate Petastorm dataset
# TODO: Remove filenames when PR #393 is merged in petastorm's github
image_zip = zip(test_dict["data"], test_dict["labels"], test_dict["filenames"])

CIFARSchema = Unischema('CIFARSchema', [
    UnischemaField('image', np.double, (32,32,3), CompressedImageCodec('png'), False),
    UnischemaField('label', np.int64, (), ScalarCodec(IntegerType()), False),
    UnischemaField('filename', np.str, (), ScalarCodec(StringType()), False),
])

def reshape_image(record):
    image, label, filename = record
    return {'image': image.reshape(32,32,3).astype(np.double), 'label': label, 'filename': filename}

rows_rdd = sc.parallelize(image_zip) \
            .map(reshape_image) \
            .map(lambda x: dict_to_spark_row(CIFARSchema, x))
    
imagesWithLabels = spark.createDataFrame(rows_rdd)


# In[6]:


# Split the images with labels into a train and test data
train, test = imagesWithLabels.randomSplit([0.8, 0.2], seed=123)


# In[7]:


train.show(3)
train.printSchema()


# In[ ]:


# Initializing the estimator
workspace = Workspace('e54229a3-0e6f-40b3-82a1-ae9cda6e2b81', 'mmlspark-serano', 'playground')
clusterName = 'train-target'
trainingScript = 'pytorch_train.py'
nodeCount = 1
modelPath = 'outputs/model.pt'
experimentName = 'pytorch-train-cifar'
unischema = CIFARSchema

estimator = PyTorchEstimator(workspace, clusterName, trainingScript, nodeCount, modelPath, experimentName, unischema)


# In[ ]:


model = estimator.fit(train)

