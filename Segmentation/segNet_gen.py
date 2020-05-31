from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os

import json
from keras import Model
from keras.layers import Input
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import plot_model

from keras_segmentation.models.segnet import vgg_segnet

dataDir='./COCO'
dataType='train2017'

trainImgs = "COCO/train2017/"
valImgs = "COCO/val2017/"

ckPtDir = "log/segnet"
plotDir = "log/plot"

segTrainOutDir = 'COCO/segTrain2017'
segValOutDir = 'COCO/segVal2017'

model = vgg_segnet(n_classes=91, input_height=416, input_width=608, encoder_level=3)
model.load_weights(ckPtDir + "/vgg_segnet.499")

model.summary()

# segTrainImg Gen
for (path, dirs, files) in os.walk(trainImgs):
    for filename in files:
        out = model.predict_segmentation(
            inp = path+filename,
            out_fname = segTrainOutDir + '/' + filename[:-3]+'png'
        )
        
# segValmg Gen
for (path, dirs, files) in os.walk(valImgs):
    for filename in files:
        out = model.predict_segmentation(
            inp = path+filename,
            out_fname = segValOutDir + '/' + filename[:-3]+'png'
        )

