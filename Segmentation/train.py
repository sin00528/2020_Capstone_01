from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

import json
from keras import Model
from keras.layers import Input
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import plot_model

#from keras_segmentation.models.model_utils import get_segmentation_model
from keras_segmentation.models import unet
from keras_segmentation.models.segnet import vgg_segnet
from keras_segmentation.models.pspnet import pspnet_50

from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import pspnet_50_ADE_20K

dataDir='./COCO'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

trainImgs = "COCO/train2017/"
valImgs = "COCO/val2017/"
trainAnns = "COCO/annotations/sementic_train2017/"
valAnns = "COCO/annotations/sementic_train2017/"

ckPtDir = "log/segnet"
plotDir = "log/plot"
segInDir = "gen/Inputs"
segOutDir = "gen/Outputs"

# initialize COCO api for instance annotations
coco=COCO(annFile)

model = vgg_segnet(n_classes=91, input_height=416, input_width=608, encoder_level=3)

model.summary()
plot_model(model, show_shapes=True, to_file='log/plot/vgg_segnet.png')

model.train(train_images = trainImgs,
            train_annotations = trainAnns,
            n_classes = 91,
            validate=True,
            verify_dataset=False,
            val_images = valImgs,
            val_annotations = valAnns,
            checkpoints_path = ckPtDir,
            epochs=500,
            batch_size=8)

hist = model.history

plt.figure(1)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
plt.savefig(plotDir + '/vgg_segnet_acc.png')

plt.figure(2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
plt.savefig(plotDir + '/vgg_segnet_loss.png')

from discord_webhook import DiscordWebhook
url = 'https://discordapp.com/api/webhooks/710208007618822145/4yUFIEoTa7kZFOhyJpSkalNn2NysrM6p5PFVG5iBDkt1ikJxBPwV3_J4FDYi40THgxvl'
webhook = DiscordWebhook(url=url, content='Model Train completed...')
response = webhook.execute()