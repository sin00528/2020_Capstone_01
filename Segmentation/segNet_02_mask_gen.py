from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os

import json
from keras_segmentation.predict import predict
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
model.load_weights(ckPtDir + "/vgg_segnet.199")

model.summary()

# segTrain2017 Img Gen
for (path, _, files) in os.walk(trainImgs):
    for filename in files:
        predict(model,
                overlay_img=True,
                inp = path+filename,
                out_fname = segTrainOutDir + '/' + filename[:-3]+'png'
               )
        
# segVal2017 Img Gen
for (path, _, files) in os.walk(valImgs):
    for filename in files:
        predict(model,
                overlay_img=True,
                inp = path+filename,
                out_fname = segValOutDir + '/' + filename[:-3]+'png'
               )

# send the msg to discord channel
from discord_webhook import DiscordWebhook
url = 'https://discordapp.com/api/webhooks/710208007618822145/4yUFIEoTa7kZFOhyJpSkalNn2NysrM6p5PFVG5iBDkt1ikJxBPwV3_J4FDYi40THgxvl'
webhook = DiscordWebhook(url=url, content='Mask Generate Operation is completed...')
response = webhook.execute()