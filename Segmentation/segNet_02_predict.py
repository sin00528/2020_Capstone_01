from keras import Model
from keras.layers import Input
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import plot_model
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

# not using overlay
#out = model.predict_segmentation(inp = 'test.png', out_fname = 'test_out.png')

# using overlay
predict(model, overlay_img=True, inp = 'test.png', out_fname = 'test_out.png')