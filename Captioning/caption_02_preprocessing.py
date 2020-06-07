import re
import pandas as pd
import numpy as np
import json
from bs4 import BeautifulSoup

from tqdm import tqdm
from nltk.corpus import stopwords
from sklearn.utils import shuffle

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import matplotlib.pyplot as plt

# Root Dir
COCO_DIR = '../Segmentation/COCO'

# 훈련 캡션 전처리
annotation = pd.read_csv(COCO_DIR + '/annotations/captions_train2017.csv', index_col=0)

all_captions = []
all_img_name_vector = []

for annot in tqdm(annotation.iterrows()):
    caption = '<start> ' + annot[1]['caption'] + ' <end>'
    image_id = annot[1]['image_id']
    full_img_image_path = COCO_DIR + '/segTrain2017/' + '{0:0>12}.png'.format(image_id)

    all_img_name_vector.append(full_img_image_path)
    all_captions.append(caption)

all_captions, all_img_name_vector= shuffle(all_captions,
                                           all_img_name_vector,
                                           random_state=1)

print('Number of all_img_name_vector :', len(all_img_name_vector))
print('Number of all_captions :', len(all_captions))

np.save(open(COCO_DIR + '/train_input.npy', 'wb'), all_img_name_vector)
np.save(open(COCO_DIR + '/train_label.npy', 'wb'), all_captions)

# 검증 캡션 전처리
annotation = pd.read_csv(COCO_DIR + '/annotations/captions_val2017.csv', index_col=0)

all_captions = []
all_img_name_vector = []

for annot in tqdm(annotation.iterrows()):
    caption = '<start> ' + annot[1]['caption'] + ' <end>'
    image_id = annot[1]['image_id']
    full_img_image_path = COCO_DIR + '/segTrain2017/' + '{0:0>12}.png'.format(image_id)

    all_img_name_vector.append(full_img_image_path)
    all_captions.append(caption)

all_captions, all_img_name_vector= shuffle(all_captions,
                                           all_img_name_vector,
                                           random_state=1)

print('Number of all_img_name_vector :', len(all_img_name_vector))
print('Number of all_captions :', len(all_captions))

np.save(open(COCO_DIR + '/val_input.npy', 'wb'), all_img_name_vector)
np.save(open(COCO_DIR + '/val_label.npy', 'wb'), all_captions)

# send the msg to discord channel
from discord_webhook import DiscordWebhook
url = 'https://discordapp.com/api/webhooks/710208007618822145/4yUFIEoTa7kZFOhyJpSkalNn2NysrM6p5PFVG5iBDkt1ikJxBPwV3_J4FDYi40THgxvl'
webhook = DiscordWebhook(url=url, content='Caption preprocessing is completed...')
response = webhook.execute()