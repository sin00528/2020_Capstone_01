from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import pandas as pd
import json

# for multiprocessing
from multiprocessing import Pool

# Root Dir
COCO_DIR = '../Segmentation/COCO'

# Train data dir
TRAIN = 'train2017'
TRAIN_CAP_ANN_FILE = COCO_DIR + '/annotations/captions_' + TRAIN + '.json'

# Val data dir
VAL = 'val2017'
VAL_CAP_ANN_FILE = COCO_DIR + '/annotations/captions_' + VAL + '.json'


# Extract train caption
coco_caps = COCO(TRAIN_CAP_ANN_FILE)
captions = pd.DataFrame(columns=['image_id', 'id', 'caption'])

for idx in tqdm(range(len(coco_caps.getImgIds()))):
    ids = coco_caps.getImgIds()[idx]
    capsAnnIds = coco_caps.getAnnIds(imgIds=ids)
    capsAnns = coco_caps.loadAnns(capsAnnIds)
    captions = captions.append(capsAnns, ignore_index=True)

captions.to_csv(COCO_DIR + '/annotations/captions_train2017.csv')

# Extract val caption
coco_caps = COCO(VAL_CAP_ANN_FILE)
captions = pd.DataFrame(columns=['image_id', 'id', 'caption'])

for idx in tqdm(range(len(coco_caps.getImgIds()))):
    ids = coco_caps.getImgIds()[idx]
    capsAnnIds = coco_caps.getAnnIds(imgIds=ids)
    capsAnns = coco_caps.loadAnns(capsAnnIds)
    captions = captions.append(capsAnns, ignore_index=True)

captions.to_csv(COCO_DIR + '/annotations/captions_val2017.csv')

# send the msg to discord channel
from discord_webhook import DiscordWebhook
url = 'https://discordapp.com/api/webhooks/710208007618822145/4yUFIEoTa7kZFOhyJpSkalNn2NysrM6p5PFVG5iBDkt1ikJxBPwV3_J4FDYi40THgxvl'
webhook = DiscordWebhook(url=url, content='Caption Extraction is completed...')
response = webhook.execute()