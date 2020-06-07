
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import imageio
import matplotlib.pyplot as plt
import pylab
from tqdm import tqdm
import imageio.core.util
import os
import cv2

# for multiprocessing
from multiprocessing import Pool

pylab.rcParams['figure.figsize'] = (8.0, 10.0)


# # 1. Setting Hyper-Params
# Root Dir
COCO_DIR = './COCO'


# Train data dir
TRAIN = 'train2017'
TRAIN_ANN_FILE = COCO_DIR + '/annotations/instances_' + TRAIN + '.json'
TRAIN_OUT_DIR = COCO_DIR + '/annotations/sementic_' + TRAIN + '/'

# Val data dir
VAL = 'val2017'
VAL_ANN_FILE = COCO_DIR + '/annotations/instances_' + VAL + '.json'
VAL_OUT_DIR = COCO_DIR + '/annotations/sementic_' + VAL + '/'


# Warning ignore
def ignore_warnings(*args, **kwargs):
    pass

imageio.core.util._precision_warn = ignore_warnings


# # 2.Train mask generate
# initialize COCO api for instance annotations and caption annocations
coco = COCO(TRAIN_ANN_FILE)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))


def trainAnns(idx):
    ids = coco.getImgIds()[idx]
    catids = coco.getCatIds()
    imgIds = coco.getImgIds(imgIds = ids)
    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    I = io.imread(COCO_DIR + '/' + TRAIN + '/' +img['file_name']).astype(np.uint8)
    
    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=catids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)
    anns_img = np.zeros((img['height'],img['width']))
    
    for ann in anns:
        anns_img = np.maximum(anns_img, coco.annToMask(ann)*ann['category_id'])
        
    #io.imshow(anns_img, cmap='gray');
    #io.imsave(TRAIN_OUT_DIR+'{0:0>12}.png'.format(img['id']), anns_img, check_contrast=False)
    #imageio.imwrite(TRAIN_OUT_DIR+'{0:0>12}.png'.format(img['id']), anns_img)
    cv2.imwrite(TRAIN_OUT_DIR+'{0:0>12}.png'.format(img['id']), anns_img)

# RUN on imageio==2.8.0
with Pool(processes=32) as p:
    max_ = len(coco.getImgIds())
    with tqdm(total=max_) as pbar:
        for i, _ in tqdm(enumerate(p.imap_unordered(trainAnns, range(max_)))):
            pbar.update()


# # 3.Validation mask generate
# initialize COCO api for instance annotations and caption annocations
coco = COCO(VAL_ANN_FILE)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

def valAnns(idx):
    ids = coco.getImgIds()[idx]
    catids = coco.getCatIds()
    imgIds = coco.getImgIds(imgIds = ids)
    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    I = io.imread(COCO_DIR + '/' + VAL + '/' +img['file_name']).astype(np.uint8)

    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=catids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)
    anns_img = np.zeros((img['height'],img['width']))
    
    for ann in anns:
        anns_img = np.maximum(anns_img, coco.annToMask(ann)*ann['category_id'])
    
    #io.imshow(anns_img, cmap='gray');
    #io.imsave(VAL_OUT_DIR+'{0:0>12}.png'.format(img['id']), anns_img, check_contrast=False)
    #imageio.imwrite(VAL_OUT_DIR+'{0:0>12}.png'.format(img['id']), anns_img)
    cv2.imwrite(VAL_OUT_DIR+'{0:0>12}.png'.format(img['id']), anns_img)

# RUN on imageio==2.8.0
with Pool(processes=32) as p:
    max_ = len(coco.getImgIds())
    with tqdm(total=max_) as pbar:
        for i, _ in tqdm(enumerate(p.imap_unordered(valAnns, range(max_)))):
            pbar.update()


from discord_webhook import DiscordWebhook
url = 'https://discordapp.com/api/webhooks/710208007618822145/4yUFIEoTa7kZFOhyJpSkalNn2NysrM6p5PFVG5iBDkt1ikJxBPwV3_J4FDYi40THgxvl'
webhook = DiscordWebhook(url=url, content='segment generation completed...')
response = webhook.execute()
