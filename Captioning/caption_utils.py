from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

# Setting dirs
PLOT_DIR = "gen/Outputs"

# Function define
def cache_bottlenecks(img_name_vector, image_features_extract_model):
    # unique한 image name 집합을 만듭니다.
    encode_train = sorted(set(img_name_vector))
    
    # tf.data API를 이용해서 이미지를 batch 개수(=16)만큼 불러옵니다.
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)
    
    # 동일 이미지에 대한 feature map 변환 연산을 반복수행하는 부분을 제거하기 위해서
    # 한번 feature map 형태로 변환한 값들을 disk에 저장해서 caching합니다.
    for img, path in tqdm(image_dataset):
        batch_features = image_features_extract_model(img)
        # 16x8x8x2048 이미지를 16x64x2048 형태로 reshape합니다.
        batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
        
        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())

# Inception v3의 input에 적합한 형태로 image_path 경로에서 이미지를 불러옵니다.
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    
    return img, image_path

# 전체 dataset에 존재하는 caption의 maximum length를 찾습니다.
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

# attention 결과를 시각화합니다.
def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))
    
    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.savefig(PLOT_DIR + '/imgCap_' + image.split(os.path.sep)[-1].split('.')[-2] + '_attention' + '.png')
    plt.show()