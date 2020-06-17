import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import os

from keras.models import Model
from keras.layers import Dense

from keras.preprocessing import image
from keras.applications import InceptionV3

from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence

from caption_model import BahdanauAttention, CNN_Encoder, RNN_Decoder
from caption_utils import load_image, calc_max_length, plot_attention

from pypapago import Translator

# 학습을 위한 설정값들을 지정합니다.
num_examples = 30000
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
top_k = 15000
vocab_size = top_k + 1
attention_features_shape = 64
EPOCHS = 5

# Setting dirs
TRAIN_INPUT = "../Segmentation/COCO/train_input.npy"
TRAIN_LABEL = "../Segmentation/COCO/train_label.npy"

TEST_INPUT = "../Segmentation/COCO/val_input.npy"
TEST_LABEL = "../Segmentation/COCO/val_label.npy"

CKPT_DIR = "log/imgCaption"
INPUT_DIR = "gen/Inputs"
OUTPUT_DIR = "gen/Outputs"


# Load Train sets
img_name_vector = np.load(TRAIN_INPUT)
train_captions = np.load(TRAIN_LABEL)

tf.enable_eager_execution()

# 최적화를 위한 Adam 옵티마이저를 정의합니다.
optimizer = tf.keras.optimizers.Adam()

# evaluation을 위한 function을 정의합니다.
def evaluate(image, max_length, attention_features_shape, encoder, decoder, image_features_extract_model, tokenizer):
    attention_plot = np.zeros((max_length, attention_features_shape))
    
    hidden = decoder.reset_state(batch_size=1)
    
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    
    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    
    result = []
    
    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()
        
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])
        
        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)
        
    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

image_feature_extract_base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_feature_extract_base_model.input
hidden_layer = image_feature_extract_base_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# Load Caption data and then preprocess
# 가장 빈도수가 높은 15000개의 단어를 선택해서 Vocabulary set을 만들고,
# Vocabulary set에 속하지 않은 단어들은 <unk>로 지정합니다.
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

tokenizer.fit_on_texts(train_captions)
# 가장 긴 문장보다 작은 문장들은 나머지 부분은 <pad>로 padding합니다.
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

# caption 문장을 띄어쓰기 단위로 split해서 tokenize 합니다.
train_seqs = tokenizer.texts_to_sequences(train_captions)
# 길이가 짧은 문장들에 대한 padding을 진행합니다.
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

# attetion weights를 위해서 가장 긴 문장의 길이를 저장합니다.
max_length = calc_max_length(train_seqs)


# encoder와 decoder를 선언합니다.
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

# checkpoint 데이터를 저장할 경로를 지정합니다.
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=5)

start_epoch = 0
#assert ckpt_manager.latest_checkpoint != True
start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
# checkpoint_path에서 가장 최근의 checkpoint를 restore합니다.
ckpt.restore(ckpt_manager.latest_checkpoint)

image_path =  INPUT_DIR + '/test.png'
result, attention_plot = evaluate(image_path, max_length, attention_features_shape, encoder, decoder, image_features_extract_model, tokenizer)

predicted_caption = ' '.join(result)
predicted_caption = predicted_caption[:-5]
translator = Translator()
translated_caption = translator.translate(predicted_caption)

print('Prediction Caption:', predicted_caption)
print('Korean Caption : ',  translated_caption)

# 한국어 캡션 저장
f = open(OUTPUT_DIR+"/test.txt", mode='wt', encoding='utf-8')
f.write(translated_caption)
f.close()

# Attention Plot
plot_attention(image_path, result, attention_plot)

