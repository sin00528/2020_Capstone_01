import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from tqdm import tqdm

import nltk
from nltk.tokenize import RegexpTokenizer

from keras.models import Model
from keras.layers import Dense

from keras.preprocessing import image
from keras.applications import InceptionV3

from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential

from caption_model import BahdanauAttention, CNN_Encoder, RNN_Decoder
from caption_utils import load_image, calc_max_length, plot_attention, cache_bottlenecks

import warnings
warnings.filterwarnings(action='ignore')

# Setting dirs
TRAIN_INPUT = "../Segmentation/COCO/train_input.npy"
TRAIN_LABEL = "../Segmentation/COCO/train_label.npy"

TEST_INPUT = "../Segmentation/COCO/val_input.npy"
TEST_LABEL = "../Segmentation/COCO/val_label.npy"

CKPT_DIR= "log/imgCaption"
PLOT_DIR = "log/plot"

# Load Train sets
img_name_vector = np.load(TRAIN_INPUT)
train_captions = np.load(TRAIN_LABEL)


# Load Test sets
test_img_name_vector = np.load(TEST_INPUT)
test_captions = np.load(TEST_LABEL)


tf.enable_eager_execution()

# Setting Hyperprams
# 학습을 위한 설정값들을 지정합니다.
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
top_k = 15000
vocab_size = top_k + 1
attention_features_shape = 64
EPOCHS = 20

# sparse cross-entropy 손실 함수를 정의합니다.
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)


# 최적화를 위한 Adam 옵티마이저를 정의합니다.
optimizer = tf.keras.optimizers.Adam()


# 최적화를 위한 function을 정의합니다.
@tf.function
def train_step(img_tensor, target, tokenizer, encoder, decoder):
    loss = 0
    
    # 매 batch마다 hidden state를 0으로 초기화합니다.
    hidden = decoder.reset_state(batch_size=target.shape[0])
    
    # <start>로 decoding 문장을 시작합니다.
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)
    
    with tf.GradientTape() as tape:
        features = encoder(img_tensor)
        
        for i in range(1, target.shape[1]):
            # feature를 decoder의 input으로 넣습니다.
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            
            loss += loss_function(target[:, i], predictions)
            
            # teacher forcing 방식으로 학습을 진행합니다.
            dec_input = tf.expand_dims(target[:, i], 1)
            
    total_loss = (loss / int(target.shape[1]))
        
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    
    return loss, total_loss

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


# Load InceptionV3 model
image_feature_extract_base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_feature_extract_base_model.input
hidden_layer = image_feature_extract_base_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# Cache #
#cache_bottlenecks(test_img_name_vector, image_features_extract_model)

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


# 데이터의 80%를 training 데이터로, 20%를 validation 데이터로 split합니다.
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)

print('train image size:', len(img_name_train), 'train caption size:', len(cap_train))
print('validation image size:',len(img_name_val), 'validation caption size:', len(cap_val))

num_steps = len(img_name_vector) // BATCH_SIZE

# Load Cashed data
# disk에 caching 해놓은 numpy 파일들을 읽습니다.
def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap


dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
# numpy 파일들을 병렬적(parallel)으로 불러옵니다.
dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32]),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)


# tf.data API를 이용해서 데이터를 섞고(shuffle) batch 개수(=64)로 묶습니다.
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# encoder와 decoder를 선언합니다.
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

# checkpoint 데이터를 저장할 경로를 지정합니다.
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=5)


start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # CKPT_DIR에서 가장 최근의 checkpoint를 restore합니다.
    ckpt.restore(ckpt_manager.latest_checkpoint)

loss_plot = []

# 지정된 epoch 횟수만큼 optimization을 진행합니다.
for epoch in range(start_epoch+1, EPOCHS+1):
    start = time.time()
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target, tokenizer, encoder, decoder)
        total_loss += t_loss
        
        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch, batch, batch_loss.numpy() / int(target.shape[1])))
            
    # 추후에 plot을 위해서 epoch별 loss값을 저장합니다.
    loss_plot.append(total_loss / num_steps)
        
    # 5회 반복마다 파라미터값을 저장합니다.
    if epoch % 5 == 0:
        ckpt_manager.save(checkpoint_number=epoch)

    print ('Epoch {} Loss {:.6f}'.format(epoch, total_loss/num_steps))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    
print('Training Finished !')


# caption 문장을 띄어쓰기 단위로 split해서 tokenize 합니다.
test_seqs = tokenizer.texts_to_sequences(test_captions)
# 길이가 짧은 문장들에 대한 padding을 진행합니다.
test_cap = tf.keras.preprocessing.sequence.pad_sequences(test_seqs, padding='post')

regexTokenizer = RegexpTokenizer("[\w]+")

BLEUscores = []
# Test set에대한 BLEU Score를 계산합니다.
for idx in tqdm(range(len(test_img_name_vector))):
#for idx in tqdm(range(1000)):
    image = test_img_name_vector[idx]
    ground_truth_caption = ' '.join([tokenizer.index_word[i] for i in test_cap[idx] if i not in [0]])
    result, _ = evaluate(image, max_length, attention_features_shape, encoder, decoder,
                         image_features_extract_model, tokenizer)
    result = ' '.join(result)[:-5]
    tok_caption = regexTokenizer.tokenize(ground_truth_caption[7:-5])
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([tok_caption], result)
    BLEUscores.append(BLEUscore)

# Average Bleu Score 
avgBLEU = np.sum(BLEUscores)/len(BLEUscores)
print("Average BLEU Score: {:.6f}".format(avgBLEU))

# bleu Plot
sns.distplot(BLEUscores, kde=False)
plt.title("Average BLEU Score: {:.6f}".format(avgBLEU))
plt.ylabel('Frequency')
plt.xlabel('BLEU score')
plt.savefig('log/plot/imgCap_bleu.png')

# send a msg to discord channel
from discord_webhook import DiscordWebhook
url = 'https://discordapp.com/api/webhooks/710208007618822145/4yUFIEoTa7kZFOhyJpSkalNn2NysrM6p5PFVG5iBDkt1ikJxBPwV3_J4FDYi40THgxvl'
webhook = DiscordWebhook(url=url, content='Calculating BLUE Score is completed...')
response = webhook.execute()