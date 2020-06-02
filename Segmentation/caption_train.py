#!/usr/bin/env python
# coding: utf-8

# In[1]:

import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.models import Model
from keras.layers import Dense
from keras.utils import plot_model

from keras.preprocessing import image
from keras.applications import InceptionV3

from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation, LSTM, CuDNNGRU, CuDNNLSTM
from keras.layers import Dropout, Conv1D, MaxPooling1D, GlobalMaxPool1D
from keras.layers import Bidirectional, InputLayer

from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint


# ## Setting dirs

# In[2]:


TRAIN_INPUT = "COCO/train_input.npy"
TRAIN_LABEL = "COCO/train_label.npy"

TEST_INPUT = "COCO/val_input.npy"
TEST_LABEL = "COCO/val_label.npy"

CKPT_DIR= "log/imgCaption"
PLOT_DIR = "log/plot"


# ## Load Train sets

# In[3]:


img_name_vector = np.load(TRAIN_INPUT)
train_captions = np.load(TRAIN_LABEL)


# ## Load Test sets

# In[4]:


test_img_name_vector = np.load(TEST_INPUT)
test_captions = np.load(TEST_LABEL)


# In[5]:


tf.enable_eager_execution()


# ## Setting Hyperprams

# In[6]:


# 학습을 위한 설정값들을 지정합니다.
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
top_k = 5000
vocab_size = top_k + 1
attention_features_shape = 64
EPOCHS = 20


# ## Model Define

# In[7]:


# tf.keras.Model을 이용해서 Attention 모델을 정의합니다.
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # sum 이후에 context_vector shape == (batch_size, embedding_dim)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# In[8]:


# tf.keras.Model을 이용해서 CNN Encoder 모델을 정의합니다.
class CNN_Encoder(tf.keras.Model):
    # 이미 Inception v3 모델로 특징 추출된 Feature map이 인풋으로 들어오기 때문에
    # Fully connected layer를 이용한 Embedding만 수행합니다.
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)
    
    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


# In[9]:


# tf.keras.Model을 이용해서 RNN Decoder 모델을 정의합니다.
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)
        
    def call(self, x, features, hidden):
        # attention은 별도의 모델로 정의합니다.
        context_vector, attention_weights = self.attention(features, hidden)

        # embedding 이후에 x shape == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # concatenation 이후에 x shape == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # concatenated vector를 GRU에 넣습니다.
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


# ## Function define

# In[10]:


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


# In[11]:


# sparse cross-entropy 손실 함수를 정의합니다.
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


# In[12]:


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)


# In[13]:


# 최적화를 위한 Adam 옵티마이저를 정의합니다.
optimizer = tf.keras.optimizers.Adam()


# In[14]:


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


# In[15]:


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


# In[16]:


# Inception v3의 input에 적합한 형태로 image_path 경로에서 이미지를 불러옵니다.
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    
    return img, image_path


# In[17]:


# 전체 dataset에 존재하는 caption의 maximum length를 찾습니다.
def calc_max_length(tensor):
    return max(len(t) for t in tensor)


# In[18]:


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
    plt.savefig(image.split(os.path.sep)[-1].split('.')[-2] + ' attention' + '.png')
    plt.show()


# # Load InceptionV3 model

# In[19]:


img_feature_extract_base_model = tf.keras.applications.InceptionV3(include_top=False, weights=None)
new_input = img_feature_extract_base_model.input
hidden_layer = img_feature_extract_base_model.layers[-1].output

img_feature_extract_model = tf.keras.Model(new_input, hidden_layer)


# In[20]:


img_feature_extract_model.summary()


# In[21]:


#cache_bottlenecks(img_name_vector, img_feature_extract_model)


# ## Load Caption data and then preprocess

# In[22]:


# 가장 빈도수가 높은 5000개의 단어를 선택해서 Vocabulary set을 만들고,
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


# In[23]:


# 데이터의 80%를 training 데이터로, 20%를 validation 데이터로 split합니다.
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)


# In[24]:


print('train image size:', len(img_name_train), 'train caption size:', len(cap_train))
print('validation image size:',len(img_name_val), 'validation caption size:', len(cap_val))


# In[25]:


num_steps = len(img_name_train) // BATCH_SIZE


# ## Load Cashed data

# In[26]:


# disk에 caching 해놓은 numpy 파일들을 읽습니다.
def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap


# In[27]:


dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
# numpy 파일들을 병렬적(parallel)으로 불러옵니다.
dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32]),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)


# In[28]:


# tf.data API를 이용해서 데이터를 섞고(shuffle) batch 개수(=64)로 묶습니다.
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# In[29]:


# encoder와 decoder를 선언합니다.
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)


# In[30]:


# checkpoint 데이터를 저장할 경로를 지정합니다.
checkpoint_path = "./log/imgCaption"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)


# In[31]:


start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # checkpoint_path에서 가장 최근의 checkpoint를 restore합니다.
    ckpt.restore(ckpt_manager.latest_checkpoint)

loss_plot = []


# In[ ]:


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


# In[ ]:


plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.savefig('Loss plot.png')
#plt.show()
plt.savefig(plotDir + 'log/imgCaption/img_cap_loss.png')

from discord_webhook import DiscordWebhook
url = 'https://discordapp.com/api/webhooks/710208007618822145/4yUFIEoTa7kZFOhyJpSkalNn2NysrM6p5PFVG5iBDkt1ikJxBPwV3_J4FDYi40THgxvl'
webhook = DiscordWebhook(url=url, content='Model Train completed...')
response = webhook.execute()

