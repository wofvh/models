#BERT 

import os 
import re
import json
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import *
import keras
from keras_preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tokenize,tokenizers
tf.random.set_seed(111)#랜덤시드 고정
np.random.seed(111)
BATCH_SIZE = 32
NUM_EPOCHS = 3 #num
VALID_SPLIT = 0.2 #validation set을 만들기 위해 train set을 8:2로 나눈다.
MAX_LEN = 39 


#BERT 모델 불러오기

from transformers import BertTokenizer, TFBertModel
import urllib.request
#한글 데이터
train_file = urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
test_file = urllib.request.urlopen("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt")

train_data = pd.read_table(train_file)
test_data = pd.read_table(test_file) #read_table은 deprecated 되었으므로 read_csv로 대체

#데이터 전처리
# train_data.drop_duplicates(subset=['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
test_data = test_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(train_data.head())
print(test_data.head())


# def Bert_Tokenizer(sentence, MAX_LEN):
#     encoded_dict = tokenize.encode_plus(
#                         sentence,  # Sentence to encode.           
#                         add_special_tokens = True, # Add '[CLS]' and '[SEP]'
#                         max_length = MAX_LEN,           # Pad & truncate all sentences.
#                         pad_to_max_length = True, # Pad & truncate all sentences.
#                         return_attention_mask = True,   # Construct attn. masks.
#                         return_tensors = 'tf',     # Return pytorch tensors.
#                    )
    
#     input_id = encoded_dict['input_ids']
#     attention_mask = encoded_dict['attention_mask']
#     token_type_id = encoded_dict['token_type_ids']
    
#     return input_id, attention_mask, token_type_id

# input_ids = []
# attention_masks = []
# token_type_ids = []
# train_data_labels = []

# for train_sentence, train_label in tqdm(zip(train_data['document'], train_data['label']), total=len(train_data)):
#     try:
#         input_id, attention_mask, token_type_id = Bert_Tokenizer(train_sentence, MAX_LEN)
        
#         input_ids.append(input_id) #    numpy()는 tensor를 numpy array로 변환
#         attention_masks.append(attention_mask)
#         token_type_ids.append(token_type_id)
#         train_data_labels.append(train_label)
#     except Exception as e:
#         print(e)
#         print(train_sentence)
#         pass

