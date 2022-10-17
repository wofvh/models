from keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding

#1. 데이터
docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글세요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밋네요', '민수가 못 생기긴 했어요',
        '안결 혼해요'
]
test_set = ['나는 형권이가 정말 재미없다 너무 정말']


# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0]) # (14,)

token = Tokenizer()
token.fit_on_texts(docs)
token.fit_on_texts(test_set)
print(token.word_index)
# {'너무': 1, '참': 2, '재미없다': 3, '정말': 4, '재밋어요': 5, '최고
# 에요': 6, '잘': 7, '만든': 8, '영화에요': 9, '추천하고': 10, '싶은': 11,
# '영화입니다': 12, '한': 13, '번': 14, '더': 15, '보고': 16, '싶
# 네요': 17, '글세요': 18, '별로에요': 19, '생각보다': 20, '지루해요': 21,
# '연기가': 22, '어색해요': 23, '재미없어요': 24, '재밋네요': 25, '민수가': 26,
# '못': 27, '생기긴': 28, '했어요': 29, '안결': 30, '혼해요': 31,
# '나는': 32, '형권이가': 33}

x = token.texts_to_sequences(docs)
test_set = token.texts_to_sequences(test_set)
print(x)
print(test_set)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17],
#  [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26, 27, 28], [29, 30]]

from keras.preprocessing.sequence import pad_sequences # 0채워서 와꾸맞춰줌
pad_x = pad_sequences(x, padding='pre', maxlen=5) #'post'도있음, 뒤 / truncating= 잘라내기 
pad_test_set = pad_sequences(test_set, padding='pre', maxlen=4, truncating='post') #'post'도있음, 뒤 / truncating= 잘라내기 
pad_test_set = pad_sequences(pad_test_set, padding='pre', maxlen=5) #'post'도있음, 뒤 / truncating= 잘라내기 
print(pad_x, pad_x.shape) # (14, 5)
print(pad_test_set, pad_test_set.shape) # (1, 5)

pad_x = pad_x.reshape(14,5,1)
pad_test_set = pad_test_set.reshape(1,5,1)

# [[ 0  0  0  2  3]
#  [ 0  0  0  1  4]
#  [ 0  1  5  6  7]
#  [ 0  0  8  9 10]
#  [11 12 13 14 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0  0 17]
#  [ 0  0  0 18 19]
#  [ 0  0  0 20 21]
#  [ 0  0  0  0 22]
#  [ 0  0  0  2 23]
#  [ 0  0  0  1 24]
#  [ 0 25 26 27 28]
#  [ 0  0  0 29 30]] 

word_size = len(token.word_index) # len(x,word_index) x의 인덱스 길이, 수
print("word_size : ", word_size)   # 단어사전의 갯수 : 30

print(np.unique(pad_x, return_counts=True))

# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 
# 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]), 
#  array([37,  3,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 
#  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],     
#       dtype=int64))

#2. 모델



model = Sequential()
                    #단어사전의 갯수  
# model.add(Embedding(input_dim=31, output_dim=11, input_length=5)) #embedding 에선 아웃풋딤이 뒤로 들어감
# model.add(Embedding(input_dim=31, output_dim=10)) # 인풋렝쓰는 모를 경우 안넣어줘도 자동으로 잡아줌
# model.add(Embedding(31, 10))
# model.add(Embedding(31, 10, 5)) # error
model.add(LSTM(32, return_sequences=True, input_shape = (5,1)))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Model: "sequential"
# _________________________________________________________________   
# Layer (type)                 Output Shape              Param #      
# =================================================================   
# embedding (Embedding)        (None, 5, 20)             680
# _________________________________________________________________   
# lstm (LSTM)                  (None, 5, 32)             6784
# _________________________________________________________________   
# lstm_1 (LSTM)                (None, 5, 32)             8320
# _________________________________________________________________   
# lstm_2 (LSTM)                (None, 32)                8320
# _________________________________________________________________   
# dense (Dense)                (None, 1)                 33
# =================================================================   
# Total params: 24,137
# Trainable params: 24,137
# Non-trainable params: 0
# _________________________________________________________________   
# Epoch 1/200

#3. 컴파일, 훈련

es = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, 
                              restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(pad_x, labels, epochs=200, batch_size=128 ,validation_split=0.2,callbacks=[es])

#4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
print('acc : ', acc)

#### [실습] ####
# 결과는???

y_predict = model.predict(pad_test_set)


print(y_predict)

# [[0.78815293]]

# acc :  0.6142857670783997
# [[[0.4757533 ]
#   [0.58985746]
#   [0.6378508 ]