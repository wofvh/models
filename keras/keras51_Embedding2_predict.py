from lib2to3.pgen2 import token
from keras.preprocessing.text import Tokenizer
import numpy as np

#데이터 
docs = ['너무 재밋어요',' 참 최고에요','참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다','한번 더 보고싶네요','글세요',
        '별로에요','생각보다 지루해요','연기가 어색해요'
        '재미없어요','너무 재미없다','참 재밋네요','민수가 못 생기긴 했어요',
        '안결 혼해요']

# 긍정 1 , 부정 0 
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)


# {'참': 1, '너무': 2, '재밋어요': 3, '최고에요': 4, '잘': 5, '만든': 
# 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다'
# : 10, '한번': 11, '더': 12, '보고싶네요': 13, '글세요': 14, '재미없어요': 15, 
# ' 재미없다': 16, '재밋네요': 17, '민수가': 18, '못': 19, '생기긴': 20, '했어요':
#     21, '안결': 22, '혼해요': 23}

x = token.texts_to_sequences(docs) #pad 로 앞부분 부터 0을 채우는게 좋음 
print(x)

# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13], [14], [15], [16, 17],
#  [18, 19], [2, 20], [1, 21], [ [22, 23, 24, 25], [26, 27]]

from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x,padding='pre',maxlen=5, truncating='pre')  #
print(pad_x)
print(pad_x.shape)  #(13, 5) #rehape 해서 3~4차원으로 LSTM 가능 

word_size = len(token.word_index)
print('word_size:',word_size)    #word_size: 27 #단어사전의 갯수 : 30개 / 현재 27개
# print(np.unique(pad_x, return_counts=True))


# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 
# 15, 16,
#        17, 18, 19, 20, 21, 22, 23]), array([29,  3,  2,  1,  1,  1, 
#  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#         1,  1,  1,  1,  1,  1,  1], dtype=int64))

#.모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LSTM,Embedding #Embedding 통상 레이어스를 많이씀 

model = Sequential()
                  #input dim은 단어 사전에 겟수 output_dim
# model.add(LSTM()) 
model.add(Embedding(input_dim=31, output_dim=10, input_length=5)) #길이  input_length=5
# model.add(Embedding(input_dim=31, output_dim=10)) #길이  input_length=5(명시를 안 해도 알아서 잡아줌)
# model.add(Embedding(31, 10)) #(단어 사전의 갯수 31 아웃풋 10)
# model.add(Embedding(31, 10, 5)) 
# model.add(Embedding(31, 10, input_length=5)) # 10개짜리가 5개씩 묶여있음 (인풋랭스는 정확하게 명시 해야함)
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid')) 
model.summary()

#3.컴파일훈련 
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels ,epochs=16, batch_size=16)



# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 5, 3)              93 
# _________________________________________________________________
# lstm (LSTM)                  (None, 32)                4608
# _________________________________________________________________
# dense (Dense)                (None, 1)                 33
# =================================================================
# Total params: 4,734
# Trainable params: 4,734
# Non-trainable params: 0
# _________________________________________________________________

#4. 평가 ,예측 
acc = model.evaluate(pad_x,labels)[1]  #<<<<<< [0]을 넣으면 loss [1]을 넣으면 accrcy
print('acc:',acc)

###############실습#######################
x_predict = ['나는 형권이가 정말 재미없다 너무 정말']

#결과는 ??? 긍정????? 부정?????d,;sf]skf,wfㅇ니으ㅈ