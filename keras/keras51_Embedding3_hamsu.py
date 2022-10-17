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


{'참': 1, '너무': 2, '재밋어요': 3, '최고에요': 4, '잘': 5, '만든': 
6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다'
: 10, '한번': 11, '더': 12, '보고싶네요': 13, '글세요': 14, '재미없어요': 15, 
' 재미없다': 16, '재밋네요': 17, '민수가': 18, '못': 19, '생기긴': 20, '했어요':
    21, '안결': 22, '혼해요': 23}

x = token.texts_to_sequences(docs) #pad 로 앞부분 부터 0을 채우는게 좋음 
print(x)

# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13], [14], [15], [16, 17],
#  [18, 19], [2, 20], [1, 21], [ [22, 23, 24, 25], [26, 27]]

from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x,padding='pre',maxlen=6 ,truncating= 'pre' ) #maxlen 최대글자 설정 
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
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense,LSTM,Embedding,Input #Embedding 통상 레이어스를 많이씀

input1 = Input(shape=(5,))
dense1 = (Embedding(input_dim=31, output_dim=10))(input1) #길이  input_length=5(명시를 안 해도 알아서 잡아줌)
dense2 = Dense(18,activation='relu')(dense1)
dense3 = Dense(20,activation='sigmoid')(dense2)
dense4 = Dense(20,activation='sigmoid')(dense3)
output1 = Dense(20,activation='sigmoid')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()


'''
#3. 컴파일. 훈련

model.compile(loss='mae', optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)   
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=800, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time()

model.save("./_save/keras22_hamsu01_boston.h5")
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
   
print('loss : ', loss)
print('r2스코어 : ', r2)
#StandardScaler()
'''