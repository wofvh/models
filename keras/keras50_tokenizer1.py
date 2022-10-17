from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder

text = '진짜 매우 나는 나는 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)

x = token.texts_to_sequences([text])
print(x)
# [[4, 2, 3, 3, 2, 5, 6, 7, 1, 1, 1, 8]]

from tensorflow.python.keras.utils.np_utils import to_categorical

# x = to_categorical(x)
# print(x)
# print(x.shape)  

# [[4, 2, 3, 3, 2, 5, 6, 7, 1, 1, 1, 8]]
# [[[0. 0. 0. 0. 1. 0. 0. 0. 0.]        
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]        
#   [0. 0. 0. 1. 0. 0. 0. 0. 0.]        
#   [0. 0. 0. 1. 0. 0. 0. 0. 0.]        
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1.]]]
# (1, 12, 9)    <<<<차원 rnn LSTM , 


ohe = OneHotEncoder(sparse=True)
x = ohe.fit_transform(x.reshape(-1,1,0))
x = np.array

print(x)
