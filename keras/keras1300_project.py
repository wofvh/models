
from tabnanny import verbose
import numpy as np
import pandas as pd
import os
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50, ResNet101
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image


#1. 데이터 로드

train = pd.read_csv('./_data/project/cls_data.csv',index_col=0)
# print(train.head)
train_img = 'C:/study/_data/test/yoonyeojeong/'


# 이미지 데이터 가져오기
img_result = []

for file in os.listdir(train_img): 
    img_file = file
    img_result.append(img_file) 
# print(len(img_result))  # 2039


# 라벨링 tokenizer    
labels = train['label']
print(labels.head)

y = np.array(labels[0])
for i in range(1,30):
    y = np.vstack((y, labels[i]))
y = np.array(y)
print(y.shape) #(30, 1)


# Image DataGenerator
'''
import tqdm
from tensorflow.keras.utils import load_img, img_to_array
train_image = []
for i in tqdm.tqdm(range(train.shape[0])):
    img = load_img(train_img + str(i + 1) +'.jpg', target_size=(100, 100, 3))
    img = img_to_array(img)
    img = img/255
    train_image.append(img)
x = np.array(train_image)
print(x.shape) # (3227, 50, 60, 3), (2298, 50, 60, 3), (1830, 50, 60, 3), (1539, 50, 60, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=False) 




# test_datagen = ImageDataGenerator(
#     rescale=1./255,)  

# xy_train = test_datagen.flow(x_train,y_train,
#                                   batch_size=4380,shuffle=False)


# xy_test = test_datagen.flow(x_test,y_test,
#                        batch_size=4380,shuffle=False)
# x_train = xy_train[0][0]
# y_train = xy_train[0][1]
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]

print(x_train.shape)  
print(y_train.shape) 
print(x_test.shape)   
print(y_test.shape)
# (3492, 4, 2)
# (3492, 2)
# (873, 4, 2)
# (873, 2)
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense , Conv2D , Flatten,MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(2, 2), padding='same', input_shape=(4,2), activation='relu'))
# model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3), activation='relu'))
# model.add(MaxPooling2D())
model.add(Conv2D(256,(3,3), activation='relu'))
# model.add(MaxPooling2D())
model.add(Conv2D(128,(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()


#3. 컴파일, 훈련\
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, validation_split = 0.2, epochs=10 , batch_size=32)



#4. 평가, 예측
loss = model.evaluate([train_gen ]) 
y_predict = model.predict(valid_gen)
print('loss : ', loss)

y_predict=model.predict([valid_gen])

from sklearn.metrics import r2_score
r2 = r2_score(y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)

loss = model.evaluate(valid_gen)
print('loss :', loss)
y_predict = np.argmax(y_predict,axis=1)
print('y_predict :',y_predict) 
from random import *

# 조건를 충족하는 데이터를 필터링하여 새로운 변수에 저장합니다.
# bal = df[valid_gen]
# i = randrange(40)  # 0부터 39 사이의 임의의 정수
# print(i)
# bal = '{} - {}'.format(bal['title'][i],bal['artist'][i])
# 결과를 출력합니다.
if y_predict[0]   ==   1  : print('사회성 좋은 화개살 강한 관상\
갈매기 상의 이마는 슬기롭고 지혜로운 여성을 말합니다. 재벌가의 눈썹을 지녔고 
부모복이 좋고 남편복이 좋습니다. 눈이 매혹적이고 화개살이
있는 눈이라 사람들에게 사랑받고 이성운이 강한 눈이나 출세하지 
않으면 필요없는 남자들이 많이 붙을 수 있습니다. 이성을 많이 꼬시는 눈이고 
예능기질이 강하며 성격이 온순하고 착하고 선한 성격입니다')
elif y_predict[0] ==   2  : print('머리가 뛰어난 천재형 관상
이마가 짱구이마에 가까워 머리가 좋고 지혜롭습니다. 노력은 잘 안하는편이지만 
머리는 뛰어난 재능충 유형입니다. 코는 재복이 좋고 여복이 좋으며 연애운이 좋은 
눈입니다. 강인하며 직업운이 좋고 말년운이 깔끔하며
재능이 다재다능한 상입니다. 원숭이상에 가까운
상이라 재능을 통해 화려한 인생을 살 수도 있습니다. ')
elif y_predict[0] ==   3  : print('마음만 먹으면 뭐든지 할 수 있는 관상
코와 귀가 아주 재복이 좋고 특히 귀는 재복에 완벽한 귀입니다. 
낭비벽이 약간 있으나 신경쓰지 않아도 됩니다. 얼굴형은 미남형이라 연예인을 하면
적성에 맞을 것이며 이마가 사각형이라 사회성도 매우 좋습니다. 
눈은 잘생김을 넘어서 아름다운 눈입니다. 아름다운 눈은 정에 약한 눈입니다.
눈이 들어간 형태라 판단력과 머리 회전이 빠릅니다.
하지만 약간 외로운 상이며 결혼은 하고 싶을 때 하면 됩니다.
전체적으로 정적이며 과묵한 관상입니다. ')
elif y_predict[0] ==   4  : print('강한 기를 가지고 있는 관상
이마는 적당하며 당돌한 면이 있고 눈썹이 길고 진하여 재복이 좋습니다. 
눈은 이성을 유혹하는 눈으로서 기가 강하며 연애운이 강합니다.
이성을 꼬시는 재주가 좋고 자존심이 강한 입입니다
볼은 욕심이 많고 고집이 강하며 이기적인 면도 있지만 재복이 
좋고 생활력이 강하며 인복이 좋습니다. 전체적으로 흠잡을 곳이 없는 
완벽한 관상이지만 단점으로는 얼굴에서 나오는 기가 강합니다. ')
elif y_predict[0] ==   5  : print('평범한 인생을 살지 않을 관상
이마를 볼때 성격은 착하고 인정이 많습니다. 하지만 뺀질이 기질도 있고 남의 말을 
잘 안 듣는 성향이강합니다. 눈은 다정하면서 차분하고 사색적이고 반항아기질이 있고 개성이 강하고
주관이 뚜렷한 눈입니다. 남과 다른 인생을 살려하고 평범한 인생을 싫어하는 눈입니다. 
턱은 강인한편이라 재복이 좋고 말년운이 좋지만 예민한면도 강합니다. ')
elif y_predict[0] ==   6  : print('평생 재물운은 타고난 관상
좋은 이마를 가졌고 성격이 강인하며 정적입니다. 눈이 봉황눈에 가깝기 때문에 재복이 좋은 눈이고 
사람을 끌어들입니다. 눈썹을 보니 성격이 강직하니 자존심이 강하며 고집이 있습니다. 귀는 인복이
좋으며 정이 많은 귀입니다. 턱선이 약간 갸름하여 성격이 원만하고 입은 무겁습니다
전체적으로 사람을 끌어들이는 기운이 강하여 성공운이 열린 얼굴입니다. ')
elif y_predict[0] ==   7  : print('이성운이 강한 관상
이성운이 좋고 두뇌가 좋습니다.
자아도취적인 경향도 강하며 의지력이 강하고
무엇이든지 집중하는 성격입니다. 눈을 보면 이성운이 강하며 재물에 대한 집착이 강합니다
화려하고 동시에 깨끗하고 수려한 것을 추구하는 상입니다. 낙천적이고 
긍정적인 마인드를 볼 수 있는 입꼬리와 입이 미세하게 나와있어 말재주가 좋습니다.
')  
elif y_predict[0] ==   8  : print('섬세하고 끈기가 있으며 의지력이 강한 관상
눈썹은 굵고 진하여 재물복이 좋습니다. 끈기가 강한 눈썹입니다. 눈썹과 눈 사이에 살이 별로 없는데
이기주의적인 성향이 조금 있으나 성격이 세심합니다.
날카로운 눈은 예리하고 관찰력이 좋으며 눈치가 빠릅니다.
코를 보면 재물복이 좋으며 재물에 대한 욕심이 있습니다.
재주와 재능이 좋은 인중이며 예능계열쪽으로 좋은 하관입니다. 긍정적인 성향으로 말년운이 무난합니다.
')
elif y_predict[0] ==   9  : print('머리가 좋고 지혜로운 관상
이마가 갈매기상이라 현모양처상이며 지혜롭습니다. 얼굴해 비해 코가
크지만 그에 비해 콧구멍은 보이지 않아 재물이 들어올 상이며 귀를 보아하니 
출세를 해야 재복이 산더미처럼 들어올 귀입니다. 턱도 갸름하니 성격이 좋아보이며 
이마가 너무 훨칠하여 남편과 트러블이
발생할 수도 있지만 지혜롭기 때문에 모든지 지혜롭게 대쳐해 나갈 수 있는 관상입니다.
 ')
elif y_predict[0] ==   10 : print('무엇을 받으면 몇 배로 베푸는 진정한 이타주의자입니다. 이들은 종종 의료 부분이나 학문,
혹은 사회단체와 같이 오랜 역사나 전통과 관련된 분야에 종사합니다. 다만 업적이나 실적을 다른 
사람들이 알아차리게 하는 데 어려움을 느낍니다.
이들은 종종 자신이 이룬 성취를 과소평가하는 경향이 있는데,
이러한 겸손한 태도로 종종 존경을 받지만 때로는 이용당할 수도 있습니다.
 ')
elif y_predict/[0] ==   11 : print('상상력이 풍부하고 철두철미한 계획을 세우는 전략가형 전체 인구의 2%도 안되는 극소수의 성격 유형입니다. 
오랜 시간 방대한 지식을 쌓아 온 똑똑하고 자신감 넘치는 이들이지만 
인간관계만큼은 이들이 자신 있어 하는 분야는 아닙니다. 진리나 깊이 있는 지식을 좇는
이들에게 선의의 거짓말이나 가벼운 잡담은 그저 낯설기만 합니다. ')
elif y_predict[0] ==   12 : print('사실에 근거하여 사고하며 결정에 한 치 의심없는 현실주의자, 논리주의자형은 가장 다수의 사람이
속하는 성격 유형으로 인구의 대략 13%를 차지합니다. 
법 규제 기관 혹은 군대와 같이 전통이나 질서를 중시하는 조직에서 핵심 구성원 역할을 합니다.
 ')
elif y_predict[0] ==   13 : print('갑자기 흥얼거리며 즉흥적으로 춤을 추기 시작하는 누군가가 있다면 그게 당신입니다.
천부적으로 스타성 기질을 타고난 이들은 그들에게 쏟아지는 스포트라이트를 즐기며 
어디를 가나 모든 곳이 이들에게는 무대입니다.
어디에 콕 박혀 친구나 사람들과 어울리지 못하는 자신을 발견하는 것을 괴로워합니다.
 ')  
elif y_predict[0] ==   14 : print('외향성이 높아서 즐겁고 친절하면 사람 만나기를 좋아합니다 하지만 많은 사람 앞에서 
의견을 강하게 주장하고 분위기를 주도할 정도로 높은 것은 아닙니다
책임의식과 자제력이 있고 체계적이며 의욕이 있습니다 또 개방성이 높아서 새로운 것에 열려 있는 태도를 보이며
새로운 시도를 하는 것을 좋아하고 호기심이 많으며 다방면에 관심이 잇습니다
')
elif y_predict[0] ==   15 : print('대담하면서도 상상력이 풍부하고 당신은 천성적으로 타고난 리더입니다
넘치는 카리스마와 자신감으로 목표를 향해 사람들을 이끌 수 있으며
목표 성취가 삶의 원동력입니다. 
또한 진정성 있는 인간관계를 구축해나가는 경향이 있습니다 ')
elif y_predict[0] ==   16 : print('
사물이나 사람을 관리하는데 탁월한 능력을 가진 사람, 전 세계 유명 비즈니스 리더나 
정치인 중 상당수가 이 유형에 속합니다. 기본적으로 사람들과의 약속을 충실히 이행하는 
이들의 기본 성향 때문에 함께 일하는 동업자나 부하의 무능력함, 태만,
심지어는 부정직함으로 이들을 시험에 들게 하는 경우 심한 불호령도 마다하지 않습니다.
')
elif y_predict[0] ==   17 : print('상상력이 풍부하고 철두철미한 계획을 세우는 전략가형. 
전체 인구의 2%도 안되는 극소수의 성격 유형입니다. 오랜 시간 방대한 지식을 쌓아 온 똑똑하고
자신감 넘치는 이들이지만, 인간관계만큼은 이들이 자신 있어 하는 
분야는 아닙니다. 진리나 깊이 있는 지식을 좇는
이들에게 선의의 거짓말이나 가벼운 잡담은 그저 낯설기만 합니다.
 ')
elif y_predict[0] ==   18 : print('이마가 적당하면서 약간 특이한 것이 남편을 좌지우지 하는 이마입니다.
눈썹과 코는 재복이 아주 좋습니다. 또한 귀와 눈꼬리가 옆에 있기 때문에 현실주의자입니다.
얼굴에 비해서 입이 약간 작은 것이 자기 속마음을 남에게 잘 말하지 않지만 남자와 달리 여자는
입이 작아도 괜찮습니다. 
눈이 약간 부리부리하니 자기 주관대로 가정을 꾸려나갈 눈입니다. ')
elif y_predict[0] ==   19 : print('초년 중년 말년이 대체로 안정적이다
그러므로 그 기운을 많은 사람들이 같이 나누기를 원한다
주변에서 음해하는 경우가 생길수 있으니 다른 사람들보다
스스로 자신을 보호하는 능력을 가져야 한다 감정을 중시하고 천진하고 착하다, 동정심이 
많아 금전이나 애정문제로 남에게 쉽게 이용당하기도 한다')  
elif y_predict[0] ==   20 : print('상냥하고 이타적인 밝은 사회건설에 앞장서는 낭만형
하지만 이들은 종종 깊은 생각의 나락으로 자신을 내몰아 이론적 가설이나 혹은 
철학적 논리에 빠지기도 하는데 꾸준한 관심을 가지고 이들을 지켜보지 않으면 
이들은 연락을 끊고 은둔자 생활을 하기도 합니다 ')
elif y_predict[0] ==   21 : print('가장 흔치 않은 성격 유형으로 인구의 채 1%도 되지 않습니다. 그럼에도 불구하고 나름의 고유 성향으로 세상에서 
그들만의 입지를 확고히 다집니다. 
이들 안에는 깊이 내재한 이상향이나 도덕적 관념이 자리하고 있는데, 
다른 외교형 사람과 다른 점은 이들은 단호함과 결단력이 있다는 것입니다. 바라는 
이상향을 꿈꾸는데 절대 게으름 피우는 법이 없으며, 목적을 달성하고 지속적으로 \
긍정적인 영향을 미치고자 구체적으로 계획을 세워 이행해 나갑니다.
 :')  
elif y_predict[0] ==   22 : print('남자가 꼬이고 남편때문에 마음고생할 수 있는 이마입니다. 성격이 남자다운면도 있고 서글서글하고 붙임성이 
좋고 사회성이 좋습니다. 연애운도 좋고 눈썹이 가지런하여 여성스럽고 차분합니다.
좋은 성격은 다가지고 있다고 보시면 됩니다.
하지만 돌아서면 아주 냉담한 면도 있습니다.
얼굴이 개성이 강한 얼굴이라 사람 마음을 끌어당기는 면이 강합니다. :')  
elif y_predict[0] ==   24 : print('자유로운 사고의 소유자입니다. 종종 분위기 메이커 역할을 하기도 하는 이들은 단순한 인생의 
즐거움이나 그때그때 상황에서 주는 일시적인 만족이 아닌 타인과 사회적,
정서적으로 깊은 유대 관계를 맺음으로써 행복을 느낍니다.
매력적이며 독립적인 성격으로 활발하면서도 인정이 많은 이들은 인구의 
대략 7%에 속하며, 어느 모임을 가든 어렵지 않게 만날 수 있습니다.
:')  
elif y_predict[0] ==   25 : print('진실하게 행동하는 자신의 모습에서 자부심을 느끼며, 자기 생각을 솔직하게 이야기하고 어떤 것에 헌신하기로 한 경우 최선을 다합니다.
현실주의자는 인구의 상당 부분을 차지합니다. 화려한 삶이나 다른 사람의 주의를 끄는 일에는 관심이 없으며, 안정된 사회를 위해 자신의 
몫보다 많은 기여를 하곤 합니다. 이들은 가족이나 주변 사람들로부터 믿음직한 사람이라는 평판을 받을 때가 많으며,
현실 감각이 뛰어나 스트레스가 극심한 상황에서도 현실적이고 논리적인 태도를 유지하는 사람으로 인정받곤 합니다.
 :')  
elif y_predict[0] ==   26 : print(성격이 어질고 동정심이 많습니다 그러나 의지가 강하여 완고한 면도 있죠
친화성이 높아 배려심이 있고 긍정적인 관계를 유지하려하며 조금의 다툼이라도 있으면 마음이 불편해지는 
경향이 있어서 먼저 화해를 시도하거나 관계를 개선시키려고 노력하는 편입니다
개방성이 조금 높아서 어떤 형식에 얽매이는 것을 싫어하고 통찰력이 있는 편이면 혼자만의
환상과 공상을 즐기기도 한다 또 어떤 문제를 만나면 근본적인 해결책을 찾고자 합니다
 :')  
elif y_predict[0] ==   27 : print('이마는 적당하면서 훤칠하여 결혼운이 좋고 배우자복이 좋은 이마입니다 눈썹은 길고 진하면 재복이 좋고 미간은 적당하고
눈은 예술 감각이 좋고 감정이 풍부하고 선하고 여자복 좋고 연애운도 좋은 눈입니다 
재물복이 좋으나 대인관계에 주의 하셔야합니다 또 인정이 많고 심성이 착한 편이지만 이성보는 눈이 높은편입니다
 :')  
elif y_predict[0] ==   28 : print('외향성이 높아서 성공에 대한 욕구가 강하고 다른사람 사람들의 관심과 주목을 받고 싶어하면
경쟁심과 승부욕 이 강한 편이다 또 개방성이 높아서 창의적이고 독특한 사고방식을 가지고 있다
개인사업은 어울리지 않을수도 있습니다 또 감각적이고 증흥적인 스타일 입니다 그래서 인생이 복잡하고 
다채로우면서 파란만장하고 기복이 심한 편임 :')  
elif y_predict[0] ==   29 : print('눈썹이 재복이 좋은 눈썹입니다. 눈은 내조를 잘하는 눈입니다. 
입은 자기 속마음을 잘 말하지 않아 보이고 
턱은 말년에 운이 좋아 연말에 행운을 기대해도 좋을 것 같습니다.
귀가 낭비벽이 심한 귀라 절제를 해야합니다. 하나 문제라면 이마때문에 
남편이 사고칠 수 있는 이마인데 착한 남편을 만나면 좋을 관상입니다. :')  
'''
