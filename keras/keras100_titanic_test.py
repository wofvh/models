#작업흐름도 
#데이터 분류 *사람이 하는작업
#연관성 찾아서 분류 입력값과 예측값
#텍스트(이름 등 ) 숫자로 바꾸기 2빠진 값을 넣어서 데이터를 한눈에 알아볼수 있도록하기

from pydoc import describe
from bitarray import test
import numpy as np 
import pandas as pd #pandas >panal data 분석스템
import seaborn as sns 
import matplotlib.pyplot as plt
#머신러닝 튤 소환 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#작업 흐름도 work flow
#훈련세트 테스트 세트 두가지 있는데 불러오기
#훈련세트 데스크 데이터를 불러와서 각각 train, test 에 집어 넣기 이것을 
#패널데이터 분석 도구로 불러오기 (데이터 소환)
#맨왠쪽 숫자는 인덱스 넘버(항목 번호) PassengerID는 승객 들을 나열한 번호
#ticker 넘버 중요하지 않음 
#fare >> 요금 돈 얼마냇는지 따라 생존율 차이 예상
#cabin > 객실 >pclass 객실등급 >결과적으로 중요함 돈 낸만큼 좋은 객실 타입에 배정되기 때문 
#embarked


#1.데이터 처리 
path = './_data/kaggle_titanic/'

train = pd.read_csv(path +'train.csv')
test = pd.read_csv(path +'test.csv')

# train.info()
# print(300)       
# test.info()
#[891 rows x 12 columns]
#[418 rows x 11 columns]

#훈련세트에서 생존자 데이터와 나머지 항목들의 상관관계를 수학적 알고리즘으로 파악한우에 훈련
#데이터 셋트에 생존자 데이터 세트를 빼고 결과를 예측 
#옛날 자료다보니 빠진 데이터값이 당연하게 생김 
#정보를 소환 

#  #   Column       Non-Null Count  Dtype  
# ---  ------       --------------  -----  
#  0   PassengerId  891 non-null    int64  
#  1   Survived     891 non-null    int64  
#  2   Pclass       891 non-null    int64  
#  3   Name         891 non-null    object 
#  4   Sex          891 non-null    object 
#  5   Ag  *(빠진숫자)714 non-null    float64
#  6   SibSp        891 non-null    int64  
#  7   Parch        891 non-null    int64  
#  8   Ticket       891 non-null    object 
#  9   Fare         891 non-null    float64
#  10  Cabin        204 non-null    object 
#  11  Embarked     889 non-null    object 
# dtypes: float64(2), int64(5), object(5)  
# memory usage: 83.7+ KB

# Data columns (total 11 columns):
#  #   Column       Non-Null Count  Dtype  
# ---  ------       --------------  -----  
#  0   PassengerId  418 non-null    int64  
#  1   Pclass       418 non-null    int64  
#  2   Name         418 non-null    object 
#  3   Sex          418 non-null    object (글자로 나와서)
#  4   Age          332 non-null    float64
#  5   SibSp        418 non-null    int64  
#  6   Parch        418 non-null    int64   
#  7   Ticket       418 non-null    object  
#  8   Fare         417 non-null    float64 (연속된수자) 
#  9   Cabin        91 non-null     object  
#  10  Embarked     418 non-null    object  
# dtypes: float64(2), int64(4), object(5)   
# memory usage: 36.0+ KB

print(train.isnull().sum()) #isnull #sum (결측값 확인 법)

#빠진 값 >나이 177 개 , 객실687개, 승선지 2 개 Age 177, Cabin 687,Embarked 2
#자료를 알아보기 위해 컬럼들은 분류할거임 
#손가락으로 셀수있는 범위가 나와있는 categorical(엑셀처럼 치면 나오는거라 간단히 파악가능)
# /numeric(시각화 자료를 통해 파악함) 
#기계한테 설명시키기 describe 시키기

print(train.describe())

#        PassengerId  ...        Fare
# count   891.000000  ...  891.000000       
# mean    446.000000  ...   32.204208       
# std     257.353842  ...   49.693429       
# min       1.000000  ...    0.000000       
# 25%     223.500000  ...    7.910400       
# 50%     446.000000  ...   14.454200       
# 75%     668.500000  ...   31.000000       
# max     891.000000  ...  512.329200  
#사실 누가 생존했고 누가 돌아가셨는지 다 알고있음 
#사고로 인해서 돌아가신건지,실종되서 돌아가신지 불분명할수있다 

#총 2,224 이 탔는데 훈련 샘플은 891개임.>총 데이터중 40%가 제공됨
#. 살았는지 죽었는지 여부는 0과1로 구분된다
#타이타닉 침몰사고 생존률은 32%임 >descride 항목에 제공된 훈력세트의 평균 생존률 $38로
#실제 생존률이 아닌 훈련 데이터에 주어진 사람들의 생존률
# print(train.describe(include=0))
