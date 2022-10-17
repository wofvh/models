
#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
# x_train = np.array([1,2,3,4,5,6,7])
# x_test = np.array([8,9,10])
# y_train = np.array([1,2,3,4,5,6,7])
# y_test = np.array([8,9,10])

#[검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,                                                    
                                                    test_size=0.4, 
                                                    train_size=0.7,
                                                    #shuffle=true,
                                                    random_state=66)
print(x_train) #[2 7 6 3 4 8 5]