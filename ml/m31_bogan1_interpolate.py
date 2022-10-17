#타이핑 치기 진짜 힘두렁 T.T 
#결측치처리리리리이
#1. 행 또는 열 삭제 (열 삭제는 무식한 방법이긴 상관관계 관련 없는 열은 제거 )
#2. 임의의 값 평균값,중위값,0,앞의값,뒤의값

# 평균: mean
# 중위:median
# 0 : fillna
# 앞의값:fill
# 뒤의값:bfill
# 특정값:.....
# 기타등등:.... 
#3. 보간 - Interpolate (선형회귀 방식임 nan값을 linear방식으로 찾음)
#4. 모델 - predict model.predict 
#5 부스팅계열 - 결측치,이상치에 대해 자유롭다 . 믿거나 말거나ㅋ 

import pandas as pd
import numpy as np
from datetime import datetime

dates = ['8/10/2022','8/11/2022','8/12/2022','8/13/2022','8/14/2022']

dates = pd.to_datetime(dates)
print(dates)

ts = pd.Series([2, np.nan, np.nan,8,10],index=dates)  #Series >컬럼하나 백터하나 존재
print(ts)


print('===========================')
ts = ts.interpolate()
print(ts)