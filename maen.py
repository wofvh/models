
import numpy as np
import pandas as pd
# # data = pd.read_csv('C:\datawather/2019_12.31강수량.csv', encoding='cp949', engine='python')

# # print (data)

# # data.info()
# # data.isnull().sum()

#1. 데이터
path = 'C:\datawather/'

# filelist = ['2019_12.31강수량.csv', '2019_12.31기온.csv', '2019_12.31습도.csv', '2019_12.31일조.csv', 
#             '2019_12.31적설.csv', '2019_12.31지면온도.csv', '2019_12.31풍속.csv',
#             '2020_12.31강수량.csv', '2020_12.31기온.csv', '2020_12.31습도.csv', '2020_12.31일조.csv',
#             '2020_12.31적설.csv', '2020_12.31지면온도.csv', '2020_12.31풍속.csv',
#             '2021_01.31강수량.csv', '2021_01.31기온.csv', '2021_01.31습도.csv', '2021_01.31일조.csv',
#             '2021_01.31적설.csv', '2021_01.31지면온도.csv', '2021_01.31풍속.csv']

name = ['기온', '강수량', '습도', '일조', '적설', '지면온도', '풍속']
year_month = ['2021_12']
year = ['2021']
# df = pd.read_csv(path + '2021_01.31풍속.csv',encoding='cp949') # + 명령어는 문자를 앞문자와 더해줌  index_col=n n번째 컬럼을 인덱스로 인식

for i in range(len(year)):
    for j in range(len(name)):
        df = pd.read_csv(path + year_month[i] + '_31' + name[j] + '.csv',encoding='cp949')

        df.columns = ['지점','위치', '날짜','기온']

        df['날짜'] = pd.to_datetime(df['날짜']) 

        df.index = df['날짜']

        # DatetimeIndex = pd.date_range(start=f'{year[i]}-01-01', end=f'{year_month[i]}-31', freq='H')
        if year_month[i] == '2019_12':
            DatetimeIndex = pd.date_range(start='2019-01-01 00:00:00', end='2019-12-31 23:00:00', freq='H')

        elif year_month[i] == '2020_12':
            DatetimeIndex = pd.date_range(start='2020-01-01 00:00:00', end='2020-12-31 23:00:00', freq='H')
        
        elif year_month[i] == '2021_01':
            DatetimeIndex = pd.date_range(start='2021-01-01 00:00:00', end='2021-01-31 23:00:00', freq='H')
        
        elif year_month[i] == '2021_12':
            DatetimeIndex = pd.date_range(start='2021-03-01 00:00:00', end='2021-12-31 23:00:00', freq='H')

        DatetimeIndex = pd.DataFrame(DatetimeIndex, columns=['날짜'])

        DatetimeIndex = DatetimeIndex.set_index('날짜')

        DatetimeIndex = DatetimeIndex[~DatetimeIndex.index.isin(df.index)]
        print(DatetimeIndex.shape)

        df = pd.concat([df, DatetimeIndex])

        df = df.sort_index()

        
        if name[j] == ['기온', '습도', '지면온도', '풍속']:
            df['기온'] = df['기온'].interpolate(method='time')
            df = df.fillna(0)
        else:
            df = df.fillna(0)
        
        print(df.shape)
        print(df.isnull().sum())

        df_numpy = df.to_numpy()

        print(df_numpy.shape) # (8760, 4)

        append_data = np.empty((0,4))

        print(append_data)

        for x in range(len(df)):
            whole_list = np.reshape(df_numpy[x], (1,4))
            print(whole_list.shape) # (4,)
            for y in range(6):
                append_data = np.append(append_data, whole_list, axis=0)
                

        print(append_data)


        append_list = pd.DataFrame(append_data, columns=['지점','위치', '날짜','기온'])
        print(append_list)

        print(append_list.shape) # (52560, 4)

        # append_list.to_csv("./전처리데이터/" + year[i] + name[j] + '.csv')
        append_list.to_csv("./전처리데이터/" + year[i] + name[j] + '.csv', index=False)


exit()