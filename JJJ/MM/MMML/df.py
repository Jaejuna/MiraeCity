import pandas as pd
import numpy as np

# 초기 데이터
data = {
    'index': [1],
    'command_1': ['go'],
    'pronunciation_1': ['ɑpʰɯɾo'],
    'similarity_1': [None],
    'command_2': ['go'],
    'pronunciation_2': ['ɑpʰɯɾo kɑ'],
    'similarity_2': [None],
    'command_3': ['stop'],
    'pronunciation_3': ['mʌmtɕʰwʌ'],
    'similarity_3': [None],
    'command_4': ['stop'],
    'pronunciation_4': ['kɯmɑn'],
    'similarity_4': [None],
    'command_5': ['stop'],
    'pronunciation_5': ['sɯtʰɑp'],
    'similarity_5': [None],
    'command_6': ['back'],
    'pronunciation_6': ['twiɾo'],
    'similarity_6': [None],
    'command_7': ['back'],
    'pronunciation_7': ['toɾɑɡɑ'],
    'similarity_7': [None],
    'command_8': ['back'],
    'pronunciation_8': ['twiɾo kɑ'],
    'similarity_8': [None],
    'command_9': ['back'],
    'pronunciation_9': ['p*ɑɡu'],
    'similarity_9': [None],
    'command_10': ['goback'],
    'pronunciation_10': ['pokk*wi'],
    'similarity_10': [None],
    'command_11': ['goback'],
    'pronunciation_11': ['wʌnwitɕʰi'],
    'similarity_11': [None],
    'label': [None]
}

# DataFrame 생성
df = pd.DataFrame(data)

# index 값만 바꾸면서 15,000개의 행으로 확장
new_df = pd.concat([df] * 15000, ignore_index=True)
new_df['index'] = range(1, 15001)

# 각 similarity_n 열에 대해
for col in new_df.columns:
    if 'similarity' in col:
        new_df[col] = np.round(np.random.rand(15000), 2)

# 'similarity_'로 시작하는 열을 선택
similarity_cols = [col for col in new_df.columns if col.startswith('similarity')]

# 각 행에서 최대값을 갖는 similarity 열의 인덱스를 구하고 그에 따른 command 열의 값을 label로 지정
new_df['label'] = new_df[similarity_cols].idxmax(axis=1).apply(lambda x: new_df[f"command_{x.split('_')[-1]}"].iloc[0])

# 결과 확인
print(new_df.head())

# CSV 파일로 저장
new_df.to_csv('expanded_data.csv', index=False)