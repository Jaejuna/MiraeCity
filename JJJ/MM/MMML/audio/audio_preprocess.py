import random
import pandas as pd
import numpy as np
import os
import pysrt
from tqdm.auto import tqdm
from pathlib import Path

import librosa

import json
from pandas import json_normalize

##### hyperparameter
CFG = {
    'SR':16000, # sampling rate
    'N_MFCC':128, # Melspectrogram 벡터를 추출할 개수
    'SEED':42
}

#### fixed random seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED']) # seed 고정

""" #### data preprocessing
def get_filelist(subfolder, file_extension):
    data_path = Path.cwd()/subfolder
    
    return list(data_path.glob('**/*' + file_extension))

rootdir = './data/ann'

#### json handling
# 이 파일이 위치해있는 폴더의 하위폴더 'data'에 있는 확장자명이 '.json'인 모든 파일을 불러옵니다
files = get_filelist(rootdir+ '*' ,'json')

# 저장할 데이터 항목의 이름을 입력합니다. json 파일에 적힌 항목(key)과 같아야합니다.
column_names = ['dataSet', 'version', 'mediaUrl', 'date', 'typeInfo', 'conversationType', 'speakerNumber', 'speakers', 'dialogs', 'samplingRate', 'recStime', 'recLen', 'recDevice']
result = pd.DataFrame(columns=column_names)   

for json_file in files:
    df = pd.read_json(json_file)
    row_data = pd.json_normalize(data=df['row'])
    
    result = pd.concat([result,df])

# 현재 이 파일이 위치한 폴더의 하위 폴더 data 에 'result.csv'로 저장
result.to_csv(Path.cwd()/'data'/'json_sample_category.csv', index=None)

print(result.head(2)) #데이터가 잘 불러와지는지 확인하는 출력   

##### annotation organize -> json to csv
rootdir = './data/ann'

file_list = [f for f in os.scandir(rootdir) if f.is_file() and f.name.endswith('.json')]
dataframes = []

for file in file_list:
    with open(file, 'r') as f:
        json_data = json.load(f)
        
        # Flatten 'typeInfo', 'speakers' and 'dialogs' separately
        typeInfo_df = json_normalize(json_data, record_path='typeInfo', meta=['dataSet', 'version', 'mediaUrl', 'date', 'conversationType', 'speakerNumber'], errors='ignore')
        speakers_df = json_normalize(json_data, record_path='speakers', meta=['dataSet', 'version', 'mediaUrl', 'date', 'conversationType', 'speakerNumber'], errors='ignore')
        dialogs_df = json_normalize(json_data, record_path='dialogs', meta=['dataSet', 'version', 'mediaUrl', 'date', 'conversationType', 'speakerNumber'], errors='ignore')
        
        # Concatenate all data into one DataFrame
        dataframes.append(pd.concat([typeInfo_df, speakers_df, dialogs_df], axis=1))

# Concatenate all data from different JSON files
total_dataFrame = pd.concat(dataframes, ignore_index=True)

# Save the DataFrame to CSV
total_dataFrame.to_csv('./data/ann.csv', index=False) 


##### annotation organize -> srt to csv
def srt_to_df(file_path):
    subs = pysrt.open(file_path, encoding='utf-8')
    data = {
        'start': [],
        'end': [],
        'text': [],
    }
    
    for sub in subs:
        data['start'].append(str(sub.start))
        data['end'].append(str(sub.end))
        data['text'].append(sub.text)

    df = pd.DataFrame(data)
    return df

# a list to store each individual dataframe
dfs = []

# iterate over all srt files in directory
for filename in os.listdir(rootdir):
    if filename.endswith(".srt"):
        srt_file_path = os.path.join(rootdir, filename)
        
        # load the SRT file
        df = srt_to_df(srt_file_path)

        # add to list of dataframes
        dfs.append(df)

# concatenate all dataframes into one
df_all = pd.concat(dfs, ignore_index=True)

# save the concatenated dataframe to a single CSV file
df_all.to_csv('./data/final.csv', index=False) """

# mfcc & mel feature extract function
##### mfcc feature extract function
rootdir = './data/raw'

def get_mfcc_feature(df):
    features = []
    for path in tqdm(df['mediaUrl'].astype(str)):
        full_path = os.path.join(rootdir, path)
        try:
            y, sr = librosa.load(full_path, sr=CFG['SR'])
        except FileNotFoundError:
            #print(f"File {full_path} not found.")
            continue
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CFG['N_MFCC'])
        features.append({
            'mfcc_mean': np.mean(mfcc, axis=1),
            'mfcc_max': np.max(mfcc, axis=1),
            'mfcc_min': np.min(mfcc, axis=1),
        })
    if not features:  # If features list is empty
        print("No valid audio files found.")
        return pd.DataFrame()  # Return an empty DataFrame
    else:
        print("Found features")

    mfcc_df = pd.DataFrame(features)
    mfcc_mean_df = pd.DataFrame(mfcc_df['mfcc_mean'].tolist(), columns=[f'mfcc_mean_{i}' for i in range(CFG['N_MFCC'])])
    mfcc_max_df = pd.DataFrame(mfcc_df['mfcc_max'].tolist(), columns=[f'mfcc_max_{i}' for i in range(CFG['N_MFCC'])])
    mfcc_min_df = pd.DataFrame(mfcc_df['mfcc_min'].tolist(), columns=[f'mfcc_min_{i}' for i in range(CFG['N_MFCC'])])

    return pd.concat([mfcc_mean_df, mfcc_max_df, mfcc_min_df], axis=1)

##### mel feature extract function
def get_feature_mel(df):
    features = []
    for path in tqdm(df['mediaUrl'].astype(str)):
        full_path = os.path.join(rootdir, path)
        try:
            y, sr = librosa.load(full_path, sr=CFG['SR'])
        except FileNotFoundError:
            #print(f"File {full_path} not found.")
            continue
        n_fft = 2048
        win_length = 2048
        hop_length = 1024
        n_mels = 128

        D = np.abs(librosa.stft(y, n_fft=n_fft, win_length = win_length, hop_length=hop_length))
        mel = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length)

        features.append({
            'mel_mean': mel.mean(axis=1),
            'mel_max': mel.min(axis=1),
            'mel_min': mel.max(axis=1),
        })
        
    if not features:  # If features list is empty
        print("No valid audio files found.")
        return pd.DataFrame()  # Return an empty DataFrame
    else:
        print("Found features")

    mel_df = pd.DataFrame(features)
    mel_mean_df = pd.DataFrame(mel_df['mel_mean'].tolist(), columns=[f'mel_mean_{i}' for i in range(n_mels)])
    mel_max_df = pd.DataFrame(mel_df['mel_max'].tolist(), columns=[f'mel_max_{i}' for i in range(n_mels)])
    mel_min_df = pd.DataFrame(mel_df['mel_min'].tolist(), columns=[f'mel_min_{i}' for i in range(n_mels)])

    return pd.concat([mel_mean_df, mel_max_df, mel_min_df], axis=1)

# train_mf = get_mfcc_feature(train_df)
# train_mel = get_feature_mel(train_df)
# train_x = pd.concat([train_mel, train_mf], axis=1)
# train_x.to_csv('./data/train_data.csv', index=False)
# train_data = TabularDataset(train_x)