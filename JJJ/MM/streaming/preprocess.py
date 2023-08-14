import pandas as pd

# importing data & concatenating bi-modal
# 파일 경로 정의
# audio_path = './audio/data/audio_data.csv'
# gesture_path = './gesture/data/label/gest_data.csv'

# csv 파일 읽기
audio_df = pd.read_csv(audio_path)
gesture_df = pd.read_csv(gesture_path)

# audio_df 행 개수 늘리기
repeat_times = gesture_df.shape[0] // audio_df.shape[0]
audio_df = pd.concat([audio_df] * repeat_times, ignore_index=True)

# 나머지 부분 채우기
remaining_rows = gesture_df.shape[0] - audio_df.shape[0]
audio_df = pd.concat([audio_df, audio_df.iloc[:remaining_rows]], ignore_index=True)

# 두 데이터프레임을 concatenate
result_df = pd.concat([gesture_df, audio_df], axis=1)

# NaN 값들을 각 열의 평균값으로 채우기
result_df = result_df.fillna(result_df.mean())

# 두 데이터프레임을 가로로 합침
# concatenated_df = pd.concat([audio_df, gesture_df], axis=1)

# 결과 저장 (필요한 경우 저장 경로를 변경하세요)
result_df.to_csv('./final_1.csv', index=False)
