import cv2
import os
import pandas as pd
from glob import glob
from tqdm import tqdm  

data_root = 'E:\\data\\01data\\1.Training\\original'
ann_file_train = 'E:\\data\\new_train_ann.txt'
output_path = 'E:\\data\\output.csv'

scales = [1, 0.875, 0.75, 0.66]

# 레이블 데이터를 읽어들입니다
with open(ann_file_train, 'r') as f:
    lines = f.readlines()

# 각 라인을 처리하여, 파일 경로와 레이블을 분리하고, 딕셔너리에 저장합니다
labels = {}
for line in lines:
    path, label = line.rsplit(' ', 1)
    labels[path] = int(label)

dfs = []  # 각 이미지의 결과를 담을 리스트를 초기화합니다

# data_root 내의 모든 파일을 처리합니다
for root, _, filenames in tqdm(os.walk(data_root), desc="Processing images"):  # tqdm을 사용하여 진행 막대를 표시합니다
    for filename in filenames:
        # 파일의 절대 경로를 만듭니다
        image_path = os.path.join(root, filename)

        # 파일이 이미지인지 확인합니다
        if not os.path.isfile(image_path) or not any(image_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp']):
            continue 
        image = cv2.imread(image_path)
        
        # Resize
        rescaled_image = cv2.resize(image, (256, 256))
        
        tensors = []  # 이 이미지에 대한 텐서를 담을 리스트를 초기화합니다
        
        # MultiScaleCrop
        height, width, _ = image.shape
        for scale in scales:
            new_height, new_width = int(height * scale), int(width * scale)
            resized = cv2.resize(image, (new_width, new_height))
            start_row, start_col = int((new_height - 224) / 2), int((new_width - 224) / 2)
            cropped = resized[start_row:start_row + 224, start_col:start_col + 224]
            
            # Flip
            flipped_image = cv2.flip(cropped, 1)
            
            # 이미지를 1차원 텐서로 변환하고 저장합니다
            tensor = flipped_image.flatten()
            tensors.append(tensor.tolist())
        
        # 이 이미지에 대한 레이블을 가져옵니다
        label = labels.get(image_path, None)  # 만약 딕셔너리에 레이블이 없다면 None을 반환합니다
        
        df = pd.DataFrame({'filename': [filename], 'label': [label], 'tensors': [tensors]})
        dfs.append(df)

# 모든 처리가 끝났으므로, 결과 DataFrame들을 결합합니다
result_df = pd.concat(dfs, ignore_index=True)

# 결과를 CSV 파일로 저장합니다
result_df.to_csv(output_path, index=False)
