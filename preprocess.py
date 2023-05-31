import glob
import pandas as pd
from pandas import DataFrame
import json
import zipfile

path_to_zip_file = glob.glob('C:/Users/user/git/MiraeCity/SR/noise_env_recognition_data/01.데이터/***')
directory_to_extract_to = 'C:/Users/user/git/MiraeCity/SR/noise_env_recognition_data'

with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)