import random
import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
from pathlib import Path
from autogluon.tabular import TabularDataset, TabularPredictor

import librosa

import json

#### data load 
###### multiprocess & multithread program to MM model 

#### data check

#### autogluon
label = 'label'
eval_metric = 'accuracy'
time_limit = 3600 * 1 # hrs

predictor = TabularPredictor(
    label=label, eval_metric=eval_metric
).fit(train_data, presets='best_quality', time_limit=time_limit, ag_args_fit={'num_gpus': 0, 'num_cpus': 12})