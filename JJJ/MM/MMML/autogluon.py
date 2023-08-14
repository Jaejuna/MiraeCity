from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd

train_df = pd.read_csv('./final_drop.csv')
train_data = TabularDataset(train_df)
time_limit = 3600 * 0.5 #hrs

#### autogluon
label = 'label'
eval_metric = 'accuracy'

predictor = TabularPredictor(
    label=label, eval_metric=eval_metric
).fit(train_data, presets='best_quality', time_limit=time_limit, ag_args_fit={'num_gpus': 0, 'num_cpus': 12})

### result (leaderboard)
print(predictor.leaderboard(silent=False))