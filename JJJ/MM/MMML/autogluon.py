from autogluon.tabular import TabularDataset, TabularPredictor

#### autogluon
label = 'label'
eval_metric = 'accuracy'
time_limit = 3600 * 1 # hrs

predictor = TabularPredictor(
    label=label, eval_metric=eval_metric
).fit(train_data, presets='best_quality', time_limit=time_limit, ag_args_fit={'num_gpus': 0, 'num_cpus': 12})

### result (leaderboard)
predictor.leaderboard(silent=True)

# result (final csv shape)
print(concatenated_df.shape)