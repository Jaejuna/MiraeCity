from autogluon.multimodal import MultiModalPredictor
from autogluon.core.utils.loaders import load_zip
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


#### time series
data = TimeSeriesDataFrame('https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/train.csv')

predictor = TimeSeriesPredictor(target='target', prediction_length=48).fit(data)
predictions = predictor.predict(data)

#### multimodal 
data_zip = "https://automl-mm-bench.s3.amazonaws.com/object_detection_dataset/" + "tiny_motorbike_coco.zip"
load_zip.unzip(data_zip, unzip_dir=".")

train_path = "./tiny_motorbike/Annotations/trainval_cocoformat.json"
test_path = "./tiny_motorbike/Annotations/test_cocoformat.json"

predictor = MultiModalPredictor(
  problem_type="object_detection",
  sample_data_path=train_path
)

predictor.fit(train_path)
score = predictor.evaluate(test_path)

pred = predictor.predict({"image": ["./tiny_motorbike/JPEGImages/000038.jpg"]})