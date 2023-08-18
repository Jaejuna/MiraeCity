import threading
import queue
from autogluon.tabular import TabularPredictor
import pandas as pd

class PredictionThread:

    def __init__(self, model_path):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.thread = threading.Thread(target=self._predict_data)
        self.loaded_predictor = TabularPredictor.load(model_path)
        self.thread.start()

    def _predict_data(self):
        while True:
            transformed_data = self.input_queue.get()
            if transformed_data == "exit":
                break

            # Ensure the input data is in the correct format for the predictor
            df = pd.DataFrame(transformed_data)
            predictions = self.loaded_predictor.predict(df)
            self.output_queue.put(predictions)

    def put_transformed_data(self, data):
        self.input_queue.put(data)

    def get_prediction(self):
        return self.output_queue.get()

    def stop(self):
        self.input_queue.put("exit")
        self.thread.join()
