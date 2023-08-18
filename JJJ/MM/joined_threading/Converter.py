import threading
import queue

class Converter:

    def __init__(self):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.thread = threading.Thread(target=self._transform_data)
        self.thread.start()

    def _transform_data(self):
        while True:
            input_data = self.input_queue.get()
            if input_data == "exit":
                break

            output_data = {
                'index': [1],
                'label': [None]
            }
            idx = 1
            for command_data in input_data:
                for command, pronunciations in command_data.items():
                    for pronunciation, similarity in pronunciations.items():
                        output_data[f'command_{idx}'] = [command]
                        output_data[f'pronunciation_{idx}'] = [pronunciation]
                        output_data[f'similarity_{idx}'] = [similarity]
                        idx += 1

            self.output_queue.put(output_data)

    def put_data(self, data):
        self.input_queue.put(data)

    def get_transformed_data(self):
        return self.output_queue.get()

    def stop(self):
        self.input_queue.put("exit")
        self.thread.join()
