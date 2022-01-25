import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only TF ERRORs printed (https://stackoverflow.com/a/40982782/3696141)

from typing import Any, AsyncGenerator
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
from tensorflow import Tensor
from numpy import ndarray
import cv2
import urllib.request
import numpy as np
import time
import asyncio
import logging
import typer

# Profiling modules
import yappi

# Getting some unknown linter errors, disable everything to get this to production asap
# pylint: disable-all

model_url = 'https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1'
labels_url = 'https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv'

# Logging configuration
logging.basicConfig(
    filename="classifier_log.txt",
    filemode='a',  # append logs to logfile
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=logging.DEBUG,
    datefmt='%m/%d/%Y %I:%M:%S %p',
)

logger = logging.getLogger("mylogger")
logging.getLogger("classifier.logger").disabled = True


# TODO: Docstrings to the BirdClassifier class and all the methods
class BirdClassifier:
    @staticmethod
    def load_model() -> hub.KerasLayer:
        try:
            model = hub.KerasLayer(model_url)
            logging.info("Loaded model successfully.")
            return model
        except Exception as e:
            msg = "Loading model failed with exception: {}".format(e)
            logging.exception(msg)
            exit(1)  # Unsuccessful exit

    def load_labels_raw(self) -> Any:
        try:
            bird_labels_raw = urllib.request.urlopen(labels_url)
            logging.info("Loaded labels from labels_url successfully.")
            return bird_labels_raw
        except Exception as e:
            msg = "Requesting raw bird labels failed with exception: {}".format(e)
            logging.exception(msg)
            exit(1)  # Unsuccessful exit

    def clean_labels(self, bird_labels_raw: Any) -> dict:
        try:
            bird_labels_lines = [line.decode('utf-8').replace('\n', '') for line in bird_labels_raw.readlines()]
            bird_labels_lines.pop(0)  # remove header (id, name)
            logging.info("Decoded labels successfully.")
            bird_labels_cleaned = dict()
            for bird_line in bird_labels_lines:
                bird_id = int(bird_line.split(',')[0])
                bird_name = bird_line.split(',')[1]
                bird_labels_cleaned[bird_id] = {'name': bird_name}
            logging.info("Bird labels successfully cleaned.")
            return bird_labels_cleaned
        except Exception as e:
            msg = "Decoding and cleaning the labels failed with exception: {}".format(e)
            logging.exception(msg)
            exit(1)  # Unsuccessful exit

    def get_image_response(self, image_url: str) -> Any:
        try:
            image_response = urllib.request.urlopen(image_url)
            logging.info("Loaded image from image_url successfully.")
            return image_response
        except Exception as e:
            msg = "Loading image request failed with exception: {}".format(e)
            logging.exception(msg)
            exit(1)  # Unsuccessful exit

    def get_image_array(self, image_response: Any) -> ndarray:
        try:
            image_array = np.asarray(bytearray(image_response.read()), dtype=np.uint8)
            logging.info("Turned image to bytearray successfully.")
            return image_array
        except Exception as e:
            msg = "Turning the image to bytearray failed with exception: {}".format(e)
            logging.exception(msg)
            exit(1)  # Unsuccessful exit

    def change_image(self, image_array: ndarray) -> ndarray:
        try:
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            logging.info("Decoded image into array successfully.")
            image = cv2.resize(image, (224, 224))
            logging.info("Image resized successfully.")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logging.info("Color converted to image successfully")
            image = image / 255
            logging.info("Image rescaled successfully.")
            return image
        except Exception as e:
            msg = "Decoding image into array failed with exception: {}".format(e)
            logging.exception(msg)
            exit(1)  # Unsuccessful exit

    def convert_image_to_tensor(self, image: ndarray) -> Tensor:
        try:
            image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
            logging.info("Converted image to tensor successfully.")
            return image_tensor
        except Exception as e:
            msg = "Converting image to tensor failed with exception :{}".format(e)
            logging.exception(msg)
            exit(1)  # Unsuccessful exit

    def add_outer_batch_axis_to_tensor(self, image_tensor: Tensor) -> Tensor:
        try:
            image_tensor_expanded = tf.expand_dims(image_tensor, 0)
            logging.info("Tensor dimension expanded successfully.")
            return image_tensor_expanded
        except Exception as e:
            msg = "Tensor dimension expansion failed with exception :{}".format(e)
            logging.exception(msg)
            exit(1)  # Unsuccessful exit

    def get_model_raw_output(self, bird_model: hub.KerasLayer, image_tensor_expanded: Tensor) -> ndarray:
        try:
            model_raw_output = bird_model.call(image_tensor_expanded).numpy()
            logging.info("Model raw output created successfully.")
            return model_raw_output
        except Exception as e:
            msg = "Model raw output failed with exception :{}".format(e)
            logging.exception(msg)
            exit(1)  # Unsuccessful exit

    def sort_birds_by_result_score(self, model_raw_output: ndarray, bird_labels: dict) -> list:
        try:
            for index, value in np.ndenumerate(model_raw_output):
                bird_index = index[1]
                bird_labels[bird_index]['score'] = value
            logging.info("Ordered birds by results core successfully.")
            sorted_birds_by_score = sorted(bird_labels.items(), key=lambda x: x[1]['score'])
            logging.info("Sorted bird-list successfully.")
            return sorted_birds_by_score
        except Exception as e:
            msg = "Ordering birds by result score failed with exception: {}".format(e)
            logging.exception(msg)
            exit(1)  # Unsuccessful exit

    def get_top_n_result(self, top_index: int, sorted_birds_by_score: list) -> tuple[str, str]:
        try:
            sorted_bird_dict = sorted_birds_by_score[top_index*(-1)][1]
            bird_name = sorted_bird_dict['name']
            bird_score = sorted_bird_dict['score']
            logging.info("Objects bird_name and bird_score calculated successfully.")
            return bird_name, bird_score
        except Exception as e:
            msg = "Getting top n result, bird name or bird score failed with exception: {}".format(e)
            logging.exception(msg)
            exit(1)  # Unsuccessful exit

    def print_results_to_k8s_logs(self, index: int, sorted_birds_by_score: list) -> None:
        try:
            print('Run: {}'.format(int(index + 1)))
            bird_name, bird_score = self.get_top_n_result(1, sorted_birds_by_score)
            print('Top match: {} with score: {}'.format(bird_name, bird_score))
            bird_name, bird_score = self.get_top_n_result(2, sorted_birds_by_score)
            print('Top match: {} with score: {}'.format(bird_name, bird_score))
            bird_name, bird_score = self.get_top_n_result(3, sorted_birds_by_score)
            print('Top match: {} with score: {}'.format(bird_name, bird_score))
            print('\n')
            logging.info("Printed results to k8s logs successfully.")
        except Exception as e:
            msg = "Printing results to k8s logs failed with exception: {}".format(e)
            logging.exception(msg)
            exit(1)  # Unsuccessful exit

    async def async_generator_of_urls(self, urls: list) -> AsyncGenerator:
        try:
            for url in urls:
                yield url
            logging.info("Async generator created successfully.")
        except Exception as e:
            msg = "Async generator of urls failed with exception: {}".format(e)
            logging.exception(msg)
            exit(1)  # Unsuccessful exit

    async def main(self, image_urls) -> None:
        try:
            print(image_urls[0])
            logging.info("List of images non empty.")
        except Exception as e:
            msg = "Main function failed with exception since image list is empty. Exception: {}".format(e)
            logging.exception(msg)
            exit(1)  # Unsuccessful exit
        try:
            # Loading bird model and bird labels
            bird_model = self.load_model()
            bird_labels_raw = self.load_labels_raw()
            bird_labels_cleaned = self.clean_labels(bird_labels_raw)
            # Initiating async loop
            index = 0
            async for image_url in self.async_generator_of_urls(image_urls):
                # Getting image from url
                image_response = self.get_image_response(image_url)
                # Getting image array
                image_array = self.get_image_array(image_response)
                # Changing each image
                image = self.change_image(image_array)
                # Get bird names with the results ordered
                image_tensor = self.convert_image_to_tensor(image)
                # Expand tensor
                image_tensor_expanded = self.add_outer_batch_axis_to_tensor(image_tensor)
                # Get raw model
                model_raw_output = self.get_model_raw_output(bird_model, image_tensor_expanded)
                # Get sorted per score bird list
                sorted_birds_by_score = self.sort_birds_by_result_score(model_raw_output, bird_labels_cleaned)
                # Print results to kubernetes log
                self.print_results_to_k8s_logs(index, sorted_birds_by_score)
                index += 1
            logging.info("Main method finished successfully !")
        except Exception as e:
            msg = "Main function failed during model and label modelling, with exception : {}".format(e)
            logging.exception(msg)
            exit(1)  # Unsuccessful exit


# TODO: Docstring
def async_main(c: BirdClassifier, image_urls: list) -> None:
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(c.main(image_urls))
        logging.info("Asyncio event loop created successfully.")
    except Exception as e:
        msg = "Asyncio loop failed with exception: {}".format(e)
        logging.exception(msg)
        loop.shutdown_asyncgens()
        exit(1)  # Unsuccessful exit
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        logging.info("Asyncio loop closed successfully.")
        logging.info("Classifier finished successfully.")


# TODO: Docstring
def using_yappi_async_profiler(clock_type: str, c: BirdClassifier, image_urls: list) -> None:
    yappi.set_clock_type(clock_type)
    try:
        with yappi.run():
            async_main(c, image_urls)
        logging.info("Yappi profiled the classifier successfully.")
        yappi.get_func_stats().print_all()
    except Exception as e:
        msg = "Yappi profiling failed with exception: {}".format(e)
        logging.exception(msg)
        exit(1)  # Unsuccessful exit


if __name__ == "__main__":
    image_urls= [
        'https://i.imgur.com/8eGMhGP.jpg',
        'https://i.imgur.com/TRVxZAZ.jpg',
        'https://i.imgur.com/kBHq8Xt.jpg',
        'https://i.imgur.com/wmEaY0t.jpg',
        'https://i.imgur.com/olSQAGI.jpg'
    ]
    start_time = time.time()
    classifier = BirdClassifier()
    # using_yappi_async_profiler("WALL", classifier) # running the classifier with the yappi profiler.
    async_main(classifier, image_urls)  # Running the classifier without the yappi profiler above.
    total_time = time.time() - start_time
    time_per_image = total_time / len(image_urls)
    print('Time spent: {} sec'.format(total_time))
    print('Time spent per image was: {}  sec'.format(time_per_image))
    logging.info('Time spent classifying {} images was: {} sec'.format(len(image_urls), total_time))
    logging.info('Time spent per image was: {}  sec'.format(time_per_image))
    logging.info("===================================================================")
