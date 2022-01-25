# Purpose of this file is to call the async_main() method from the `classifier.py` backend.
from src.classifier import BirdClassifier, async_main
import logging
import time
import typer
app = typer.Typer()


def main(image_urls: list[str]) -> None:
    """
    Classifies a list of images according to the model 'https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1 and
    the labels: 'https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv'.

    :param image_urls: list of comma separated strings of urls that point to a bird image
    :return: Three top matches along with the confidence score
    """
    start_time = time.time()
    classifier = BirdClassifier()
    # using_yappi_async_profiler("WALL", classifier) # running the classifier with the yappi profiler.
    async_main(classifier, image_urls)  # Running the classifier without the yappi profiler above.
    total_time = time.time() - start_time
    time_per_image = total_time / len(image_urls)
    print('Total time spent: {} sec'.format(total_time))
    print('Average time spent per image was: {}  sec'.format(time_per_image))
    logging.info('Total time spent classifying {} images was: {} sec'.format(len(image_urls), total_time))
    logging.info('Average time spent per image was: {}  sec'.format(time_per_image))
    logging.info("===================================================================")


if __name__ == "__main__":
    logging.info("======RUNNER STARTS======")
    typer.run(main)
    logging.info("======RUNNER ENDS=========")
