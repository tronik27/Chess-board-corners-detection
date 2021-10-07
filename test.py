import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import cv2 as cv
from config import MODEL_PATH, WORK_DATA_PATH
import os


class KPDWork:

    def __init__(self, path_to_model: str, path_to_images: str) -> None:
        """
        key points detection class.
        :param path_to_model: folder where the model is stored.
        :param path_to_images: path to folder containing data.
        """
        try:
            self.nn = tf.keras.models.load_model(path_to_model)
        except FileNotFoundError:
            raise ValueError('There is no trained model! Try to train the model first.')
        self.path_to_images = path_to_images

    def predict_and_show(self, path_to_save: str) -> None:
        """
        Method for showing predictions.
        """
        data = self.__get_data()
        norm_data = self.__normalize(data)
        predicted_keypoints = self.nn(norm_data, training=False).numpy()
        np.save(os.path.join(path_to_save, 'predicted_keypoints.npy'), np.asarray(predicted_keypoints))
        for i in range(predicted_keypoints.shape[0]):
            img = data[i, :, :, :]
            keypoints = predicted_keypoints[i: i+1, :]
            pts = self.__pts_prepare(keypoints, img.shape[1])
            image = self.__draw_keypoints(self.__image_prepare(img), pts)
            plt.axis('off')
            plt.imshow(image)
            plt.title('Predicted keypoints')
            plt.show()

    def __get_data(self) -> np.array:
        """
        Get images array.
        :return image array
        """
        try:
            images = np.load(self.path_to_images)
        except MemoryError:
            images = np.load(self.path_to_images, mmap_mode='r')
        return images

    @staticmethod
    def __normalize(batch) -> np.array:
        """
        Normalising batch.
        :param batch: batch of images.
        :return: normalised batch tensor.
        """
        norm_batch = (batch - np.mean(batch, axis=(1, 2, 3), keepdims=True)) / np.std(batch, axis=(1, 2, 3),
                                                                                      keepdims=True)
        return norm_batch

    @staticmethod
    def __pts_prepare(keypoints: np.array, shape: int) -> np.array:
        """
        Method for drawing keypoints.
        :param keypoints: keypoints array.
        :param shape: shape of image.
        :return keypoints coordinates array.
        """
        return (shape * keypoints).astype(int).reshape((keypoints.shape[1] // 2, 1, 2))

    @staticmethod
    def __draw_keypoints(image: np.array, pts: np.array) -> np.array:
        """
        Method for drawing keypoints.
        :param image: image array.
        :param pts: keypoints coordinates.
        :return image with keypoints drawn .
        """
        image = cv.polylines(image, [pts], isClosed=True, color=(255, 255, 90), thickness=1)
        for i in range(pts.shape[0]):
            image = cv.circle(image, tuple(pts[i, 0, :]), int(image.shape[0] * 0.02), (0, 255, 0), -1)
        return image

    @staticmethod
    def __image_prepare(x: np.array) -> np.array:
        """
        Method for image preparing.
        :param x: image array.
        :return prepared image array.
        """
        x = ((x - x.min()) * (1 / (x.max() - x.min()) * 255)).astype('uint8')
        x = cv.cvtColor(x, cv.COLOR_GRAY2RGB)
        return x


if __name__ == '__main__':
    #  Creating the key points detector
    detector = KPDWork(
        path_to_model=MODEL_PATH,
        path_to_images=WORK_DATA_PATH,
    )
    #  Getting predictions for images
    detector.predict_and_show(path_to_save=os.path.dirname(WORK_DATA_PATH))
