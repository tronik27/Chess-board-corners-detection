import cv2
import numpy as np
import matplotlib.pyplot as plt
from Augmentation import image_augmentation
import tensorflow as tf
import random
from typing import Tuple
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from config import PATH_TO_TRAIN_IMAGES, PATH_TO_VAL_IMAGES, PATH_TO_TRAIN_KEYPOINTS, PATH_TO_VAL_KEYPOINTS


class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self, path_to_images: str, path_to_keypoints: str, batch_size: int, aug_config: list = []) -> None:
        """
        Data generator for the image classification and segmentation task.
        :param path_to_images: str containing path for images array.
        :param path_to_keypoints: str containing path for keypoints array.
        :param batch_size: size of the batch of images fed to the input of the neural network.
        :param aug_config: a dictionary containing the parameter values for augmentation.
        """
        self.images, self.keypoints = self.__get_data(path_to_images, path_to_keypoints)
        self.batch_size = batch_size
        self.aug_config = aug_config
        self.shape = self.images.shape[1:]
        self.number_of_images = self.images.shape[0]

    def on_epoch_end(self):
        """
        Random shuffling of training data at the end of each epoch during training.
        """
        if self.__augmentation:
            self.images, self.keypoints = shuffle(self.images, self.keypoints)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Getting batch.
        :param index: batch number.
        :return: images and keypoints tensors.
        """
        image_batch = self.images[index * self.batch_size:(index + 1) * self.batch_size, :, :, :]
        keypoints_batch = self.keypoints[index * self.batch_size:(index + 1) * self.batch_size, :]
        if self.aug_config:
            image_batch, keypoints_batch = self.__get_aug_data(image_batch[:, :, :, 0], keypoints_batch)
        image_batch = self.__normalize(image_batch)
        return image_batch, keypoints_batch

    def __len__(self):
        return self.number_of_images // self.batch_size

    @staticmethod
    def __normalize(batch: np.array) -> np.array:
        """
        Normalising batch.
        :param batch: batch of images.
        :return: normalised batch tensor.
        """
        norm_batch = (batch - np.mean(batch, axis=(1, 2, 3), keepdims=True)) / np.std(batch, axis=(1, 2, 3),
                                                                                      keepdims=True)
        return norm_batch

    @staticmethod
    def __get_data(path_to_images: str, path_to_keypoints: str) -> Tuple[np.array, np.array]:
        """
        Get images and keypoints arrays.
        :param path_to_images: str containing path for images array.
        :param path_to_keypoints: str containing path for keypoints array.
        :return: loaded images and keypoints tensors.
        """
        try:
            images = np.load(path_to_images)
            keypoints = np.load(path_to_keypoints)
        except MemoryError:
            images = np.load(path_to_images, mmap_mode='r')
            keypoints = np.load(path_to_keypoints, mmap_mode='r')

        keypoints[keypoints < 0] = 0
        keypoints[keypoints > 1] = 1

        return images, keypoints

    def __get_aug_data(self, images_arr: np.array, keypoints_arr: np.array) ->\
            Tuple[np.array, np.array]:
        """
        Making batch of augmented data.
        :param images_arr: numpy array containing images included in the batch.
        :param keypoints_arr: numpy array containing keypoints included in the batch.
        :return: images and keypoints tensors.
        """
        images_batch = np.empty(shape=(images_arr.shape[0], images_arr.shape[1], images_arr.shape[2], 1))
        keypoints_batch = np.empty_like(keypoints_arr)
        for i in range(images_arr.shape[0]):
            image, keypoints = self.__augmentation(self.__image_prepare(images_arr[i, :, :], convert_to_rgb=False),
                                                   self.__convert_for_albumentations(keypoints_arr[i, :]))
            images_batch[i, :, :, :] = image / 255
            keypoints_batch[i, :] = self.__convert_from_albumentations(keypoints)

        return images_batch, keypoints_batch

    def __augmentation(self, image: np.array, keypoints: np.array) -> np.array:
        """
        Apply augmentation to the image and keypoints.
        :param image: image array.
        :param keypoints: keypoints array.
        :return: augmented image and keypoints.
        """
        augmentation = image_augmentation(config=self.aug_config)
        transform = augmentation(image=image, keypoints=keypoints)
        return np.expand_dims(transform['image'], axis=-1), transform['keypoints']

    def __convert_for_albumentations(self, keypoints: np.array) -> list:
        """
        Convert keypoints in albumentations format.
        :param keypoints: keypoints array.
        :return: keypoints list.
        """
        keypoints = (self.shape[0] * keypoints)
        keypoints = list(keypoints.reshape((keypoints.shape[0] // 2, 2)))
        return [tuple(keypoint) for keypoint in keypoints]

    def __convert_from_albumentations(self, keypoints: list) -> np.array:
        """
        Convert keypoints from albumentations format in numpy array.
        :param keypoints: keypoints list.
        :return: keypoints array.
        """
        keypoints = np.array(keypoints) / self.shape[0]
        return keypoints.flatten()

    def show_image_data(self, num_of_examples: int = 3) -> None:
        """
        Method for showing original and augmented image with labels.
        :param num_of_examples: number of images to display.
        """
        for _ in range(num_of_examples):
            j = random.randint(0, self.number_of_images)
            image, keypoints = self.images[j, :, :, 0], self.keypoints[j, :]
            augmented_image, augmented_keypoints = self.__get_aug_data(images_arr=np.expand_dims(image, axis=0),
                                                                       keypoints_arr=np.expand_dims(keypoints, axis=0))

            image, pts = self.__image_prepare(image), self.__pts_prepare(np.expand_dims(keypoints, axis=0))
            image = self.__draw_keypoints(image, pts)

            augmented_image = self.__image_prepare(np.squeeze(augmented_image, axis=0))
            augmented_pts = self.__pts_prepare(augmented_keypoints)
            augmented_image = self.__draw_keypoints(augmented_image, augmented_pts)
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            fig.suptitle('Original and augmented image', fontsize=16)
            if image.shape[-1] == 1:
                image = np.squeeze(image, axis=-1)
            axes[0].imshow(image)
            axes[0].set_title('Original', size=12)
            axes[0].axis('off')
            axes[1].imshow(augmented_image)
            axes[1].set_title('Augmented', size=12)
            axes[1].axis('off')
            plt.show()

    def __pts_prepare(self, keypoints: np.array) -> np.array:
        """
        Method for drawing keypoints.
        :param keypoints: keypoints array.
        :return keypoints coordinates array.
        """
        return (self.shape[0] * keypoints).astype(int).reshape((keypoints.shape[1] // 2, 1, 2))

    @staticmethod
    def __draw_keypoints(image: np.array, pts: np.array) -> np.array:
        """
        Method for drawing keypoints.
        :param image: image array.
        :param pts: keypoints coordinates.
        :return image with keypoints drawn .
        """
        image = cv2.polylines(image, [pts], isClosed=True, color=(255, 255, 90), thickness=1)
        for i in range(pts.shape[0]):
            image = cv2.circle(image, tuple(pts[i, 0, :]), int(image.shape[0] * 0.02), (0, 255, 0), -1)
        return image

    @staticmethod
    def __image_prepare(x: np.array, convert_to_rgb: bool = True) -> np.array:
        """
        Method for image preparing.
        :param x: image array.
        :param convert_to_rgb: parameter indicating whether to convert the image to rgb or not.
        :return prepared image array.
        """
        x = ((x - x.min()) * (1 / (x.max() - x.min()) * 255)).astype('uint8')
        if convert_to_rgb:
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
        return x


def prepare_dataset(path_to_images: str, path_to_keypoints: str, valid_size: float) -> None:
    """
    Method of splitting a dataset into datasets for training and validation.
    :param path_to_images: str containing path for images array.
    :param path_to_keypoints: str containing path for keypoints array.
    :param valid_size: part of the data that will be allocated to the validation set.
    """
    try:
        images = np.load(path_to_images)
        keypoints = np.load(path_to_keypoints)
    except MemoryError:
        images = np.load(path_to_images, mmap_mode='r')
        keypoints = np.load(path_to_keypoints, mmap_mode='r')

    train_images, val_images, train_keypoints, test_keypoints = train_test_split(images, keypoints, shuffle=True,
                                                                                 test_size=valid_size, random_state=42)
    np.save(PATH_TO_TRAIN_IMAGES, train_images)
    np.save(PATH_TO_VAL_IMAGES, val_images)
    np.save(PATH_TO_TRAIN_KEYPOINTS, train_keypoints)
    np.save(PATH_TO_VAL_KEYPOINTS, test_keypoints)
