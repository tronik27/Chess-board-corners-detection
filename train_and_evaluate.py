import tensorflow as tf
import matplotlib.pyplot as plt
from Data_Preprocessing import CustomDataGen
from cnn_model import CustomResNet18
import numpy as np
from multiprocessing import cpu_count
import random
from typing import Tuple
import cv2 as cv


class KeyPointsDetection:
    def __init__(self,
                 batch_size: int,
                 target_size: Tuple[int, int, int],
                 num_predictions: int,
                 num_filters: int,
                 learning_rate: float,
                 model_name: str,
                 input_name: str,
                 output_name: str,
                 path_to_model_weights: str,
                 regularization: float) -> None:
        """
        Chess board corners detection class.
        :param batch_size: size of the batch of images fed to the input of the neural network.
        :param target_size: the size to which all images in the dataset are reduced.
        :param num_predictions: number of classes of images in dataset.
        :param num_filters: network expansion factor, determines the number of filters in start layer.
        :param learning_rate: learning rate when training the model.
        :param model_name: name of model.
        :param input_name: name of the input tensor.
        :param output_name: name of the output tensor.
        :param path_to_model_weights: folder where the weights of the model will be saved after the epoch at which it
         showed the best result.
        :param regularization: pass float < 1 (good is 0.0005) to make regularization or None to ignore it.
        """
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_predictions = num_predictions
        self.path_to_model_weights = path_to_model_weights
        self.nn = CustomResNet18(
            input_shape=self.target_size,
            num_filters=num_filters,
            regularization=regularization,
            input_name=input_name,
            output_name=output_name,
            num_predictions=self.num_predictions
        ).build()
        self.nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        loss=tf.keras.losses.MeanSquaredError(),
                        metrics=tf.keras.metrics.CosineSimilarity())
        self.model_name = model_name
        self.model_summary = self.nn.summary()

    def train(self,
              path_to_train_images: str,
              path_to_train_keypoints: str,
              path_to_val_images: str,
              path_to_val_keypoints: str,
              augmentation: list = [],
              epochs: int = 100,
              show_learning_curves: bool = False,
              show_image_data: bool = False,
              num_of_examples: int = 3
              ) -> None:
        """
        Method for training the model.
        :param path_to_train_images: path to array containing train image data.
        :param path_to_train_keypoints: path to array containing train key points data.
        :param path_to_val_images: path to array containing validation image data.
        :param path_to_val_keypoints: path to array containing validation key points data.
        :param augmentation: list of transforms to be applied to the training image.
        :param epochs: number of epochs to train the model.
        :param show_learning_curves: indicates whether to show show learning curves or not.
        :param show_image_data: indicates whether to show original and augmented image with labels or not.
        :param num_of_examples: number of original and augmented image examples to display.
        """

        train_datagen = CustomDataGen(
            path_to_images=path_to_train_images,
            path_to_keypoints=path_to_train_keypoints,
            batch_size=self.batch_size,
            aug_config=augmentation
        )
        validation_datagen = CustomDataGen(
            path_to_images=path_to_val_images,
            path_to_keypoints=path_to_val_keypoints,
            batch_size=self.batch_size,
        )

        if show_image_data:
            print('[INFO] displaying images from dataset. Close the window to continue...')
            train_datagen.show_image_data(num_of_examples=num_of_examples)

        print('[INFO] training network...')
        history = self.nn.fit(
            train_datagen,
            validation_data=validation_datagen,
            steps_per_epoch=train_datagen.number_of_images // self.batch_size,
            callbacks=self.__get_callbacks(),
            epochs=epochs,
            workers=cpu_count(),
        )

        if show_learning_curves:
            print('[INFO] displaying information about learning process. Close the window to continue...')
            self.__plot_learning_curves(history)

    def evaluate(self, path_to_test_images: str, path_to_test_keypoints: str, show_image_data: bool, num_examples: int)\
            -> None:
        """
        Method for evaluating a model on a test set.
        :param path_to_test_images: path to array containing test image data.
        :param path_to_test_keypoints: path to array containing test key points data.
        :param show_image_data: indicates whether to show predictions for set images or not.
        :param num_examples: number of predictions examples to display
        """
        test_datagen = CustomDataGen(
            path_to_images=path_to_test_images,
            path_to_keypoints=path_to_test_keypoints,
            batch_size=self.batch_size
        )

        try:
            self.nn.load_weights(self.path_to_model_weights)
        except FileNotFoundError:
            raise ValueError('There are no weights to evaluate the trained model! Try to train the model first.')

        print('[INFO] evaluating network...')
        results = self.nn.evaluate(test_datagen, batch_size=self.batch_size, verbose=0, use_multiprocessing=True)
        for i, metric in enumerate(self.nn.metrics_names):
            print('{}: {:.05f}'.format(metric, results[i]))
        if show_image_data:
            self.__show_image_data(test_datagen, num_examples=num_examples)

    def save_model(self, path_to_save: str, save_format: str = '') -> None:
        """
        Method for saving the whole model.
        :param path_to_save: folder where the model will be stored.
        :param save_format: format to save the model.
        """
        print('[INFO] saving network model...')
        try:
            self.nn.load_weights(self.path_to_model_weights)
        except FileNotFoundError:
            raise ValueError('There are no weights to save the trained model! Try to train the model first.')
        if save_format == 'h5':
            self.nn.save(path_to_save, save_format='h5')
        else:
            self.nn.save(path_to_save)

    def __show_image_data(self, generator: tf.keras.utils.Sequence, num_examples: int = 10) -> None:
        """
        Method for showing predictions.
        :param generator: data generator.
        :param num_examples: number of predictions examples to display
        """
        for _ in range(num_examples):
            j = random.randint(0, len(generator))
            images, keypoints = generator[j]
            if images.shape[0] > 3:
                images = images[:3, :, :, :]
            predict_keypoints = self.nn.predict(images)

            fig, axes = plt.subplots(nrows=1, ncols=images.shape[0], figsize=(12, 4))
            fig.suptitle('Network prediction results:\n (GT keypoints: red, Predicted keypoints: green)',
                         fontsize=14, fontweight="bold")
            for i in range(images.shape[0]):
                image = self.__image_prepare(images[i, :, :, :])

                pts = self.__pts_prepare(np.expand_dims(keypoints[i, :], axis=0))
                image = self.__draw_keypoints(image, pts, color=(255, 0, 0))

                predict_pts = self.__pts_prepare(np.expand_dims(predict_keypoints[i, :], axis=0))
                image = self.__draw_keypoints(image, predict_pts)

                axes[i].imshow(image)
                axes[i].axis('off')
            plt.show()

    def __pts_prepare(self, keypoints) -> np.array:
        """
        Method for drawing keypoints.
        :param keypoints: keypoints array.
        :return keypoints coordinates array.
        """
        return (self.target_size[0] * keypoints).astype(int).reshape((keypoints.shape[1] // 2, 1, 2))

    @staticmethod
    def __draw_keypoints(image: np.array, pts: np.array, color: Tuple = (0, 255, 0)) -> np.array:
        """
        Method for drawing keypoints.
        :param image: image array.
        :param pts: keypoints coordinates.
        :param color: color of keypoints.
        :return image with keypoints drawn .
        """
        image = cv.polylines(image, [pts], isClosed=True, color=(255, 255, 90), thickness=1)
        for i in range(pts.shape[0]):
            image = cv.circle(image, tuple(pts[i, 0, :]), int(image.shape[0] * 0.02), color, -1)
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

    def __get_callbacks(self) -> list:
        """
        Method for creating a list of callbacks.
        :return: list containing callbacks.
        """

        def scheduler(epoch, lr):
            if epoch < 1:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        callbacks = list()
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', min_delta=0.000001,
                                                         factor=0.1, patience=3, min_lr=0.000001)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.path_to_model_weights, save_weights_only=True,
                                                        save_best_only=True, monitor='val_loss', mode='min')
        stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.00001)
        callbacks += [reduce_lr, checkpoint, stop]
        return callbacks

    def __plot_learning_curves(self, metric_data) -> None:
        """
        Method for plotting learning curves.
        :param metric_data: dictionary containing metric an loss logs.
        """
        print(metric_data)
        figure, axes = plt.subplots(len(metric_data.history) // 2, 1, figsize=(7, 10))
        for axe, metric in zip(axes, self.nn.metrics_names):
            name = metric.replace("_", " ").capitalize()
            axe.plot(metric_data.epoch, metric_data.history[metric], label='Train')
            axe.plot(metric_data.epoch, metric_data.history['val_' + metric], linestyle="--",
                     label='Validation')
            axe.set_xlabel('Epoch')
            axe.set_ylabel(name)
            axe.grid(color='coral', linestyle='--', linewidth=0.5)
            axe.legend()
        plt.show()
