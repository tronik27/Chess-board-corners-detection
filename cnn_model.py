from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, GlobalAveragePooling2D, Input, Add, LeakyReLU,\
    MaxPooling2D, ReLU, Dropout
from tensorflow.python.framework.ops import Tensor


class CustomResNet18:
    def __init__(self, input_shape: Tuple[int, int, int], num_predictions: int, num_filters: int,
                 regularization: Optional[float], input_name: str, output_name: str) -> None:
        """
        Custom implementation of the stripped-down ResNet for key points detection task.
        :param input_shape: input shape (height, width, channels).
        :param num_predictions: number of output predictions.
        :param num_filters: network expansion factor, determines the number of filters in start layer.
        :param regularization: pass float < 1 (good is 0.0005) to make regularization or None to ignore it.
        :param input_name: name of the input tensor.
        :param output_name: name of the output tensor.
        """
        self.input_shape = input_shape
        self.num_predictions = num_predictions
        self.input_name = input_name
        self.output_name = output_name
        self.num_filters = num_filters
        self.ker_reg = None if regularization is None else tf.keras.regularizers.l2(regularization)
        self.conv_kwargs = {'use_bias': False, 'padding': 'same', 'kernel_regularizer': self.ker_reg}

    def build(self) -> tf.keras.models.Model:
        """
        Building CNN model for key points detection task.
        :return: keras.model.Model() object.
        """
        inputs = Input(shape=self.input_shape, name=self.input_name)
        x = BatchNormalization()(inputs)
        x = Conv2D(self.num_filters, (7, 7), strides=2, **self.conv_kwargs)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        for _ in range(2):
            x = self.res_block(x, self.num_filters)

        for i in range(5):
            x = self.res_block(x, self.num_filters * 2 ** (i + 1), 2)
            x = self.res_block(x, self.num_filters * 2 ** (i + 1))

        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_predictions, kernel_regularizer=self.ker_reg, activation='sigmoid', name=self.output_name)(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)

    def res_block(self, x: Tensor, filters: int, stride: int = 1) -> Tensor:
        """
        Residual block. If stride == 1, then there are no any transformations in one of the branches.
        If stride > 1, then there are convolution with 1x1 filters in one of the branches.
        :param x: input tensor.
        :param filters: number of filters in output tensor.
        :param stride: convolution stride.
        :return: output tensor.
        """
        x1 = Conv2D(filters, (3, 3), strides=stride, **self.conv_kwargs)(x)
        x1 = BatchNormalization()(x1)
        x1 = LeakyReLU()(x1)
        x1 = Conv2D(filters, (3, 3), **self.conv_kwargs)(x1)
        if stride == 1:
            x2 = x
        else:
            x2 = Conv2D(filters, (1, 1), strides=stride, **self.conv_kwargs)(x)
        x_out = Add()([x1, x2])
        x_out = BatchNormalization()(x_out)
        x_out = LeakyReLU()(x_out)
        return x_out
