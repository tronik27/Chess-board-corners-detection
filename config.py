import os

# Paths to data.
# PATH = 'path/to/folder/containing/dataset'
PATH_TO_IMAGES = r'E:\DATASETS\idchess_zadanie\xtrain.npy'
PATH_TO_KEYPOINTS = r'E:\DATASETS\idchess_zadanie\ytrain.npy'
PATH_TO_TRAIN_IMAGES = os.path.join(os.path.dirname(PATH_TO_IMAGES), 'train_images.npy')
PATH_TO_VAL_IMAGES = os.path.join(os.path.dirname(PATH_TO_IMAGES), 'val_images.npy')
PATH_TO_TRAIN_KEYPOINTS = os.path.join(os.path.dirname(PATH_TO_KEYPOINTS), 'train_keypoints.npy')
PATH_TO_VAL_KEYPOINTS = os.path.join(os.path.dirname(PATH_TO_KEYPOINTS), 'val_keypoints.npy')
# WORK_DATA_PATH = 'path/to/folder/containing/images/for/classification'
WORK_DATA_PATH = r'E:\DATASETS\idchess_zadanie\xtest.npy'
VALID_SIZE = 0.3

# Training parameters.
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 135

# Custom model parameters.
NUM_FILTERS = 4
MODEL_NAME = 'keypoints_resnet_detector'
NUM_PREDICTIONS = 8
INPUT_SHAPE = (256, 256, 1)
REGULARIZATION = 0.0005
INPUT_NAME = 'input'
OUTPUT_NAME = 'keypoints'

# Paths to weights and saved model.
WEIGHTS_PATH = 'keypoints_detection_model/weights_2'
MODEL_PATH = '{} (trained_model2)'.format(MODEL_NAME)

# augmentation configuration
# AUG_CONFIG = ['vertical_flip', 'horizontal_flip', 'sharpen', 'blur']
AUG_CONFIG = ['vertical_flip', 'horizontal_flip', 'blur']

# service parameters
SHOW_LEARNING_CURVES = True
SHOW_IMAGE_DATA = True
NUM_OF_EXAMPLES = 10