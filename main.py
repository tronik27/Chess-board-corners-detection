from train_and_evaluate import KeyPointsDetection
from Data_Preprocessing import prepare_dataset
from config import PATH_TO_IMAGES, PATH_TO_KEYPOINTS, PATH_TO_TRAIN_IMAGES, PATH_TO_VAL_IMAGES,\
    PATH_TO_TRAIN_KEYPOINTS, PATH_TO_VAL_KEYPOINTS, BATCH_SIZE, INPUT_SHAPE, NUM_PREDICTIONS, NUM_FILTERS,\
    LEARNING_RATE, MODEL_NAME, NUM_EPOCHS, AUG_CONFIG, MODEL_PATH, SHOW_IMAGE_DATA, VALID_SIZE, SHOW_LEARNING_CURVES,\
    NUM_OF_EXAMPLES, INPUT_NAME, OUTPUT_NAME, WEIGHTS_PATH, REGULARIZATION


def main(prepare_data: bool, train: bool, evaluate: bool, save: bool):
    if prepare_data:
        prepare_dataset(path_to_images=PATH_TO_IMAGES, path_to_keypoints=PATH_TO_KEYPOINTS, valid_size=VALID_SIZE)
    #  Creating key points detection neural network
    detector = KeyPointsDetection(
        batch_size=BATCH_SIZE,
        target_size=INPUT_SHAPE,
        num_predictions=NUM_PREDICTIONS,
        num_filters=NUM_FILTERS,
        learning_rate=LEARNING_RATE,
        model_name=MODEL_NAME,
        input_name=INPUT_NAME,
        output_name=OUTPUT_NAME,
        path_to_model_weights=WEIGHTS_PATH,
        regularization=REGULARIZATION
    )
    if train:
        # Training neural network
        detector.train(
            path_to_train_images=PATH_TO_TRAIN_IMAGES,
            path_to_train_keypoints=PATH_TO_TRAIN_KEYPOINTS,
            path_to_val_images=PATH_TO_VAL_IMAGES,
            path_to_val_keypoints=PATH_TO_VAL_KEYPOINTS,
            epochs=NUM_EPOCHS,
            augmentation=AUG_CONFIG,
            show_learning_curves=SHOW_LEARNING_CURVES,
            show_image_data=SHOW_IMAGE_DATA,
            num_of_examples=NUM_OF_EXAMPLES
        )
    if evaluate:
        #  Testing neural network
        detector.evaluate(
            path_to_test_images=PATH_TO_VAL_IMAGES,
            path_to_test_keypoints=PATH_TO_VAL_KEYPOINTS,
            show_image_data=SHOW_IMAGE_DATA,
            num_examples=NUM_OF_EXAMPLES
        )
    if save:
        #  Saving trained neural network
        detector.save_model(path_to_save=MODEL_PATH)


if __name__ == '__main__':
    main(
        prepare_data=False,
        train=True,
        evaluate=True,
        save=True
    )
