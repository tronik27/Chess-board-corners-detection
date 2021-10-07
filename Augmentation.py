import albumentations as aug


def image_augmentation(config: list) -> aug.core.composition.Compose:
    """
    Method for creating a list of transforms according to the configuration.
    :param config: list with a set of parameters for augmentation.
    :return: composed transforms.
    """
    augmentation_list = []
    augmentations = {'vertical_flip': aug.VerticalFlip(),
                     'horizontal_flip': aug.HorizontalFlip(),
                     'sharpen': aug.Sharpen(alpha=(0.2, 0.5), lightness=(0.1, 0.3)),
                     'blur': aug.OneOf([aug.GaussianBlur(blur_limit=(3, 5)), aug.MedianBlur(blur_limit=5)]),
                     'contrast': aug.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=0.5)}
    for augmentation in config:
        augmentation_list.append(augmentations[augmentation])
    transform = aug.Compose(augmentation_list, keypoint_params=aug.KeypointParams(format='xy', remove_invisible=False))
    return transform
