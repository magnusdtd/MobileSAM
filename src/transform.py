import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: tuple[int, int]) -> tuple[A.Compose, A.Compose]:
    train_transform = A.Compose(
        [
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )

    return train_transform, val_transform
