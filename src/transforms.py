from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    NormalizeIntensityd,
    ResizeD,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    EnsureTyped,
)


# --------------------------------------------------
# Classification transforms (OASIS / Parkinson)
# --------------------------------------------------

def get_classification_train_transforms(image_size=(96, 96, 96)):
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityd(keys=["image"]),
            ResizeD(keys=["image"], spatial_size=image_size),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image"], prob=0.5, max_k=3),
            RandGaussianNoised(keys=["image"], prob=0.2),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_classification_val_transforms(image_size=(96, 96, 96)):
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityd(keys=["image"]),
            ResizeD(keys=["image"], spatial_size=image_size),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


# --------------------------------------------------
# MSSEG segmentation transforms
# --------------------------------------------------

def get_msseg_train_transforms(image_size=(128, 128, 128)):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ResizeD(keys=["image", "label"], spatial_size=image_size),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_msseg_val_transforms(image_size=(128, 128, 128)):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ResizeD(keys=["image", "label"], spatial_size=image_size),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
