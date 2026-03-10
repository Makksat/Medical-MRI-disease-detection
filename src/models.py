from monai.networks.nets import DenseNet121, UNet


def build_classification_model(
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 1,
):
    model = DenseNet121(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    return model


def build_msseg_unet(
    spatial_dims: int = 3,
    in_channels: int = 4,
    out_channels: int = 1,
):
    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    return model
