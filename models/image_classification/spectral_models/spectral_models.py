import torch.nn as nn
from spectral import DctII2d, iDctII2d, Dft2d, iDft2d
from models.registry import Model


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Module parts are defined here
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Flatten(nn.Module):
    """
    Flatten a convolution block into a simple vector.

    Replaces the flattening line (view) often found into forward() methods of networks. This makes it
    easier to navigate the network with introspection
    """

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


# ----------------------------------------------------------------------------------------------------------------------
# DCT
# ----------------------------------------------------------------------------------------------------------------------
class DiscreteCosine2dConvBlock(nn.Module):

    def __init__(self,
                 in_channels,  # Input channels on the conv layer
                 out_channels,  # Output channels of the conv layer
                 kernel_size,
                 stride,
                 padding,
                 spectral_width,  # This corresponds to the width of the output of the conv layer
                 spectral_height,  # This corresponds to the height of the output of the conv layer
                 weight_normalization=True,  # Normalize the weights of the spectral matrices
                 fixed=False,  # Freeze the spectral weights
                 random_init=False,  # Initialize the spectral weights at random (so not spectral at all!)
                 scaling_factor=1.0,  # Scale the spectral values with a constant
                 groups_conv=1,
                 ):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv, padding=padding)
        self.cos = DctII2d(nrows=spectral_width, ncols=spectral_height, weight_normalization=weight_normalization,
                           fixed=fixed, random_init=random_init, scaling_factor=scaling_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.cos(x)
        return x


class InverseDiscreteCosine2dConvBlock(nn.Module):

    def __init__(self,
                 in_channels,  # Input channels on the conv layer
                 out_channels,  # Output channels of the conv layer
                 kernel_size,
                 stride,
                 padding,
                 spectral_width,  # This corresponds to the width of the output of the conv layer
                 spectral_height,  # This corresponds to the height of the output of the conv layer
                 weight_normalization=True,  # Normalize the weights of the spectral matrices
                 fixed=False,  # Freeze the spectral weights
                 random_init=False,  # Initialize the spectral weights at random (so not spectral at all!)
                 scaling_factor=1.0,  # Scale the spectral values with a constant
                 groups_conv=1,
                 ):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv, padding=padding)
        self.cos = iDctII2d(nrows=spectral_width, ncols=spectral_height, weight_normalization=weight_normalization,
                            fixed=fixed,random_init=random_init, scaling_factor=scaling_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.cos(x)
        return x


# ----------------------------------------------------------------------------------------------------------------------
# FFT
# ----------------------------------------------------------------------------------------------------------------------
class DiscreteFourier2dConvBlock(nn.Module):

    def __init__(self,
                 in_channels,  # Input channels on the conv layer
                 out_channels,  # Output channels of the conv layer
                 kernel_size,
                 stride,
                 padding,
                 spectral_width,  # This corresponds to the width of the output of the conv layer
                 spectral_height,  # This corresponds to the height of the output of the conv layer
                 weight_normalization=True,  # Normalize the weights of the spectral matrices
                 fixed=False,  # Freeze the spectral weights
                 random_init=False,  # Initialize the spectral weights at random (so not spectral at all!)
                 scaling_factor=1.0,  # Scale the spectral values with a constant
                 groups_conv=1,
                 ):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv, padding=padding)
        self.dft = Dft2d(nrows=spectral_width, ncols=spectral_height, weight_normalization=weight_normalization,
                         fixed=fixed,random_init=random_init, scaling_factor=scaling_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.dft(x)
        return x


class InverseDiscreteFourier2dConvBlock(nn.Module):

    def __init__(self,
                 in_channels,  # Input channels on the conv layer
                 out_channels,  # Output channels of the conv layer
                 kernel_size,
                 stride,
                 padding,
                 spectral_width,  # This corresponds to the width of the output of the conv layer
                 spectral_height,  # This corresponds to the height of the output of the conv layer
                 weight_normalization=True,  # Normalize the weights of the spectral matrices
                 fixed=False,  # Freeze the spectral weights
                 random_init=False,  # Initialize the spectral weights at random (so not spectral at all!)
                 scaling_factor=1.0,  # Scale the spectral values with a constant
                 groups_conv=1,
                 ):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv, padding=padding)
        self.dft = iDft2d(nrows=spectral_width, ncols=spectral_height, weight_normalization=weight_normalization,
                          fixed=fixed,random_init=random_init, scaling_factor=scaling_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.dft(x)
        return x


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Bidirectional Spectral Models
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# DCT
# ----------------------------------------------------------------------------------------------------------------------
@Model
class DCTBidir(nn.Module):

    def __init__(self, output_channels=10, in_channels=3, ocl1=32,  # output channels layer 1
                 fixed=False, **kwargs):
        super().__init__()

        self.expected_input_size = (149, 149)
        self.features = []

        self.encoder = nn.Sequential(
            DiscreteCosine2dConvBlock(in_channels, ocl1, kernel_size=8, stride=3, padding=0,
                                      spectral_width=48, spectral_height=48, fixed=fixed),
            nn.LeakyReLU(),
            InverseDiscreteCosine2dConvBlock(ocl1, ocl1 * 2, kernel_size=5, stride=3, padding=1,
                                             spectral_width=16, spectral_height=16, fixed=fixed),
            nn.LeakyReLU(),
            DiscreteCosine2dConvBlock(ocl1 * 2, ocl1 * 4, kernel_size=3, stride=1, padding=1,
                                      spectral_width=16, spectral_height=16, fixed=fixed),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=16, stride=1),
            Flatten(),
            nn.Linear(ocl1 * 4, output_channels)
        )

    def forward(self, x):
        self.features = self.encoder(x)
        return self.classifier(self.features)


@Model
class DCTBidir_Fixed(DCTBidir):
    def __init__(self, output_channels=10, in_channels=3, ocl1=32, **kwargs):
        super().__init__(output_channels=output_channels, in_channels=in_channels, ocl1=ocl1,  fixed=True, **kwargs)

# ----------------------------------------------------------------------------------------------------------------------
# FFT
# ----------------------------------------------------------------------------------------------------------------------
@Model
class FFTBidir(nn.Module):

    def __init__(self, output_channels=10, in_channels=3, ocl1=26,  # output channels layer 1
                 fixed=False, scaling_factor=1.0, **kwargs):
        super().__init__()

        self.expected_input_size = (149, 149)
        self.features = []

        self.encoder = nn.Sequential(
            DiscreteFourier2dConvBlock(in_channels, ocl1, kernel_size=8, stride=3, padding=0,
                                       spectral_width=48, spectral_height=48, fixed=fixed),
                                       #scaling_factor=scaling_factor, weight_normalization=False),
            nn.LeakyReLU(),
            InverseDiscreteFourier2dConvBlock(ocl1 * 2, ocl1 * 2, kernel_size=5, stride=3, padding=1,
                                              spectral_width=16, spectral_height=16, fixed=fixed),
                                              #scaling_factor=scaling_factor, weight_normalization=False),
            nn.LeakyReLU(),
            DiscreteFourier2dConvBlock(ocl1 * 2, ocl1 * 4, kernel_size=3, stride=1, padding=1,
                                       spectral_width=16, spectral_height=16, fixed=fixed),
                                       #scaling_factor=scaling_factor, weight_normalization=False),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=16, stride=1),
            Flatten(),
            nn.Linear(ocl1 * 8, output_channels)
        )

    def forward(self, x):
        self.features = self.encoder(x)
        return self.classifier(self.features)


@Model
class FFTBidir_Fixed(FFTBidir):
    def __init__(self, output_channels=10, in_channels=3, ocl1=27, **kwargs):
        super().__init__(output_channels=output_channels, in_channels=in_channels, ocl1=ocl1,  fixed=True,
                         scaling_factor=2, **kwargs)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# First Block Spectral Models
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# DCT
# ----------------------------------------------------------------------------------------------------------------------
@Model
class DCTFirst(nn.Module):

    def __init__(self, output_channels=10, in_channels=3, ocl1=32,  # output channels layer 1
                 fixed=False, **kwargs):
        super().__init__()

        self.expected_input_size = (149, 149)
        self.features = []

        self.encoder = nn.Sequential(
            DiscreteCosine2dConvBlock(in_channels, ocl1, kernel_size=8, stride=3, padding=0,
                                      spectral_width=48, spectral_height=48, fixed=fixed),
            nn.LeakyReLU(),
            nn.Conv2d(ocl1, ocl1 * 2, kernel_size=5, stride=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(ocl1 * 2, ocl1 * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=16, stride=1),
            Flatten(),
            nn.Linear(ocl1 * 4, output_channels)
        )

    def forward(self, x):
        self.features = self.encoder(x)
        return self.classifier(self.features)


@Model
class DCTFirst_Fixed(DCTFirst):
    def __init__(self, output_channels=10, in_channels=3, ocl1=32, **kwargs):
        super().__init__(output_channels=output_channels, in_channels=in_channels, ocl1=ocl1,  fixed=True, **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
# FFT
# ----------------------------------------------------------------------------------------------------------------------
@Model
class FFTFirst(nn.Module):

    def __init__(self, output_channels=10, in_channels=3, ocl1=26,  # output channels layer 1
                 fixed=False, scaling_factor=1.0, **kwargs):
        super().__init__()

        self.expected_input_size = (149, 149)
        self.features = []

        self.encoder = nn.Sequential(
            DiscreteFourier2dConvBlock(in_channels, ocl1, kernel_size=8, stride=3, padding=0,
                                       spectral_width=48, spectral_height=48, fixed=fixed),
                                       #scaling_factor=scaling_factor, weight_normalization=False),
            nn.LeakyReLU(),
            nn.Conv2d(ocl1 * 2, ocl1 * 2, kernel_size=5, stride=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(ocl1 * 2, ocl1 * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=16, stride=1),
            Flatten(),
            nn.Linear(ocl1 * 4, output_channels)
        )
    def forward(self, x):
        self.features = self.encoder(x)
        return self.classifier(self.features)


@Model
class FFTFirst_Fixed(FFTFirst):
    def __init__(self, output_channels=10, in_channels=3, ocl1=27, **kwargs):
        super().__init__(output_channels=output_channels, in_channels=in_channels, ocl1=ocl1,  fixed=True,
                         scaling_factor=2, **kwargs)


