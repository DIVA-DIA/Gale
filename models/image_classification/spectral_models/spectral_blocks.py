import torch.nn as nn
from spectral import DctII2d, iDctII2d, Dft2d, iDft2d


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
