import torch.nn as nn

from models.image_classification.spectral_models.spectral_blocks import DiscreteCosine2dConvBlock, \
    InverseDiscreteCosine2dConvBlock, DiscreteFourier2dConvBlock, InverseDiscreteFourier2dConvBlock, Flatten
from models.registry import Model

# ----------------------------------------------------------------------------------------------------------------------
# DCT
# ----------------------------------------------------------------------------------------------------------------------
@Model
class DCT_1(nn.Module):
    expected_input_size = (149, 149)

    def __init__(self, num_classes, in_channels=3, ocl1=32, fixed=False, random_init=False, **kwargs):
        super().__init__()
        self.features = []

        self.encoder = nn.Sequential(
            DiscreteCosine2dConvBlock(
                in_channels,     ocl1, kernel_size=8, stride=3, padding=0, spectral_width=48, spectral_height=48, fixed=fixed, random_init=random_init,
            ),
            nn.LeakyReLU(),

            nn.Conv2d(
                ocl1       , ocl1 * 2, kernel_size=5, stride=3, padding=1
            ),
            nn.LeakyReLU(),

            nn.Conv2d(
                ocl1 * 2   , ocl1 * 4, kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=16, stride=1),
            Flatten(),
            nn.Linear(ocl1 * 4, num_classes)
        )

    def forward(self, x):
        self.features = self.encoder(x)
        return self.classifier(self.features)


@Model
class DCT_1_Fixed(DCT_1):
    def __init__(self, **kwargs):
        super().__init__(fixed=True, **kwargs)

@Model
class DCT_2(nn.Module):
    expected_input_size = (149, 149)

    def __init__(self, num_classes, in_channels=3, ocl1=32, fixed=False, random_init=False, **kwargs):
        super().__init__()
        self.features = []

        self.encoder = nn.Sequential(
            DiscreteCosine2dConvBlock(
                in_channels,     ocl1, kernel_size=8, stride=3, padding=0, spectral_width=48, spectral_height=48, fixed=fixed, random_init=random_init,
            ),
            nn.LeakyReLU(),

            InverseDiscreteCosine2dConvBlock(
                       ocl1, ocl1 * 2, kernel_size=5, stride=3, padding=1, spectral_width=16, spectral_height=16, fixed=fixed, random_init=random_init,
            ),
            nn.LeakyReLU(),

            nn.Conv2d(
                   ocl1 * 2, ocl1 * 4, kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=16, stride=1),
            Flatten(),
            nn.Linear(ocl1 * 4, num_classes)
        )

    def forward(self, x):
        self.features = self.encoder(x)
        return self.classifier(self.features)


@Model
class DCT_2_Fixed(DCT_2):
    def __init__(self, **kwargs):
        super().__init__(fixed=True, **kwargs)


@Model
class DCT_3(nn.Module):
    expected_input_size = (149, 149)

    def __init__(self, num_classes, in_channels=3, ocl1=32, fixed=False, random_init=False, **kwargs):
        super().__init__()
        self.features = []

        self.encoder = nn.Sequential(
            DiscreteCosine2dConvBlock(
                in_channels,     ocl1, kernel_size=8, stride=3, padding=0, spectral_width=48, spectral_height=48, fixed=fixed, random_init=random_init,
            ),
            nn.LeakyReLU(),

            InverseDiscreteCosine2dConvBlock(
                ocl1       , ocl1 * 2, kernel_size=5, stride=3, padding=1, spectral_width=16, spectral_height=16, fixed=fixed, random_init=random_init,
            ),
            nn.LeakyReLU(),

            DiscreteCosine2dConvBlock(
                ocl1 * 2   , ocl1 * 4, kernel_size=3, stride=1, padding=1, spectral_width=16, spectral_height=16, fixed=fixed, random_init=random_init,
            ),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=16, stride=1),
            Flatten(),
            nn.Linear(ocl1 * 4, num_classes)
        )

    def forward(self, x):
        self.features = self.encoder(x)
        return self.classifier(self.features)


@Model
class DCT_3_Fixed(DCT_3):
    def __init__(self, **kwargs):
        super().__init__(fixed=True, **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
# FFT
# ----------------------------------------------------------------------------------------------------------------------
@Model
class FFT_1(nn.Module):

    expected_input_size = (149, 149)

    def __init__(self, num_classes, in_channels=3, ocl1=26, fixed=False, **kwargs):
        super().__init__()


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
            nn.Linear(ocl1 * 4, num_classes)
        )
    def forward(self, x):
        self.features = self.encoder(x)
        return self.classifier(self.features)


@Model
class FFT_1_Fixed(FFT_1):
    def __init__(self, **kwargs):
        super().__init__(fixed=True, **kwargs)



@Model
class FFT_2(nn.Module):
    expected_input_size = (149, 149)

    def __init__(self, num_classes, in_channels=3, ocl1=26, fixed=False, **kwargs):
        super().__init__()

        self.features = []

        self.encoder = nn.Sequential(
            DiscreteFourier2dConvBlock(
                in_channels,     ocl1, kernel_size=8, stride=3, padding=0, spectral_width=48, spectral_height=48, fixed=fixed
            ),
            nn.LeakyReLU(),

            InverseDiscreteFourier2dConvBlock(
                   ocl1 * 2, ocl1 * 2, kernel_size=5, stride=3, padding=1, spectral_width=16, spectral_height=16, fixed=fixed
            ),
            nn.LeakyReLU(),

            nn.Conv2d(
                   ocl1 * 2, ocl1 * 4, kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=16, stride=1),
            Flatten(),
            nn.Linear(ocl1 * 4, num_classes)
        )

    def forward(self, x):
        self.features = self.encoder(x)
        return self.classifier(self.features)


@Model
class FFT_2_Fixed(FFT_2):
    def __init__(self, **kwargs):
        super().__init__(fixed=True, **kwargs)

@Model
class FFT_3(nn.Module):
    expected_input_size = (149, 149)

    def __init__(self, num_classes, in_channels=3, ocl1=26, fixed=False, **kwargs):
        super().__init__()

        self.features = []

        self.encoder = nn.Sequential(
            DiscreteFourier2dConvBlock(
                in_channels,     ocl1, kernel_size=8, stride=3, padding=0, spectral_width=48, spectral_height=48, fixed=fixed
            ),
            nn.LeakyReLU(),

            InverseDiscreteFourier2dConvBlock(
                   ocl1 * 2, ocl1 * 2, kernel_size=5, stride=3, padding=1, spectral_width=16, spectral_height=16, fixed=fixed
            ),
            nn.LeakyReLU(),

            DiscreteFourier2dConvBlock(
                   ocl1 * 2, ocl1 * 4, kernel_size=3, stride=1, padding=1, spectral_width=16, spectral_height=16, fixed=fixed
            ),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=16, stride=1),
            Flatten(),
            nn.Linear(ocl1 * 8, num_classes)
        )

    def forward(self, x):
        self.features = self.encoder(x)
        return self.classifier(self.features)


@Model
class FFT_3_Fixed(FFT_3):
    def __init__(self, **kwargs):
        super().__init__(fixed=True, **kwargs)

# ----------------------------------------------------------------------------------------------------------------------
# RND
# ----------------------------------------------------------------------------------------------------------------------
@Model
class RND_1(DCT_1):
    def __init__(self, **kwargs):
        super().__init__(random_init=True, **kwargs)

@Model
class RND_2(DCT_2):
    def __init__(self, **kwargs):
        super().__init__(random_init=True, **kwargs)

@Model
class RND_3(DCT_3):
    def __init__(self, **kwargs):
        super().__init__(random_init=True, **kwargs)