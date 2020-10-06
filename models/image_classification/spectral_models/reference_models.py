import torch.nn as nn

from models.registry import Model
from models.image_classification.spectral_models.spectral_models import DiscreteCosine2dConvBlock, InverseDiscreteCosine2dConvBlock


class Flatten(nn.Module):
    """
    Flatten a convolution block into a simple vector.

    Replaces the flattening line (view) often found into forward() methods of networks. This makes it
    easier to navigate the network with introspection
    """

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


@Model
class BaselineConv(nn.Module):
    expected_input_size = (149, 149)

    def __init__(self, num_classes, in_channels=3, ocl1=32,  # output channels layer 1
                 **kwargs):
        super(BaselineConv, self).__init__()
        self.features = []

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels,     ocl1, kernel_size=8, stride=3, padding=0), nn.LeakyReLU(),
            nn.Conv2d(       ocl1, ocl1 * 2, kernel_size=5, stride=3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(   ocl1 * 2, ocl1 * 4, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
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
class BaselineDeep(nn.Module):

    expected_input_size = (149, 149)

    def __init__(self, num_classes, in_channels=3, ocl1=32,  # output channels layer 1
                 **kwargs):
        super(BaselineDeep, self).__init__()


        self.features = []

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels,     ocl1, kernel_size=8, stride=3, padding=0), nn.LeakyReLU(),
            nn.Conv2d(       ocl1,     ocl1, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
            nn.Conv2d(       ocl1, ocl1 * 2, kernel_size=5, stride=3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(   ocl1 * 2, ocl1 * 4, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=16, stride=1),
            Flatten(),
            nn.Linear(ocl1 * 4, num_classes)
        )

    def forward(self, x):
        self.features = self.encoder(x)
        return self.classifier(self.features)