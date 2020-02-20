import torch.nn as nn
from torch import sigmoid

from models.registry import Model
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * sigmoid(x)

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
@Model
class InitBaseline(nn.Module):
    expected_input_size = (149, 149)

    def __init__(self, num_classes, **kwargs):
        super(InitBaseline, self).__init__()

        ocl1 = 32

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=ocl1, kernel_size=8, stride=3, padding=0),
            Swish(),
        )
        # Second layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=ocl1, out_channels= ocl1 * 2,  kernel_size=5, stride=3, padding=1),
            Swish(),
        )
        # Third layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=ocl1 * 2, out_channels=ocl1 * 4,  kernel_size=3, stride=1, padding=1),
            Swish(),
            nn.AvgPool2d(kernel_size=16, stride=1),
            Flatten(),
        )

        # Classification layer
        self.fc = nn.Sequential(
            nn.Linear(ocl1 * 4, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x


####################################################################################################
####################################################################################################
####################################################################################################
@Model
class LDA_Simple(nn.Module):
    expected_input_size = (32, 32)

    def __init__(self, num_classes, **kwargs):
        super(LDA_Simple, self).__init__()

        layer_1_neurons = 48

        # First layer
        self.conv1 = nn.Sequential(  # in: 32x32x3 out: 8 x 8 x layer_1_neurons
            nn.Conv2d(in_channels=3, out_channels=layer_1_neurons, kernel_size=4, stride=4,
                      padding=0),
            Swish()
            # nn.Softsign()
        )
        # Classification layer
        self.cl = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=8 * 8 * layer_1_neurons, out_features=num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.cl(x)
        return x

