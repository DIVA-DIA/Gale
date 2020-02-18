import torch.nn as nn
from models.registry import Model

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
class LDA_CIFAR(nn.Module):
    """
    :var conv1   : torch.nn.Conv2d
    :var conv2   : torch.nn.Conv2d
    :var conv3   : torch.nn.Conv2d
        The first three convolutional layers of the network

    :var fc      : torch.nn.Linear
        Final fully connected layer
    """

    def __init__(self, num_classes, input_channels=3, **kwargs):
        """
        :param num_classes: the number of classes in the dataset
        """
        super(LDA_CIFAR, self).__init__()

        self.expected_input_size = (32, 32)

        num_filters_conv1 = 150
        num_filters_conv2 = 150

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, num_filters_conv1, kernel_size=5, stride=3),
            nn.LeakyReLU(),
        )
        # Second layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_filters_conv1, num_filters_conv2, kernel_size=3, stride=2),
            nn.LeakyReLU(),
        )
        # Third layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_filters_conv2, 72, kernel_size=3, stride=1),
            nn.LeakyReLU(),
        )

        # Classification layer
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(288, num_classes)
        )

    def forward(self, x):
        """
        Computes forward pass on the network
        :param x: torch.Tensor
            The input to the model
        :return: torch.Tensor
            Activations of the fully connected layer
        """
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

        layer_1_neurons = 16

        # First layer
        self.conv1 = nn.Sequential(  # in: 32x32x3 out: 8 x 8 x layer_1_neurons
            nn.Conv2d(in_channels=3, out_channels=layer_1_neurons, kernel_size=4, stride=4,
                      padding=0),
            nn.Softsign()
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
