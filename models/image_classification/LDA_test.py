"""
CNN with 6 conv layers and 3 fully connected classification layer
Designed for CIFAR (input: 32x32x3)
"""

import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
class LDA_Deep(nn.Module):
    """
    :var conv1-n   : torch.nn.Conv2d
        The first n convolutional layers of the network
    :var fc      : torch.nn.Linear
        Fully connected layer
    :var cl      : torch.nn.Linear
        Final fully connected layer for classification
    """

    # new_size = (width - filter + 2padding) / stride + 1
    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super(LDA_Deep, self).__init__()

        self.expected_input_size = (32, 32)

        # Num filters conv1
        nfconv1 = 64

        # Layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, nfconv1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(nfconv1),
        )
        # Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(nfconv1, nfconv1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(nfconv1),
        )
        # Layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(nfconv1, nfconv1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(nfconv1),
        )
        # Layer 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(nfconv1, nfconv1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(nfconv1),
        )
        # Layer 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(nfconv1, nfconv1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(nfconv1),
        )
        # Layer 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(nfconv1, nfconv1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(nfconv1),
        )
        # Layer 7
        self.conv7 = nn.Sequential(
            nn.Conv2d(nfconv1, nfconv1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(nfconv1),
        )
        # Layer 8
        self.conv8 = nn.Sequential(
            nn.Conv2d(nfconv1, nfconv1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(nfconv1),
        )
        # Layer 9
        self.conv9 = nn.Sequential(
            nn.Conv2d(nfconv1, nfconv1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(nfconv1),
        )
        # Layer 10
        self.conv10 = nn.Sequential(
            nn.Conv2d(nfconv1, nfconv1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(nfconv1),
        )
        # Layer 11
        self.conv11 = nn.Sequential(
            nn.Conv2d(nfconv1, nfconv1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(nfconv1),
        )
        # Layer 12
        self.conv12 = nn.Sequential(
            nn.Conv2d(nfconv1, nfconv1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(nfconv1),
        )
        # Layer 13
        self.conv13 = nn.Sequential(
            nn.Conv2d(nfconv1, nfconv1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(nfconv1),
        )
        # Layer 14
        self.conv14 = nn.Sequential(
            nn.Conv2d(nfconv1, nfconv1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(nfconv1),
        )
        # Layer 15
        self.conv15 = nn.Sequential(
            nn.Conv2d(nfconv1, nfconv1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(nfconv1),
        )
        # Layer 16
        self.conv16 = nn.Sequential(
            nn.Conv2d(nfconv1, nfconv1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(nfconv1),
        )
        # Layer 17
        self.conv17 = nn.Sequential(
            nn.Conv2d(nfconv1, nfconv1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(nfconv1),
        )
        # Layer 18
        self.conv18 = nn.Sequential(
            nn.Conv2d(nfconv1, nfconv1, kernel_size=3, stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(nfconv1),
            nn.AvgPool2d(kernel_size=10)
        )

        # Classification layer
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=nfconv1, out_features=output_channels),
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
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)

        x = self.fc(x)
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

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        """
        :param output_channels: the number of classes in the dataset
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
            nn.Linear(288, output_channels)
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


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
class LDA_Simple(nn.Module):
    """
    :var conv1   : torch.nn.Conv2d
        The convolutional layer of the network
    :var cl      : torch.nn.Linear
        Final fully connected layer for classification
    """

    def __init__(self, output_channels=10, **kwargs):
        super(LDA_Simple, self).__init__()

        self.expected_input_size = (32, 32)

        # First layer
        self.conv1 = nn.Sequential(  # in: 32x32x3 out: 32x32x16
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=4,
                      padding=0),
            nn.Softsign()
        )
        # Classification layer
        self.cl = nn.Sequential(
            nn.Linear(in_features=8 * 8 * 16, out_features=output_channels),
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
        x = x.view(x.size(0), -1)
        x = self.cl(x)
        return x
