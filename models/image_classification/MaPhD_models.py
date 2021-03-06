import math

import torch
import torch.nn as nn
from torch import sigmoid

from models.registry import Model


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
    expected_input_size = (32, 32)

    def __init__(self, num_classes, **kwargs):
        super(InitBaseline, self).__init__()

        f = 32  # Initial number of dimensions

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(3, f, kernel_size=3, stride=1, padding=1), Swish())
        # Block 1: 32x32
        self.conv2 = nn.Sequential(nn.Conv2d(f, f, kernel_size=3, stride=1, padding=1), Swish())
        # Last conv + GAP + flatten
        self.conv3 = nn.Sequential(
            nn.Conv2d(f, f, kernel_size=3, stride=1, padding=1), Swish(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Flatten(),
        )

        # Classification layer
        self.fc = nn.Sequential(
            nn.Linear(f, num_classes)
        )

        # Initialize the weights of all layers. For Conv2d and Linear we use "Kaiming .He"
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                n = m.in_features
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
@Model
class InitBaselineVGGLike(nn.Module):
    expected_input_size = (32, 32)

    def __init__(self, num_classes, activation_function='swish', **kwargs):
        super(InitBaselineVGGLike, self).__init__()

        af = Swish if 'swish' in activation_function else nn.Softsign

        cb = True  # Enable/Disable bias for convolutional layers
        f = 64  # Initial number of dimensions

        # First layer: bring 32x32 to 28x28
        self.conv_in = nn.Sequential(nn.Conv2d(3    , f    , bias=cb, kernel_size=5), af())
        # Block 1: 28x28
        self.conv_b1 = nn.Sequential(
            nn.Conv2d(f,     f, bias=cb, kernel_size=3, padding=1), af(),
            nn.Conv2d(f,     f, bias=cb, kernel_size=3, padding=1), af(),
            nn.Conv2d(f,     f, bias=cb, kernel_size=3, padding=1), af(),
            nn.Conv2d(f,     f, bias=cb, kernel_size=3, padding=1), af(),
            nn.Conv2d(f, f * 2, bias=cb, kernel_size=3, padding=1), af(),
        )
        # Block 2: 28x28
        self.conv_b2 = nn.Sequential(
            nn.Conv2d(f * 2, f * 2, bias=cb, kernel_size=3, padding=1), af(),
            nn.Conv2d(f * 2, f * 2, bias=cb, kernel_size=3, padding=1), af(),
            nn.Conv2d(f * 2, f * 2, bias=cb, kernel_size=3, padding=1), af(),
            nn.Conv2d(f * 2, f * 2, bias=cb, kernel_size=3, padding=1), af(),
            nn.Conv2d(f * 2, f * 4, bias=cb, kernel_size=3, padding=1, stride=2), af(),
        )
        # Block 3: 14x14
        self.conv_b3 = nn.Sequential(
            nn.Conv2d(f * 4, f * 4, bias=cb, kernel_size=3, padding=1), af(),
            nn.Conv2d(f * 4, f * 4, bias=cb, kernel_size=3, padding=1), af(),
            nn.Conv2d(f * 4, f * 4, bias=cb, kernel_size=3, padding=1), af(),
            nn.Conv2d(f * 4, f * 4, bias=cb, kernel_size=3, padding=1), af(),
            nn.Conv2d(f * 4, f * 8, bias=cb, kernel_size=3, padding=1, stride=2), af(),
        )
        # Block 4: 7x7
        self.conv_b4 = nn.Sequential(
            nn.Conv2d(f * 8, f * 8, bias=cb, kernel_size=3, padding=1), af(),
            nn.Conv2d(f * 8, f * 8, bias=cb, kernel_size=3, padding=1), af(),
            nn.Conv2d(f * 8, f * 8, bias=cb, kernel_size=3, padding=1), af(),
            nn.Conv2d(f * 8, f * 8, bias=cb, kernel_size=3, padding=1), af(),
            nn.Conv2d(f * 8, f * 8, bias=cb, kernel_size=3, padding=1), af(),
        )
        self.gap = nn.Sequential(
            # GAP + flatten
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Flatten(),
        )

        # Classification layer
        self.fc = nn.Sequential(
            nn.Linear(f * 8, num_classes)
        )

        # Initialize the weights of all layers. For Conv2d and Linear we use "Kaiming .He"
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0, math.sqrt(1. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                n = m.in_features
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        x = self.conv_in(x)

        x = self.conv_b1(x)
        x = self.conv_b2(x)
        x = self.conv_b3(x)
        x = self.conv_b4(x)
        x = self.gap(x)
        x = self.fc(x)
        return x

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
@Model
class LDApaper(nn.Module):
    expected_input_size = (23, 23)

    def __init__(self, num_classes, activation_function='swish', **kwargs):
        super(LDApaper, self).__init__()

        af = Swish if 'swish' in activation_function else nn.Softsign

        cb = True  # Enable/Disable bias for convolutional layers
        f = 24  # Initial number of dimensions

        # First layer: 28x28
        self.conv_1 = nn.Sequential(nn.Conv2d(3    , f    , bias=cb, kernel_size=5, stride=3), af())
        # Second layer:
        self.conv_2 = nn.Sequential(nn.Conv2d(f    , f * 2, bias=cb, kernel_size=3, stride=2), af())
        # Third layer:
        self.conv_3 = nn.Sequential(nn.Conv2d(f * 2, f * 3, bias=cb, kernel_size=3), af(),
            Flatten(),
        )

        # Classification layer
        self.fc = nn.Sequential(
            nn.Linear(f * 3, num_classes)
        )

        # Initialize the weights of all layers. For Conv2d and Linear we use "Kaiming .He"
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     if isinstance(m, nn.Linear):
        #         n = m.in_features
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x):

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)

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

