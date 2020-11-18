import math
import torch.nn as nn
from torch import sigmoid

from models.registry import Model

class Flatten(nn.Module):
    """
    Flatten a convolution block into a simple vector.

    Replaces the flattening line (view) often found into forward() methods of networks. This makes it
    easier to navigate the network with introspection
    """
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class Swish(nn.Module):
    """Activation function from: https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * sigmoid(x)


class _BasicBlock(nn.Module):
    """This is the basic block of the network. For the order of the operation see: https://arxiv.org/abs/1603.05027"""

    def __init__(self, in_filters, out_filters, stride=1, downsample=None):
        super(_BasicBlock, self).__init__()
        # BN + Swish(1)
        self.bn1 = nn.BatchNorm2d(in_filters)
        self.swish1 = Swish()
        self.conv1 = nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=3, stride=stride, padding=1, bias=False)
        # BN + Swish + Conv  (2)
        self.bn2 = nn.BatchNorm2d(out_filters)
        self.swish2 = Swish()
        self.conv2 = nn.Conv2d(in_channels=out_filters, out_channels=out_filters, kernel_size=3, stride=1, padding=1, bias=False)
        # Downsample (can be None!) THIS MUST BE LAST added to the list of the --init won't work!
        self.downsample = downsample

    def forward(self, x):
        """
        Computes forward pass on the block.

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on

        Returns
        -------
        Variable
            Activations of the block summed to the residual connection values
        """
        residual = x

        x = self.bn1(x)
        x = self.swish1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.swish2(x)
        x = self.conv2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        return x

class RaieResNet(nn.Module):
    r"""
    ResNet model architecture adapted from `<https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py>`
    It is better suited for smaller images as the expected input size is 32x32
    """
    expected_input_size = (32, 32)

    def __init__(self, num_block, num_classes, **kwargs):
        """
        Creates a RaieResNet model from the scratch.

        RaieResNet differs from a regular ResNet for it has an input size of 32x32
        rather than 224x224. The intuition behind is that it makes not so much sense
        to rescale an input image from a native domain from 32x32 to 224x224 (e.g. in
        CIFAR or MNIST datasets) because of multiple reasons. To begin with,  the image
        distortion of such an upscaling is massive, but most importantly we pay an
        extremely high overhead both in terms of computation and in number of parameters
        involved. For this reason the ResNet architecture has been adapted to fit a
        smaller input size.

        Adaptations:
        The conv1 layer filters have been reduced from 7x7 to 5x5 with stride 1, the
        padding and initial maxpool removed. Additionally the number of filters has been increased to 128.
        This way, with an input of 32x32 the final output of the conv1 is then 28x28x128
        which matches the expected input size of conv3x layer. This is no coincidence.
        Since the image is already smaller than 56x56 (which is the expected size of
        conv2x layer) the conv2x layer has been dropped entirely.
        This would reduce the total number of layers in the network. In an effort to
        reduce this gap, we increased the number of blocks in the conv3x layer with
        as many blocks there where in the conv2x (we basically moved blocks from one layer
        to another).

        The order as well as which operations happen in a _basicBlock is set according to
        https://arxiv.org/abs/1603.05027 and Swish is used instead of ReLU.

        Parameters
        ----------
        num_block : List(int)
            Number of blocks to put in each layer of the network. Must be of size 3
        num_classes : int
            Number of neurons in the last layer
        """
        super(RaieResNet, self).__init__()

        self.num_input_filters = 128  # Attention: this gets updated after each conv[2,3,4] layer creation!

        # First convolutional layer, bring the input into the 28x28x128 desired size
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.num_input_filters, kernel_size=5, stride=1),
            nn.BatchNorm2d(self.num_input_filters),
            Swish(),
        )

        # Bulk part of the network with four groups of blocks (expected input size of conv2: 28x28x128)
        self.conv2 = self._make_layer(_BasicBlock, 128, num_block[0])
        self.conv3 = self._make_layer(_BasicBlock, 256, num_block[1], stride=2)
        self.conv4 = self._make_layer(_BasicBlock, 512, num_block[2], stride=2)

        # Final averaging and fully connected layer for classification (expected input size: 7x7x512)
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Flatten(),
        )
        self.fc = nn.Linear(512, num_classes)

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

    def _make_layer(self, block_type, filters, num_block, stride=1):
        """
        This function creates a convx layer for the ResNet.
        A convx layer is a group of blocks (either _BasickBlock or _BottleNeck)
        where each block is skipped by a connection. Refer to the original paper for more info.

        Parameters
        ----------
        block_type : nn.Module
            Type of the blocks to be used in the network. Either _BasickBlock or _BottleNeck.
        filters : int
            Number of filters to be used in the two convolutional layers inside a block
        num_block : List(int)
            Number of blocks to put in each layer of the network. Must be of size 4
        stride : int
            Specifies the stride. It is used also to flag when the residual dimensions have to be halved
        Returns
        -------
        torch.nn.Sequential
            The convx layer as a sequential containing all the blocks
        """
        downsample = None
        layers = []
        # Create the downsample module if its needed
        if stride != 1 or self.num_input_filters != filters:
            # This modules halves the dimension of the residual connection (dotted line Fig.3 of the paper)
            # Also beware that in the case of _Bottleneck block this could up upsampling the residual to an higher dimension!
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.num_input_filters, out_channels=filters, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(filters),
            )

        # Append the first block which can have the downsample
        layers.append(block_type(self.num_input_filters, filters, stride, downsample))

        # Add the remaining amount of BasicBlocks to this layer
        self.num_input_filters = filters
        for i in range(1, num_block):
            layers.append(block_type(self.num_input_filters, filters))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Computes forward pass on the network

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on. (input to the model)

        Returns
        -------
        Variable
            Activations of the fully connected layer
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x


@Model
def raieresnet18(**kwargs):
    """
    Constructs a _raieresnet-18 model.

    Returns
    -------
    torch.nn.Module
        The model of the network
    """
    return RaieResNet(num_block=[4, 2, 2], **kwargs)
raieresnet18.expected_input_size = RaieResNet.expected_input_size


@Model
def raieresnet34(**kwargs):
    """
    Constructs a _raieresnet-34 model.

    Returns
    -------
    torch.nn.Module
        The model of the network
    """
    return RaieResNet(num_block=[7, 6, 3], **kwargs)
raieresnet34.expected_input_size = RaieResNet.expected_input_size

