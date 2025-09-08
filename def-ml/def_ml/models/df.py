import torch
from torch import nn

from def_ml.models.utils import unsqueeze_batch
from def_ml.trace.features import Feats


# from https://github.com/Xinhao-Deng/Website-Fingerprinting-Library/blob/master/WFlib/models/DF.py
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        pool_size,
        pool_stride,
        dropout_p,
        activation,
    ):
        super(ConvBlock, self).__init__()
        padding = (
            kernel_size // 2
        )  # Calculate padding to keep the output size same as input size
        # Define a convolutional block consisting of two convolutional layers, each followed by batch normalization and activation
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=padding,
                bias=False,
            ),  # First convolutional layer
            nn.BatchNorm1d(out_channels),  # Batch normalization layer
            activation(inplace=True),  # Activation function (e.g., ELU or ReLU)
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                stride,
                padding=padding,
                bias=False,
            ),  # Second convolutional layer
            nn.BatchNorm1d(out_channels),  # Batch normalization layer
            activation(inplace=True),  # Activation function
            nn.MaxPool1d(
                pool_size, pool_stride, padding=0
            ),  # Max pooling layer to downsample the input
            nn.Dropout(p=dropout_p),  # Dropout layer for regularization
        )

    def forward(self, x):
        # Pass the input through the convolutional block
        return self.block(x)


# from https://github.com/Xinhao-Deng/Website-Fingerprinting-Library/blob/master/WFlib/models/DF.py
class DF(nn.Module):
    name: str = "df-orig"

    def __init__(self, num_classes, large_input=False):
        super(DF, self).__init__()

        # Configuration parameters for the convolutional blocks
        filter_num = [32, 64, 128, 256]  # Number of filters for each block
        kernel_size = 8  # Kernel size for convolutional layers
        conv_stride_size = 1  # Stride size for convolutional layers
        pool_stride_size = 4  # Stride size for max pooling layers
        pool_size = 8  # Kernel size for max pooling layers
        length_after_extraction = (
            18  # Length of the feature map after the feature extraction part
        )

        # Define the feature extraction part of the network using a sequential container with ConvBlock instances
        self.feature_extraction = nn.Sequential(
            ConvBlock(
                1,
                filter_num[0],
                kernel_size,
                conv_stride_size,
                pool_size,
                pool_stride_size,
                0.1,
                nn.ELU,
            ),  # Block 1
            ConvBlock(
                filter_num[0],
                filter_num[1],
                kernel_size,
                conv_stride_size,
                pool_size,
                pool_stride_size,
                0.1,
                nn.ReLU,
            ),  # Block 2
            ConvBlock(
                filter_num[1],
                filter_num[2],
                kernel_size,
                conv_stride_size,
                pool_size,
                pool_stride_size,
                0.1,
                nn.ReLU,
            ),  # Block 3
            ConvBlock(
                filter_num[2],
                filter_num[3],
                kernel_size,
                conv_stride_size,
                pool_size,
                pool_stride_size,
                0.1,
                nn.ReLU,
            ),  # Block 4
        )

        # Define the classifier part of the network
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten the tensor to a vector
            nn.Linear(
                9728 if large_input else filter_num[3] * length_after_extraction,
                512,
                bias=False,
            ),  # Fully connected layer
            nn.BatchNorm1d(512),  # Batch normalization layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.Dropout(p=0.7),  # Dropout layer for regularization
            nn.Linear(512, 512, bias=False),  # Fully connected layer
            nn.BatchNorm1d(512),  # Batch normalization layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.Dropout(p=0.5),  # Dropout layer for regularization
            nn.Linear(512, num_classes),  # Output layer
        )

    def forward(self, x: dict[Feats, torch.tensor]):

        x_ = torch.cat([x_.unsqueeze(1) for x_ in x.values()], dim=1)
        # Pass the input through the feature extraction part
        x_ = self.feature_extraction(x_)

        # Pass the output through the classifier part
        x_ = self.classifier(x_)

        return x_

    def example_input(self, x: dict[str, torch.tensor]) -> dict[str, torch.tensor]:
        return unsqueeze_batch(x)
