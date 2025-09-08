import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler

from def_ml.models.utils import unsqueeze_batch


def make_layers(cfg, in_channels=32):
    """
    Create a sequence of convolutional and pooling layers.

    Parameters:
    cfg (list): Configuration list specifying the layers.
    in_channels (int): Number of input channels.

    Returns:
    nn.Sequential: Sequential container with the layers.
    """
    layers = []

    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool1d(3), nn.Dropout(0.3)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, stride=1, padding=1)
            layers += [
                conv1d,
                nn.BatchNorm1d(v, eps=1e-05, momentum=0.1, affine=True),
                nn.ReLU(),
            ]
            in_channels = v

    return nn.Sequential(*layers)


def make_first_layers(in_channels=1, out_channel=32):
    """
    Create the initial convolutional layers.

    Parameters:
    in_channels (int): Number of input channels.
    out_channel (int): Number of output channels.

    Returns:
    nn.Sequential: Sequential container with the initial layers.
    """
    layers = []
    conv2d1 = nn.Conv2d(
        in_channels, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1)
    )
    layers += [
        conv2d1,
        nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(),
    ]

    conv2d2 = nn.Conv2d(
        out_channel, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1)
    )
    layers += [
        conv2d2,
        nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(),
    ]

    layers += [nn.MaxPool2d((1, 3)), nn.Dropout(0.1)]

    conv2d3 = nn.Conv2d(out_channel, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [
        conv2d3,
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(),
    ]

    conv2d4 = nn.Conv2d(64, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [
        conv2d4,
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(),
    ]

    layers += [nn.MaxPool2d((2, 2)), nn.Dropout(0.1)]

    return nn.Sequential(*layers)


# from https://github.com/Xinhao-Deng/Website-Fingerprinting-Library/blob/master/WFlib/models/RF.py
class RF(nn.Module):
    name: str = "rf"

    def __init__(self, num_classes=100, num_tab=1):
        """
        Initialize the RF model.

        Parameters:
        num_classes (int): Number of output classes.
        num_tab (int): Number of tabs (not used in this model).
        """
        super(RF, self).__init__()

        # Create feature extraction layers
        features = make_layers([128, 128, "M", 256, 256, "M", 512] + [num_classes])
        init_weights = True
        self.first_layer_in_channel = 1
        self.first_layer_out_channel = 32

        # Create the initial convolutional layers
        self.first_layer = make_first_layers(self.first_layer_in_channel)
        self.features = features
        self.class_num = num_classes

        # Adaptive average pooling layer for classification
        self.classifier = nn.AdaptiveAvgPool1d(1)

        # Fully connected layer to project to embedding space
        self.to_emb = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=num_classes * 65, out_features=128),
        )

        # Initialize weights
        if init_weights:
            self._initialize_weights()

    def _dict2x(self, x: dict[str, torch.tensor]) -> torch.tensor:
        return torch.cat([x_.unsqueeze(1) for x_ in x.values()], dim=1).unsqueeze(1)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after passing through the network.
        """

        x = self._dict2x(x)
        # shape [batch_size, 1, CH, NBINS]

        x = self.first_layer(x)
        x = x.view(x.size(0), self.first_layer_out_channel, -1)
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x

    def example_input(self, x: dict[str, torch.tensor]) -> dict[str, torch.tensor]:
        return unsqueeze_batch(x)

    def _initialize_weights(self):
        """
        Initialize weights for the network layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class RFLRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        last_epoch: int = -1,
        n_epochs: int = 100,
    ):
        self.n_epochs = n_epochs
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        return [lr * (0.2 ** (self.last_epoch / self.n_epochs)) for lr in self.base_lrs]
