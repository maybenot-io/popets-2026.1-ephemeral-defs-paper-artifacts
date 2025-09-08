"""
Wrappers for laserbeak models.
"""

import torch
from laserbeak.cls_cvt import ConvolutionalVisionTransformer
from laserbeak.transdfnet import DFNet

from def_ml.logging.logger import get_logger
from def_ml.models.utils import unsqueeze_batch

logger = get_logger(__name__)


class WrapDFNet(DFNet):

    def example_input(self, X: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return unsqueeze_batch(X)

    def forward(
        self,
        x: dict[str, torch.Tensor],
        sample_sizes=None,
        return_feats=False,
        *args,
        **kwargs,
    ):
        x_ = torch.cat([x_.unsqueeze(1) for x_ in x.values()], dim=1)

        return super().forward(x_, sample_sizes, return_feats, *args, **kwargs)


class CNNVisTransformer(ConvolutionalVisionTransformer):

    def example_input(self, X: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return unsqueeze_batch(X)

    def forward(self, x: torch.Tensor):
        x = torch.cat([x_.unsqueeze(1) for x_ in x.values()], dim=1)
        return super().forward(x)
