from typing import Tuple, Union

from torch import nn


class BasicModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[Tuple[int, int], int]) -> None:
        # TRY SELU
        super().__init__()
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size[0] // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)
