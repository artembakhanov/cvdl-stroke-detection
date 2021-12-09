from typing import Union, Tuple

from torch import nn

from src.models.modules.fsm import FSM
from src.utils.utils import batch_to_tensor


class SequenceFSM(nn.Module):
    def __init__(self, channels: int, sequence_size: int) -> None:
        super().__init__()
        self.fsms = nn.ModuleList(
            [FSM(channels) for _ in range(sequence_size)]
        )

    def forward(self, sequence):
        sequence = [fsm(img.unsqueeze(0)) for img, fsm in zip(sequence, self.fsms)]
        sequence = batch_to_tensor(sequence, included_y=False).squeeze(0)
        return sequence


class SequenceConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]],
                 sequence_size: int):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel, stride, padding) for _ in range(sequence_size)]
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, sequence):
        sequence = [self.relu(self.bn(conv(img.unsqueeze(0)))) for img, conv in zip(sequence, self.convs)]
        sequence = batch_to_tensor(sequence, included_y=False).squeeze(0)
        return sequence
