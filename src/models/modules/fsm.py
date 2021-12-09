import torch
from torch import nn

from src.models.modules.basic_module import BasicModule


class FSM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        reduced_channels = max(channels // 3, 4)
        intermed_channels = max(reduced_channels // 2, 2)
        self.for_x = BasicModule(channels, reduced_channels, (3, 3))
        self.for_theta = BasicModule(reduced_channels, intermed_channels, (1, 1))  # reshape
        self.for_phi = BasicModule(reduced_channels, intermed_channels, (1, 1))  # reshape
        # f = theta * phi
        self.for_g = BasicModule(reduced_channels, intermed_channels, (1, 1))  # reshape
        # y = f * g
        self.y_conv = BasicModule(intermed_channels, reduced_channels, (1, 1))  # for y before summation
        # k = g + x
        self.last_conv = BasicModule(reduced_channels, channels, (3, 3))

    def forward(self, origx):
        x = self.for_x(origx)
        bs, c, w, h = x.shape
        theta, phi, g = self.for_theta(x), self.for_phi(x), self.for_g(x)
        c = theta.shape[1]
        theta = theta.permute(0, 2, 3, 1).reshape(bs, h * w, c)
        phi = phi.reshape(bs, c, h * w)
        g = g.permute(0, 2, 3, 1).reshape(bs, h * w, c)
        f = torch.matmul(theta, phi)
        y = self.y_conv(torch.matmul(f, g).reshape(bs, c, w, h))
        z = torch.add(y, x)
        z = self.last_conv(z)
        z = torch.add(z, origx)
        return z
