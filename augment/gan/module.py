from torch.nn import functional
import torch
import torch.nn as nn
import torch
from torch import nn


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class ResBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int = None):
        """
        :param in_channels: input channels of the residual block
        :param out_channels: if None, use in_channels. Else, adds a 1x1 conv layer.
        """
        super().__init__()

        if out_channels is None or out_channels == in_channels:
            out_channels = in_channels
            self.conv_shortcut = None
        else:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding='same', bias=False)

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding='same', bias=False)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding='same', bias=False)

    def forward(self, x):

        residual = functional.silu(self.norm1(x))
        residual = self.conv1(residual)

        residual = functional.silu(self.norm2(residual))
        residual = self.conv2(residual)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x.contiguous())

        return x + residual


class Downsample(nn.Module):

    def __init__(self, kernel_size: int = 2, stride: int = 2, padding: int = 0):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        res = torch.nn.functional.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        return res


class Upsample(nn.Module):

    def __init__(self, scale_factor: float = 2.0, mode: str = 'nearest'):
        super().__init__()

        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Encoder(nn.Module):
    def __init__(self, channels: int, num_res_blocks: int, channel_multipliers: tuple, embedding_dim: int):

        super().__init__()

        self.conv_in = torch.nn.Conv2d(3, channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)

        blocks = []
        ch_in = channels

        for i in range(len(channel_multipliers)):

            ch_out = channels * channel_multipliers[i]
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch_in, ch_out))
                ch_in = ch_out

            blocks.append(Downsample())

        self.blocks = nn.Sequential(*blocks)

        self.attn = AttnBlock(in_channels=ch_in)

        self.final_residual = nn.Sequential(*[ResBlock(ch_in) for _ in range(num_res_blocks)])

        self.norm = nn.GroupNorm(num_groups=32, num_channels=ch_in)
        self.conv_out = torch.nn.Conv2d(ch_in, embedding_dim, kernel_size=(1, 1), padding='same')

    def forward(self, x):

        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.attn(x)
        x = self.final_residual(x)
        x = self.norm(x)
        x = functional.silu(x)
        x = self.conv_out(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels: int, num_res_blocks: int, channel_multipliers: tuple, embedding_dim: int):

        super().__init__()

        ch_in = channels * channel_multipliers[-1]

        self.conv_in = torch.nn.Conv2d(embedding_dim, ch_in, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.attn = AttnBlock(in_channels=ch_in)
        self.initial_residual = nn.Sequential(*[ResBlock(ch_in) for _ in range(num_res_blocks)])

        blocks = []
        for i in reversed(range(len(channel_multipliers))):

            ch_out = channels * channel_multipliers[i - 1] if i > 0 else channels

            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch_in, ch_out))
                ch_in = ch_out

            blocks.append(Upsample())

        self.blocks = nn.Sequential(*blocks)

        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.conv_out = torch.nn.Conv2d(channels, 3, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.attn(x)
        x = self.initial_residual(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = functional.silu(x)
        x = self.conv_out(x)
        x = (nn.functional.tanh(x) + 1) / 2

        return x
