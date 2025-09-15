"""
@author: zjf
@create time: 2024/11/29 10:06
@desc:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from mamba_ssm import Mamba


def get_dwconv_layer(
        in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
        bias: bool = False, spatial_dims=3
):
    depth_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=in_channels,
                             strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True, groups=in_channels)
    point_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels,
                             strides=stride, kernel_size=1, bias=bias, conv_only=True, groups=1)
    return torch.nn.Sequential(depth_conv, point_conv)


def FM_Init(in_channels: int, out_channels: int):
    return get_dwconv_layer(in_channels=in_channels, out_channels=out_channels)


def FM_Output(in_channels: int, out_channels: int):
    return torch.nn.Sequential(
        get_norm_layer(name=("group", {"num_groups": 8}), spatial_dims=3, channels=in_channels),
        get_act_layer(name=("RELU", {"inplace": True})),
        get_dwconv_layer(in_channels, out_channels, kernel_size=1, bias=True)
    )


class Down(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_channels, in_channels * 2, (3, 3, 3), (1, 2, 2), (1, 1, 1)),
            nn.BatchNorm3d(in_channels * 2)
        )
        self.max_pooling = nn.Sequential(
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(in_channels, in_channels * 2, 3, 1, 1),
            nn.BatchNorm3d(in_channels * 2)
        )
        self.RELU = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.RELU(self.conv_down(x) + self.max_pooling(x))


def get_down_sample(in_channels: int, out_channels: int):
    return Down(in_channels)


class Up(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_t = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=(1, 2, 2),
                                         stride=(1, 2, 2))

    def forward(self, x):
        return self.conv_t(x)


def get_up_sample(in_channels: int, out_channels: int):
    return Up(in_channels)


class mamba_block(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # if x.dtype == torch.float16:
        #     x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


class conv_layer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dwconv = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 7, 7), padding=(1, 3, 3), groups=in_channels)
        self.norm1 = get_norm_layer(name=("group", {"num_groups": 8}), spatial_dims=3, channels=in_channels)
        self.pwconv1 = nn.Conv3d(in_channels, in_channels * 4, kernel_size=1, groups=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(in_channels * 4, in_channels, kernel_size=1, groups=1)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.norm1(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        return x


class Down_Block(nn.Module):

    def __init__(self, in_channels: int, ) -> None:
        """
        Args:
            in_channels: number of input channels.
        """

        super().__init__()

        self.conv_layer = conv_layer(in_channels)
        self.mamba_layer = mamba_block(in_channels, in_channels)

    def forward(self, x):
        conv = self.conv_layer(x)
        mamba = self.mamba_layer(x)

        return conv + mamba


class Up_Block(nn.Module):

    def __init__(self, in_channels: int, ) -> None:
        """
        Args:
            in_channels: number of input channels.
        """

        super().__init__()

        self.conv_layer = conv_layer(in_channels)
        self.mamba_layer = mamba_block(in_channels, in_channels)

    def forward(self, x):
        conv = self.conv_layer(x)
        mamba = self.mamba_layer(x)

        return conv + mamba


class FFS_T(nn.Module):

    def __init__(
            self,
            init_filters: int = 8,  # 网络整体宽度
            in_channels: int = 3,  # 输入维度
            out_channels: int = 1,  # 输出维度
            blocks_down: tuple = (2, 2, 2, 4),
            blocks_up: tuple = (2, 2, 2),
    ):
        super().__init__()

        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.ConvInit = FM_Init(in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.ConvFinal = FM_Output(init_filters, out_channels)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, filters = (self.blocks_down, self.init_filters)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2 ** i
            down_sample = (
                get_down_sample(layer_in_channels // 2, layer_in_channels)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                down_sample,
                *[Down_Block(layer_in_channels) for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        blocks_up, filters = (self.blocks_up, self.init_filters)
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[
                        Up_Block(sample_in_channels // 2) for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    get_up_sample(sample_in_channels, sample_in_channels // 2)
                )
            )
        return up_layers, up_samples

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.ConvInit(x)
        down_x = []

        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)

        x = self.ConvFinal(x)
        return x

    def forward_focus_dists(self, FS_output, Focus_Dists):
        _, T, Height, Width = Focus_Dists.shape
        FS_output = torch.squeeze(FS_output, 1)
        FS_output = F.interpolate(FS_output, [Height, Width], mode='bilinear')
        FS_output = F.softplus(FS_output) + 1e-6
        FS_output = FS_output / FS_output.sum(axis=1, keepdim=True)
        depth = torch.sum(Focus_Dists * FS_output, dim=1)
        return depth

    def forward(self, x: torch.Tensor, Focus_Dists: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        # 列表翻转
        down_x.reverse()
        x = self.decode(x, down_x)
        x = self.forward_focus_dists(x, Focus_Dists)
        return x

# zjf 测试模型
images = torch.randn(size=(4, 3, 10, 128, 128)).cuda()
Focus_Dists = torch.randn(size=(4, 10, 128, 128)).cuda()
model = FFS_T().cuda()
depth = model(images, Focus_Dists)
# pass
print(depth)
print(depth.shape)
