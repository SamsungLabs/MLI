__all__ = ['DoubleConvResnet']

from torch import nn

from lib.networks.blocks import DoubleConvBlock
from lib.networks.blocks.ffc import ConcatTupleLayer, FFCResnetBlock
from lib.networks.blocks.norm import LayerNormTotal


class DoubleConvResnet(nn.Module):
    def __init__(self,
                 input_dim: int,
                 dims: list,
                 conv_dim: int = 2,
                 norm: str = 'ln',
                 activation: str = 'relu',
                 pad_type: str = 'replicate',
                 output_activation: bool = False,
                 output_norm: bool = True,
                 use_residual: bool = False,
                 weight_norm: str = 'none'
                 ):
        """
        Stacked DoubleConv blocks.

        Args:
            input_dim: num input channels
            dims: list of list format [[ inner_dim_block1, output_dim_block1 ], [ inner_dim_block2, output_dim_block2 ] ...]
            conv_dim: dimension of convolutions 3D/2D/1D
            norm: list of supported normalizations here :func:`~lib.networks.blocks.base.get_norm_layer`
            activation: list of supported normalizations here :func:`~lib.networks.blocks.base.get_activation`
            output_activation: If output activation is not needed, set False, else it is set same as activation.
            output_norm: If output normalization is not needed, set False, else it is set same as normalization.
            use_residual: enable residual connection between input and output
            weight_norm: type of weight normalization
        """
        super().__init__()

        self.model = []

        block_input_dim = input_dim
        for block_dims in dims[:-1]:
            self.model.append(
                DoubleConvBlock(
                    input_dim=block_input_dim,
                    output_dim=block_dims[1],
                    inner_dim=block_dims[0],
                    conv_dim=conv_dim,
                    norm=norm,
                    activation=activation,
                    pad_type=pad_type,
                    output_activation=True,
                    output_norm=output_norm,
                    use_residual=use_residual,
                    weight_norm=weight_norm
                )
            )
            block_input_dim = block_dims[1]

        self.model.append(
            DoubleConvBlock(
                input_dim=block_input_dim,
                output_dim=dims[-1][1],
                inner_dim=dims[-1][0],
                conv_dim=conv_dim,
                norm=norm,
                activation=activation,
                pad_type=pad_type,
                output_activation=output_activation,
                output_norm=output_norm,
                use_residual=use_residual,
                weight_norm=weight_norm
            )
        )

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class DoubleFFConvResnet(nn.Module):
    def __init__(self,
                 input_dim: int,
                 dims: list,
                 pad_type: str = 'reflect',
                 ratio_gin: float = 0.0,
                 ratio_ginter: float = 0.75,
                 ratio_gout: float = 0.0,
                 ):
        """
        Stacked DoubleConv blocks.

        Args:
            input_dim: num input channels
            dims: list of list format [[ inner_dim_block1, output_dim_block1 ], [ inner_dim_block2, output_dim_block2 ] ...]
            conv_dim: dimension of convolutions 3D/2D/1D
            norm: list of supported normalizations here :func:`~lib.networks.blocks.base.get_norm_layer`
            activation: list of supported normalizations here :func:`~lib.networks.blocks.base.get_activation`
            output_activation: If output activation is not needed, set False, else it is set same as activation.
            output_norm: If output normalization is not needed, set False, else it is set same as normalization.
            use_residual: enable residual connection between input and output
            weight_norm: type of weight normalization
        """
        super().__init__()

        self.model = []
        self.input_dim = input_dim
        self.dims = dims
        block_input_dim = input_dim
        for i, block_dims in enumerate(dims[:-1]):
            if i == 0:
                ratio_gin_block = ratio_gin
            else:
                ratio_gin_block = ratio_ginter

            self.model.append(
                FFCResnetBlock(
                    input_dim=block_input_dim,
                    output_dim=block_dims[1],
                    inner_dim=block_dims[0],
                    padding_type=pad_type,
                    norm_layer=LayerNormTotal,
                    activation_layer=nn.ReLU,
                    ratio_gin=ratio_gin_block,
                    ratio_ginter=ratio_ginter,
                    ratio_gout=ratio_ginter,
                )
            )
            block_input_dim = block_dims[1]

        self.model.append(
            FFCResnetBlock(
                input_dim=block_input_dim,
                output_dim=dims[-1][1],
                inner_dim=dims[-1][0],
                padding_type=pad_type,
                norm_layer=LayerNormTotal,
                activation_layer=nn.ReLU,
                ratio_gin=ratio_ginter,
                ratio_ginter=ratio_ginter,
                ratio_gout=ratio_gout,
            )
        )
        # self.model.append(ConcatTupleLayer())
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        x = self.model(x)[0]
        return x
