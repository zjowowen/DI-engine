from typing import Optional, Dict, Union, List
from functools import reduce
import operator
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from ding.torch_utils import ResFCBlock, ResBlock, Flatten, normed_linear, normed_conv2d
from ding.torch_utils.network.dreamer import Conv2dSame, DreamerLayerNorm
from ding.utils import SequenceType

from ding.nn.parameter import Parameter

def prod(iterable):
    """
    Product of all elements.(To be deprecated soon.)
    This function denifition is for supporting python version that under 3.8.
    In python3.8 and larger, 'math.prod()' is recommended.
    """
    return reduce(operator.mul, iterable, 1)


class SobelConcatModule(nn.Module):
    def __init__(self, input_channel:int=1):
        super(SobelConcatModule, self).__init__()
        self.input_channel = input_channel

        self.sobel_x_filter = Parameter(torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).repeat(self.input_channel, 1, 1, 1), requires_grad=False)
        self.sobel_y_filter = Parameter(torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).repeat(self.input_channel, 1, 1, 1), requires_grad=False)
        

    def forward(self, x):
        sobel_x = F.conv2d(x, self.sobel_x_filter, padding=1, groups=self.input_channel)
        sobel_y = F.conv2d(x, self.sobel_y_filter, padding=1, groups=self.input_channel)
        gradient_magnitude = torch.sqrt(sobel_x**2 + sobel_y**2)

        output_tensor = torch.cat([x, gradient_magnitude], dim=1)
        return output_tensor

class ConvEncoder(nn.Module):
    """
    Overview:
        The ``Convolution Encoder`` used to encode raw 2-dim observations.
    Interfaces:
        ``__init__``, ``_get_flatten_size``, ``forward``.
    """

    def __init__(
            self,
            obs_shape: SequenceType,
            hidden_size_list: SequenceType = [32, 64, 64, 128],
            activation: Optional[nn.Module] = nn.ReLU(),
            kernel_size: SequenceType = [8, 4, 3],
            stride: SequenceType = [4, 2, 1],
            padding: Optional[SequenceType] = None,
            layer_norm: Optional[bool] = False,
            sobel_operator: Optional[bool] = False,
            norm_type: Optional[str] = None
    ) -> None:
        """
        Overview:
            Init the ``Convolution Encoder`` according to the provided arguments.
        Arguments:
            - obs_shape (:obj:`SequenceType`): Sequence of ``in_channel``, plus one or more ``input size``.
            - hidden_size_list (:obj:`SequenceType`): Sequence of ``hidden_size`` of subsequent conv layers \
                and the final dense layer.
            - activation (:obj:`nn.Module`): Type of activation to use in the conv ``layers`` and ``ResBlock``. \
                Default is ``nn.ReLU()``.
            - kernel_size (:obj:`SequenceType`): Sequence of ``kernel_size`` of subsequent conv layers.
            - stride (:obj:`SequenceType`): Sequence of ``stride`` of subsequent conv layers.
            - padding (:obj:`SequenceType`): Padding added to all four sides of the input for each conv layer. \
                See ``nn.Conv2d`` for more details. Default is ``None``.
            - layer_norm (:obj:`bool`): Whether to use ``DreamerLayerNorm``.
            - norm_type (:obj:`str`): Type of normalization to use. See ``ding.torch_utils.network.ResBlock`` \
                for more details. Default is ``None``.
        """
        super(ConvEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.act = activation
        self.hidden_size_list = hidden_size_list
        if padding is None:
            padding = [0 for _ in range(len(kernel_size))]

        layers = []
        self.sobel_operator = sobel_operator
        if self.sobel_operator:
            input_size = obs_shape[0] * 2
            layers.append(SobelConcatModule(input_channel=obs_shape[0]))
        else:
            input_size = obs_shape[0]  # in_channel

        for i in range(len(kernel_size)):
            if layer_norm:
                layers.append(
                    Conv2dSame(
                        in_channels=input_size,
                        out_channels=hidden_size_list[i],
                        kernel_size=(kernel_size[i], kernel_size[i]),
                        stride=(2, 2),
                        bias=False,
                    )
                )
                layers.append(DreamerLayerNorm(hidden_size_list[i]))
                layers.append(self.act)
            else:
                layers.append(nn.Conv2d(input_size, hidden_size_list[i], kernel_size[i], stride[i], padding[i]))
                layers.append(self.act)
            input_size = hidden_size_list[i]
        if len(self.hidden_size_list) >= len(kernel_size) + 2:
            assert self.hidden_size_list[len(kernel_size) - 1] == self.hidden_size_list[
                len(kernel_size)], "Please indicate the same hidden size between conv and res block"
        assert len(
            set(hidden_size_list[len(kernel_size):-1])
        ) <= 1, "Please indicate the same hidden size for res block parts"
        for i in range(len(kernel_size), len(self.hidden_size_list) - 1):
            layers.append(ResBlock(self.hidden_size_list[i - 1], activation=self.act, norm_type=norm_type))
        layers.append(Flatten())
        self.main = nn.Sequential(*layers)

        flatten_size = self._get_flatten_size()
        self.output_size = hidden_size_list[-1]  # outside to use
        self.mid = nn.Linear(flatten_size, hidden_size_list[-1])

    def _get_flatten_size(self) -> int:
        """
        Overview:
            Get the encoding size after ``self.main`` to get the number of ``in-features`` to feed to ``nn.Linear``.
        Returns:
            - outputs (:obj:`torch.Tensor`): Size ``int`` Tensor representing the number of ``in-features``.
        Shapes:
            - outputs: :math:`(1,)`.
        """
        test_data = torch.randn(1, *self.obs_shape)
        with torch.no_grad():
            output = self.main(test_data)
        return output.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Return output embedding tensor of the env observation.
        Arguments:
            - x (:obj:`torch.Tensor`): Env raw observation.
        Returns:
            - outputs (:obj:`torch.Tensor`): Output embedding tensor.
        Shapes:
            - outputs: :math:`(B, N)`, where ``N = hidden_size_list[-1]``.
        """
        x = self.main(x)
        x = self.mid(x)
        return x


class FCEncoder(nn.Module):
    """
    Overview:
        The ``FCEncoder`` used in models to encode raw 1-dim observations.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(
            self,
            obs_shape: int,
            hidden_size_list: SequenceType,
            res_block: bool = False,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            dropout: Optional[float] = None
    ) -> None:
        """
        Overview:
            Init the FC Encoder according to arguments.
        Arguments:
            - obs_shape (:obj:`int`): Observation shape.
            - hidden_size_list (:obj:`SequenceType`): Sequence of ``hidden_size`` of subsequent FC layers.
            - res_block (:obj:`bool`): Whether use ``res_block``. Default is ``False``.
            - activation (:obj:`nn.Module`): Type of activation to use in ``ResFCBlock``. Default is ``nn.ReLU()``.
            - norm_type (:obj:`str`): Type of normalization to use. See ``ding.torch_utils.network.ResFCBlock`` \
                for more details. Default is ``None``.
            - dropout (:obj:`float`): Dropout rate of the dropout layer. If ``None`` then default no dropout layer.
        """
        super(FCEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.act = activation
        self.init = nn.Linear(obs_shape, hidden_size_list[0])

        if res_block:
            assert len(set(hidden_size_list)) == 1, "Please indicate the same hidden size for res block parts"
            if len(hidden_size_list) == 1:
                self.main = ResFCBlock(hidden_size_list[0], activation=self.act, norm_type=norm_type, dropout=dropout)
            else:
                layers = []
                for i in range(len(hidden_size_list)):
                    layers.append(
                        ResFCBlock(hidden_size_list[0], activation=self.act, norm_type=norm_type, dropout=dropout)
                    )
                self.main = nn.Sequential(*layers)
        else:
            layers = []
            for i in range(len(hidden_size_list) - 1):
                layers.append(nn.Linear(hidden_size_list[i], hidden_size_list[i + 1]))
                layers.append(self.act)
                if dropout is not None:
                    layers.append(nn.Dropout(dropout))
            self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Return output embedding tensor of the env observation.
        Arguments:
            - x (:obj:`torch.Tensor`): Env raw observation.
        Returns:
            - outputs (:obj:`torch.Tensor`): Output embedding tensor.
        Shapes:
            - outputs: :math:`(B, N)`, where ``N = hidden_size_list[-1]``.
        """
        x = self.act(self.init(x))
        x = self.main(x)
        return x


class StructEncoder(nn.Module):

    def __init__(self, obs_shape: Dict[str, Union[int, List[int]]]) -> None:
        super(StructEncoder, self).__init__()
        # TODO concrete implementation
        raise NotImplementedError


class IMPALACnnResidualBlock(nn.Module):
    """
    Residual basic block (without batchnorm) in IMPALA cnn encoder.
    Preserves channel number and shape
    """

    def __init__(self, in_channnel, scale=1, batch_norm=False):
        """
        Overview:
            Init every impala cnn residual block.
        Arguments:
            - in_channnel (:obj:`int`): Channel number of input features.
            - scale (:obj:`float`): Scale of module.
        """
        super().__init__()
        self.in_channnel = in_channnel
        self.batch_norm = batch_norm
        s = math.sqrt(scale)
        self.conv0 = normed_conv2d(self.in_channnel, self.in_channnel, 3, padding=1, scale=s)
        self.conv1 = normed_conv2d(self.in_channnel, self.in_channnel, 3, padding=1, scale=s)
        if self.batch_norm:
            self.bn0 = nn.BatchNorm2d(self.in_channnel)
            self.bn1 = nn.BatchNorm2d(self.in_channnel)

    def residual(self, x):
        # inplace should be False for the first relu, so that it does not change the input,
        # which will be used for skip connection.
        # getattr is for backwards compatibility with loaded models
        if self.batch_norm:
            x = self.bn0(x)
        x = F.relu(x, inplace=False)
        x = self.conv0(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv1(x)
        return x

    def forward(self, x):
        return x + self.residual(x)


class IMPALACnnDownStack(nn.Module):
    """
    Downsampling stack from Impala CNN
    """

    def __init__(self, in_channnel, nblock, out_channel, scale=1, pool=True, **kwargs):
        """
        Overview:
            Init every impala cnn block of the Impala Cnn Encoder.
        Arguments:
            - in_channnel (:obj:`int`): Channel number of input features.
            - nblock (:obj:`int`): Residual Block number in each block.
            - out_channel (:obj:`int`): Channel number of output features.
            - scale (:obj:`float`): Scale of the module.
            - pool (:obj:`bool`): Whether to use maxing pooling after first conv layer.
        """
        super().__init__()
        self.in_channnel = in_channnel
        self.out_channel = out_channel
        self.pool = pool
        self.firstconv = normed_conv2d(in_channnel, out_channel, 3, padding=1)
        s = scale / math.sqrt(nblock)
        self.blocks = nn.ModuleList([IMPALACnnResidualBlock(out_channel, scale=s, **kwargs) for _ in range(nblock)])

    def forward(self, x):
        x = self.firstconv(x)
        if self.pool:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for block in self.blocks:
            x = block(x)
        return x

    def output_shape(self, inshape):
        c, h, w = inshape
        assert c == self.in_channnel
        if self.pool:
            return (self.out_channel, (h + 1) // 2, (w + 1) // 2)
        else:
            return (self.out_channel, h, w)


class IMPALAConvEncoder(nn.Module):
    name = "IMPALAConvEncoder"  # put it here to preserve pickle compat

    def __init__(
        self, obs_shape, channels=(16, 32, 32), outsize=256, scale_ob=255.0, nblock=2, final_relu=True, **kwargs
    ):
        """
        Overview:
            Init the Encoder described in paper, \
            IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures, \
            https://arxiv.org/pdf/1802.01561.pdf,
        Arguments:
            - obs_shape (:obj:`int`): Observation shape.
            - channels (:obj:`SequenceType`): Channel number of each impala cnn block. \
                The size of it is the number of impala cnn blocks in the encoder
            - outsize (:obj:`int`): Out feature of the encoder.
            - scale_ob (:obj:`float`): Scale of each pixel.
            - nblock (:obj:`int`): Residual Block number in each block.
            - final_relu (:obj:`bool`): Whether to use Relu in the end of encoder.
        """
        super().__init__()
        self.scale_ob = scale_ob
        c, h, w = obs_shape
        curshape = (c, h, w)
        s = 1 / math.sqrt(len(channels))  # per stack scale
        self.stacks = nn.ModuleList()
        for out_channel in channels:
            stack = IMPALACnnDownStack(curshape[0], nblock=nblock, out_channel=out_channel, scale=s, **kwargs)
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)
        self.dense = normed_linear(prod(curshape), outsize, scale=1.4)
        self.outsize = outsize
        self.final_relu = final_relu

    def forward(self, x):
        x = x / self.scale_ob
        for (i, layer) in enumerate(self.stacks):
            x = layer(x)
        *batch_shape, h, w, c = x.shape
        x = x.reshape((*batch_shape, h * w * c))
        x = F.relu(x)
        x = self.dense(x)
        if self.final_relu:
            x = torch.relu(x)
        return x
