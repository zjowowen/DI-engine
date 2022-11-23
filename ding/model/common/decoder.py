from typing import Optional
from functools import reduce
import operator
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from ding.torch_utils import ResFCBlock, ResBlock, Flatten, normed_linear, normed_conv2d
from ding.utils import SequenceType


class IMPALAConvDecoder(nn.Module):
    name = "IMPALAConvDecoder"  # put it here to preserve pickle compat

    def __init__(
        self,
        obs_shape,
        channels=(16, 32, 32),
        outsize=256,
        scale_ob=255.0,
        nblock=2,
        final_relu=True,
        batch_norm=False,
        layer_norm=False,
        init_orthogonal=False,
        post_norm=False,
        **kwargs
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
            stack = IMPALACnnDownStack(
                curshape[0],
                curshape,
                nblock=nblock,
                out_channel=out_channel,
                scale=s,
                batch_norm=batch_norm,
                layer_norm=layer_norm,
                init_orthogonal=init_orthogonal,
                post_norm=post_norm,
                **kwargs
            )
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)
        self.dense = nn.Linear(prod(curshape), outsize)
        if init_orthogonal:
            if isinstance(self.dense, torch.nn.Linear):
                torch.nn.init.orthogonal_(self.dense.weight)
                torch.nn.init.zeros_(self.dense.bias)
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

    def report_overall_gradient_norm(self):
        norm_gradient = {}

        norm_gradient["weight"] = []
        norm_gradient["bias"] = []
        if self.dense.weight.grad is not None:
            norm_gradient["weight"].append(torch.linalg.norm(self.dense.weight.grad).item())
        if self.dense.bias.grad is not None:
            norm_gradient["bias"].append(torch.linalg.norm(self.dense.bias.grad).item())
        for stack in self.stacks:
            if stack.firstconv.weight.grad is not None:
                norm_gradient["weight"].append(torch.linalg.norm(stack.firstconv.weight.grad).item())
            if stack.firstconv.bias.grad is not None:
                norm_gradient["bias"].append(torch.linalg.norm(stack.firstconv.bias.grad).item())
            for block in stack.blocks:
                if block.conv0.weight.grad is not None:
                    norm_gradient["weight"].append(torch.linalg.norm(block.conv0.weight.grad).item())
                if block.conv0.bias.grad is not None:
                    norm_gradient["bias"].append(torch.linalg.norm(block.conv0.bias.grad).item())
                if block.conv1.weight.grad is not None:
                    norm_gradient["weight"].append(torch.linalg.norm(block.conv1.weight.grad).item())
                if block.conv1.bias.grad is not None:
                    norm_gradient["bias"].append(torch.linalg.norm(block.conv1.bias.grad).item())
                if block.batch_norm == True:
                    if block.bn0.weight.grad is not None:
                        norm_gradient["weight"].append(torch.linalg.norm(block.bn0.weight.grad).item())
                    if block.bn0.bias.grad is not None:
                        norm_gradient["bias"].append(torch.linalg.norm(block.bn0.bias.grad).item())
                    if block.bn1.weight.grad is not None:
                        norm_gradient["weight"].append(torch.linalg.norm(block.bn1.weight.grad).item())
                    if block.bn1.bias.grad is not None:
                        norm_gradient["bias"].append(torch.linalg.norm(block.bn1.bias.grad).item())
                elif block.layer_norm == True:
                    if block.bn0.weight.grad is not None:
                        norm_gradient["weight"].append(torch.linalg.norm(block.bn0.weight.grad).item())
                    if block.bn0.bias.grad is not None:
                        norm_gradient["bias"].append(torch.linalg.norm(block.bn0.bias.grad).item())
                    if block.bn1.weight.grad is not None:
                        norm_gradient["weight"].append(torch.linalg.norm(block.bn1.weight.grad).item())
                    if block.bn1.bias.grad is not None:
                        norm_gradient["bias"].append(torch.linalg.norm(block.bn1.bias.grad).item())

        total_norm_gradient = {}
        total_norm_gradient["weight"] = 0
        total_norm_gradient["bias"] = 0

        for weight_norm_gradient in norm_gradient["weight"]:
            total_norm_gradient["weight"] += weight_norm_gradient * weight_norm_gradient
        total_norm_gradient["weight"] = math.sqrt(total_norm_gradient["weight"])

        for bias_norm_gradient in norm_gradient["bias"]:
            total_norm_gradient["bias"] += bias_norm_gradient * bias_norm_gradient
        total_norm_gradient["bias"] = math.sqrt(total_norm_gradient["bias"])

        return total_norm_gradient
