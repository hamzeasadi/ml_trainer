"""
network generallayers
"""

import os
from typing import List, OrderedDict, Dict, Optional
from enum import IntEnum

import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange







class PaddingType(IntEnum):
    """
    define the type of padding
    Valid: no padding
    One: Add one line of zero padding
    Same: pad so that output is the same as input
    """
    Valid = 0
    One = 1
    Same = 2




def calculate_padding(input_size:List, output_size:List, ks:List, stride:List, symmetric:bool=True)->List:
    """
    based on provided data calculate the required padding
    output = (input_size + 2*padding - ks)/stride + 1
    input_size: [c,h,w]
    output_size: [c h, w]
    ks: [int,int]
    stride: [int, int]
    """
    padd_a = (((output_size[1] - 1)*stride[0]) + ks[0] - input_size[1])/2
    if symmetric and padd_a%2!=0:
        padd_a += 1
    
    padd_b = (((output_size[2] - 1)*stride[1]) + ks[1] - input_size[2])/2
    if symmetric and padd_b%2!=0:
        padd_b += 1
    
    return [int(padd_a//2), int(padd_b//2)]




def calc_outshape(input_size:List, outch:int, padding_type:PaddingType, ks:List, stride:List)->List:
    """
    calculate the output shape based on padding and conv info
    """
    output_size = [outch,0,0]
    if padding_type.value == 2:
        output_size[1:] = input_size[1:]
        return output_size
    
    h = int((input_size[1] - ks[0] + 2*padding_type.value)/stride[0]) + 1
    w = int((input_size[2] - ks[1] + 2*padding_type.value)/stride[1]) + 1

    output_size[1:] = [h, w]
    return output_size




def convolution(input_size:List, outch:int, ks:List=[3,3], stride:List=[1,1], padding_type:PaddingType=PaddingType.One,
                activation:Optional[nn.Module]=nn.ReLU, bn:bool=True, bias:bool=False, **activation_args):
    
    """
    create a convolution layer with specificed parameters
    args:
        ...
    return:
        conv_layer: nn.Module
        output_size: [c,h,w]
    """
    conv_layer:nn.ModuleList = nn.ModuleList()
    output_size = calc_outshape(input_size=input_size, outch=outch, padding_type=padding_type, ks=ks, stride=stride)
    
    padding = calculate_padding(input_size=input_size, output_size=output_size, ks=ks, stride=stride, symmetric=True)
    layer = nn.Conv2d(in_channels=input_size[0], out_channels=outch, kernel_size=ks, padding=padding, stride=stride, bias=bias)
    conv_layer.append(layer)
    
    if bn:
        layer = nn.BatchNorm2d(num_features=outch)
        conv_layer.append(layer)
    
    if activation:
        layer = activation(**activation_args)
        conv_layer.append(layer)

    
    return nn.Sequential(*conv_layer), output_size




class SEBlock(nn.Module):
    """
    channel attention
    """
    def __init__(self, inch:int, reduction:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=inch, out_features=inch//reduction, bias=False)
        self.fc2 = nn.Linear(in_features=inch//reduction, out_features=inch, bias=False)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avgpool(x).view(b, c)
        y = self.act(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        y = y.view(b,c,1,1)
        return x * y.expand_as(x)





class ResidualBase(nn.Module):
    """
    base residual block
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.residual_block:nn.Sequential
        self.downsample:Optional[nn.Sequential] = None

    def forward(self, x):
        out = self.residual_block(x)
        residual = x
        if self.downsample:
            residual = self.downsample(x)
        
        return out + residual




class ResidualBlock(ResidualBase):
    """
    resnet residual block without downsample
    """

    def __init__(self, num_conv:int, input_size:List, outch:int, ks:List=[3,3], stride:List=[1,1],
                 padding_type:PaddingType=PaddingType.One, activition:nn.Module=nn.ReLU, bn:bool=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        convs = nn.ModuleList()
        input_layer_size = input_size
        for _ in range(num_conv):
            layer, out_size = convolution(input_size=input_layer_size, outch=outch, ks=ks, stride=stride, padding_type=padding_type,
                                activation=activition, bn=bn)
            input_layer_size = out_size
            convs.append(layer)

        
        self.output_size = input_layer_size

        self.residual_block = nn.Sequential(*convs)


    


class InvertResBlock(nn.Module):
    """
    mobilenet v2 inversted residual block
    """
    def __init__(self, input_size:List, outch:int, ks:List=[3,3], stride:List=[1,1], padding_type:PaddingType=PaddingType.One,
                activation:Optional[nn.Module]=nn.ReLU, bn:bool=True, bias:bool=False, **activation_args) -> None:
        super().__init__()

        conv0 = nn.Conv2d(in_channels=input_size[0], out_channels=outch, kernel_size=1, stride=1, bias=False)
        conv1 = nn.Conv2d(in_channels=outch, out_channels=outch, kernel_size=ks, stride=stride, padding=1, groups=outch)
        seblk = SEBlock(inch=outch, reduction=outch//4)
        batch_norm = nn.BatchNorm2d(num_features=outch)
        act = nn.ReLU6()
        conv2 = nn.Conv2d(in_channels=outch, out_channels=input_size[0], kernel_size=1, stride=1, bias=False)
        batch_norm1 = nn.BatchNorm2d(num_features=input_size[0])
        self.blk = nn.Sequential(conv0, conv1, batch_norm, act, seblk, conv2, batch_norm1)

        self.output_size = input_size

    def forward(self, x):
        return self.blk(x)+x





class SandboxBlock(nn.Module):
    """
    sandboc block
    """
    def __init__(self, input_size:List, outch:int, ks:List=[3,3], stride:List=[1,1], padding_type:PaddingType=PaddingType.One,
                activation:Optional[nn.Module]=nn.ReLU, bn:bool=True, bias:bool=False, **activation_args) -> None:
        super().__init__()
        conv0 = nn.Sequential(
            nn.Conv2d(in_channels=input_size[0], out_channels=input_size[0], kernel_size=ks, stride=stride, padding=1, groups=input_size[0]),
            nn.BatchNorm2d(num_features=input_size[0]), nn.ReLU6(),
            nn.Conv2d(in_channels=input_size[0], out_channels=outch, kernel_size=1, stride=1, padding=0), 
            nn.BatchNorm2d(num_features=outch))
        conv1 = nn.Sequential(
            nn.Conv2d(in_channels=outch, out_channels=outch, kernel_size=ks, stride=stride, padding=1, groups=outch),
            SEBlock(inch=outch, reduction=outch//4),nn.BatchNorm2d(num_features=outch), nn.ReLU6(),
            nn.Conv2d(in_channels=outch, out_channels=input_size[0], kernel_size=1, stride=1, padding=0), 
            nn.BatchNorm2d(num_features=input_size[0]))
        conv2 = nn.Sequential(
            nn.Conv2d(in_channels=input_size[0], out_channels=input_size[0], kernel_size=ks, stride=stride, padding=1, groups=input_size[0]),
            nn.BatchNorm2d(num_features=input_size[0]), nn.ReLU6()
        )

        self.blk = nn.Sequential(conv0, conv1, conv2)

        self.output_size = input_size

    def forward(self, x):
        return self.blk(x) + x






def attdotprdt(k:torch.Tensor, q:torch.Tensor, v:torch.Tensor):
    dim_size = q.shape[-1]
    qv_att_weight = torch.bmm(q, k.transpose(2,1))/(dim_size**0.5)
    qv_att = torch.softmax(qv_att_weight, dim=-1)
    att_v = torch.einsum("blt,btv->blv", [qv_att, v])
    return att_v




class SDPAttention(nn.Module):
    """
    docs
    """
    def __init__(self, kv_mapper:nn.Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kv_mapper = kv_mapper
        self.output_size = kv_mapper.output_size


    def forward(self, x):
        b,c,h,w = x.shape
        v = self.kv_mapper(x)
        values = rearrange(v, "b c h w -> b (h w) c")
        queries = rearrange(x, "b c h w -> b (h w) c")
        weighted_value = attdotprdt(k=values, q=queries, v=values)
        x = rearrange(weighted_value, "b (h w) c -> b c h w", h=h, w=w)

        return x








if __name__ == "__main__":
    print(__file__)

    input_size = [3, 32, 32]
    ks = [3,3]
    stride = [1,1]
    pad_type = PaddingType.One
    x = torch.randn(size=(1, *input_size))

    conv = InvertResBlock(input_size=input_size, outch=10, padding_type=pad_type, stride=stride)
    att = SDPAttention(kv_mapper=conv)
    print(att)
    out = att(x)
    
    print(att.output_size)
    print(out.shape)
    






