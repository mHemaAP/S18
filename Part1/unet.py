import torch
import torch.nn as nn
import torch.nn.functional as F

######## UNet Network Architecture ######## 

class ContractingBlock(nn.Module):

  def __init__(self, in_channels, out_channels, stride, max_pooling=True):
    super(ContractingBlock,self).__init__()

    self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1)
    self.bn1   = nn.BatchNorm2d(out_channels)
    self.relu1 = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3, padding=1)
    self.bn2   = nn.BatchNorm2d(out_channels)
    self.relu2 = nn.ReLU(inplace=True)

    self.maxpool  = nn.MaxPool2d(kernel_size=2,stride=2)
    self.strided_convolution = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)

    self.max_pooling = max_pooling
    self.stride  = stride

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu2(x)

    skip = x
    if self.max_pooling:
      if self.stride:
        x = self.strided_convolution(x)
      else:
        x = self.maxpool(x)

    return x, skip

class ExpandingBlock(nn.Module):

  def __init__(self, in_channels, out_channels, conv_transpose):
    super(ExpandingBlock,self).__init__()

    self.conv2d_transpose = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
    self.upsample  = nn.Upsample(scale_factor = 2, mode='bilinear')
    self.conv2d    = nn.Conv2d(in_channels, in_channels //2, kernel_size=1)

    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) # 512,256
    self.bn1   = nn.BatchNorm2d(out_channels)
    self.relu1 = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) # 256,256
    self.bn2   = nn.BatchNorm2d(out_channels)
    self.relu2 = nn.ReLU(inplace=True)
    self.conv_transpose = conv_transpose

  def forward(self,x,skip):

    if self.conv_transpose:
      x = self.conv2d_transpose(x)
    else:
      x = self.upsample(x)
      x = self.conv2d(x)

    # dim=1 is the channel
    x = torch.cat((x,skip), dim = 1)


    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu2(x)

    return x

class UNet(nn.Module):

  def __init__(self, in_channels, out_channels, stride, conv_transpose):
    super(UNet,self).__init__()

    self.contract1 = ContractingBlock(in_channels, 32, stride)
    self.contract2 = ContractingBlock(32, 64, stride)
    self.contract3 = ContractingBlock(64, 128, stride)
    self.contract4 = ContractingBlock(128, 256, stride)
    self.contract5 = ContractingBlock(256, 512, stride=False, max_pooling=False)

    self.expand1 = ExpandingBlock(512, 256, conv_transpose)
    self.expand2 = ExpandingBlock(256, 128, conv_transpose)
    self.expand3 = ExpandingBlock(128, 64, conv_transpose)
    self.expand4 = ExpandingBlock(64, 32, conv_transpose)

    self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

  def forward(self,x):
    x,skip1 = self.contract1(x)

    x,skip2 = self.contract2(x)

    x,skip3 = self.contract3(x)

    x,skip4 = self.contract4(x)

    x,skip5 = self.contract5(x)

    x = self.expand1(x,skip4)

    x = self.expand2(x,skip3)
    
    x = self.expand3(x,skip2)
    
    x = self.expand4(x,skip1)
    
    x = self.final_conv(x)

    return x