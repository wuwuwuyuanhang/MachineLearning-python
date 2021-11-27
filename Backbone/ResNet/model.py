# @Auther : wuwuwu 
# @Time : 2021/9/18 
# @File : model.py
# @Description :


import torch
import torch.nn as nn

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class BasicBlock(nn.Module):
    # ResNet-18，ResNet-34 使用，输入输出的通道数不变
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # 第一个卷积模块
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        # 第二个卷积模块，ReLU 在相加之后进行操作
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        # 是否需要下采样，不需要时直接连接
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # ResNet-50，ResNet-101，ResNet-152 使用，输出通道数是输入通道数的 4 倍
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1,
                 base_width=64):
        super(Bottleneck, self).__init__()

        # 计算中间过程通道数
        width = int(out_channel * (base_width / 64.)) * groups

        # 论文中第一个 1*1 卷积核步长为2，而官方代码中第一个 1*1 卷积核步长为 1，第二个 3*3 卷积核步长为 2
        # 可在 top-1 上提高 0.5% 准确率
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)

        # 最后一层输出通道数是 中间通道数的 expansion 倍
        self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel * self.expansion, groups=groups,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channel = 64
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, channels, block_num, stride=1):
        """
        构建 Block 层结构
        :param block: 选择网络结构，ResNet-18/34 选择 BasicBlock，ResNet-50/101/152 选择 Bottleneck
        :param channels: 输入网络通道数
        :param block_num: 每个 block 循环次数
        :param stride: 步长，判断是否需要下采样
        :return: nn.Sequential() 网络结构
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        # 如果步长不为 1，即需要下采样；或者 输入 X 通道数 和 输出 F(X) 通道数 不一致时，使用 1 * 1 卷积进行修正
        if stride != 1 or self.in_channel != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(channels * block.expansion),
            )

        # 搭建 Block 结构
        layers = []
        # 第一个卷积层
        layers.append(block(self.in_channel, channels, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        # 叠加第二个 Block 开始，输入通道数为前一个 Block 的输出通道数，即原始通道数的 expansion 倍
        self.in_channel = channels * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channels, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

