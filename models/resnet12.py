import torch.nn as nn

from .models import register
from models.MPNCOV import CovpoolLayer, SqrtmLayer, TriuvecLayer


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)


class Block(nn.Module):

    def __init__(self, inplanes, planes, downsample, scale=True):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)

        self.downsample = downsample

        self.maxpool = nn.MaxPool2d(2)
        self.scale = scale

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.scale:
            out = self.maxpool(out)
        else:
            out = out

        return out


class ResNet12(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.inplanes = 3

        self.layer1 = self._make_layer(channels[0], scale=True)
        self.layer2 = self._make_layer(channels[1], scale=True)
        self.layer3 = self._make_layer(channels[2], scale=False)
        self.layer4 = self._make_layer(channels[3], scale=False)

        # self.out_dim = channels[3]

        self.layer_reduce_1 = nn.Conv2d(channels[3], 256, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.layer_reduce_bn_1 = nn.BatchNorm2d(256)
        self.layer_reduce_relu_1 = nn.ReLU(inplace=True)

        self.layer_reduce_2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.layer_reduce_bn_2 = nn.BatchNorm2d(128)
        self.layer_reduce_relu_2 = nn.ReLU(inplace=True)


        self.out_dim = int(128*(128+1)/2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, scale):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes),
        )
        block = Block(self.inplanes, planes, downsample, scale)
        self.inplanes = planes
        return block

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        x = self.layer_reduce_1(x)
        x = self.layer_reduce_bn_1(x)
        x = self.layer_reduce_relu_1(x)

        x = self.layer_reduce_2(x)
        x = self.layer_reduce_bn_2(x)
        x = self.layer_reduce_relu_2(x)

        x = CovpoolLayer(x)
        x = SqrtmLayer(x, 5)
        x = TriuvecLayer(x)
        x = x.view(x.size(0), -1)
        return x


@register('resnet12')
def resnet12():
    return ResNet12([64, 128, 256, 512])


@register('resnet12-wide')
def resnet12_wide():
    return ResNet12([64, 160, 320, 640])

