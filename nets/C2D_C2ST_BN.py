import torch
import torch.nn as nn
# from .utils import load_state_dict_from_url
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

from nets.block_zoo import C as EF1
from nets.block_zoo import ST as EF2

import pdb
__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)


# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_ef=False, cdiv=2, num_segments=8):
        super(Bottleneck, self).__init__()
        self.use_ef = use_ef
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if self.use_ef:
            # print('=> Using VB with cdiv: {}'.format(cdiv))
            self.eft1 = EF1(num_segments, use_BN=True)
            self.eft2 = EF2(num_segments, use_BN=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # x = [bcz*n_seg, c, h, w]
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_ef:
            out = self.eft1(out)
            out = self.eft2(out)
        #
        
        #
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, cdiv=2, num_segments=8):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], cdiv=cdiv, n_seg=num_segments)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, cdiv=cdiv, n_seg=num_segments)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, cdiv=cdiv, n_seg=num_segments)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, cdiv=cdiv, n_seg=num_segments)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for name, m in self.named_modules():
            if 'eft' not in name:
                # if 'deconv' in name:
                #     nn.init.xavier_normal_(m.weight)
                # else:
                #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, cdiv=2, n_seg=8):
        print('=> Processing stage with {} blocks'.format(blocks))
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, True, cdiv=cdiv, num_segments=n_seg))
        self.inplanes = planes * block.expansion
        #
        n_round = 1
        if blocks >= 23:
            n_round = 2
            print('=> Using n_round {} to insert Element Filter -T'.format(n_round))
        #
        for i in range(1, blocks):
            if i % n_round == 0:
                use_ef = True
            else:
                use_ef = False
            layers.append(block(self.inplanes, planes, use_ef=use_ef, cdiv=cdiv, num_segments=n_seg))

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
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # EF_name = getattr(EF_zoo, EF)
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        checkpoint = model_zoo.load_url(model_urls['resnet50'])
        # checkpoint_keys = list(checkpoint.keys())
        model_dict = model.state_dict()
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)

    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # EF_name = getattr(EF_zoo, EF)
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        checkpoint = model_zoo.load_url(model_urls['resnet101'])
        # checkpoint_keys = list(checkpoint.keys())
        model_dict = model.state_dict()
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model


if __name__ == "__main__":
    inputs = torch.rand(8, 3, 224, 224)  # [btz, channel, T, H, W]
    # inputs = torch.rand(1, 64, 4, 112, 112) #[btz, channel, T, H, W]
    net = resnet50(False, num_classes=174, num_segments=8)
    net.eval()
    output = net(inputs)
    print(output.size())
    from thop import profile

    flops, params = profile(net, inputs=(inputs,), custom_ops={net: net})
    print(flops)
    print(params)
    # output = net(inputs)
    # print output.size()
