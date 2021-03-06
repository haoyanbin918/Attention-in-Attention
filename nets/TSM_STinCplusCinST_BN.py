import torch
import torch.nn as nn
# from .utils import load_state_dict_from_url
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

from nets.block_zoo import STinC as EF1
from nets.block_zoo import CinST as EF2

__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    # 'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    # 'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    # 'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    # 'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, n_segment=8, fold_div=8, place=None,
                 use_ef=False, cdiv=16):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.n_segment = n_segment
        self.fold_div = fold_div
        self.place = place
        self.use_ef = use_ef

        if self.use_ef:
            # print('=> Using VB with cdiv: {}'.format(cdiv))
            self.eft1 = EF1(n_segment, use_BN=True)
            self.eft2 = EF2(n_segment, use_BN=True)

        if place in ['block', 'blockres']:
            print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        residual = x

        if self.place == 'blockres':
            out = self.shift(x, self.n_segment, fold_div=self.fold_div)
            out = self.conv1(out)
        else:
            out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_ef:
            out = self.eft1(out) * 0.5 + self.eft2(out) * 0.5

        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        #
        if self.place == 'block':
            out = self.shift(out, self.n_segment, fold_div=self.fold_div)

        return out

    #
    @staticmethod
    def shift(x, n_segment, fold_div=8, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)


class ResNet(nn.Module):

    def __init__(self, block, layers, n_segments, num_classes=1000, fold_div=8, place='blockres', cdiv=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #
        self.layer1 = self._make_layer(block, 64, layers[0], n_seg=n_segments[0], fold_d=fold_div, pla=place,
                                       cdiv=cdiv)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, n_seg=n_segments[1], fold_d=fold_div,
                                       pla=place, cdiv=cdiv)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, n_seg=n_segments[2], fold_d=fold_div,
                                       pla=place, cdiv=cdiv)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, n_seg=n_segments[3], fold_d=fold_div,
                                       pla=place, cdiv=cdiv)
        #
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #
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

    def _make_layer(self, block, planes, blocks, stride=1, n_seg=8, fold_d=8, pla=None, cdiv=2):
        print('=> Processing stage with {} blocks'.format(blocks))
        #
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, n_seg, fold_d, pla, True, cdiv))
        self.inplanes = planes * block.expansion
        #
        n_round = 1
        if blocks >= 23:
            n_round = 2
            print('=> Using n_round {} to insert Element Filter -T'.format(n_round))
            print('=> Using n_round {} to insert temporal shift'.format(n_round))
        #
        for i in range(1, blocks):
            if i % n_round == 0:
                layers.append(block(self.inplanes, planes, n_segment=n_seg, fold_div=fold_d, place=pla, use_ef=True,
                                    cdiv=cdiv))
            else:
                layers.append(block(self.inplanes, planes, use_ef=False))

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


# def _resnet(arch, block, layers, pretrained, progress, **kwargs):
#     model = ResNet(block, layers, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model


def resnet50(pretrained=False, num_segments=8, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # EF_name = getattr(EF_zoo, EF)
    n_segment_list = [num_segments] * 4
    print('=> construct a TSM-VB based on Resnet-50 model')
    print('=> n_segment per stage: {}'.format(n_segment_list))
    #
    model = ResNet(Bottleneck, [3, 4, 6, 3], n_segment_list, **kwargs)
    if pretrained:
        checkpoint = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model


def resnet101(pretrained=False, num_segments=8, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # EF_name = getattr(EF_zoo, EF)
    n_segment_list = [num_segments] * 4
    print('=> construct a TSM-VB based on Resnet-101 model')
    print('=> n_segment per stage: {}'.format(n_segment_list))
    #
    model = ResNet(Bottleneck, [3, 4, 23, 3], n_segment_list, **kwargs)
    if pretrained:
        checkpoint = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model


# def resnet152(pretrained=False, n_segment=8, **kwargs):
#     """Constructs a ResNet-152 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     n_segment_list = [n_segment] * 4
#     print('=> construct a TSM-EFT based on Resnet-152 model')
#     print('=> n_segment per stage: {}'.format(n_segment_list))
#     #
#     model = ResNet(Bottleneck, [3, 8, 36, 3], n_segment_list, **kwargs)
#     if pretrained:
#         checkpoint = model_zoo.load_url(model_urls['resnet152'])
#         model_dict = model.state_dict()
#         model_dict.update(checkpoint)
#         model.load_state_dict(model_dict)
#     return model

if __name__ == "__main__":
    inputs = torch.rand(8, 3, 224, 224)  # [btz, channel, T, H, W]
    # inputs = torch.rand(1, 64, 4, 112, 112) #[btz, channel, T, H, W]
    net = resnet50(False, num_segments=8, fold_div=8, place='blockres', cdiv=16)
    net.eval()

    from thop import profile

    flops, params = profile(net, inputs=(inputs,), custom_ops={net: net})
    print(flops)
    print(params)
