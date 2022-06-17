import sys

import torch
import torch.nn as nn
import pdb

class AvgMaxPool(nn.Module):
    def forward(self, x, d):
        return torch.cat((torch.max(x, d)[0].unsqueeze(d), torch.mean(x, d).unsqueeze(d)), dim=d)


class THWaC4(nn.Module):
    def __init__(self, num_segments=8, use_BN=True):
        super(THWaC4, self).__init__()
        self.num_segments = num_segments
        self.use_BN = use_BN

        self.am_pool = AvgMaxPool()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.C_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.C_bn = nn.BatchNorm3d(1)

        self.T_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.T_bn = nn.BatchNorm3d(1)

        self.H_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.H_bn = nn.BatchNorm3d(1)

        self.W_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.W_bn = nn.BatchNorm3d(1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.C_conv.weight)
        nn.init.kaiming_normal_(self.T_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.H_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.W_conv.weight, mode='fan_out', nonlinearity='relu')

        print("Use THWaC ing ===")

        if self.use_BN:
            nn.init.constant_(self.C_bn.weight, 1)
            nn.init.constant_(self.C_bn.bias, 0)
            nn.init.constant_(self.T_bn.weight, 1)
            nn.init.constant_(self.T_bn.bias, 0)
            nn.init.constant_(self.H_bn.weight, 1)
            nn.init.constant_(self.H_bn.bias, 0)
            nn.init.constant_(self.W_bn.weight, 1)
            nn.init.constant_(self.W_bn.bias, 0)

    def forward(self, x):
        bn, c, h, w = x.size()
        b = bn // self.num_segments
        t = self.num_segments
        x = x.view(b, self.num_segments, c, h, w)
        #
        # T
        yt = self.am_pool(x, d=1)  # b 2 c h w
        yt = self.T_conv(yt)
        if self.use_BN:
            yt = self.T_bn(yt)
        #        yt = self.relu(yt)  # b t=1 c h w
        yt = self.sigmoid(yt)
        yt = yt.mean(dim=2).unsqueeze(2)  # b t=1 c=1 h w  Avgpool
        yt = yt.permute(0, 2, 1, 3, 4).contiguous()  # b c=1 t=1 h w
        yt = yt.expand(b, 1, t, h, w)  # b c=1 t h w

        # H
        yh = self.am_pool(x, d=3)  # b t c 2 w
        yh = yh.permute(0, 3, 1, 2, 4).contiguous()  # b h=2 t c w
        yh = self.H_conv(yh)
        if self.use_BN:
            yh = self.H_bn(yh)
        #        yh = self.relu(yh)
        yh = self.sigmoid(yh)
        yh = yh.mean(dim=3).unsqueeze(3)  # b h=1 t c=1 w
        yh = yh.permute(0, 3, 2, 1, 4).contiguous()  # b c=1 t h=1 w
        yh = yh.expand(b, 1, t, h, w)  # b c=1 t h w

        # W
        yw = self.am_pool(x, d=4)  # b t c h 2
        yw = yw.permute(0, 4, 1, 3, 2).contiguous()  # b w=2 t h c
        yw = self.W_conv(yw)
        if self.use_BN:
            yw = self.W_bn(yw)
        #        yw = self.relu(yw)
        yw = self.sigmoid(yw)
        yw = yw.mean(dim=4).unsqueeze(4)  # b w=1 t h c=1
        yw = yw.permute(0, 4, 2, 3, 1).contiguous()  # b c=1 t h w
        yw = yw.expand(b, 1, t, h, w)  # b c=1 t h w

        # C
        yc = x.permute(0, 2, 1, 3, 4).contiguous()
        #        tmp = yc
        #        yc = yc + 1 / 3 * (yt + yh + yw)
        yc = yc * 1 / 3 * (yt + yh + yw)
        yc1 = self.am_pool(yc, d=1)
        # yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b c=2 t h w
        # yc = torch.cat((yc, yt, yh, yw), dim=1)  # b c=5 t h w
        yc1 = self.C_conv(yc1)  # b 1 t h w
        if self.use_BN:
            yc1 = self.C_bn(yc1)
        yc1 = self.sigmoid(yc1)  # b 1 t h w
        yc1 = yc1.permute(0, 2, 1, 3, 4).contiguous()  # b t c=1 h w
        yc = yc.permute(0, 2, 1, 3, 4).contiguous()

        #        x = x * yc.expand_as(x)  # b t c h w
        x = yc * yc1.expand_as(x)

        return x.view(bn, c, h, w)


class STinC(nn.Module):
    def __init__(self, num_segments=8, use_BN=True):
        super(STinC, self).__init__()
        self.num_segments = num_segments
        self.use_BN = use_BN

        self.am_pool = AvgMaxPool()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.C_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.C_bn = nn.BatchNorm3d(1)

        self.T_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.T_bn = nn.BatchNorm3d(1)

        self.H_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.H_bn = nn.BatchNorm3d(1)

        self.W_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.W_bn = nn.BatchNorm3d(1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.C_conv.weight)
        nn.init.kaiming_normal_(self.T_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.H_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.W_conv.weight, mode='fan_out', nonlinearity='relu')

        print("Use THWaC ing ===")

        if self.use_BN:
            nn.init.constant_(self.C_bn.weight, 1)
            nn.init.constant_(self.C_bn.bias, 0)
            nn.init.constant_(self.T_bn.weight, 1)
            nn.init.constant_(self.T_bn.bias, 0)
            nn.init.constant_(self.H_bn.weight, 1)
            nn.init.constant_(self.H_bn.bias, 0)
            nn.init.constant_(self.W_bn.weight, 1)
            nn.init.constant_(self.W_bn.bias, 0)

    def forward(self, x):
        bn, c, h, w = x.size()
        b = bn // self.num_segments
        t = self.num_segments
        x = x.view(b, self.num_segments, c, h, w)
        #
        # T
        yt = self.am_pool(x, d=1)  # b 2 c h w
        yt = self.T_conv(yt)
        if self.use_BN:
            yt = self.T_bn(yt)
        #        yt = self.relu(yt)  # b t=1 c h w
        yt = self.sigmoid(yt)
        yt = yt.mean(dim=2).unsqueeze(2)  # b t=1 c=1 h w  Avgpool
        yt = yt.permute(0, 2, 1, 3, 4).contiguous()  # b c=1 t=1 h w
        yt = yt.expand(b, 1, t, h, w)  # b c=1 t h w

        # H
        yh = self.am_pool(x, d=3)  # b t c 2 w
        yh = yh.permute(0, 3, 1, 2, 4).contiguous()  # b h=2 t c w
        yh = self.H_conv(yh)
        if self.use_BN:
            yh = self.H_bn(yh)
        #        yh = self.relu(yh)
        yh = self.sigmoid(yh)
        yh = yh.mean(dim=3).unsqueeze(3)  # b h=1 t c=1 w
        yh = yh.permute(0, 3, 2, 1, 4).contiguous()  # b c=1 t h=1 w
        yh = yh.expand(b, 1, t, h, w)  # b c=1 t h w

        # W
        yw = self.am_pool(x, d=4)  # b t c h 2
        yw = yw.permute(0, 4, 1, 3, 2).contiguous()  # b w=2 t h c
        yw = self.W_conv(yw)
        if self.use_BN:
            yw = self.W_bn(yw)
        #        yw = self.relu(yw)
        yw = self.sigmoid(yw)
        yw = yw.mean(dim=4).unsqueeze(4)  # b w=1 t h c=1
        yw = yw.permute(0, 4, 2, 3, 1).contiguous()  # b c=1 t h w
        yw = yw.expand(b, 1, t, h, w)  # b c=1 t h w

        # C
        yc = x.permute(0, 2, 1, 3, 4).contiguous()
        #        yc = yc + 1 / 3 * (yt + yh + yw)
        yc = yc * 1 / 3 * (yt + yh + yw)

        yc = self.am_pool(yc, d=1)
        # yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b c=2 t h w
        # yc = torch.cat((yc, yt, yh, yw), dim=1)  # b c=5 t h w
        yc = self.C_conv(yc)  # b 1 t h w
        if self.use_BN:
            yc = self.C_bn(yc)
        yc = self.sigmoid(yc)  # b 1 t h w
        yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b t c=1 h w

        x = x * yc.expand_as(x)  # b t c h w

        return x.view(bn, c, h, w)

class CinST_weight(nn.Module):
    def __init__(self, num_segments=8, use_BN=True):
        super(CinST_weight, self).__init__()
        self.num_segments = num_segments
        self.use_BN = use_BN
        # self.inplanes = inplanes
        # self.planes = planes

        self.am_pool = AvgMaxPool()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.C_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.C_bn = nn.BatchNorm3d(1)

        self.T_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.T_bn = nn.BatchNorm3d(1)

        self.H_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.H_bn = nn.BatchNorm3d(1)

        self.W_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.W_bn = nn.BatchNorm3d(1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.C_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.T_conv.weight)
        nn.init.xavier_normal_(self.H_conv.weight)
        nn.init.xavier_normal_(self.H_conv.weight)
        print("Use CaTHW ing ===")

        if self.use_BN:
            nn.init.constant_(self.C_bn.weight, 1)
            nn.init.constant_(self.C_bn.bias, 0)
            nn.init.constant_(self.T_bn.weight, 1)
            nn.init.constant_(self.T_bn.bias, 0)
            nn.init.constant_(self.H_bn.weight, 1)
            nn.init.constant_(self.H_bn.bias, 0)
            nn.init.constant_(self.W_bn.weight, 1)
            nn.init.constant_(self.W_bn.bias, 0)

    def forward(self, x):
        bn, c, h, w = x.size()
        b = bn // self.num_segments
        t = self.num_segments
        x = x.view(b, self.num_segments, c, h, w)

        # C
        yc = self.am_pool(x, d=2)
        yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b c t h w
        yc = self.C_conv(yc)  # b 1 t h w
        if self.use_BN:
            yc = self.C_bn(yc)
        # yc = self.relu(yc)  # b 1 t h w
        yc = self.sigmoid(yc)
        yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b t c h w
        yc = yc.expand(b, t, c, h, w)  # b t c h w (repeat)

        # T
        # tmp = x + yc
        yt = self.am_pool(x * yc, d=1)  # b 2 c h w
        yt = self.T_conv(yt)
        if self.use_BN:
            yt = self.T_bn(yt)
        yt = self.sigmoid(yt)  # b t=1 c h w
        # yt = yt.permute(0, 2, 1, 3, 4).contiguous() # b t=1 c h w

        # H
        yh = self.am_pool(x * yc, d=3)  # b t c 2 w
        yh = yh.permute(0, 3, 1, 2, 4).contiguous()  # b h=2 t c w
        yh = self.H_conv(yh)
        if self.use_BN:
            yh = self.H_bn(yh)
        yh = self.sigmoid(yh)
        yh = yh.permute(0, 2, 3, 1, 4).contiguous()  # b t c h=1 w

        # W
        yw = self.am_pool(x * yc, d=4)  # b t c h 2
        yw = yw.permute(0, 4, 1, 3, 2).contiguous()  # b w=2 t h c
        yw = self.W_conv(yw)
        if self.use_BN:
            yw = self.W_bn(yw)
        yw = self.sigmoid(yw)
        yw = yw.permute(0, 2, 4, 3, 1).contiguous()  # b t c h w=1]\
        # print(yt.size(), yh.size(), yw.size())

        # x = 1 / 3 * x * (yt.expand_as(x) + yh.expand_as(x) + yw.expand_as(x))

        # return x.view(bn, c, h, w)
        # pdb.set_trace()


        return 1 / 3  * (yt + yh + yw).view(bn, c, h, w)



class STinC_weight(nn.Module):
    def __init__(self, num_segments=8, use_BN=True):
        super(STinC_weight, self).__init__()
        self.num_segments = num_segments
        self.use_BN = use_BN

        self.am_pool = AvgMaxPool()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.C_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.C_bn = nn.BatchNorm3d(1)

        self.T_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.T_bn = nn.BatchNorm3d(1)

        self.H_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.H_bn = nn.BatchNorm3d(1)

        self.W_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.W_bn = nn.BatchNorm3d(1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.C_conv.weight)
        nn.init.kaiming_normal_(self.T_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.H_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.W_conv.weight, mode='fan_out', nonlinearity='relu')

        print("Use THWaC ing ===")

        if self.use_BN:
            nn.init.constant_(self.C_bn.weight, 1)
            nn.init.constant_(self.C_bn.bias, 0)
            nn.init.constant_(self.T_bn.weight, 1)
            nn.init.constant_(self.T_bn.bias, 0)
            nn.init.constant_(self.H_bn.weight, 1)
            nn.init.constant_(self.H_bn.bias, 0)
            nn.init.constant_(self.W_bn.weight, 1)
            nn.init.constant_(self.W_bn.bias, 0)

    def forward(self, x):
        bn, c, h, w = x.size()
        b = bn // self.num_segments
        t = self.num_segments
        x = x.view(b, self.num_segments, c, h, w)
        #
        # T
        yt = self.am_pool(x, d=1)  # b 2 c h w
        yt = self.T_conv(yt)
        if self.use_BN:
            yt = self.T_bn(yt)
        #        yt = self.relu(yt)  # b t=1 c h w
        yt = self.sigmoid(yt)
        yt = yt.mean(dim=2).unsqueeze(2)  # b t=1 c=1 h w  Avgpool
        yt = yt.permute(0, 2, 1, 3, 4).contiguous()  # b c=1 t=1 h w
        yt = yt.expand(b, 1, t, h, w)  # b c=1 t h w

        # H
        yh = self.am_pool(x, d=3)  # b t c 2 w
        yh = yh.permute(0, 3, 1, 2, 4).contiguous()  # b h=2 t c w
        yh = self.H_conv(yh)
        if self.use_BN:
            yh = self.H_bn(yh)
        #        yh = self.relu(yh)
        yh = self.sigmoid(yh)
        yh = yh.mean(dim=3).unsqueeze(3)  # b h=1 t c=1 w
        yh = yh.permute(0, 3, 2, 1, 4).contiguous()  # b c=1 t h=1 w
        yh = yh.expand(b, 1, t, h, w)  # b c=1 t h w

        # W
        yw = self.am_pool(x, d=4)  # b t c h 2
        yw = yw.permute(0, 4, 1, 3, 2).contiguous()  # b w=2 t h c
        yw = self.W_conv(yw)
        if self.use_BN:
            yw = self.W_bn(yw)
        #        yw = self.relu(yw)
        yw = self.sigmoid(yw)
        yw = yw.mean(dim=4).unsqueeze(4)  # b w=1 t h c=1
        yw = yw.permute(0, 4, 2, 3, 1).contiguous()  # b c=1 t h w
        yw = yw.expand(b, 1, t, h, w)  # b c=1 t h w

        # C
        yc = x.permute(0, 2, 1, 3, 4).contiguous()
        #        yc = yc + 1 / 3 * (yt + yh + yw)
        yc = yc * 1 / 3 * (yt + yh + yw)

        # pdb.set_trace()
        yc = self.am_pool(yc, d=1)
        # yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b c=2 t h w
        # yc = torch.cat((yc, yt, yh, yw), dim=1)  # b c=5 t h w
        yc = self.C_conv(yc)  # b 1 t h w
        if self.use_BN:
            yc = self.C_bn(yc)
        yc = self.sigmoid(yc)  # b 1 t h w
        yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b t c=1 h w

        # x = x * yc.expand_as(x)  # b t c h w

        # return x.view(bn, c, h, w)

        return yc.expand_as(x).view(bn, c, h, w)

class CinST(nn.Module):
    def __init__(self, num_segments=8, use_BN=True):
        super(CinST, self).__init__()
        self.num_segments = num_segments
        self.use_BN = use_BN
        # self.inplanes = inplanes
        # self.planes = planes

        self.am_pool = AvgMaxPool()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.C_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.C_bn = nn.BatchNorm3d(1)

        self.T_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.T_bn = nn.BatchNorm3d(1)

        self.H_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.H_bn = nn.BatchNorm3d(1)

        self.W_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.W_bn = nn.BatchNorm3d(1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.C_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.T_conv.weight)
        nn.init.xavier_normal_(self.H_conv.weight)
        nn.init.xavier_normal_(self.H_conv.weight)
        print("Use CaTHW ing ===")

        if self.use_BN:
            nn.init.constant_(self.C_bn.weight, 1)
            nn.init.constant_(self.C_bn.bias, 0)
            nn.init.constant_(self.T_bn.weight, 1)
            nn.init.constant_(self.T_bn.bias, 0)
            nn.init.constant_(self.H_bn.weight, 1)
            nn.init.constant_(self.H_bn.bias, 0)
            nn.init.constant_(self.W_bn.weight, 1)
            nn.init.constant_(self.W_bn.bias, 0)

    def forward(self, x):
        bn, c, h, w = x.size()
        b = bn // self.num_segments
        t = self.num_segments
        x = x.view(b, self.num_segments, c, h, w)

        # C
        yc = self.am_pool(x, d=2)
        yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b c t h w
        yc = self.C_conv(yc)  # b 1 t h w
        if self.use_BN:
            yc = self.C_bn(yc)
        # yc = self.relu(yc)  # b 1 t h w
        yc = self.sigmoid(yc)
        yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b t c h w
        yc = yc.expand(b, t, c, h, w)  # b t c h w (repeat)

        # T
        # tmp = x + yc
        yt = self.am_pool(x * yc, d=1)  # b 2 c h w
        yt = self.T_conv(yt)
        if self.use_BN:
            yt = self.T_bn(yt)
        yt = self.sigmoid(yt)  # b t=1 c h w
        # yt = yt.permute(0, 2, 1, 3, 4).contiguous() # b t=1 c h w

        # H
        yh = self.am_pool(x * yc, d=3)  # b t c 2 w
        yh = yh.permute(0, 3, 1, 2, 4).contiguous()  # b h=2 t c w
        yh = self.H_conv(yh)
        if self.use_BN:
            yh = self.H_bn(yh)
        yh = self.sigmoid(yh)
        yh = yh.permute(0, 2, 3, 1, 4).contiguous()  # b t c h=1 w

        # W
        yw = self.am_pool(x * yc, d=4)  # b t c h 2
        yw = yw.permute(0, 4, 1, 3, 2).contiguous()  # b w=2 t h c
        yw = self.W_conv(yw)
        if self.use_BN:
            yw = self.W_bn(yw)
        yw = self.sigmoid(yw)
        yw = yw.permute(0, 2, 4, 3, 1).contiguous()  # b t c h w=1]\
        # print(yt.size(), yh.size(), yw.size())

        x = 1 / 3 * x * (yt.expand_as(x) + yh.expand_as(x) + yw.expand_as(x))

        return x.view(bn, c, h, w)

# CTHW

class CaTHW(nn.Module):
    def __init__(self, num_segments=8, use_BN=True):
        super(CaTHW, self).__init__()
        self.num_segments = num_segments
        self.use_BN = use_BN
        # self.inplanes = inplanes
        # self.planes = planes

        self.am_pool = AvgMaxPool()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.C_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.C_bn = nn.BatchNorm3d(1)

        self.T_conv = nn.Conv3d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.T_bn = nn.BatchNorm3d(1)

        self.H_conv = nn.Conv3d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.H_bn = nn.BatchNorm3d(1)

        self.W_conv = nn.Conv3d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.W_bn = nn.BatchNorm3d(1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.C_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.T_conv.weight)
        nn.init.xavier_normal_(self.H_conv.weight)
        nn.init.xavier_normal_(self.H_conv.weight)

        if self.use_BN:
            nn.init.constant_(self.C_bn.weight, 1)
            nn.init.constant_(self.C_bn.bias, 0)
            nn.init.constant_(self.T_bn.weight, 1)
            nn.init.constant_(self.T_bn.bias, 0)
            nn.init.constant_(self.H_bn.weight, 1)
            nn.init.constant_(self.H_bn.bias, 0)
            nn.init.constant_(self.W_bn.weight, 1)
            nn.init.constant_(self.W_bn.bias, 0)

    def forward(self, x):
        bn, c, h, w = x.size()
        b = bn // self.num_segments
        t = self.num_segments
        x = x.view(b, self.num_segments, c, h, w)

        # C
        yc = self.am_pool(x, d=2)
        yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b c t h w
        yc = self.C_conv(yc)  # b 1 t h w
        if self.use_BN:
            yc = self.C_bn(yc)
        yc = self.relu(yc)  # b 1 t h w

        # T
        yct = yc.mean(dim=2).unsqueeze(2)  # b c=1 t=1 h w
        yct = yct.permute(0, 2, 1, 3, 4).contiguous()  # b t=1 c=1 h w
        yct = yct.expand(b, 1, c, h, w)
        yt = self.am_pool(x, d=1)  # b 2 c h w
        yt = torch.cat((yt, yct), dim=1)  # b t=3 c h w
        yt = self.T_conv(yt)
        if self.use_BN:
            yt = self.T_bn(yt)
        yt = self.sigmoid(yt)  # b t=1 c h w
        # yt = yt.permute(0, 2, 1, 3, 4).contiguous() # b t=1 c h w
        # H
        ych = yc.mean(dim=3).unsqueeze(3)  # b c=1 t h=1 w
        ych = ych.permute(0, 3, 2, 1, 4).contiguous()  # b h=1 t c=1 w
        ych = ych.expand(b, 1, t, c, w)  # b h=1 t c w
        yh = self.am_pool(x, d=3)  # b t c 2 w
        yh = yh.permute(0, 3, 1, 2, 4).contiguous()  # b h=2 t c w
        yh = torch.cat((yh, ych), dim=1)  # b h=3 t c w
        yh = self.H_conv(yh)
        if self.use_BN:
            yh = self.H_bn(yh)
        yh = self.sigmoid(yh)
        yh = yh.permute(0, 2, 3, 1, 4).contiguous()  # b t c h=1 w
        # W
        ycw = yc.mean(dim=4).unsqueeze(4)  # b c=1 t h w=1
        ycw = ycw.permute(0, 4, 2, 3, 1).contiguous()  # b w=1 t h c=1
        ycw = ycw.expand(b, 1, t, h, c)  # b w=1 t h c
        yw = self.am_pool(x, d=4)  # b t c h 2
        yw = yw.permute(0, 4, 1, 3, 2).contiguous()  # b w=2 t h c
        yw = torch.cat((yw, ycw), dim=1)  # b w=3 t h c
        yw = self.W_conv(yw)
        if self.use_BN:
            yw = self.W_bn(yw)
        yw = self.sigmoid(yw)
        yw = yw.permute(0, 2, 4, 3, 1).contiguous()  # b t c h w=1

        x = 1 / 3 * x * (yt.expand_as(x) + yh.expand_as(x) + yw.expand_as(x))

        return x.view(bn, c, h, w)


# videoBlock
# 1 t h w
class THWaC(nn.Module):
    def __init__(self, num_segments=8, use_BN=True):
        super(THWaC, self).__init__()
        self.num_segments = num_segments
        self.use_BN = use_BN

        self.am_pool = AvgMaxPool()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.C_conv = nn.Conv3d(5, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.C_bn = nn.BatchNorm3d(1)

        self.T_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.T_bn = nn.BatchNorm3d(1)

        self.H_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.H_bn = nn.BatchNorm3d(1)

        self.W_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.W_bn = nn.BatchNorm3d(1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.C_conv.weight)
        nn.init.kaiming_normal_(self.T_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.H_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.W_conv.weight, mode='fan_out', nonlinearity='relu')

        print("Use THWaC ing ===")

        if self.use_BN:
            nn.init.constant_(self.C_bn.weight, 1)
            nn.init.constant_(self.C_bn.bias, 0)
            nn.init.constant_(self.T_bn.weight, 1)
            nn.init.constant_(self.T_bn.bias, 0)
            nn.init.constant_(self.H_bn.weight, 1)
            nn.init.constant_(self.H_bn.bias, 0)
            nn.init.constant_(self.W_bn.weight, 1)
            nn.init.constant_(self.W_bn.bias, 0)

    def forward(self, x):
        bn, c, h, w = x.size()
        b = bn // self.num_segments
        t = self.num_segments
        x = x.view(b, self.num_segments, c, h, w)
        #
        # T
        yt = self.am_pool(x, d=1)  # b 2 c h w
        yt = self.T_conv(yt)
        if self.use_BN:
            yt = self.T_bn(yt)
        yt = self.relu(yt)  # b t=1 c h w
        yt = yt.mean(dim=2).unsqueeze(2)  # b t=1 c=1 h w  Avgpool
        yt = yt.permute(0, 2, 1, 3, 4).contiguous()  # b c=1 t=1 h w
        yt = yt.expand(b, 1, t, h, w)  # b c=1 t h w

        # H
        yh = self.am_pool(x, d=3)  # b t c 2 w
        yh = yh.permute(0, 3, 1, 2, 4).contiguous()  # b h=2 t c w
        yh = self.H_conv(yh)
        if self.use_BN:
            yh = self.H_bn(yh)
        yh = self.relu(yh)
        yh = yh.mean(dim=3).unsqueeze(3)  # b h=1 t c=1 w
        yh = yh.permute(0, 3, 2, 1, 4).contiguous()  # b c=1 t h=1 w
        yh = yh.expand(b, 1, t, h, w)  # b c=1 t h w

        # W
        yw = self.am_pool(x, d=4)  # b t c h 2
        yw = yw.permute(0, 4, 1, 3, 2).contiguous()  # b w=2 t h c
        yw = self.W_conv(yw)
        if self.use_BN:
            yw = self.W_bn(yw)
        yw = self.relu(yw)
        yw = yw.mean(dim=4).unsqueeze(4)  # b w=1 t h c=1
        yw = yw.permute(0, 4, 2, 3, 1).contiguous()  # b c=1 t h w
        yw = yw.expand(b, 1, t, h, w)  # b c=1 t h w

        # C

        yc = self.am_pool(x, d=2)
        yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b c=2 t h w
        yc = torch.cat((yc, yt, yh, yw), dim=1)  # b c=5 t h w
        yc = self.C_conv(yc)  # b 1 t h w
        if self.use_BN:
            yc = self.C_bn(yc)
        yc = self.sigmoid(yc)  # b 1 t h w
        yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b t c=1 h w

        x = x * yc.expand_as(x)  # b t c h w

        return x.view(bn, c, h, w)


class THWaC2(nn.Module):
    def __init__(self, num_segments=8, use_BN=True):
        super(THWaC2, self).__init__()
        self.num_segments = num_segments
        self.use_BN = use_BN

        self.am_pool = AvgMaxPool()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.C_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.C_bn = nn.BatchNorm3d(1)

        self.T_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.T_bn = nn.BatchNorm3d(1)

        self.H_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.H_bn = nn.BatchNorm3d(1)

        self.W_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.W_bn = nn.BatchNorm3d(1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.C_conv.weight)
        nn.init.kaiming_normal_(self.T_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.H_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.W_conv.weight, mode='fan_out', nonlinearity='relu')

        print("Use THWaC ing ===")

        if self.use_BN:
            nn.init.constant_(self.C_bn.weight, 1)
            nn.init.constant_(self.C_bn.bias, 0)
            nn.init.constant_(self.T_bn.weight, 1)
            nn.init.constant_(self.T_bn.bias, 0)
            nn.init.constant_(self.H_bn.weight, 1)
            nn.init.constant_(self.H_bn.bias, 0)
            nn.init.constant_(self.W_bn.weight, 1)
            nn.init.constant_(self.W_bn.bias, 0)

    def forward(self, x):
        bn, c, h, w = x.size()
        b = bn // self.num_segments
        t = self.num_segments
        x = x.view(b, self.num_segments, c, h, w)
        #
        # T
        yt = self.am_pool(x, d=1)  # b 2 c h w
        yt = self.T_conv(yt)
        if self.use_BN:
            yt = self.T_bn(yt)
        yt = self.relu(yt)  # b t=1 c h w
        yt = yt.mean(dim=2).unsqueeze(2)  # b t=1 c=1 h w  Avgpool
        yt = yt.permute(0, 2, 1, 3, 4).contiguous()  # b c=1 t=1 h w
        yt = yt.expand(b, 1, t, h, w)  # b c=1 t h w

        # H
        yh = self.am_pool(x, d=3)  # b t c 2 w
        yh = yh.permute(0, 3, 1, 2, 4).contiguous()  # b h=2 t c w
        yh = self.H_conv(yh)
        if self.use_BN:
            yh = self.H_bn(yh)
        yh = self.relu(yh)
        yh = yh.mean(dim=3).unsqueeze(3)  # b h=1 t c=1 w
        yh = yh.permute(0, 3, 2, 1, 4).contiguous()  # b c=1 t h=1 w
        yh = yh.expand(b, 1, t, h, w)  # b c=1 t h w

        # W
        yw = self.am_pool(x, d=4)  # b t c h 2
        yw = yw.permute(0, 4, 1, 3, 2).contiguous()  # b w=2 t h c
        yw = self.W_conv(yw)
        if self.use_BN:
            yw = self.W_bn(yw)
        yw = self.relu(yw)
        yw = yw.mean(dim=4).unsqueeze(4)  # b w=1 t h c=1
        yw = yw.permute(0, 4, 2, 3, 1).contiguous()  # b c=1 t h w
        yw = yw.expand(b, 1, t, h, w)  # b c=1 t h w

        # C
        yc = x.permute(0, 2, 1, 3, 4).contiguous()
        yc = yc + 1 / 3 * (yt + yh + yw)
        yc = self.am_pool(yc, d=1)
        # yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b c=2 t h w
        # yc = torch.cat((yc, yt, yh, yw), dim=1)  # b c=5 t h w
        yc = self.C_conv(yc)  # b 1 t h w
        if self.use_BN:
            yc = self.C_bn(yc)
        yc = self.sigmoid(yc)  # b 1 t h w
        yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b t c=1 h w

        x = x * yc.expand_as(x)  # b t c h w

        return x.view(bn, c, h, w)


# 带有参差结构
class CaTHW2(nn.Module):
    def __init__(self, num_segments=8, use_BN=True):
        super(CaTHW2, self).__init__()
        self.num_segments = num_segments
        self.use_BN = use_BN
        # self.inplanes = inplanes
        # self.planes = planes

        self.am_pool = AvgMaxPool()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.C_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.C_bn = nn.BatchNorm3d(1)

        self.T_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.T_bn = nn.BatchNorm3d(1)

        self.H_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.H_bn = nn.BatchNorm3d(1)

        self.W_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.W_bn = nn.BatchNorm3d(1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.C_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.T_conv.weight)
        nn.init.xavier_normal_(self.H_conv.weight)
        nn.init.xavier_normal_(self.H_conv.weight)
        print("Use CaTHW2 ing ===")

        if self.use_BN:
            nn.init.constant_(self.C_bn.weight, 1)
            nn.init.constant_(self.C_bn.bias, 0)
            nn.init.constant_(self.T_bn.weight, 1)
            nn.init.constant_(self.T_bn.bias, 0)
            nn.init.constant_(self.H_bn.weight, 1)
            nn.init.constant_(self.H_bn.bias, 0)
            nn.init.constant_(self.W_bn.weight, 1)
            nn.init.constant_(self.W_bn.bias, 0)

    def forward(self, x):
        bn, c, h, w = x.size()
        b = bn // self.num_segments
        t = self.num_segments
        x = x.view(b, self.num_segments, c, h, w)

        # C
        yc = self.am_pool(x, d=2)
        yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b c t h w
        yc = self.C_conv(yc)  # b 1 t h w
        if self.use_BN:
            yc = self.C_bn(yc)
        yc = self.relu(yc)  # b 1 t h w
        yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b t c h w
        yc = yc.expand(b, t, c, h, w)  # b t c h w (repeat)

        # T
        # tmp = x + yc
        yt = self.am_pool(x + yc, d=1)  # b 2 c h w
        yt = self.T_conv(yt)
        if self.use_BN:
            yt = self.T_bn(yt)
        yt = self.sigmoid(yt)  # b t=1 c h w
        # yt = yt.permute(0, 2, 1, 3, 4).contiguous() # b t=1 c h w

        # H
        yh = self.am_pool(x + yc, d=3)  # b t c 2 w
        yh = yh.permute(0, 3, 1, 2, 4).contiguous()  # b h=2 t c w
        yh = self.H_conv(yh)
        if self.use_BN:
            yh = self.H_bn(yh)
        yh = self.sigmoid(yh)
        yh = yh.permute(0, 2, 3, 1, 4).contiguous()  # b t c h=1 w

        # W
        yw = self.am_pool(x + yc, d=4)  # b t c h 2
        yw = yw.permute(0, 4, 1, 3, 2).contiguous()  # b w=2 t h c
        yw = self.W_conv(yw)
        if self.use_BN:
            yw = self.W_bn(yw)
        yw = self.sigmoid(yw)
        yw = yw.permute(0, 2, 4, 3, 1).contiguous()  # b t c h w=1]\
        # print(yt.size(), yh.size(), yw.size())

        x = 1 / 3 * x * (yt.expand_as(x) + yh.expand_as(x) + yw.expand_as(x))

        return x.view(bn, c, h, w)


# 时空占比均为1/2
class CaTHW4(nn.Module):
    def __init__(self, num_segments=8, use_BN=True):
        super(CaTHW4, self).__init__()
        self.num_segments = num_segments
        self.use_BN = use_BN
        # self.inplanes = inplanes
        # self.planes = planes

        self.am_pool = AvgMaxPool()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.C_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.C_bn = nn.BatchNorm3d(1)

        self.T_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.T_bn = nn.BatchNorm3d(1)

        self.H_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.H_bn = nn.BatchNorm3d(1)

        self.W_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.W_bn = nn.BatchNorm3d(1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.C_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.T_conv.weight)
        nn.init.xavier_normal_(self.H_conv.weight)
        nn.init.xavier_normal_(self.H_conv.weight)
        # if 'CaTHW' in sys._getframe().f_code.co_name:
        print("Use CaTHW4 ing ===")

        if self.use_BN:
            nn.init.constant_(self.C_bn.weight, 1)
            nn.init.constant_(self.C_bn.bias, 0)
            nn.init.constant_(self.T_bn.weight, 1)
            nn.init.constant_(self.T_bn.bias, 0)
            nn.init.constant_(self.H_bn.weight, 1)
            nn.init.constant_(self.H_bn.bias, 0)
            nn.init.constant_(self.W_bn.weight, 1)
            nn.init.constant_(self.W_bn.bias, 0)

    def forward(self, x):
        bn, c, h, w = x.size()
        b = bn // self.num_segments
        t = self.num_segments
        x = x.view(b, self.num_segments, c, h, w)

        # C
        yc = self.am_pool(x, d=2)
        yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b c t h w
        yc = self.C_conv(yc)  # b 1 t h w
        if self.use_BN:
            yc = self.C_bn(yc)
        yc = self.relu(yc)  # b 1 t h w
        yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b t c h w
        yc = yc.expand(b, t, c, h, w)  # b t c h w (repeat)

        # T
        # tmp = x + yc
        yt = self.am_pool(x + yc, d=1)  # b 2 c h w
        yt = self.T_conv(yt)
        if self.use_BN:
            yt = self.T_bn(yt)
        yt = self.sigmoid(yt)  # b t=1 c h w
        # yt = yt.permute(0, 2, 1, 3, 4).contiguous() # b t=1 c h w

        # H
        yh = self.am_pool(x + yc, d=3)  # b t c 2 w
        yh = yh.permute(0, 3, 1, 2, 4).contiguous()  # b h=2 t c w
        yh = self.H_conv(yh)
        if self.use_BN:
            yh = self.H_bn(yh)
        yh = self.sigmoid(yh)
        yh = yh.permute(0, 2, 3, 1, 4).contiguous()  # b t c h=1 w

        # W
        yw = self.am_pool(x + yc, d=4)  # b t c h 2
        yw = yw.permute(0, 4, 1, 3, 2).contiguous()  # b w=2 t h c
        yw = self.W_conv(yw)
        if self.use_BN:
            yw = self.W_bn(yw)
        yw = self.sigmoid(yw)
        yw = yw.permute(0, 2, 4, 3, 1).contiguous()  # b t c h w=1]\
        # print(yt.size(), yh.size(), yw.size())

        x = 1 / 2 * x * (yt.expand_as(x) + 1 / 2 * (yh.expand_as(x) + yw.expand_as(x)))

        return x.view(bn, c, h, w)


# 底层使用双池化，高层使用单池化，5为单池化
class CaTHW5(nn.Module):
    def __init__(self, num_segments=8, use_BN=True):
        super(CaTHW5, self).__init__()
        self.num_segments = num_segments
        self.use_BN = use_BN
        # self.inplanes = inplanes
        # self.planes = planes

        self.am_pool = AvgMaxPool()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.C_conv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.C_bn = nn.BatchNorm3d(1)

        self.T_conv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.T_bn = nn.BatchNorm3d(1)

        self.H_conv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.H_bn = nn.BatchNorm3d(1)

        self.W_conv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.W_bn = nn.BatchNorm3d(1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.C_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.T_conv.weight)
        nn.init.xavier_normal_(self.H_conv.weight)
        nn.init.xavier_normal_(self.H_conv.weight)
        print("Use CaTHW5 ing ===")

        if self.use_BN:
            nn.init.constant_(self.C_bn.weight, 1)
            nn.init.constant_(self.C_bn.bias, 0)
            nn.init.constant_(self.T_bn.weight, 1)
            nn.init.constant_(self.T_bn.bias, 0)
            nn.init.constant_(self.H_bn.weight, 1)
            nn.init.constant_(self.H_bn.bias, 0)
            nn.init.constant_(self.W_bn.weight, 1)
            nn.init.constant_(self.W_bn.bias, 0)

    def forward(self, x):
        bn, c, h, w = x.size()
        b = bn // self.num_segments
        t = self.num_segments
        x = x.view(b, self.num_segments, c, h, w)

        # C
        # yc = self.am_pool(x, d=2)
        yc = x.mean(dim=2).unsqueeze(2)
        yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b c t h w
        yc = self.C_conv(yc)  # b 1 t h w
        if self.use_BN:
            yc = self.C_bn(yc)
        yc = self.relu(yc)  # b 1 t h w
        yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b t c h w
        yc = yc.expand(b, t, c, h, w)  # b t c h w (repeat)

        # T
        # tmp = x + yc
        yt = (yc+x).mean(dim=1).unsqueeze(1)
        # yt = self.am_pool(x + yc, d=1)  # b 2 c h w
        yt = self.T_conv(yt)
        if self.use_BN:
            yt = self.T_bn(yt)
        yt = self.sigmoid(yt)  # b t=1 c h w
        # yt = yt.permute(0, 2, 1, 3, 4).contiguous() # b t=1 c h w

        # H
        yh = (x + yc).mean(dim=3).unsqueeze(3)
        # yh = self.am_pool(x + yc, d=3)  # b t c 2 w
        yh = yh.permute(0, 3, 1, 2, 4).contiguous()  # b h=2 t c w
        yh = self.H_conv(yh)
        if self.use_BN:
            yh = self.H_bn(yh)
        yh = self.sigmoid(yh)
        yh = yh.permute(0, 2, 3, 1, 4).contiguous()  # b t c h=1 w

        # W
        yw = (yc+x).mean(dim=4).unsqueeze(4)
        # yw = self.am_pool(x + yc, d=4)  # b t c h 2
        yw = yw.permute(0, 4, 1, 3, 2).contiguous()  # b w=2 t h c
        yw = self.W_conv(yw)
        if self.use_BN:
            yw = self.W_bn(yw)
        yw = self.sigmoid(yw)
        yw = yw.permute(0, 2, 4, 3, 1).contiguous()  # b t c h w=1]\
        # print(yt.size(), yh.size(), yw.size())

        x = 1 / 3 * x * (yt.expand_as(x) + yh.expand_as(x) + yw.expand_as(x))

        return x.view(bn, c, h, w)


class TaCHW(nn.Module):
    def __init__(self, num_segments=8, use_BN=True):
        super(TaCHW, self).__init__()
        self.num_segments = num_segments
        self.use_BN = use_BN

        self.am_pool = AvgMaxPool()

        self.C_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.C_bn = nn.BatchNorm3d(1)

        self.T_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.T_bn = nn.BatchNorm3d(1)

        self.H_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.H_bn = nn.BatchNorm3d(1)

        self.W_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.W_bn = nn.BatchNorm3d(1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.C_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.T_conv.weight)
        nn.init.xavier_normal_(self.H_conv.weight)
        nn.init.xavier_normal_(self.H_conv.weight)
        print("Use TaCHW ing ===")

        if self.use_BN:
            nn.init.constant_(self.C_bn.weight, 1)
            nn.init.constant_(self.C_bn.bias, 0)
            nn.init.constant_(self.T_bn.weight, 1)
            nn.init.constant_(self.T_bn.bias, 0)
            nn.init.constant_(self.H_bn.weight, 1)
            nn.init.constant_(self.H_bn.bias, 0)
            nn.init.constant_(self.W_bn.weight, 1)
            nn.init.constant_(self.W_bn.bias, 0)

    def forward(self, x):
        bn, c, h, w = x.size()
        b = bn // self.num_segments
        t = self.num_segments
        x = x.view(b, self.num_segments, c, h, w)  # b t c h w

        # yt
        yt = self.am_pool(x, d=1)
        yt = self.T_conv(yt)  # b 1 c h w
        if self.use_BN:
            yc = self.T_bn(yt)
        yt = self.relu(yt)  # b 1 c h w
        yt = yt.expand(b, t, c, h, w)  # b t c h w (repeat)

        yc = self.am_pool(x + yt, d=2)  # b t c h w
        yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b c t h w
        yc = self.C_conv(yc)  # b 1 t h w
        if self.use_BN:
            yc = self.C_bn(yc)
        yc = self.relu(yc)  # b 1 t h w
        yc = yc.permute(0, 2, 1, 3, 4).contiguous()  # b t c h w
        # yc = yc.expand(b, t, c, h, w)  # b t c h w (repeat)

        yh = self.am_pool(x + yt, d=3)  # b t c 2 w
        yh = yh.permute(0, 3, 1, 2, 4).contiguous()  # b h=2 t c w
        yh = self.H_conv(yh)
        if self.use_BN:
            yh = self.H_bn(yh)
        yh = self.sigmoid(yh)
        yh = yh.permute(0, 2, 3, 1, 4).contiguous()  # b t c h=1 w

        yw = self.am_pool(x + yt, d=4)  # b t c h 2
        yw = yw.permute(0, 4, 1, 2, 3).contiguous()  # b w=2 c t h
        yw = self.W_conv(yw)
        if self.use_BN:
            yw = self.W_bn(yw)
        yw = self.sigmoid(yw)
        yw = yw.permute(0, 2, 3, 4, 1).contiguous()  # b t c h w=1
        # print(yt.size(), yh.size(), yw.size())

        x = 1 / 3 * x * (yc.expand_as(x) + yh.expand_as(x) + yw.expand_as(x))

        return x.view(bn, c, h, w)


# 1 t h w
class VB_STR(nn.Module):
    def __init__(self, inplanes, planes, use_BN, num_segments=8):
        super(VB_STR, self).__init__()
        self.num_segments = num_segments
        self.inplanes = inplanes
        self.planes = planes
        self.use_BN = use_BN

        self.pool = AvgMaxPool()

        self.conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        bn, c, h, w = x.size()
        batch_size = bn // self.num_segments
        x = x.view(batch_size, self.num_segments, c, h, w)

        y = x.permute(0, 2, 1, 3, 4).contiguous()  # b c t h w
        y = self.pool(y, 1)  # b 1 t h w

        y = self.conv(y)
        if self.use_BN:
            y = self.bn(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1, 3, 4).contiguous()

        x = x * y.expand_as(x)
        x = x.view(bn, c, h, w)

        return x


# c 1 h w
class VB_SR(nn.Module):
    def __init__(self, inplanes, planes, use_BN, num_segments=8):
        super(VB_SR, self).__init__()
        self.num_segments = num_segments
        self.inplanes = inplanes
        self.planes = planes
        self.use_BN = use_BN

        self.pool = AvgMaxPool()

        self.conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        bn, c, h, w = x.size()
        batch_size = bn // self.num_segments
        x = x.view(batch_size, self.num_segments, c, h, w)  # b t c h w

        y = self.pool(x, 1)

        y = self.conv(y)
        if self.use_BN:
            y = self.bn(y)
        y = self.sigmoid(y)

        x = x * y.expand_as(x)
        x = x.view(bn, c, h, w)

        return x


class VB_TRH(nn.Module):
    def __init__(self, inplanes, planes, use_BN, num_segments=8):
        super(VB_TRH, self).__init__()
        self.num_segments = num_segments
        self.inplanes = inplanes
        self.planes = planes
        self.use_BN = use_BN

        self.pool = AvgMaxPool()

        self.conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        bn, c, h, w = x.size()
        batch_size = bn // self.num_segments
        x = x.view(batch_size, self.num_segments, c, h, w)  # b t c h w

        y = x.permute(0, 3, 1, 2, 4).contiguous()  # b h t c w
        y = self.pool(y, 1)

        y = self.conv(y)
        if self.use_BN:
            y = self.bn(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 3, 1, 4).contiguous()

        x = x * y.expand_as(x)
        x = x.view(bn, c, h, w)

        return x


class VB_TRW(nn.Module):
    def __init__(self, inplanes, planes, use_BN, num_segments=8):
        super(VB_TRW, self).__init__()
        self.num_segments = num_segments
        self.inplanes = inplanes
        self.planes = planes
        self.use_BN = use_BN

        self.pool = AvgMaxPool()

        self.conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        bn, c, h, w = x.size()
        batch_size = bn // self.num_segments
        x = x.view(batch_size, self.num_segments, c, h, w)

        y = x.permute(0, 4, 1, 2, 3).contiguous()
        y = self.pool(y, 1)

        y = self.conv(y)
        if self.use_BN:
            y = self.bn(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 3, 4, 1).contiguous()

        x = x * y.expand_as(x)
        x = x.view(bn, c, h, w)

        return x


class VB(nn.Module):
    def __init__(self, inplanes, planes, use_BN, num_segments=8):
        super(VB, self).__init__()
        self.num_segments = num_segments
        self.inplanes = inplanes
        self.planes = planes
        self.use_BN = use_BN

        self.STR = VB_STR(self.inplanes, self.planes, self.use_BN, self.num_segments)
        self.SR = VB_SR(self.inplanes, self.planes, self.use_BN, self.num_segments)
        self.TRH = VB_TRH(self.inplanes, self.planes, self.use_BN, self.num_segments)
        self.TRW = VB_TRW(self.inplanes, self.planes, self.use_BN, self.num_segments)
        print("Use VB ing ===")

    def forward(self, x):
        y1 = self.STR(x)
        y2 = self.SR(x)
        y3 = self.TRH(x)
        y4 = self.TRW(x)
        x = 1 / 4 * (y1 + y2 + y3 + y4)
        # x = x + y
        return x


if __name__ == "__main__":
    inputs = torch.rand(32, 128, 56, 56)  # [batch*segment, channel, H, W]
    # inputs = torch.rand(16, 256, 4, 112, 112).cuda()  # [btz, channel, T, H, W]
    net = CaTHW5(8, True)
    # net = VB(128, 128, num_segments=8, use_BN=True).cuda()
    net.eval()
    # net = EF_CLLD(256, 256, 4, 8)
    output = net(inputs)
    print(output.size())
