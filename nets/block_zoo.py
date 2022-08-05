import sys
import torch
import torch.nn as nn
import pdb

class AvgMaxPool(nn.Module):
    def forward(self, x, d):
        return torch.cat((torch.max(x, d)[0].unsqueeze(d), torch.mean(x, d).unsqueeze(d)), dim=d)

class C(nn.Module):
    def __init__(self, num_segments=8, use_BN=True):
        super(C, self).__init__()
        self.num_segments = num_segments
        self.use_BN = use_BN
        # self.inplanes = inplanes
        # self.planes = planes

        self.am_pool = AvgMaxPool()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.C_conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_BN:
            self.C_bn = nn.BatchNorm3d(1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.C_conv.weight, mode='fan_out', nonlinearity='relu')
        print("Use C ing ===")

        if self.use_BN:
            nn.init.constant_(self.C_bn.weight, 1)
            nn.init.constant_(self.C_bn.bias, 0)

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
        # yc = yc.expand(b, t, c, h, w)  # b t c h w (repeat)

        x = x * yc.expand_as(x)

        return x.view(bn, c, h, w)
        
class ST(nn.Module):
    def __init__(self, num_segments=8, use_BN=True):
        super(ST, self).__init__()
        self.num_segments = num_segments
        self.use_BN = use_BN

        self.am_pool = AvgMaxPool()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)

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

        # nn.init.xavier_normal_(self.C_conv.weight)
        nn.init.kaiming_normal_(self.T_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.H_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.W_conv.weight, mode='fan_out', nonlinearity='relu')

        print("Use ST ing ===")

        if self.use_BN:
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
        # tmp = x + yc
        yt = self.am_pool(x, d=1)  # b 2 c h w
        yt = self.T_conv(yt)
        if self.use_BN:
            yt = self.T_bn(yt)
        yt = self.sigmoid(yt)  # b t=1 c h w
        # yt = yt.permute(0, 2, 1, 3, 4).contiguous() # b t=1 c h w

        # H
        yh = self.am_pool(x, d=3)  # b t c 2 w
        yh = yh.permute(0, 3, 1, 2, 4).contiguous()  # b h=2 t c w
        yh = self.H_conv(yh)
        if self.use_BN:
            yh = self.H_bn(yh)
        yh = self.sigmoid(yh)
        yh = yh.permute(0, 2, 3, 1, 4).contiguous()  # b t c h=1 w

        # W
        yw = self.am_pool(x, d=4)  # b t c h 2
        yw = yw.permute(0, 4, 1, 3, 2).contiguous()  # b w=2 t h c
        yw = self.W_conv(yw)
        if self.use_BN:
            yw = self.W_bn(yw)
        yw = self.sigmoid(yw)
        yw = yw.permute(0, 2, 4, 3, 1).contiguous()  # b t c h w=1]\
        # print(yt.size(), yh.size(), yw.size())

        # C
        # yc = x.permute(0, 2, 1, 3, 4).contiguous()
        #        yc = yc + 1 / 3 * (yt + yh + yw)
        yc = x * 1 / 3 * (yt.expand_as(x) + yh.expand_as(x) + yw.expand_as(x))

        # yc = yc.permute(0, 2, 1, 3, 4).contiguous()

        return yc.view(bn, c, h, w)


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

        print("Use STinC ing ===")

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
        # tmp = x + yc
        yt = self.am_pool(x , d=1)  # b 2 c h w
        yt = self.T_conv(yt)
        if self.use_BN:
            yt = self.T_bn(yt)
        yt = self.sigmoid(yt)  # b t=1 c h w
        # yt = yt.permute(0, 2, 1, 3, 4).contiguous() # b t=1 c h w

        # H
        yh = self.am_pool(x, d=3)  # b t c 2 w
        yh = yh.permute(0, 3, 1, 2, 4).contiguous()  # b h=2 t c w
        yh = self.H_conv(yh)
        if self.use_BN:
            yh = self.H_bn(yh)
        yh = self.sigmoid(yh)
        yh = yh.permute(0, 2, 3, 1, 4).contiguous()  # b t c h=1 w

        # W
        yw = self.am_pool(x, d=4)  # b t c h 2
        yw = yw.permute(0, 4, 1, 3, 2).contiguous()  # b w=2 t h c
        yw = self.W_conv(yw)
        if self.use_BN:
            yw = self.W_bn(yw)
        yw = self.sigmoid(yw)
        yw = yw.permute(0, 2, 4, 3, 1).contiguous()  # b t c h w=1]\
        # print(yt.size(), yh.size(), yw.size())

        # C
        
        #        yc = yc + 1 / 3 * (yt + yh + yw)
        yc = x * 1 / 3 * (yt.expand_as(x) + yh.expand_as(x) + yw.expand_as(x))
        yc = yc.permute(0, 2, 1, 3, 4).contiguous()

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


class STinC2(nn.Module):
    def __init__(self, num_segments=8, use_BN=True):
        super(STinC2, self).__init__()
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
        print("Use CinST ing ===")

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

if __name__ == "__main__":
    # inputs = torch.rand(32, 128, 56, 56)  # [batch*segment, channel, H, W]
    inputs = torch.rand(16, 256, 4, 112, 112).cuda()  # [btz, channel, T, H, W]
    net = CaTHW5(8, True)
    net.eval()
    output = net(inputs)
    print(output.size())
