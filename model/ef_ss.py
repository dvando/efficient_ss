import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def cbr_layer(in_channels, out_channels, kernel_size, groups=1, stride=1, dilation=1, activation=True, padding=None):
    """Convolution, BatchNorm, and ReLU"""
    if activation:
        if not padding:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, 
                        padding=int(kernel_size / 2), dilation=dilation,
                        groups=groups, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, 
                        padding=padding, dilation=dilation,
                        groups=groups, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=int(kernel_size / 2),
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_channels, affine=True, eps=1e-5, momentum=0.1))

class FPN(nn.Module):
    """ Feature Pyramid Network """
    def __init__(self, in_channels):
        super().__init__()
        assert len(in_channels) == 4
        for i, channel in enumerate(in_channels):
            setattr(self, f"l{i+1}", cbr_layer(channel, 256, 1, padding='same'))

    def forward(self, f1, f2, f3, f4):
        feat1 = self.l1(f1)
        feat2 = self.l2(f2) + F.interpolate(feat1, f2.size()[2:], mode='bilinear', align_corners=False)
        feat3 = self.l3(f3) + F.interpolate(feat2, f3.size()[2:], mode='bilinear', align_corners=False)
        feat4 = self.l4(f4) + F.interpolate(feat3, f4.size()[2:], mode='bilinear', align_corners=False)

        return feat1, feat2, feat3, feat4
        
class FPF(nn.Module):
    """ Feature Pyramid Fusion """
    def __init__(self):
        super().__init__()
        for i in range(4):
            setattr(self, f"p{i+1}", nn.Sequential(
                nn.Conv2d(256, 256, 3, stride=1, groups = 256, padding=1),
                nn.Conv2d(256, 256, 1, stride=1, groups = 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(inplace=True)
                )
            )

    def forward(self, f1, f2, f3, f4, f5, f6, f7, f8):
        # assert (len(feats) == 8), f"Input length should be 8, got {len(feats)}"
        p1 = self.p1(f1 + f8)
        p2 = self.p2(f2 + f7)
        p3 = self.p3(f3 + f6)
        p4 = self.p4(f4 + f5)

        return p1, p2, p3, p4

class LSFE(nn.Module):
    """ Large Scake Feature Extractor """
    def __init__(self, in_channel):
        super().__init__()
        self.mbconv1 = nn.Sequential(
            nn.Conv2d(in_channel, 256, 3, stride=1, groups = in_channel, padding='same'),
            nn.Conv2d(256, 128, 3, stride=1, groups=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.mbconv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, groups = 128, padding='same'),
            nn.Conv2d(128, 128, 3, stride=1, groups=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, p):
        p = self.mbconv1(p)
        p = self.mbconv2(p)

        return p

class DPC(nn.Module):
    """ Dense Prediction Cells """
    def __init__(self):
        super().__init__()
        config = torch.tensor([
            [1,6],
            [1,1],
            [6,21],
            [18,15],
            [6,3]
        ], dtype=torch.int8)

        for i, (rh, rw) in enumerate(config):
            setattr(self, f'dconv{i+1}',cbr_layer(256, 256, 3, dilation=(rh.item(),rw.item()), stride=1, padding='same'))
        
        self.lconv = cbr_layer(1280, 128, 1, stride=1)

    def forward(self, p):
        p1 = self.dconv1(p)
        p2 = self.dconv2(p1)
        p3 = self.dconv3(p1)
        p4 = self.dconv4(p1)
        p5 = self.dconv5(p4)

        p = torch.cat((p1, p2, p3, p4, p5), dim=1)
        p = self.lconv(p)

        return p

class MC(nn.Module):
    """Missmatch Correction"""
    def __init__(self, in_channel):
        super().__init__()
        self.mbconv1 = nn.Sequential(
            nn.Conv2d(in_channel, 256, 3, stride=1, groups = in_channel, padding='same'),
            nn.Conv2d(256, 128, 3, stride=1, groups=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.mbconv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, groups = 128, padding='same'),
            nn.Conv2d(128, 128, 3, stride=1, groups=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, target):
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        
        return F.interpolate(x, target.size()[2:], mode='bilinear', align_corners=False)


class Vixlane(nn.Module):
    """ Vixmo Lane Detector """
    def __init__(self, num_class=2):
        super().__init__()
        self.feature_extractor = timm.create_model('efficientnet_b5', features_only= True, out_indices= (0,1,2,3,4,5))
        self.bridge = cbr_layer(512, 2048, 1)
        self.fpn1 = FPN([40, 64, 176, 2048])
        self.fpn2 = FPN([2048, 176, 64, 40])
        self.fpf = FPF()
        self.lsfe1 = LSFE(256)
        self.lsfe2 = LSFE(256)
        self.dpc1 = DPC()
        self.dpc2 = DPC()
        self.mc1 = MC(128)
        self.mc2 = MC(128)
        self.out = cbr_layer(512, num_class, 1, activation=False)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        _, f2, f3, f4, f5 = self.feature_extractor(x)
        f6 = self.bridge(f5)

        feats1 = self.fpn1(f2, f3, f4, f6)
        feats2 = self.fpn2(f6, f4, f3, f2)
        p1, p2, p3, p4 = self.fpf(*(*feats1, *feats2))

        pd = self.dpc1(p4)
        pc = self.dpc2(p3)
        pb = self.lsfe1(p2)
        pa = self.lsfe2(p1)


        pdc = self.mc1(pc + F.interpolate(pd, pc.size()[2:], mode='bilinear'), pb)
        pb = pdc + pb
        pdcb = self.mc2(pb, pa)
        pa = pdcb + pa

        to_cat = [F.interpolate(p, pa.size()[2:], mode='bilinear') for p in [pd, pc, pb, pa]]

        pre_out = torch.cat(to_cat, dim=1)
        out = self.out(pre_out)
        out = F.interpolate(out, x.size()[2:], mode='bilinear', align_corners=False)
        out = self.softmax(out)

        return out