import torch
import torch.nn as nn
# 从GAN InPainting引入in painting模型
import sys
sys.path.reverse()
sys.path.append('/home/chengk/chk-root/demos/AttentionInPainting/generative-inpainting-pytorch')
sys.path.reverse()
from model.networks import CoarseGenerator

class ConvBN(nn.Module):
    def __init__(self, in_chan, out_chan, kernel=3, stride=1, padding=0):
        super(ConvBN, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_chan, out_chan, kernel, stride, padding),
                                        nn.BatchNorm2d(out_chan),
                                        nn.ReLU())
        
    def forward(self, x):
        return self.layer(x)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.origin_scale = ConvBN(3, 16, 3, 1, 1)
        self.fusion_ori = nn.Sequential(ConvBN(32, 16, 1), nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid())
        self.downscale1 = ConvBN(16, 32, 3, 2, 1)
        self.fusion1 = nn.Sequential(ConvBN(64, 32, 1), ConvBN(32, 16, 1))
        self.downscale2 = ConvBN(32, 64, 3, 2, 1)
        self.fusion2 = nn.Sequential(ConvBN(128, 64, 1), ConvBN(64, 32, 1))
        self.downscale3 = nn.Sequential(ConvBN(64, 128, 3, 2, 1), ConvBN(128, 64, 1))
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self, x):
        orig_scale = self.origin_scale(x)
        down_scale1 = self.downscale1(orig_scale)
        down_scale2 = self.downscale2(down_scale1)
        down_scale3 = self.downscale3(down_scale2)
        up_scale2 = self.fusion2(torch.cat([down_scale2, self.upscale(down_scale3)], 1))
        up_scale1 = self.fusion1(torch.cat([down_scale1, self.upscale(up_scale2)], 1))
        orig_scale = self.fusion_ori(torch.cat([orig_scale, self.upscale(up_scale1)], 1))

        return orig_scale


class InPainting(nn.Module):
    def __init__(self):
        super(InPainting, self).__init__()
        self.encoder = AutoEncoder()
        self.painter = CoarseGenerator(3, 32, False)

    def forward(self, x):
        mask = self.encoder(x)
        '''
        blank = torch.ones_like(x)
        inpaint_input = mask * blank + (1-mask) * x
        inpaint_input = torch.cat([inpaint_input, mask])
        '''
        inpaint = self.painter(x, mask)
        return inpaint


if __name__ == '__main__':
    model = InPainting()
    x = torch.randn((1, 3, 256, 256))
    out = model(x)
    print(out.size())