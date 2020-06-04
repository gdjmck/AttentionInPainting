import torch
import torch.nn as nn
import torch.nn.functional as F
# 从GAN InPainting引入in painting模型
import sys
sys.path.reverse()
sys.path.append('/home/chengk/chk-root/demos/AttentionInPainting/generative-inpainting-pytorch')
sys.path.reverse()
from model.networks import CoarseGenerator, gen_conv

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
        self.origin_scale = ConvBN(3, 32, 3, 1, 1)
        self.fusion_ori = nn.Sequential(ConvBN(64, 32, 1), nn.Conv2d(32, 1, 3, 1, 1), nn.Sigmoid())
        self.downscale1 = ConvBN(32, 64, 3, 2, 1)
        self.fusion1 = nn.Sequential(ConvBN(128, 64, 1), ConvBN(64, 32, 1))
        self.downscale2 = ConvBN(64, 128, 3, 2, 1)
        self.fusion2 = nn.Sequential(ConvBN(256, 128, 1), ConvBN(128, 64, 1))
        self.downscale3 = nn.Sequential(ConvBN(128, 256, 3, 2, 1), ConvBN(256, 128, 1))
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)


    def forward(self, x):
        orig_scale = self.origin_scale(x)
        down_scale1 = self.downscale1(orig_scale)
        down_scale2 = self.downscale2(down_scale1)
        down_scale3 = self.downscale3(down_scale2)
        up_scale2 = self.fusion2(torch.cat([down_scale2, self.upscale(down_scale3)], 1))
        up_scale1 = self.fusion1(torch.cat([down_scale1, self.upscale(up_scale2)], 1))
        orig_scale = self.fusion_ori(torch.cat([orig_scale, self.upscale(up_scale1)], 1))

        return orig_scale

class Generator(CoarseGenerator):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(Generator, self).__init__(input_dim, cnum, use_cuda, device_ids)
        self.conv1 = gen_conv(input_dim+1, cnum, 5, 1, 2)

    def forward(self, x, mask):
        ones = torch.ones(x.size(0), 1, x.size(2), x.size(3))
        if self.use_cuda:
            mask = mask.cuda()
            ones = ones.cuda()
        # 4 x 256 x 256
        x = ones * mask + x * (1-mask)
        x = self.conv1(torch.cat([x, mask], dim=1))
        x = self.conv2_downsample(x)
        # cnum*2 x 128 x 128
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        # cnum*4 x 64 x 64
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cnum*2 x 128 x 128
        x = self.conv13(x)
        x = self.conv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cnum x 256 x 256
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        # 3 x 256 x 256
        x_stage1 = torch.clamp(x, -1., 1.)

        return x_stage1

class InPainting(nn.Module):
    def __init__(self, cuda=False):
        super(InPainting, self).__init__()
        self.encoder = AutoEncoder()
        self.painter = Generator(3, 32, cuda)

    def forward(self, x):
        mask = self.encoder(x)
        '''
        blank = torch.ones_like(x)
        inpaint_input = mask * blank + (1-mask) * x
        inpaint_input = torch.cat([inpaint_input, mask])
        '''
        inpaint = self.painter(x, mask)
        return mask, inpaint


if __name__ == '__main__':
    model = InPainting()
    x = torch.randn((1, 3, 256, 256))
    out = model(x)
    print(out.size())