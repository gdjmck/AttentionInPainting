import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import shm
import unet
from torch import autograd

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
        self.fusion_ori = nn.Sequential(ConvBN(64, 32, 1), nn.Conv2d(32, 1, 3, 1, 1))
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
        orig_scale = torch.clamp(orig_scale, 0., 1.)
        
        return orig_scale

class Discriminator(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dis_conv_module = DisConvModule(input_dim, hidden_dim)
        self.ext_conv_module = nn.Sequential(Conv2dBlock(hidden_dim*4, hidden_dim*2, 1, 1),
                                             Conv2dBlock(hidden_dim*2, hidden_dim, 3, 2, 1),
                                             Conv2dBlock(hidden_dim, hidden_dim, 1, 1),
                                             nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1, groups=hidden_dim),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(hidden_dim, 1, 1, 1), nn.InstanceNorm2d(1))
        #self.pooling = nn.Aver((4, 4))
        self.linear = nn.Linear(16, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        # hidden_dim*4 x 16 x 16
        x = self.ext_conv_module(x)
        # 1 x 4 x 4
        #x = self.pooling(x)
        x = x.view((x.size(0), -1))
        # 1 x 1 x 1
        x = self.linear(x)
        return x

# Calculate gradient penalty
def calc_gradient_penalty(netD, real_data, fake_data):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(fake_data.device)

    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.requires_grad_().clone()

    disc_interpolates = netD(interpolates)
    grad_outputs = torch.ones(disc_interpolates.size())

    grad_outputs = grad_outputs.to(fake_data.device)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=grad_outputs, create_graph=True,
                                retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty

class InPainting(nn.Module):
    def __init__(self, cuda=False, mask_clipping=False):
        super(InPainting, self).__init__()
        self.clipping = mask_clipping
        self.encoder = unet.UNet(3, 1)
        self.painter = unet.UNet(4, 3)

    def forward(self, x):
        mask = self.encoder(x)
        if self.clipping:
            mask_mean = mask.mean().detach()
            # avoid early clipping, results in no gradient
            if mask_mean >= 0.4:
                mask_mean /= 2
            mask_clone = mask.clone()
            mask_clone[mask<mask_mean] = 0
            mask = mask_clone
        # randomly p=0.3 dilates the mask
        # mask got detach from there &
        # generates a new mask
        '''
        if np.random.random() < 0.3:
            kernel = torch.ones((1, 1, 5, 5)).to(x.device)
            mask_ = nn.functional.conv2d(mask, kernel, padding=(2, 2)).clamp(0, 1).detach()
        else:
            mask_ = mask
        '''
        ones = torch.ones((x.size(0), 1, x.size(2), x.size(3))).to(x.device)
        mask_clamp = mask.clamp(0., 1.)
        # x = ones * mask_clamp + x * (1-mask_clamp)
        # 这样要限制(1-mask)区域的跟原图输入一致
        x = torch.cat([x, mask_clamp], dim=1)
        inpaint = self.painter(x)
        return mask, inpaint

    def test_inference(self, x):
        mask = self.encoder(x)
        import pdb; pdb.set_trace()
        kernel = torch.ones((1, 1, 5, 5)).to(x.device)
        mask = torch.nn.functional.conv2d(mask, kernel, padding=(2, 2)).clamp(0, 1)
        inpaint = self.painter(x, mask)
        return mask, inpaint

if __name__ == '__main__':
    import sys
    #model = InPainting()
    model = Discriminator()
    '''
    for n, p in [(n, p) for (n, p) in model.named_parameters() if 'fine_painter' not in n]:
        print(n, p.size())
    sys.exit(0)
    for n, p in model.named_parameters():
        if p not in model.fine_painter.parameters():
            print(n, p.size())
    print('====================================================')
    for n, p in model.fine_painter.named_parameters():
        print(n, p.size())
    '''
    x = torch.randn((1, 3, 256, 256))
    out = model(x)
    print(out.size())