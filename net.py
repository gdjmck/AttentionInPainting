import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import shm
import unet
from torch import autograd
# 从GAN InPainting引入in painting模型
import sys
sys.path.append('/home/chengk/chk-root/demos/AttentionInPainting/sub/generative_inpainting')
from sub.generative_inpainting.model.networks import CoarseGenerator, gen_conv, FineGenerator, DisConvModule, Conv2dBlock

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

class Generator(CoarseGenerator):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(Generator, self).__init__(input_dim, cnum, use_cuda, device_ids)
        self.conv1 = gen_conv(input_dim+1, cnum, 5, 1, 2)
        self.conv_d1_merge = gen_conv(cnum*6, cnum*4, 1, 1)
        self.conv17 = Conv2dBlock(cnum//2, input_dim, 1, 1, conv_padding=0, activation='none', norm='none', use_bias=False)


    def forward(self, x, mask):
        ones = torch.ones(x.size(0), 1, x.size(2), x.size(3))
        if self.use_cuda:
            mask = mask.cuda()
            ones = ones.cuda()
        # 4 x 256 x 256
        x = ones * mask + x * (1-mask)
        #print('x:', x.size())
        x = self.conv1(torch.cat([x, mask], dim=1))
        # cnum x 256 x 256
        #print('conv1:', x.size())
        x_d1 = self.conv2_downsample(x)
        #print('x_d1:', x_d1.size())
        # cnum*2 x 128 x 128
        x = self.conv3(x_d1)
        #print('conv3:', x.size())
        x = self.conv4_downsample(x)
        #print('conv4_down:', x.size())
        # cnum*4 x 64 x 64
        x = self.conv5(x)
        #print('conv5:', x.size())
        x = self.conv6(x)
        #print('conv6:', x.size())
        x = self.conv7_atrous(x)
        #print('conv7_atrous:', x.size())
        x = self.conv8_atrous(x)
        #print('conv8:', x.size())
        x = self.conv9_atrous(x)
        #print('conv9:', x.size())
        x = self.conv10_atrous(x)
        #print('conv10:', x.size())
        x = self.conv11(x)
        #print('conv11:', x.size())
        x = self.conv12(x)
        #print('conv12:', x.size())
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        # concat 96 & 192 channels
        x = self.conv_d1_merge(torch.cat([x_d1, x], dim=1))
        #print('conv_d1_merge:', x.size())
        # cnum*2 x 128 x 128
        x = self.conv13(x)
        #print('conv13:', x.size())
        x = self.conv14(x)
        #print('conv14:', x.size())
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        # cnum x 256 x 256
        x = self.conv15(x)
        #print('conv15:', x.size())
        x = self.conv16(x)
        #print('conv16:', x.size())
        x = self.conv17(x)
        #print('conv17:', x.size())
        # 3 x 256 x 256
        # x_stage1 = torch.clamp(x, -1., 1.)
        #import pdb; pdb.set_trace()

        return x

class FineGenerator(FineGenerator):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(FineGenerator, self).__init__(input_dim, cnum, use_cuda, device_ids)
        self.conv1 = gen_conv(input_dim + 1, cnum, 5, 1, 2)
        self.pmconv1 = gen_conv(input_dim + 1, cnum, 5, 1, 2)
        self.allconv11 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.allconv17 = Conv2dBlock(cnum//2, input_dim, 3, 1, 1, activation='none', norm='none')

    def forward(self, xin, x_stage1, mask):
        # mask = mask.detach()
        x1_inpaint = x_stage1 * mask + xin * (1-mask)
        x_in = torch.cat([x1_inpaint, mask], dim=1)
        x = self.conv1(x_in)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_hallu = x
        # attention branch
        x = self.pmconv1(x_in)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        '''
        print('before attention:', x.size())
        x, offset_flow = self.contextul_attention(x, x, mask)
        print('self-attention:', x.size())
        '''
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        #import pdb; pdb.set_trace()
        #x = torch.cat([x_hallu, pm], dim=1)
        x = x_hallu + pm
        # merge two branches
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv13(x)
        x = self.allconv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        #x_stage2 = torch.clamp(x, -1., 1.)
        
        return x


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