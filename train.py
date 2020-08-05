import net, util
import torch
import dataset
import argparse
import sys
import os
import shutil
import traceback
import pdb
import ssim
from tensorboardX import SummaryWriter

eps = 1e-6

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./images/', help='training data folder')
    parser.add_argument('--path_origin', type=str, default='original_images/train', 
                        help='Large-scale watermark dataset original images training set')
    parser.add_argument('--path_wm', type=str, default='watermarked_images/train', 
                        help='Large-scale watermark dataset watermarked images training set')
    parser.add_argument('--path_anno', type=str, default='watermarked_images/train_imageID.txt')
    parser.add_argument('--batchsize', '-b', type=int, default=8, help='batchsize')
    parser.add_argument('--epochs', type=int, default=200, help='epochs to train')
    parser.add_argument('--resume', type=str, default='', help='checkpoint file')
    parser.add_argument('--save_dir', type=str, default='./ckpt', help='folder to save checkpoint file')
    parser.add_argument('--use_cuda', action='store_true', help='activate GPU for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--fix_mask', action='store_true', help='freeze mask generator')
    parser.add_argument('--fix_recon', action='store_true', help='freeze reconstruction model')
    parser.add_argument('--gan_method', action='store_true', help='use gan based method')
    return parser.parse_args()

def dis_forward(discriminator, real, fake):
    batch_size_real = real.size(0)
    batch = torch.cat([real, fake], dim=0)
    batch_out = discriminator(batch)
    batch_out = batch_out.clamp(0, 1)
    pred_real, pred_fake = torch.split(batch_out, batch_size_real, dim=0)
    return pred_real, pred_fake

def main():
    args = parse_arg()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    writer = SummaryWriter(os.path.join(args.save_dir, 'tb'))
    if args.use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = net.InPainting(args.use_cuda, mask_clipping=False).to(device)
    model_dis = net.Discriminator().to(device)
    model.train()
    model_dis.train()
    #train_params = [p for (n, p) in model.named_parameters() if 'fine_painter' not in n]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_dis = torch.optim.Adam(model_dis.parameters(), lr=args.lr)
    if args.resume:
        print('Resume training from ', args.resume)
        ckpt = torch.load(args.resume)
        try:
            model.load_state_dict(ckpt['ckpt'])
            model.load_state_dict(ckpt['ckpt_dis'])
            optimizer.load_state_dict(ckpt['optim'])
            optimizer_dis.load_state_dict(ckpt['optim_dis'])
        except Exception as e:
            print(traceback.format_exc())
            print('Missing keys')
            model.load_state_dict({k: v for k, v in ckpt['ckpt'].items() if 'encoder' in k}, strict=False)
    if args.fix_mask:
        print('fix mask prediction')
        for p in model.encoder.parameters():
            p.requires_grad = False
    elif args.fix_recon:
        print('fix painter')
        for p in model.painter.parameters():
            p.requires_grad = False

    loss = torch.nn.L1Loss(reduction='none')
    loss_bce = torch.nn.BCELoss(reduction='none')
    loss_l2 = torch.nn.MSELoss()
    ssim_window = ssim.create_window(11, 3).to(device)

    #data = dataset.Dataset(args.path)
    data = dataset.LargeScaleWatermarkDataset(folder_origin=os.path.join(args.path, args.path_origin),
                                                folder_watermarked=os.path.join(args.path, args.path_wm),
                                                anno_file=os.path.join(args.path, args.path_anno))
    train_loader = torch.utils.data.DataLoader(dataset=data,
                                           batch_size=args.batchsize,
                                           shuffle=True,
                                           num_workers=4)

    try:
        batch_per_epoch = len(train_loader)
        best_loss = 100
        for i in range(args.epochs):
            epoch_loss = 0
            print('Epoch %d'%i)
            #item = next(iter(train_loader))
            for j, item in enumerate(train_loader):
                step = i*batch_per_epoch+j
                img_raw, img_wm, mask_wm = item
                img_raw, img_wm, mask_wm = img_raw.to(device), img_wm.to(device), mask_wm.to(device)
                mask, recon, recon_coarse = model(img_wm)
                # 加入discriminator
                if args.gan_method:
                    # optimize D
                    dis_real, dis_recon = dis_forward(model_dis, img_raw, recon.detach())
                    dis_wm = model_dis(img_wm)
                    assert dis_recon.size() == dis_wm.size()
                    dis_fake = dis_recon
                    #dis_fake = 0.5*(dis_recon + dis_wm)
                    loss_disc = util.hinge_loss(1-dis_real, lower_bound=0, upper_bound=None) + \
                                util.hinge_loss(dis_fake, lower_bound=0, upper_bound=None)
                    loss_gp = 0.1*net.calc_gradient_penalty(model_dis, img_raw, recon.detach())
                    #loss_d = loss_gp + loss_disc
                    loss_d = loss_disc
                    # enables discriminator gradient
                    '''
                    for p in model_dis.parameters():
                        p.requires_grad = True
                    '''
                    optimizer.zero_grad()
                    optimizer_dis.zero_grad()
                    loss_d.backward()
                    torch.nn.utils.clip_grad_value_(model_dis.parameters(), 0.5)
                    optimizer_dis.step()
                    # 画各层的梯度分布图
                    if j % 100 == 0:
                        writer.add_figure('discriminator_grad', util.plot_grad_flow_v2(model_dis.named_parameters()), global_step=step)
                        for n, p in model_dis.named_parameters():
                            writer.add_histogram('D_'+n, p.grad.data.cpu().numpy(), step)

                    # optimize G through D
                    dis_real, dis_recon = dis_forward(model_dis, img_raw, recon)
                    loss_g = util.hinge_loss(1-dis_recon, lower_bound=0, upper_bound=None)
                    # print('loss_gp:%.4f \t loss_disc:%.4f \t loss_g:%.4f'% \
                    #             (loss_gp.item(), loss_disc.item(), loss_g.item()))

                loss_mask_reg = 0.1*mask.clamp(0, 1).mean()
                # loss_mask = 1000*util.exclusion_loss(mask)
                try:
                    mask_wm = mask_wm.float().clamp(0., 1.).view(mask.size())
                    mask_wm_neg = 1 - util.torch_dilate(mask_wm)
                    loss_mask = loss(mask.clamp(0., 1.)*mask_wm, mask_wm).sum() / (mask_wm.sum()+eps) + \
                                loss(mask.clamp(0, 1)*mask_wm_neg, mask_wm*mask_wm_neg).sum() / (mask_wm_neg.sum()+eps)
                    loss_mask = 0.1 * loss_mask
                except Exception:
                    pdb.set_trace()
                    if not (mask>=0. & mask <=1.).all():
                        print('错误出在生成的mask')
                    if not (mask_wm>=0. & mask_wm<=1.).all():
                        print('错误出在gt水印')
                
                try:
                    loss_recon = loss_l2(recon, img_raw)
                    loss_recon_c = 0.5*loss_l2(recon_coarse, img_raw)
                    loss_ssim = 0.2*(1-ssim._ssim(0.5*(1+img_raw), 0.5*(1+recon.clamp(-1., 1.)), ssim_window, 11, 3, True))
                    loss_weighted_recon = util.weighted_l1(recon, img_raw, mask)
                    loss_ = loss_recon + loss_mask + loss_ssim + loss_recon_c
                    if args.gan_method:
                        loss_ += loss_g

                    optimizer_dis.zero_grad()
                    optimizer.zero_grad()
                    loss_.backward()
                    if args.gan_method:
                        # disable discriminator gradient
                        '''
                        for p in model_dis.parameters():
                            p.requires_grad = False
                        '''
                        torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
                    #import pdb; pdb.set_trace()
                    optimizer.step()
                except Exception as e:
                    print(str(e))
                    #import pdb; pdb.set_trace()
                
                epoch_loss += loss_.item()
                # scalar log
                if j % 5 == 0:
                    scalar_dict = {'recon_l1': loss_recon.item(),
                                                'ssim': loss_ssim.item(),
                                                'exclusion': loss_mask.item(), 
                                                'mask_reg': loss_mask_reg.item()}
                    if args.gan_method:
                        scalar_dict['G'] = loss_g.item()
                        scalar_dict['D'] = loss_d.item()
                    writer.add_scalars('loss', scalar_dict, step)
                if j % 10 == 0:
                    print('Loss: %.3f (recon: %.3f \t ssim: %.3f \t mask: %.3f \t)'%
                            (loss_.item(), loss_recon.item(), loss_ssim.item(), loss_mask.item()))
                    if args.gan_method:
                        print('disc: %.3f \t gp: %.3f \t gen: %.3f' % (loss_disc.item(), loss_gp.item(), loss_g.item()))
                # 记录mask和原图
                if j % 50 == 0:
                    #import pdb; pdb.set_trace()
                    writer.add_images('images', [torch.cat(3*[mask[0].float().to(device)]), 
                                                torch.cat(3*[mask_wm[0].squeeze().float().to(device).unsqueeze(0)]), util.denormalize(img_wm[0]), 
                                                util.denormalize(recon[0].clamp(-1, 1)).clamp(0, 1)], global_step=step, dataformats='CHW')
                    '''
                    writer.add_image('mask', mask[0], step)
                    writer.add_image('img', util.denormalize(img_wm[0]), step)
                    writer.add_image('recon_c', util.denormalize(recon_coarse[0]).clamp(0, 1), step)
                    writer.add_image('recon_f', util.denormalize(recon_fine[0]).clamp(0, 1), step)
                    writer.add_image('mask_gt', mask_wm[0], step)
                    '''
                # 画各层的梯度分布图
                if j % 100 == 0:
                    writer.add_figure('grad_flow', util.plot_grad_flow_v2(model.named_parameters()), global_step=step)
                    for n, p in model.named_parameters():
                        writer.add_histogram(n, p.grad.data.cpu().numpy(), step)
                    if args.gan_method:
                        for n, p in model_dis.named_parameters():
                            writer.add_histogram('G_'+n, p.grad.data.cpu().numpy(), step)

            ckpt = {'ckpt': model.state_dict(),
                'optim': optimizer.state_dict()}
            if args.gan_method:
                ckpt['ckpt_dis'] = model_dis.state_dict()
            torch.save(ckpt, os.path.join(args.save_dir, 'latest.pth'))
            # 记录所有最好的epoch weight
            if epoch_loss / (j+1) < best_loss:
                best_loss = epoch_loss / (j+1)
                shutil.copy(os.path.join(args.save_dir, 'latest.pth'), 
                            os.path.join(args.save_dir, 'epoch_'+str(i)+'.pth'))
    except Exception as e:
        ckpt = {'ckpt': model.state_dict(),
                'optim': optimizer.state_dict()}
        torch.save(ckpt, os.path.join(args.save_dir, 'latest.pth'))
        print('Save temporary checkpoints to %s'% args.save_dir)
        print(str(e), traceback.print_exc())
        sys.exit(0)
    print('Done training.')
    shutil.copyfile(os.path.join(args.save_dir, 'latest.pth'),
                    os.path.join(args.save_dir, 'epoch_%d.pth'%(i+1)))

if __name__ == '__main__':
    main()