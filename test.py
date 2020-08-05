from train import *
import ssim

def main():
    args = parse_arg()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    writer = SummaryWriter(os.path.join(args.save_dir, 'tb'))

    if args.use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = net.InPainting(args.use_cuda).to(device)
    # Load model checkpoint
    assert args.resume != ''
    ckpt = torch.load(args.resume)
    try:
        model.load_state_dict(ckpt['ckpt'])
    except RuntimeError:
        print(traceback.format_exc())
        model.load_state_dict(ckpt['ckpt'], strict=False)
    model.eval()
    # Eval metric
    loss = torch.nn.L1Loss()

    data = dataset.LargeScaleWatermarkDataset(folder_origin=os.path.join(args.path, args.path_origin),
                                                folder_watermarked=os.path.join(args.path, args.path_wm),
                                                anno_file=os.path.join(args.path, args.path_anno))
    eval_loader = torch.utils.data.DataLoader(dataset=data,
                                           batch_size=args.batchsize,
                                           shuffle=False,
                                           num_workers=2)

    avg_loss = 0
    avg_ssim = 0
    ssim_window = ssim.create_window(11, 3).to(device)
    step = 0
    for i, (img_raw, img_wm, mask_wm) in enumerate(eval_loader):
        img_raw, img_wm = img_raw.to(device), img_wm.to(device)
        with torch.no_grad():
            mask, recon = model.test_inference(img_wm)
            loss_recon = loss(recon, img_raw)
            avg_loss += loss_recon.item()
            avg_ssim += ssim._ssim(img_raw, recon, ssim_window, 11, 3, True)
        
        if i % 5 == 0:
            step += 1
            writer.add_images('images', [torch.cat(3*[mask[0].float().to(device)]), 
                                         torch.cat(3*[mask_wm[0].float().to(device).unsqueeze(0)]), 
                                         util.denormalize(img_wm[0]), 
                                         util.denormalize(recon[0]).clamp(0, 1)], global_step=step, dataformats='CHW')

    print('avg loss:', avg_loss / i)
    print('avg ssim:', avg_ssim / i)

if __name__ == '__main__':
    main()