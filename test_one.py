from train import *
import ssim
import glob
import cv2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='input image or folder of images')
    parser.add_argument('--ckpt', type=str, help='trained model')
    parser.add_argument('--use_cuda', action='store_true', help='activate gpu command')
    parser.add_argument('--save_dir', type=str, help='folder to save test result')
    parser.add_argument('--use_tb', action='store_true', help='use tensorboard or not')
    return parser.parse_args()

def center_crop(img):
    w, h = img.size
    t = min(w, h)
    return img.crop(((w-t)//2, (h-t)//2, t+(w-t)//2, t+(h-t)//2)).resize((256, 256))

if __name__ == '__main__':
    args = parse_args()
    if args.use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.use_tb:
        writer = SummaryWriter(args.save_dir)
    model = net.InPainting(args.use_cuda).to(device)
    assert os.path.isfile(args.ckpt)
    ckpt = torch.load(args.ckpt)
    try:
        model.load_state_dict(ckpt['ckpt'])
    except RuntimeError:
        print(traceback.format_exc())
    model.eval()
    
    if os.path.isfile(args.data):
        files = [args.data]
    else:
        files = glob.glob(os.path.join(args.data, '*'))

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    for i, file in enumerate(files):
        img = Image.open(file)
        img = center_crop(img)

        img = util.normalize(to_tensor(img))
        with torch.no_grad():
            mask, img_recon = model(img.unsqueeze(0))
        img_recon = util.denormalize(img_recon.squeeze(0))
        
        try:
            if args.use_tb:
                writer.add_iamges('test', [img_recon,
                                            util.denormalize(img), torch.cat(3*[mask[0].float().to(device)])],
                                            global_step=i, dataformats='CHW')
            else:
                file_parts = file.rsplit('.', 1)
                to_pil(img_recon).save(file_parts[0] + '_recon.' + file_parts[1])
                to_pil(mask[0]).save(file_parts[0]+'_mask.'+file_parts[1])
                '''
                cv2.imwrite(file_parts[0] + '_recon.' + file_parts[1], 
                            (img_recon.detach().numpy()*255).astype(np.uint8))
                cv2.imwrite(file_parts[0]+'_mask.'+file_parts[1],
                            (mask[0].detach().numpy()*255).astype(np.uint8))
                '''
        except Exception:
            import pdb; pdb.set_trace()