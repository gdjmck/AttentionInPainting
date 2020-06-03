import net, util
import torch
import dataset
import argparse
import sys
import os

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./images/', help='training data folder')
    parser.add_argument('--batchsize', '-b', type=int, default=8, help='batchsize')
    parser.add_argument('--epochs', type=int, default=200, help='epochs to train')
    parser.add_argument('--resume', type=str, default='', help='checkpoint file')
    parser.add_argument('--save_dir', type=str, default='./ckpt', help='folder to save checkpoint file')
    return parser.parse_args()

def main():
    args = parse_arg()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = net.InPainting()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = torch.nn.L1Loss()

    data = dataset.Dataset(args.path)
    train_loader = torch.utils.data.DataLoader(dataset=data,
                                           batch_size=args.batchsize,
                                           shuffle=True,
                                           num_workers=1)

    try:
        for i in range(args.epochs):
            print('Epoch %d'%i)
            for j, item in enumerate(train_loader):
                img_raw, img_wm = item
                recon = model(img_wm)
                loss_ = loss(recon, img_raw)
                optimizer.zero_grad()
                loss_.backward()
                optimizer.step()
                if j % 10 == 0:
                    print(loss_.item())
    except:
        ckpt = {'ckpt': model.state_dict(),
                'optim': optimizer.state_dict()}
        torch.save(ckpt, os.path.join(args.save_dir, 'latest.pth'))
        print('Save temporary checkpoints to %s'% args.save_dir)
        sys.exit(0)
    
    print('Done training.')
    ckpt = {'ckpt': model.state_dict(),
        'optim': optimizer.state_dict()}
    torch.save(ckpt, os.path.join(args.save_dir, 'epoch_%d.pth'%i))

if __name__ == '__main__':
    main()