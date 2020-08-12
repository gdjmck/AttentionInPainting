import string
import glob
import numpy as np
import torch
import os.path as osp
from PIL import ImageDraw, ImageOps, Image, ImageFont
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager as fm, rcParams
import torchvision.transforms as T

tv_transform = T.Compose([
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

tv_inv_transform = T.Compose([T.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225])],
                             [T.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])])

font_resources = glob.glob(os.path.join(rcParams['datapath'], 'fonts/ttf/*.ttf'))
prints = list(string.printable)[:62]
def random_text(img_pil):
    w, h = img_pil.size
    text_str = np.random.choice(prints, np.random.randint(low=4, high = 8))
    text_str = "".join(text_str)
    if np.random.random() > 0.7:
        text_str = '@' + text_str
    # draw the watermark on a blank
    font_size = np.random.randint(16, 40)
    print('font size:', font_size)
    TTF_FONT = np.random.choice(font_resources, 1)[0]
    font = ImageFont.truetype(TTF_FONT, font_size)
    text_width, text_height = font.getsize(text_str)
    # draw watermark on img_temp
    img_temp = Image.new('L', (int(1.2*text_width),
                                int(1.2*text_height)))
    # use a pen to draw what we want
    draw_temp = ImageDraw.Draw(img_temp) 
    opac = np.random.randint(low=110, high=256)
    draw_temp.text((0, 0), text_str,  font=font, fill=opac)
    # rotate the watermark
    rot_int = np.random.randint(low = 0, high = 8)
    print('rotate:', rot_int)
    rotated_text = img_temp.rotate(rot_int,  expand=1)
    '''
    '''
    col_1 = (np.random.randint(50, 180),
            np.random.randint(50, 180),
            np.random.randint(50, 180))
    col_2 = (np.random.randint(255, 256),
            np.random.randint(255, 256),
            np.random.randint(255, 256))
    # watermarks are drawn on the input image with white color
    '''
    col_1 = (255,255,255)
    col_2 = (255,255,255)
    '''
    #rand_loc = tuple(np.random.randint(low=0,high=max(min(h, w)-max(text_width, text_height), 1), size = (2,)))
    #print(w, text_height, h, text_height)
    rand_loc = (np.random.randint(0, max(1, w-text_width)),
                np.random.randint(0, max(1, h-text_height)))
    img_pil.paste(ImageOps.colorize(rotated_text, col_1, col_2), rand_loc,  rotated_text)
    #img_pil = Image.alpha_composite(img_pil.convert('RGBA'), ImageOps.colorize(rotated_text, col_1, col_2))
    
    # 计算watermark在img_pil的位置
    text_mask = np.array(rotated_text)
    ys, xs = text_mask.nonzero()
    if len(ys) == 0:
        return img_pil, (0, 0, 0, 0, 0)
    x_min, x_max = xs.min(), xs.max() + 1
    y_min, y_max = ys.min(), ys.max() + 1
    
    '''
    return img_pil, (rand_loc[0]+(x_min+x_max)/2,
                    rand_loc[1]+(y_min+y_max)/2,
                    x_max-x_min, y_max-y_min, rot_int)
    '''
    return img_pil, (rand_loc[0]+x_min, rand_loc[1]+y_min, rand_loc[0]+x_max, rand_loc[1]+y_max, 1)

# Change the values of tensor x from range [0, 1] to [-1, 1]
def normalize(x):
    return x.mul_(2).add_(-1)

def denormalize(x):
    return x.add_(1).mul_(0.5)

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

def inspect_image(img):
    if type(img) is str:
        try:
            img = Image.open(img)
        except:
            return False
    return len(img.mode) == 3

def search_files(root, recursive=False, filter_func=None):
    files = []
    
    for item in glob.glob(osp.join(root, '*')):
        if osp.isfile(item):
            if filter_func is not None and not filter_func(item):
                continue
            files.append(item)
        elif recursive:
            files += search_files(item, recursive, filter_func)
    return files

def convert_to_3dim(img_pil):
    img = np.array(img_pil)
    if len(img.shape) == 2:
        img = img[..., None]
    img = np.concatenate([img, img, img], -1)
    return Image.fromarray(img)

def exclusion_loss(x):
    b, c, h, w = x.size()
    x = x.view((b*c, -1))
    return 1.0/(x - 0.5).abs().sum()

def torch_dilate(x, kernel_size=3):
    kernel = torch.ones((1, 1, kernel_size, kernel_size)).to(x.device)
    x_dilated = torch.nn.functional.conv2d(x, kernel, padding=(kernel_size//2, kernel_size//2)).clamp(0, 1)
    #x_dilated = x_dilated.to(x.device)
    return x_dilated

def hinge_loss(x, upper_bound=1, lower_bound=0):
    return torch.clamp(x, min=lower_bound, max=upper_bound).mean()

def weighted_l1(pred, gt, mask):
    assert pred.size() == gt.size()
    mask = mask.clamp(0, 1).detach()
    if mask.size() != pred.size():
        mask = mask.expand_as(pred)
    loss = ((pred - gt) * mask).abs().sum() / mask.sum()
    return loss

def iou(mask1, mask2):
    unions = mask1 | mask2
    union_count = (unions > 0).sum()

def plot_grad_flow_v2(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if p.grad is None:
                #print('no grad:', n)
                continue
            n = n.replace('.weight', '').replace('inception', '').replace('branch', 'B').replace('conv', 'c')
            # 如果长度大于10，每10个字符左右插入换行
            if len(n) > 10:
                n = n[:10] + '\n' + n[10:]
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    #print('layers:', layers)
    fig = plt.figure(1, figsize=(40, 5))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical", fontsize='xx-small')
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    return fig

if __name__ == '__main__':
    '''
        用来做接口测试
    '''
    # test `search_files`
    #print(search_files('.', True, inspect_image))
    # test `exclusion_loss`
    x = torch.randn((3, 1, 10, 10))
    print(exclusion_loss(x))