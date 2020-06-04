import string
import glob
import numpy as np
import torch
import os.path as osp
from PIL import ImageDraw, ImageOps, Image, ImageFont

prints = list(string.printable)[0:84]
def random_text(img_pil):
    w, h = img_pil.size
    text_str = np.random.choice(prints, np.random.randint(low=4, high = 8))
    text_str = "".join(text_str)
    # draw the watermark on a blank
    font_size = np.random.randint(12, 50)
    font = ImageFont.truetype('/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc', font_size)
    text_width, text_height = font.getsize(text_str)
    # draw watermark on img_temp
    img_temp = Image.new('L', (int(1.2*text_width),
                                int(1.2*text_height)))
    # use a pen to draw what we want
    draw_temp = ImageDraw.Draw(img_temp) 
    opac = np.random.randint(low=255, high=256)
    draw_temp.text((0, 0), text_str,  font=font, fill=opac)
    # rotate the watermark
    rot_int = np.random.randint(low = 0, high = 8)
    rotated_text = img_temp.rotate(rot_int,  expand=1)
    '''
    '''
    col_1 = (100,100,100)
    col_2 = (np.random.randint(180, 255),
            np.random.randint(180, 255),
            np.random.randint(180, 255))
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
    x_min, x_max = xs.min(), xs.max() + 1
    y_min, y_max = ys.min(), ys.max() + 1
    
    '''
    return img_pil, (rand_loc[0]+(x_min+x_max)/2,
                    rand_loc[1]+(y_min+y_max)/2,
                    x_max-x_min, y_max-y_min, rot_int)align_corners=False
    '''
    return img_pil, (rand_loc[0]+x_min, rand_loc[1]+y_min, rand_loc[0]+x_max, rand_loc[1]+y_max, 1)

# Change the values of tensor x from range [0, 1] to [-1, 1]
def normalize(x):
    return x.mul_(2).add_(-1)

def denormalize(x):
    return x.add_(1).mul_(0.5)

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
    return 1.0/(x - 0.5).abs_().sum()

if __name__ == '__main__':
    '''
        用来做接口测试
    '''
    # test `search_files`
    #print(search_files('.', True, inspect_image))
    # test `exclusion_loss`
    x = torch.randn((3, 1, 10, 10))
    print(exclusion_loss(x))