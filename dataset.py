import glob
import os
import torch
import util
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

def load_image(image_file, target_size=(256, 256)):
    img = Image.open(image_file)

    w, h = img.size
    if type(target_size) != tuple:
        target_size = (target_size, target_size)

class Dataset(data.Dataset):
    def __init__(self, path, image_shape=(256, 256)):
        super(Dataset, self).__init__()
        self.image_shape = image_shape
        self.imgs = util.search_files(path, True)
        self.resize = transforms.Resize(min(self.image_shape))
        self.crop = transforms.RandomCrop(self.image_shape)
        self.flip = transforms.RandomHorizontalFlip()
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        w, h = img.size
        try:
            assert len(img.mode) == 3
        except AssertionError:
            print('Wrong channels:', self.imgs[idx])
            img = util.convert_to_3dim(img)
        if h < self.image_shape[0] or w < self.image_shape[1]:
            img = self.resize(img)
        img = self.crop(img)
        img = self.flip(img)
        img_watermark = util.random_text(img.copy())[0]
        img_watermark = util.tv_transform(self.to_tensor(img_watermark))
        img = util.tv_transform(self.to_tensor(img))
        mask_wm = np.any(np.array(img) != np.array(img_watermark), axis=-1).astype(np.float)

        return img, img_watermark, mask_wm

import traceback
class LargeScaleWatermarkDataset(data.Dataset):
    def __init__(self, folder_origin, folder_watermarked, anno_file, img_shape=(256, 256)):
        super(LargeScaleWatermarkDataset, self).__init__()
        self.img_shape = img_shape
        self.folder_origin = folder_origin
        self.folder_watermarked = folder_watermarked
        self.img_pair = {} # origin_filename: watermarked_filename
        self.resize = transforms.Resize(int(min(img_shape)*1.15))
        self.flip = transforms.RandomHorizontalFlip(1)
        self.to_tensor = transforms.ToTensor()

        try:
            with open(anno_file, 'r') as f:
                anno = f.readlines()
            annos = {}
            for line in anno:
                '''
                    img_id origin_filename
                '''
                img_id, origin_fn = line.rsplit(' ', 1)
                annos[img_id] = origin_fn[:-1]
            print('number of anno files:', len(annos))
        except Exception as e:
            print(str(e), traceback.format_exc())
            import pdb; pdb.set_trace()

        for label_file in glob.glob(os.path.join(folder_watermarked, '*.txt')):
            '''
                watermark_id img_id c_x c_y width height
            '''
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        print(label_file)
                    for line in lines:
                        if line == '': continue
                        img_id = line.split(' ')[1]
                        # link original image & watermark image
                        if img_id in annos.keys():
                            img_file = label_file.replace('txt', 'png')
                            if os.path.exists(os.path.join(folder_watermarked, img_file)):
                                self.img_pair[annos[img_id]] = img_file
            except Exception as e:
                print(str(e))
                continue
        self.original_files = list(self.img_pair.keys())
        print('number of training samples:', len(self.original_files))

    def __len__(self):
        return len(self.img_pair)

    def random_crop(self, img_a, img_b, w, h):
        assert img_a.size == img_b.size
        w_ori, h_ori = img_a.size
        x_start = 0 if w_ori==w else np.random.randint(w_ori-w)
        y_start = 0 if h_ori==h else np.random.randint(h_ori-h)
        target_box = (x_start, y_start, x_start+w, y_start+h)
        return img_a.crop(target_box), img_b.crop(target_box)

    def random_flip(self, img_a, img_b):
        if np.random.random() > 0.5:
            img_a = img_a.transpose(Image.FLIP_LEFT_RIGHT)
            img_b = img_b.transpose(Image.FLIP_LEFT_RIGHT)
        return img_a, img_b

    def __getitem__(self, idx):
        img_ori = Image.open(os.path.join(self.folder_origin, self.original_files[idx])).convert('RGB')
        img_wm = Image.open(os.path.join(self.folder_watermarked, self.img_pair[self.original_files[idx]])).convert('RGB')
        img_ori, img_wm = self.random_flip(self.resize(img_ori), self.resize(img_wm))
        img_ori, img_wm = self.random_crop(img_ori, img_wm, self.img_shape[0], self.img_shape[1])
        # 10% of the watermarked images are original image identities
        if np.random.random() < 0.1:
            img_wm = img_ori
        # 比较img_ori和img_wm获取水印的mask
        mask_wm = np.any(np.array(img_ori) != np.array(img_wm), axis=-1).astype(np.float)
        img_ori = util.normalize(self.to_tensor(img_ori))
        img_wm = util.normalize(self.to_tensor(img_wm))

        return img_ori, img_wm, mask_wm

if __name__ == '__main__':
    dataset = LargeScaleWatermarkDataset(folder_origin='/home/chengk/chk/data/Large-scale_Visible_Watermark_Dataset/original_images/train/',
                                        folder_watermarked='/home/chengk/chk/data/Large-scale_Visible_Watermark_Dataset/watermarked_images/train/',
                                        anno_file='/home/chengk/chk/data/Large-scale_Visible_Watermark_Dataset/watermarked_images/train_imageID.txt')
    print(len(dataset))
    img_ori, img_wm = dataset[0]
    print(img_ori.size(), img_wm.size())
    img_ori, img_wm = dataset[1]
    print(img_ori.size(), img_wm.size())
