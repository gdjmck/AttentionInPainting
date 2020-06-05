import glob
import os
import torch
import util
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

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
        img = util.normalize(self.to_tensor(img))
        if np.random.random() > 0.1:
            img_watermark = util.random_text(img.copy())[0]
            img_watermark = util.normalize(self.to_tensor(img_watermark))
        else:
            img_watermark = img.clone()

        return img, img_watermark
