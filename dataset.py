import glob
import os
import torch
import util
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

class Dataset(data.Dataset):
    def __init__(self, path, image_shape=(256, 256)):
        super(Dataset, self).__init__()
        self.image_shape = image_shape
        self.imgs = glob.glob(os.path.join(path, '*'))
        self.resize = transforms.Resize(min(self.image_shape))
        self.crop = transforms.RandomCrop(self.image_shape)
        self.flip = transforms.RandomHorizontalFlip()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        w, h = img.size
        if h < self.image_shape[0] or w < self.image_shape[1]:
            img = self.resize(img)
        img = self.crop(img)
        img = self.flip(img)
        img_watermark = util.random_text(img.copy())[0]

        img = util.normalize(transforms.ToTensor()(img))
        img_watermark = util.normalize(transforms.ToTensor()(img_watermark))

        return img, img_watermark
