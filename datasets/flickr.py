
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from torchvision.io import read_image
from configs.data import Flickr30kCfg
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets import VisionDataset

class Flickr30kDataset(Dataset):
    def __init__(self, annotations_path, img_dir, transform=None, target_transform=None):
        self.annotations = pd.read_table(annotations_path, sep='\t', header=None, names=['image', 'caption'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images=self.annotations[['image']].values
        self.captions = self.annotations[['caption']].values

    def __len__(self):
        assert len(self.captions) % 5 == 0, 'at least one image do not have 5 captions, check the annotations'
        len_dataset=int(len(self.captions)/5)
        return len_dataset

    def __getitem__(self, idx):
        img_filename,_=self.images[idx*5].item().split('#')
        captions_one_img=[]
        for cap_idx in range(0,5):
            captions_one_img.append(self.captions[idx*5+cap_idx])
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     captions_one_img = [self.target_transform(caps) for caps in captions_one_img]
        return img_filename, captions_one_img

class Flickr30kDatasetRet(Dataset):
    def __init__(self, is_train=None, transform=None, target_transform=None, preprocess=None):
        self.annotations = pd.read_table(Flickr30kCfg.annotations_path, sep='\t', header=None, names=['image', 'caption'])
        self.img_dir = Flickr30kCfg.img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.preprocess=preprocess
        self.is_train=is_train
        if is_train is None: # the whole dataset
            self.captions = self.annotations[['caption']].values
            image_filenames= self.annotations[['image']].values
            image_paths = [os.path.join(self.img_dir, image_filename.item().split('#')[0]) for image_filename in image_filenames]
            self.images = [self.preprocess(Image.open(image_path)) for image_path in image_paths]
        else:
            image_filenames = self.annotations[['image']].values
            temp_cap = self.annotations[['caption']].values

            self.val_idx=[]
            val_imgs = []
            with open(Flickr30kCfg.val_filename, encoding="utf8") as f:
                for line in f:
                    val_imgs.append(line.split(',')[0])
            del val_imgs[0]
            i=len(image_filenames)-1
            num_val=0
            while i>0 and num_val<1000:
                if any([image_filenames[i].item().startswith(img) for img in val_imgs]):
                    self.val_idx.extend([i - 4, i - 3, i - 2, i - 1, i])
                    num_val+=1
                i -= 5

            if is_train:
                self.captions= np.delete(temp_cap, np.array(self.val_idx), 0)
                images_train= np.delete(image_filenames, np.array(self.val_idx), 0)
                images_train=[images_train[i*5] for i in range(0, int(len(self.captions)/5))]
                image_paths = [os.path.join(self.img_dir, image_filename.item().split('#')[0]) for image_filename in images_train]
                self.images = [self.preprocess(Image.open(image_path)) for image_path in image_paths]
            else:
                self.captions= np.array([temp_cap[i].item() for i in self.val_idx])
                images_val= np.array([image_filenames[i].item() for i in self.val_idx])
                images_val = [images_val[i * 5] for i in range(0, int(len(self.captions) / 5))]
                image_paths = [os.path.join(self.img_dir, image_filename.item().split('#')[0]) for image_filename in images_val]
                self.images = [self.preprocess(Image.open(image_path)) for image_path in image_paths]


    def __len__(self):
        assert len(self.captions) % 5 == 0, 'at least one image do not have 5 captions, check the annotations'
        return int(len(self.captions)/5)

    def __getitem__(self, idx):
        caption = [self.captions[idx*5 + i].item() for i in range(0, 5)]
        image=self.images[idx]

        return image, caption

class Flickr30kDatasetRetFix(VisionDataset):

    def __init__(self, img_dir: str, ann_file: str, preprocess, transform=None, target_transform=None):
        super().__init__(img_dir, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)
        self.preprocess=preprocess
        img_names=[]
        self.captions=[]
        with open(ann_file) as fd:
            fd.readline()
            for line in fd:
                line = line.strip()
                if line:
                    # some lines have comma in the caption, se we make sure we do the split correctly
                    img, caption = line.strip().split('.jpg,')
                    img = img + '.jpg'
                    img_names.append(img)
                    self.captions.append(caption)
        # preprocess images
        # fixme: add code for transform and target_transform
        img_names = [img_names[i * 5] for i in range(0, int(len(self.captions) / 5))]
        image_paths = [os.path.join(img_dir, img) for img in img_names]
        self.images = [self.preprocess(Image.open(image_path)) for image_path in image_paths]

    def __getitem__(self, idx: int):
        captions = [self.captions[idx * 5 + i] for i in range(0, 5)]
        image = self.images[idx]

        return image, captions

    def __len__(self) -> int:
        return len(self.images)

class Flickr30kDatasetRetFixGray(VisionDataset):

    def __init__(self, img_dir: str, ann_file: str, preprocess, transform=None, target_transform=None):
        super().__init__(img_dir, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)
        self.preprocess=preprocess
        img_names=[]
        self.captions=[]
        with open(ann_file) as fd:
            fd.readline()
            for line in fd:
                line = line.strip()
                if line:
                    # some lines have comma in the caption, se we make sure we do the split correctly
                    img, caption = line.strip().split('.jpg,')
                    img = img + '.jpg'
                    img_names.append(img)
                    self.captions.append(caption)
        # preprocess images
        # fixme: add code for transform and target_transform
        img_names = [img_names[i * 5] for i in range(0, int(len(self.captions) / 5))]
        image_paths = [os.path.join(img_dir, img) for img in img_names]
        self.images = [self.preprocess(Image.open(image_path)) for image_path in image_paths]

    def __getitem__(self, idx: int):
        captions = [self.captions[idx * 5 + i] for i in range(0, 5)]
        image = self.images[idx]

        return image, captions

    def __len__(self) -> int:
        return len(self.images)


class Flickr30kDatasetConfusionErrorInfo(VisionDataset):

    def __init__(self, img_dir: str, ann_file: str, preprocess, transform=None, target_transform=None):
        super().__init__(img_dir, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)
        self.preprocess=preprocess
        img_names=[]
        self.captions=[]
        with open(ann_file) as fd:
            fd.readline()
            for line in fd:
                line = line.strip()
                if line:
                    # some lines have comma in the caption, se we make sure we do the split correctly
                    img, caption = line.strip().split('.jpg,')
                    img = img + '.jpg'
                    img_names.append(img)
                    self.captions.append(caption)
        # preprocess images
        # fixme: add code for transform and target_transform
        img_names = [img_names[i * 10] for i in range(0, int(len(self.captions) / 10))]
        image_paths = [os.path.join(img_dir, img) for img in img_names]
        self.images = [self.preprocess(Image.open(image_path)) for image_path in image_paths]

    def __getitem__(self, idx: int):
        captions = [self.captions[idx * 10 + i] for i in range(0, 10)]
        image = self.images[idx]

        return image, captions

    def __len__(self) -> int:
        return len(self.images)

class Flickr30kDatasetConfRandNcaps(VisionDataset):

    def __init__(self, img_dir: str, ann_file: str, preprocess, cap_num, transform=None, target_transform=None):
        super().__init__(img_dir, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)
        self.preprocess=preprocess
        self.cap_num = cap_num
        img_names=[]
        self.captions=[]
        with open(ann_file) as fd:
            for line in fd:
                line = line.strip()
                if line:
                    if line == 'image,caption':
                        continue
                    # some lines have comma in the caption, se we make sure we do the split correctly
                    img, caption = line.strip().split('.jpg,')
                    img = img + '.jpg'
                    img_names.append(img)
                    self.captions.append(caption)
        # preprocess images
        # fixme: add code for transform and target_transform
        img_names = [img_names[i * self.cap_num] for i in range(int(len(self.captions) / self.cap_num))]
        image_paths = [os.path.join(img_dir, img) for img in img_names]
        self.images = [self.preprocess(Image.open(image_path)) for image_path in image_paths]

    def __getitem__(self, idx: int):
        captions = [self.captions[idx * self.cap_num + i] for i in range(self.cap_num)]
        image = self.images[idx]

        return image, captions

    def __len__(self) -> int:
        return len(self.images)
