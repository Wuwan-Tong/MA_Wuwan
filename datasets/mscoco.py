import os
import json
from PIL import Image
from torchvision.datasets import VisionDataset
import cv2
import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
class Multilingual_MSCOCO(VisionDataset):

    def __init__(self, img_dir: str, ann_file: str, preprocess, transform=None, target_transform=None):
        super().__init__(img_dir, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)
        self.preprocess = preprocess
        # load annotation file
        with open(ann_file, 'r') as fp:
            data = json.load(fp)
        img_names = data['image_paths']
        self.captions = data['annotations']
        # preprocess images
        # fixme: add code for transform and target_transform
        img_names = [img_names[i * 5] for i in range(0, int(len(self.captions) / 5))]
        image_paths = [os.path.join(img_dir, img) for img in img_names]
        self.images = [self.preprocess(Image.open(image_path)) for image_path in image_paths]


    def __getitem__(self, idx):
        captions = [self.captions[idx * 5 + i] for i in range(0, 5)]
        image = self.images[idx]


        return image, captions

    def __len__(self) -> int:
        return len(self.images)

class MSCOCODatasetRetFix(VisionDataset):

    def __init__(self, img_dir: str, ann_file: str, preprocess, transform=None, target_transform=None):
        super().__init__(img_dir, transform=transform, target_transform=target_transform)
        self.preprocess = preprocess
        image_paths = []
        self.captions = []
        with open(ann_file) as fd:
            # fd.readline()
            for line in fd:
                data=line.split('++')
                img_id = data[0]
                image_paths.append(os.path.join(img_dir, img_id))
                for i in range(1, 6, 1):
                    self.captions.append(data[i])
            fd.close()

        # preprocess images
        self.images = [self.preprocess(Image.open(image_path)) for image_path in image_paths]

    def __getitem__(self, idx: int):
        captions = [self.captions[idx * 5 + i] for i in range(0, 5)]
        image = self.images[idx]

        return image, captions

    def __len__(self) -> int:
        return len(self.images)

class MSCOCODatasetRetFixConf(VisionDataset):

    def __init__(self, img_dir: str, ann_file: str, preprocess, cap_num, transform=None, target_transform=None):
        super().__init__(img_dir, transform=transform, target_transform=target_transform)
        self.preprocess = preprocess
        self.cap_num = cap_num
        image_paths = []
        self.captions = []
        with open(ann_file) as fd:
            # fd.readline()
            for line in fd:
                data=line.split('++')
                img_id = data[0]
                image_paths.append(os.path.join(img_dir, img_id))
                for i in range(1, cap_num, 1):
                    self.captions.append(data[i])
                self.captions.append(data[cap_num].split('\n')[0])
            fd.close()

        # preprocess images
        self.images = [self.preprocess(Image.open(image_path)) for image_path in image_paths]

    def __getitem__(self, idx: int):
        captions = [self.captions[idx * self.cap_num + i] for i in range(0, self.cap_num)]
        image = self.images[idx]

        return image, captions

    def __len__(self) -> int:
        return len(self.images)