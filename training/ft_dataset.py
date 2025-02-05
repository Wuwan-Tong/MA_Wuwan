
import os
from PIL import Image
from torchvision.datasets import VisionDataset
from dataclasses import dataclass
from open_clip import create_model_and_transforms

@dataclass
class Flickr30kCfg:
    annotations_path_train: str = 'C:/Users/INDA_HIWI/Desktop/MasterThesis/datasets/flickr30k/flickr30k_train_karpathy.txt'
    annotations_path_val: str = 'C:/Users/INDA_HIWI/Desktop/MasterThesis/datasets/flickr30k/flickr30k_val_karpathy.txt'
    annotations_path_test: str = 'C:/Users/INDA_HIWI/Desktop/MasterThesis/datasets/flickr30k/flickr30k_test_karpathy.txt'
    img_dir: str = "C:/Users/INDA_HIWI/Desktop/flickr30k/flickr30k-images"

@dataclass
class MSCOCOCfg:
    annotations_path_train: str = '../datasets/mscoco/coco_train.txt'
    # annotations_path_val: str = '../datasets/mscoco/coco_val_karpathy.json'
    annotations_path_test: str = '../datasets/mscoco/coco_test1k.txt'
    img_dir: str = "C:/Users/INDA_HIWI/Desktop/archive/coco2014/images/mscoco_images"

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
        idx_img = int(idx % len(self.images))
        idx_cap = idx % 5
        captions = self.captions[int(idx_img * 5 + idx_cap)]
        image = self.images[idx_img]

        return image, captions

    def __len__(self) -> int:
        if len(self.images)==1000: return 1000
        return len(self.images)*5


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
        idx_img = int(idx % len(self.images))
        idx_cap = idx % 5
        captions = self.captions[int(idx_img * 5 + idx_cap)]
        image = self.images[idx_img]

        return image, captions

    def __len__(self) -> int:
        if len(self.images) == 1000: return 1000
        return len(self.images)*5
