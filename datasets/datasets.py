import torch

from datasets.flickr import Flickr30kDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split



def build_dataloader(ds_name:str,dataset_cfg, is_train, batch_size=64, shuffle=False, num_workers=0):
    if ds_name=='flickr':
        flickr30k_dataset = Flickr30kDataset(annotations_path=dataset_cfg.annotations_path,
                                             img_dir=dataset_cfg.img_dir)
        train_dataset, val_dataset = random_split(dataset=flickr30k_dataset,
                                                   lengths=[30783, 1000],
                                                   generator=torch.Generator().manual_seed(0))

        if is_train:
            sampler_train = torch.utils.data.RandomSampler(train_dataset)
            dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    sampler=sampler_train,
                                    num_workers=num_workers)
        else:
            sampler_val = torch.utils.data.SequentialSampler(val_dataset)
            dataloader = DataLoader(dataset=val_dataset,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    sampler=sampler_val,
                                    num_workers=num_workers)
        return dataloader
    else:
        RuntimeError('Incorrect dataset name, please enter a name from "flickr"')