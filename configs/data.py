from dataclasses import dataclass

@dataclass
class MSCOCOCfg:
    annotations_path_train: str = '../datasets/mscoco/coco_train.txt'
    # annotations_path_val: str = '../datasets/mscoco/coco_val_karpathy.json'
    annotations_path_test: str = '../datasets/mscoco/coco_test1k.txt'
    img_dir: str = "C:/Users/INDA_HIWI/Desktop/archive/coco2014/images/mscoco_images"

@dataclass
class MSCOCOCfgConf:
    # annotations_path_train: str = '../datasets/mscoco/coco_train.txt'
    # annotations_path_val: str = '../datasets/mscoco/coco_val_karpathy.json'
    annotations_path_test: str = '../datasets/mscoco/mix/nothall_corr2345.txt'
    img_dir: str = "C:/Users/INDA_HIWI/Desktop/archive/coco2014/images/mscoco_images"


@dataclass
class Flickr30kCfg:
    annotations_path_train: str = '../datasets/flickr30k/flickr30k_train_karpathy.txt'
    annotations_path_val: str = '../datasets/flickr30k/flickr30k_val_karpathy.txt'
    annotations_path_test: str = '../datasets/flickr30k/flickr30k_test_karpathy.txt'
    annotations_path: str = "C:/Users/INDA_HIWI/Desktop/flickr30k/results_20130124.token"
    img_dir: str = "C:/Users/INDA_HIWI/Desktop/flickr30k/flickr30k-images"
    val_filename: str = '../temp_code/flickr30k_cn_test.txt'
    log_dir: str = 'C:/Users/INDA_HIWI/Desktop/plot_oc_flickr_noae_notrain/ori_image'
@dataclass
class Flickr30kCfgGray:
    annotations_path_train: str = '../datasets/flickr30k/flickr30k_train_karpathy.txt'
    annotations_path_val: str = '../datasets/flickr30k/flickr30k_val_karpathy.txt'
    annotations_path_test: str = '../datasets/flickr30k/flickr30k_test_karpathy.txt'
    annotations_path: str = "C:/Users/INDA_HIWI/Desktop/flickr30k/results_20130124.token"
    img_dir: str = "C:/Users/INDA_HIWI/Desktop/flickr30k/flickr30k-images_gray"
    log_dir: str = 'C:/Users/INDA_HIWI/Desktop/plot_oc_flickr_noae_notrain/gray_image'
@dataclass
class Flickr30kCfgDs:
    ds_ratio: int = 32 # choose from 2/4/8/16/32
    annotations_path_train: str = '../datasets/flickr30k/flickr30k_train_karpathy.txt'
    annotations_path_val: str = '../datasets/flickr30k/flickr30k_val_karpathy.txt'
    annotations_path_test: str = '../datasets/flickr30k/flickr30k_test_karpathy.txt'
    annotations_path: str = "C:/Users/INDA_HIWI/Desktop/flickr30k/results_20130124.token"
    img_dir: str = "C:/Users/INDA_HIWI/Desktop/flickr30k/flickr30k-images_ds"+f'{ds_ratio}'
    log_dir: str = f'C:/Users/INDA_HIWI/Desktop/plot_oc_flickr_noae_notrain/ds_image/ds{ds_ratio}_'

@dataclass
class Flickr30kCfgConfusionInfo:
    annotations_path_train: str = '../datasets/flickr30k/flickr30k_train_karpathy.txt'
    annotations_path_val: str = '../datasets/flickr30k/flickr30k_val_karpathy.txt'
    annotations_path_test: str = '../datasets/flickr30k/flickr30k_test_confusion.txt'
    annotations_path: str = "C:/Users/INDA_HIWI/Desktop/flickr30k/results_20130124.token"
    img_dir: str = "C:/Users/INDA_HIWI/Desktop/flickr30k/flickr30k-images"
    val_filename: str = '../temp_code/flickr30k_cn_test.txt'
    log_dir: str = 'C:/Users/INDA_HIWI/Desktop/plot_oc_flickr_noae_notrain/ori_image'


@dataclass
class Flickr30kCfgRandErrInfo:
    annotations_path_train: str = '../datasets/flickr30k/flickr30k_train_karpathy.txt'
    annotations_path_val: str = '../datasets/flickr30k/flickr30k_val_karpathy.txt'
    annotations_path_test: str = '../datasets/flickr30k/flickr30k_test_rand.txt'
    annotations_path: str = "C:/Users/INDA_HIWI/Desktop/flickr30k/results_20130124.token"
    img_dir: str = "C:/Users/INDA_HIWI/Desktop/flickr30k/flickr30k-images"
    val_filename: str = '../temp_code/flickr30k_cn_test.txt'
    log_dir: str = 'C:/Users/INDA_HIWI/Desktop/plot_oc_flickr_noae_notrain/ori_image'


class Flickr30kCfgErrorInfo:
    annotations_path_train: str = '../datasets/flickr30k/flickr30k_train_karpathy.txt'
    annotations_path_val: str = '../datasets/flickr30k/flickr30k_val_karpathy.txt'
    annotations_path_test: str = '../datasets/flickr30k/flickr30k_test_pencil.txt'
    annotations_path: str = "C:/Users/INDA_HIWI/Desktop/flickr30k/results_20130124.token"
    img_dir: str = "C:/Users/INDA_HIWI/Desktop/flickr30k/flickr30k-images"
    val_filename: str = '../temp_code/flickr30k_cn_test.txt'
    log_dir: str = 'C:/Users/INDA_HIWI/Desktop/plot_oc_flickr_noae_notrain/ori_image'

@dataclass
class Flickr30kCfgCorrConfNoth:
    annotations_path_test: str = '../datasets/flickr30k/mix/conf2_corr2345.txt'
    img_dir: str = "C:/Users/INDA_HIWI/Desktop/flickr30k/flickr30k-images"