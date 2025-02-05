import torch
import os
import open_clip
import random
import math

import numpy as np
from datetime import datetime
from datasets.flickr import Flickr30kDatasetRet, Flickr30kDatasetRetFix, Flickr30kDatasetConfusionErrorInfo, Flickr30kDatasetConfRandNcaps
from configs.data import Flickr30kCfg, Flickr30kCfgGray, Flickr30kCfgDs, Flickr30kCfgConfusionInfo, Flickr30kCfgRandErrInfo, Flickr30kCfgErrorInfo, Flickr30kCfgCorrConfNoth
from utils.model_util import signal_power  # , add_noise
from torch.utils.data import DataLoader, random_split
from utils.utils import write_log, get_checkpoint_openclip
from PIL import Image

'''
Compare the caption feature distance for one caption
'''

# without the setting here, the cpu automatically choose tf16
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# device
device = 'cuda'

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', precision='fp32')
model = get_checkpoint_openclip('C:/Users/INDA_HIWI/Desktop/plot_oc_flickr_train_ae/oc/logs/2024_05_30-15_05_41-model_ViT-B-32-lr_1e-08-b_100-j_0-p_fp32/checkpoints/epoch_51.pt', model)
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.to(device)

img_dir= "C:/Users/INDA_HIWI/Desktop/flickr30k/flickr30k-images"
image_path = os.path.join(img_dir, '1007129816.jpg')
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

caps = {}
feature_caps = {}
dist = {}
dist_img = {}
prod = {}
# caps['conf1']='The woman with pierced ears is wearing glasses and an orange hat.'
# caps['conf2']='A man with glasses is wearing a beer can crocheted hat.'
# caps['noth1']='The nothing with pierced ears is wearing glasses and an orange hat.'
# caps['noth2']='The nothing with pierced ears is wearing glasses and an nothing hat.'
# caps['ori']='The man with pierced ears is wearing glasses and an orange hat.'

caps['conf1']='The bird with pierced ears is wearing glasses and an orange hat.'
caps['conf2']='A man with glasses is wearing a beer can crocheted hat.'
caps['noth1']='The nothing with pierced ears is wearing glasses and an orange hat.'
caps['noth2']='The nothing with pierced ears is wearing glasses and an nothing hat.'
caps['ori']='The man with pierced ears is wearing glasses and an orange hat.'
with torch.no_grad():
    for key, cap in caps.items():
        captions_tok = tokenizer(cap).to(device)
        # encode using pretrained open clip
        image_features, text_features, logit_scale = model(image, captions_tok)
        feature_caps[key] = text_features.cpu().detach().numpy()
for key, feature in feature_caps.items():
    img_feature = image_features.cpu().detach().numpy()
    dist[key] = np.linalg.norm(feature - feature_caps['ori'], ord=2)
    dist_img[key] = np.linalg.norm(feature - img_feature, ord=2)
    prod[key] = feature@img_feature.T

    print(key)
    print(dist[key])
    print(dist_img[key])
    print(prod[key])