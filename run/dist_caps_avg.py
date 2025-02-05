import torch
import os
import open_clip
import random
import math

import numpy as np
from datetime import datetime

from datasets.flickr import Flickr30kDatasetRet, Flickr30kDatasetRetFix, Flickr30kDatasetConfusionErrorInfo, Flickr30kDatasetConfRandNcaps
from configs.data import Flickr30kCfg, Flickr30kCfgGray, Flickr30kCfgDs, Flickr30kCfgConfusionInfo, Flickr30kCfgRandErrInfo, Flickr30kCfgErrorInfo, Flickr30kCfgCorrConfNoth
from utils.model_util import signal_power, add_noise
from torch.utils.data import DataLoader, random_split
from utils.utils import write_log, get_checkpoint_openclip
from PIL import Image

'''
Compare the average caption feature distance for testing set Flickr30k
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

annotations_path_ori = '../datasets/flickr30k/flickr30k_test_karpathy.txt'
annotations_path_conf1 = '../datasets/flickr30k/one_cap/flickr30k_test_conf1.txt'
annotations_path_conf2 = '../datasets/flickr30k/one_cap/flickr30k_test_conf2_new.txt'
annotations_path_noth1 = '../datasets/flickr30k/one_cap/flickr30k_test_noth1.txt'
annotations_path_noth2 = '../datasets/flickr30k/one_cap/flickr30k_test_noth2.txt'
val_dataset = Flickr30kDatasetConfRandNcaps(img_dir=Flickr30kCfg.img_dir, ann_file=annotations_path_ori, preprocess=preprocess, cap_num=5)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_dataset.__len__())
# channel_snr=-18
random.seed(2)
for channel_snr in [None, 24, 18, 12, 6, 0, -6, -12, -18, -24]:
    print(f'channel_snr={channel_snr}')
    prod_cap_img_avg = {}
    prod_cap_img_angle_avg = {}
    diff_prod = {}
    diff_angle = {}
    for key in ['ori', 'conf1', 'conf2', 'noth1', 'noth2']:
        diff_prod[key] = 0
        diff_angle[key] = 0
        prod_cap_img_avg[key] = 0
        prod_cap_img_angle_avg[key] = 0

    for seed in range(4):
        cap_features = {}
        for _, (images, captions) in enumerate(val_dataloader):
            images = images.to(torch.float32).to(device)
            with torch.no_grad():
                captions_tok = tokenizer(captions[0]).to(device)
                image_features, text_features, logit_scale = model(images, captions_tok)
                if channel_snr is not None:
                    image_features = add_noise(image_features, channel_snr=channel_snr, is_val=True, seed=random.randint(1, 10000))
                    text_features = add_noise(text_features, channel_snr=channel_snr, is_val=True, seed=random.randint(1, 10000))

        img_features = image_features.cpu().detach().numpy()
        cap_features['ori'] = text_features.cpu().detach().numpy()
        del captions_tok
        del image_features
        del text_features
        torch.cuda.empty_cache()

        captions = []
        with open(annotations_path_conf1) as fd:
            for line in fd:
                line = line.strip()
                if line:
                    # some lines have comma in the caption, se we make sure we do the split correctly
                    _, caption = line.strip().split('.jpg,')
                    captions.append(caption)
            fd.close()
        with torch.no_grad():
            captions_tok = tokenizer(captions).to(device)
            _, text_features, logit_scale = model(images, captions_tok)
            if channel_snr is not None:
                text_features = add_noise(text_features, channel_snr=channel_snr, is_val=True, seed=seed)
        cap_features['conf1'] = text_features.cpu().detach().numpy()
        del captions_tok
        del text_features
        torch.cuda.empty_cache()

        captions = []
        with open(annotations_path_conf2) as fd:
            for line in fd:
                line = line.strip()
                if line:
                    # some lines have comma in the caption, se we make sure we do the split correctly
                    _, caption = line.strip().split('.jpg,')
                    captions.append(caption)
            fd.close()
        with torch.no_grad():
            captions_tok = tokenizer(captions).to(device)
            _, text_features, logit_scale = model(images, captions_tok)
            if channel_snr is not None:
                text_features = add_noise(text_features, channel_snr=channel_snr, is_val=True, seed=seed)
        cap_features['conf2'] = text_features.cpu().detach().numpy()
        del captions_tok
        del text_features
        torch.cuda.empty_cache()

        captions = []
        with open(annotations_path_noth1) as fd:
            for line in fd:
                line = line.strip()
                if line:
                    # some lines have comma in the caption, se we make sure we do the split correctly
                    _, caption = line.strip().split('.jpg,')
                    captions.append(caption)
            fd.close()
        with torch.no_grad():
            captions_tok = tokenizer(captions).to(device)
            _, text_features, logit_scale = model(images, captions_tok)
            if channel_snr is not None:
                text_features = add_noise(text_features, channel_snr=channel_snr, is_val=True, seed=seed)
        cap_features['noth1'] = text_features.cpu().detach().numpy()

        captions = []
        with open(annotations_path_noth2) as fd:
            for line in fd:
                line = line.strip()
                if line:
                    # some lines have comma in the caption, se we make sure we do the split correctly
                    _, caption = line.strip().split('.jpg,')
                    captions.append(caption)
            fd.close()
        with torch.no_grad():
            captions_tok = tokenizer(captions).to(device)
            _, text_features, logit_scale = model(images, captions_tok)
            if channel_snr is not None:
                text_features = add_noise(text_features, channel_snr=channel_snr, is_val=True, seed=seed)
        cap_features['noth2'] = text_features.cpu().detach().numpy()
        del captions_tok
        del text_features
        torch.cuda.empty_cache()

        # dist_cap_ori = {}
        # dist_cap_img = {}
        prod_cap_img = {}
        diff_angle_temp = {}
        prod_cap_img_angle_temp = {}
        # dist_cap_ori_avg = {}
        # dist_cap_img_avg = {}


        for key in ['ori', 'conf1', 'conf2', 'noth1', 'noth2']:
            # dist_cap_ori[key] = np.zeros((1, 1000))
            # dist_cap_img[key] = np.zeros((1, 1000))
            prod_cap_img[key] = np.zeros((1, 1000))
            diff_angle_temp[key] = np.zeros((1, 1000))
            prod_cap_img_angle_temp[key] = np.zeros((1, 1000))

            for i in range(1000):
                # dist_cap_ori[key][0, i] = np.linalg.norm(cap_features[key][i, :] - cap_features['ori'][i, :], ord=2)
                # dist_cap_img[key][0, i] = np.linalg.norm(cap_features[key][i, :] - img_features[i, :], ord=2)
                # prod_cap_img[key][0, i] = cap_features[key][i, :] @ img_features[i, :].T
                # cap_feature = cap_features[key][i, :]
                # img_feature = img_features[i, :]
                # print(np.linalg.norm(cap_feature))
                # print(np.linalg.norm(img_feature))
                # print(np.linalg.norm(cap_features[key][i, :]))
                # print(np.linalg.norm(img_features[i, :]))
                prod_cap_img[key][0, i] = (cap_features[key][i, :]/np.linalg.norm(cap_features[key][i, :])) @ (img_features[i, :].T/np.linalg.norm(img_features[i, :]))
                # print(cap_features[key][i, :] @ img_features[i, :].T)
                # print(prod_cap_img[key][0, i])
                diff_angle_temp[key][0, i] = np.arccos(prod_cap_img[key][0, i])/np.pi*180 - np.arccos(prod_cap_img['ori'][0, i])/np.pi*180
                prod_cap_img_angle_temp[key][0, i] = np.arccos(prod_cap_img[key][0, i])/np.pi*180


            # dist_cap_ori_avg[key] = dist_cap_ori[key].mean()
            # dist_cap_img_avg[key] = dist_cap_img[key].mean()
            prod_cap_img_avg[key] += prod_cap_img[key].mean()
            prod_cap_img_angle_avg[key] += prod_cap_img_angle_temp[key].mean()

            diff_prod[key] += (prod_cap_img[key]- prod_cap_img['ori']).mean()
            diff_angle[key] += diff_angle_temp[key].mean()

    for key in ['ori', 'conf1', 'conf2', 'noth1', 'noth2']:
        prod_cap_img_avg[key] /=4
        diff_prod[key] /=4
        diff_angle[key] /=4
        prod_cap_img_angle_avg[key] /=4

        print(key)
        # print(dist_cap_ori_avg[key])
        # print(dist_cap_img_avg[key])
        print('%.5f' % prod_cap_img_avg[key])
        print('%.5f' % diff_prod[key])
        print('angle')
        print('%.5f' % prod_cap_img_angle_avg[key])
        print('%.5f' % diff_angle[key])


##################

