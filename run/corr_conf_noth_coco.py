import torch
import os
import open_clip
import random
import math

import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import nn
from datetime import datetime
from datasets.mscoco import MSCOCODatasetRetFixConf
from configs.data import MSCOCOCfgConf
from utils.model_util import signal_power  # , add_noise
from torch.utils.data import DataLoader, random_split
from utils.utils import write_log, get_checkpoint_openclip


def add_noise(feature, channel_snr, is_val, seed, is_cap=False):
    feature_power = signal_power(feature)
    if is_val:
        random.seed(a=seed)
        noise = np.zeros(feature.shape)
        if is_cap:
            for n in np.nditer(noise, op_flags=['readwrite']):
                n[...] = random.gauss(0, math.sqrt(10 ** (-channel_snr / 10) * feature_power * 4))
            noise = torch.from_numpy(noise / 2)
            noise = noise.to(torch.float32).to('cuda')
        else:
            for n in np.nditer(noise, op_flags=['readwrite']):
                n[...] = random.gauss(0, math.sqrt(10 ** (-channel_snr / 10) * feature_power))
            noise = torch.from_numpy(noise)
            noise = noise.to(torch.float32).to('cuda')
    else:
        noise = torch.normal(0, math.sqrt(10 ** (-channel_snr / 10) * feature_power), size=feature.shape).to(torch.float32).to('cuda')

    feature = feature + noise

    return feature


def train(ret_ats,
          channel_snrs,
          rand_valset=False,
          device='cuda',
          log_dir='C:/Users/INDA_HIWI/Desktop/',
          dataset_cfg=MSCOCOCfgConf,
          seeds=[1234, 2345, 3456, 4567, 5678],
          cap_num=5):
    # without the setting here, the cpu automatically choose tf16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # time stamper
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    # path to save logs
    plot_dir = log_dir + date_str
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    log_path = plot_dir + '/log_result' + time_str + '.txt'

    # log running parameters
    # write_log(f'channel_snrs: {channel_snrs}', log_path)

    # init pretrained model
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', precision='fp32')
    model = get_checkpoint_openclip('C:/Users/INDA_HIWI/Desktop/plot_oc_flickr_train_ae/oc/coco_logs/2024_08_29-16_18_17-model_ViT-B-32-lr_1e-08-b_100-j_0-p_fp32/checkpoints/epoch_50.pt', model)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.to(device)

    # prepare dataset
    # log
    # write_log(f'[{datetime.now().strftime("%H:%M:%S")}] loading dataset...... \n', log_path)
    val_dataset = MSCOCODatasetRetFixConf(img_dir=dataset_cfg.img_dir, ann_file=dataset_cfg.annotations_path_test, preprocess=preprocess, cap_num=cap_num)
    len_valset = val_dataset.__len__()

    # prepare train dataloader
    # write_log(f'[{datetime.now().strftime("%H:%M:%S")}] preparing dataloader......\n', log_path)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_dataset.__len__())
    # test
    rate = {}
    count = {}
    for ic in ['cap']:
        for r in ret_ats:
            rate[f'{ic}ret_r{r}_corr'] = []
            rate[f'{ic}ret_r{r}_conf'] = []
            rate[f'{ic}ret_r{r}_noth'] = []
            rate[f'{ic}ret_r{r}_err'] = []
            count[f'{ic}ret_r{r}_corr'] = []
            count[f'{ic}ret_r{r}_conf'] = []
            count[f'{ic}ret_r{r}_noth'] = []
            count[f'{ic}ret_r{r}_err'] = []

    for channel_snr in channel_snrs:
        # # log
        # write_log(f'--------------------------channel_snr: {channel_snr}------------------------------\n',log_path)
        ################################# testing #################################
        # init
        val_loss = 0
        if cap_num == 6:
            count_conf = {}
            rate_conf = {}
        count_corr = {}
        count_noth = {}
        count_err = {}
        rate_corr = {}
        rate_noth = {}
        rate_err = {}
        temp_cap_logit = {}
        for ret_at in ret_ats:
            if cap_num == 6:
                count_conf[f'R@{ret_at}'] = 0
                rate_conf[f'R@{ret_at}'] = 0
            count_corr[f'R@{ret_at}'] = 0
            count_noth[f'R@{ret_at}'] = 0
            count_err[f'R@{ret_at}'] = 0
            rate_corr[f'R@{ret_at}'] = 0
            rate_noth[f'R@{ret_at}'] = 0
            rate_err[f'R@{ret_at}'] = 0
            temp_cap_logit[f'R@{ret_at}'] = None

        for seed in seeds:
            for ret_at in ret_ats:
                temp_cap_logit[f'R@{ret_at}'] = None
            for _, (images, captions) in enumerate(val_dataloader):
                # init
                img_num = len(images)
                images = images.to(torch.float32).to(device)
                #
                # temp_acc_cap = {}
                # temp_acc_cap[f'R@1'], temp_acc_cap[f'R@5'], temp_acc_cap[f'R@10'] = None, None, None

                for i in range(0, cap_num):
                    # collect caption sub batch
                    captions_batch = [captions[i][j] for j in range(0, img_num)]
                    # tokenize each caption
                    captions_tok = tokenizer(captions_batch).to(device)

                    with torch.no_grad():
                        # encode using pretrained open clip
                        image_features, text_features, logit_scale = model(images, captions_tok)
                        if channel_snr[0] is not None:
                            image_features = add_noise(image_features, channel_snr[0], is_val=False, seed=seed)
                        if channel_snr[1] is not None:
                            text_features = add_noise(text_features, channel_snr[1], is_val=False, seed=seed, is_cap=True)
                        # get val loss from logits
                        logit_scale = logit_scale.mean()
                        logits_per_image = logit_scale * image_features @ text_features.T
                        # logits_per_text = logits_per_image.T

                        # get val acc
                        groundtruth = np.arange(img_num)
                        for ret_at in ret_ats:
                            # cat all cap digits
                            if temp_cap_logit[f'R@{ret_at}'] is None:
                                temp_cap_logit[f'R@{ret_at}'] = logits_per_image.cpu().detach().numpy()
                            else:
                                temp_cap_logit[f'R@{ret_at}'] = np.concatenate((temp_cap_logit[f'R@{ret_at}'], logits_per_image.cpu().detach().numpy()), axis=-1)

                for ret_at in ret_ats:
                    temp_max_cap_index = np.argsort(temp_cap_logit[f'R@{ret_at}'], axis=-1)[:, -ret_at:]
                    temp_corr_cap_index = np.zeros(temp_max_cap_index.shape)
                    temp_noth_cap_index = np.zeros(temp_max_cap_index.shape)
                    if cap_num == 6:
                        temp_conf_cap_index = np.zeros(temp_max_cap_index.shape)
                    temp_err_cap_index = np.zeros(temp_max_cap_index.shape)
                    for idx in range(0, ret_at):
                        temp_corr_cap_index[:, idx] = temp_max_cap_index[:, idx]
                        temp_noth_cap_index[:, idx] = temp_max_cap_index[:, idx]
                        temp_err_cap_index[:, idx] = temp_max_cap_index[:, idx]
                        temp_err_cap_index[:, idx] = temp_err_cap_index[:, idx] % img_num - groundtruth
                        if cap_num == 6:
                            temp_conf_cap_index[:, idx] = temp_max_cap_index[:, idx]

                            if temp_corr_cap_index[0, idx] < 2 * img_num:
                                temp_corr_cap_index[0, idx] = 2 * img_num + 100
                            pos = np.argwhere(temp_corr_cap_index[:, idx] < (cap_num - 4) * img_num)
                            temp_corr_cap_index[pos, idx] = 0
                            temp_corr_cap_index[:, idx] = temp_corr_cap_index[:, idx] % img_num - groundtruth

                            if temp_noth_cap_index[0, idx] >= img_num:
                                temp_noth_cap_index[0, idx] = 100
                            pos = np.argwhere(temp_noth_cap_index[:, idx] > img_num)
                            temp_noth_cap_index[pos, idx] = 0
                            temp_noth_cap_index[:, idx] = temp_noth_cap_index[:, idx] % img_num - groundtruth

                            if temp_conf_cap_index[0, idx] >= 2 * img_num or temp_conf_cap_index[0, idx] < img_num:
                                temp_conf_cap_index[0, idx] = 1 * img_num + 100
                            pos = np.argwhere(temp_conf_cap_index[:, idx] >= 2 * img_num)
                            temp_conf_cap_index[pos, idx] = 0
                            pos = np.argwhere(temp_conf_cap_index[:, idx] < img_num)
                            temp_conf_cap_index[pos, idx] = 0
                            temp_conf_cap_index[:, idx] = temp_conf_cap_index[:, idx] % img_num - groundtruth
                        elif cap_num == 5:
                            if temp_corr_cap_index[0, idx] < img_num:
                                temp_corr_cap_index[0, idx] = (cap_num - 4) * img_num + 100
                            pos = np.argwhere(temp_corr_cap_index[:, idx] < (cap_num - 4) * img_num)
                            temp_corr_cap_index[pos, idx] = 0
                            temp_corr_cap_index[:, idx] = temp_corr_cap_index[:, idx] % img_num - groundtruth

                            if temp_noth_cap_index[0, idx] >= img_num:
                                temp_noth_cap_index[0, idx] = 100
                            pos = np.argwhere(temp_noth_cap_index[:, idx] > img_num)
                            temp_noth_cap_index[pos, idx] = 0
                            temp_noth_cap_index[:, idx] = temp_noth_cap_index[:, idx] % img_num - groundtruth
                        else:
                            print('!!!!! check cap num')

                    rate_corr[f'R@{ret_at}'] += np.any(temp_corr_cap_index == 0, axis=-1).sum()
                    rate_noth[f'R@{ret_at}'] += np.any(temp_noth_cap_index == 0, axis=-1).sum()
                    rate_err[f'R@{ret_at}'] += np.any(temp_err_cap_index != 0, axis=-1).sum()
                    count_corr[f'R@{ret_at}'] += np.count_nonzero(temp_corr_cap_index == 0, axis=-1).sum()
                    count_noth[f'R@{ret_at}'] += np.count_nonzero(temp_noth_cap_index == 0, axis=-1).sum()
                    count_err[f'R@{ret_at}'] += np.count_nonzero(temp_err_cap_index != 0, axis=-1).sum()
                    if cap_num == 6:
                        rate_conf[f'R@{ret_at}'] += np.any(temp_conf_cap_index == 0, axis=-1).sum()
                        count_conf[f'R@{ret_at}'] += np.count_nonzero(temp_conf_cap_index == 0, axis=-1).sum()

        for r in ret_ats:
            # if cap_num == 6:
                # count_err[f'R@{r}'] += r * img_num * 5 - count_corr[f'R@{r}'] - count_conf[f'R@{r}'] - count_noth[f'R@{r}']
            # elif cap_num == 5:
                # count_err[f'R@{r}'] += r * img_num * 5 - count_corr[f'R@{r}'] - count_noth[f'R@{r}']
            rate[f'capret_r{r}_corr'].append(rate_corr[f'R@{r}'] / len_valset * 100 / 5)
            rate[f'capret_r{r}_noth'].append(rate_noth[f'R@{r}'] / len_valset * 100 / 5)
            rate[f'capret_r{r}_err'].append(rate_err[f'R@{r}'] / len_valset * 100 / 5)
            count[f'capret_r{r}_corr'].append(count_corr[f'R@{r}'] / 5)
            count[f'capret_r{r}_noth'].append(count_noth[f'R@{r}'] / 5)
            count[f'capret_r{r}_err'].append(count_err[f'R@{r}'] / 5)
            if cap_num == 6:
                rate[f'capret_r{r}_conf'].append(rate_conf[f'R@{r}'] / len_valset * 100 / 5)
                count[f'capret_r{r}_conf'].append(count_conf[f'R@{r}'] / 5)

    if channel_snrs[0] == [None, None]:
        print('rate corr')
        for r in ret_ats:
            print('%.2f' % rate[f'capret_r{r}_corr'][0])
        if cap_num == 6:
            print('rate conf') # cap2
            for r in ret_ats:
                print('%.2f' % rate[f'capret_r{r}_conf'][0])
        print('rate noth') # cap1
        for r in ret_ats:
            print('%.2f' % rate[f'capret_r{r}_noth'][0])
        print('rate err')
        for r in ret_ats:
            print('%.2f' % rate[f'capret_r{r}_err'][0])
        print('count corr')
        for r in ret_ats:
            print('%d' % count[f'capret_r{r}_corr'][0])
        if cap_num == 6:
            print('count conf')
            for r in ret_ats:
                print('%d' % count[f'capret_r{r}_conf'][0])
        print('count noth')
        for r in ret_ats:
            print('%d' % count[f'capret_r{r}_noth'][0])
        print('count err')
        for r in ret_ats:
            print('%d' % count[f'capret_r{r}_err'][0])
    else:
        print('rate corr')
        for r in ret_ats:
            print('[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]'
                  % (rate[f'capret_r{r}_corr'][0], rate[f'capret_r{r}_corr'][1], rate[f'capret_r{r}_corr'][2], rate[f'capret_r{r}_corr'][3], rate[f'capret_r{r}_corr'][4],
                     rate[f'capret_r{r}_corr'][5], rate[f'capret_r{r}_corr'][6], rate[f'capret_r{r}_corr'][7], rate[f'capret_r{r}_corr'][8]))
        if cap_num == 6:
            print('rate conf')
            for r in ret_ats:
                print('[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]'
                      % (rate[f'capret_r{r}_conf'][0], rate[f'capret_r{r}_conf'][1], rate[f'capret_r{r}_conf'][2], rate[f'capret_r{r}_conf'][3], rate[f'capret_r{r}_conf'][4],
                         rate[f'capret_r{r}_conf'][5], rate[f'capret_r{r}_conf'][6], rate[f'capret_r{r}_conf'][7], rate[f'capret_r{r}_conf'][8]))
        print('rate noth')
        for r in ret_ats:
            print('[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]'
                  % (rate[f'capret_r{r}_noth'][0], rate[f'capret_r{r}_noth'][1], rate[f'capret_r{r}_noth'][2], rate[f'capret_r{r}_noth'][3], rate[f'capret_r{r}_noth'][4],
                     rate[f'capret_r{r}_noth'][5], rate[f'capret_r{r}_noth'][6], rate[f'capret_r{r}_noth'][7], rate[f'capret_r{r}_noth'][8]))
        print('rate err')
        for r in ret_ats:
            print('[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]'
                  % (rate[f'capret_r{r}_err'][0], rate[f'capret_r{r}_err'][1], rate[f'capret_r{r}_err'][2], rate[f'capret_r{r}_err'][3], rate[f'capret_r{r}_err'][4],
                     rate[f'capret_r{r}_err'][5], rate[f'capret_r{r}_err'][6], rate[f'capret_r{r}_err'][7], rate[f'capret_r{r}_err'][8]))
        print('count corr')
        for r in ret_ats:
            print('[%d, %d, %d, %d, %d, %d, %d, %d, %d]'
                  % (count[f'capret_r{r}_corr'][0], count[f'capret_r{r}_corr'][1], count[f'capret_r{r}_corr'][2], count[f'capret_r{r}_corr'][3], count[f'capret_r{r}_corr'][4],
                     count[f'capret_r{r}_corr'][5], count[f'capret_r{r}_corr'][6], count[f'capret_r{r}_corr'][7], count[f'capret_r{r}_corr'][8]))
        if cap_num == 6:
            print('count conf')
            for r in ret_ats:
                print('[%d, %d, %d, %d, %d, %d, %d, %d, %d]'
                      % (count[f'capret_r{r}_conf'][0], count[f'capret_r{r}_conf'][1], count[f'capret_r{r}_conf'][2], count[f'capret_r{r}_conf'][3], count[f'capret_r{r}_conf'][4],
                         count[f'capret_r{r}_conf'][5], count[f'capret_r{r}_conf'][6], count[f'capret_r{r}_conf'][7], count[f'capret_r{r}_conf'][8]))
        print('count noth')
        for r in ret_ats:
            print('[%d, %d, %d, %d, %d, %d, %d, %d, %d]'
                  % (count[f'capret_r{r}_noth'][0], count[f'capret_r{r}_noth'][1], count[f'capret_r{r}_noth'][2], count[f'capret_r{r}_noth'][3], count[f'capret_r{r}_noth'][4],
                     count[f'capret_r{r}_noth'][5], count[f'capret_r{r}_noth'][6], count[f'capret_r{r}_noth'][7], count[f'capret_r{r}_noth'][8]))
        print('count err')
        for r in ret_ats:
            print('[%d, %d, %d, %d, %d, %d, %d, %d, %d]'
                  % (count[f'capret_r{r}_err'][0], count[f'capret_r{r}_err'][1], count[f'capret_r{r}_err'][2], count[f'capret_r{r}_err'][3], count[f'capret_r{r}_err'][4],
                     count[f'capret_r{r}_err'][5], count[f'capret_r{r}_err'][6], count[f'capret_r{r}_err'][7], count[f'capret_r{r}_err'][8]))
    return 0


if __name__ == '__main__':
    ######################### set params #########################
    cap_num = 5
    # R@?
    ret_ats = [1, cap_num, 2 * cap_num]
    # choose dataset
    dataset_cfg = MSCOCOCfgConf  # Flickr30kCfgConfusionInfo, Flickr30kCfgRandErrInfo, Flickr30kCfgErrorInfo
    seeds = [1234, 2345, 3456, 4567, 5678]

    # set snr, SNR=None if not add noise [snr_img, snr_cap]
    # channel_snrs =  [[None, None], [24, 24], [18, 18], [12, 12], [6, 6], [0, 0], [-6, -6], [-12, -12], [-18, -18], [-24, -24], [-30, -30], [-36, -36], [-42, -42]]
    # channel_snrs = [[24, None], [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None], [-24, None], [-30, None], [-36, None], [-42, None]]
    # channel_snrs = [[None, 24], [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18], [None, -24], [None, -30], [None, -36], [None, -42]]
    # img_snr=-12
    # channel_snrs = [[img_snr, 24], [img_snr, 18], [img_snr, 12], [img_snr, 6], [img_snr, 0], [img_snr, -6], [img_snr, -12], [img_snr, -18], [img_snr, -24], [img_snr, -30], [img_snr, -36], [img_snr, -42]]
    # cap_snr=-12
    # channel_snrs = [[24, cap_snr], [18, cap_snr], [12, cap_snr], [6, cap_snr], [0, cap_snr], [-6, cap_snr], [-12, cap_snr], [-18, cap_snr], [-24, cap_snr], [-30, cap_snr], [-36, cap_snr], [-42, cap_snr]]

    # random split testing set from the dataset: True or use the testing set from the paper: False
    rand_valset = False

    # path to save the logs
    log_dir = 'C:/Users/INDA_HIWI/Desktop/logs_fix_val_noise/confusion_error_compare_coco'  # dataset_cfg.log_dir #'C:/Users/INDA_HIWI/Desktop/plot_oc_flickr_noae_notrain/ds_image'

    channel_snrs = [[None, None]]
    print('both')
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     rand_valset=rand_valset,
                     log_dir=log_dir,
                     dataset_cfg=dataset_cfg,
                     seeds=seeds,
                     cap_num=cap_num)

    channel_snrs = [[24, 24], [18, 18], [12, 12], [6, 6], [0, 0], [-6, -6], [-12, -12], [-18, -18], [-24, -24]]
    print('both')
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     rand_valset=rand_valset,
                     log_dir=log_dir,
                     dataset_cfg=dataset_cfg,
                     seeds=seeds,
                     cap_num=cap_num)

    channel_snrs = [[24, None], [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None], [-24, None]]
    print('noise on image')
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     rand_valset=rand_valset,
                     log_dir=log_dir,
                     dataset_cfg=dataset_cfg,
                     seeds=seeds,
                     cap_num=cap_num)

    channel_snrs = [[None, 24], [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18], [None, -24]]
    print('noise on cap')
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     rand_valset=rand_valset,
                     log_dir=log_dir,
                     dataset_cfg=dataset_cfg,
                     seeds=seeds,
                     cap_num=cap_num)


    for img_snr in [24, 18, 12, 6, 0, -6, -12, -18, -24]:
        channel_snrs = [[img_snr, 24], [img_snr, 18], [img_snr, 12], [img_snr, 6], [img_snr, 0], [img_snr, -6], [img_snr, -12], [img_snr, -18], [img_snr, -24]]
        print('img_snr=%d' % img_snr)
        acc_both = train(ret_ats=ret_ats,
                         channel_snrs=channel_snrs,
                         rand_valset=rand_valset,
                         log_dir=log_dir,
                         dataset_cfg=dataset_cfg,
                         seeds=seeds,
                         cap_num=cap_num)
    #
    # dataset_cfg = Flickr30kCfgRandErrInfo  # Flickr30kCfgConfusionInfo, Flickr30kCfgRandErrInfo, Flickr30kCfgErrorInfo
    # channel_snrs = [[None, None]]
    # print('both')
    # acc_both = train(ret_ats=ret_ats,
    #                  channel_snrs=channel_snrs,
    #                  rand_valset=rand_valset,
    #                  log_dir=log_dir,
    #                  dataset_cfg=dataset_cfg,
    #                  seeds=seeds)
    # channel_snrs = [[24, 24], [18, 18], [12, 12], [6, 6], [0, 0], [-6, -6], [-12, -12], [-18, -18], [-24, -24]]
    # print('both')
    # acc_both = train(ret_ats=ret_ats,
    #                  channel_snrs=channel_snrs,
    #                  rand_valset=rand_valset,
    #                  log_dir=log_dir,
    #                  dataset_cfg=dataset_cfg,
    #                  seeds=seeds)
    #
    # channel_snrs = [[24, None], [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None], [-24, None]]
    # print('noise on image')
    # acc_both = train(ret_ats=ret_ats,
    #                  channel_snrs=channel_snrs,
    #                  rand_valset=rand_valset,
    #                  log_dir=log_dir,
    #                  dataset_cfg=dataset_cfg,
    #                  seeds=seeds)
    #
    # channel_snrs = [[None, 24], [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18], [None, -24]]
    # print('noise on cap')
    # acc_both = train(ret_ats=ret_ats,
    #                  channel_snrs=channel_snrs,
    #                  rand_valset=rand_valset,
    #                  log_dir=log_dir,
    #                  dataset_cfg=dataset_cfg,
    #                  seeds=seeds)
    #
    # for img_snr in [24, 18, 12, 6, 0, -6, -12, -18, -24]:
    #     channel_snrs = [[img_snr, 24], [img_snr, 18], [img_snr, 12], [img_snr, 6], [img_snr, 0], [img_snr, -6], [img_snr, -12], [img_snr, -18], [img_snr, -24]]
    #     print('img_snr=%d' % img_snr)
    #     acc_both = train(ret_ats=ret_ats,
    #                      channel_snrs=channel_snrs,
    #                      rand_valset=rand_valset,
    #                      log_dir=log_dir,
    #                      dataset_cfg=dataset_cfg,
    #                      seeds=seeds)

    # for cap_snr in [42, 36, 30, 24, 18, 12, 6, 0, -6, -12, -18, -24, -30, -36, -42]:
    #     print('cap_snr=%d' % cap_snr)
    #     channel_snrs = [[42, cap_snr], [36, cap_snr], [30, cap_snr], [24, cap_snr], [18, cap_snr], [12, cap_snr], [6, cap_snr], [0, cap_snr], [-6, cap_snr], [-12, cap_snr], [-18, cap_snr], [-24, cap_snr], [-30, cap_snr], [-36, cap_snr],
    #                     [-42, cap_snr]]
    #     acc_both = train(ret_ats=ret_ats,
    #                      channel_snrs=channel_snrs,
    #                      rand_valset=rand_valset,
    #                      log_dir=log_dir,
    #                      dataset_cfg=dataset_cfg)
