import torch
import os
import open_clip

import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import nn
from datetime import datetime
from datasets.mscoco import MSCOCODatasetRetFix
from configs.data import MSCOCOCfg
from utils.model_util import AutoEncoderChannel, save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader, random_split
from utils.utils import write_log, get_checkpoint_openclip, random_seed
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, SequentialLR, ConstantLR, CosineAnnealingWarmRestarts

'''
4 scenarios: training or testing with/without noise
'''
def val(ret_ats, channel_snrs, out_feature_dims=None,device='cuda', log_dir='', load_ckpt_path='', seeds=[1234, 2345, 3456, 4567, 5678]):
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs, without this setting, the cpu automatically choose tf16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # time stamper
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    # path to save logs and checkpoints
    plot_dir=log_dir+f'/dim{out_feature_dims[0][0]}_{out_feature_dims[1][0]}_'+date_str
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    log_path=plot_dir+'/log_result'+time_str+'.txt'

    # log running parameters
    write_log(f'training without noise, test with noise\n trained model path: {load_ckpt_path} \nchannel_snrs: {channel_snrs}, out_feature_dims: {out_feature_dims}, ', log_path)
    # init pretrained model
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', precision='fp32')
    model = get_checkpoint_openclip('C:/Users/INDA_HIWI/Desktop/plot_oc_flickr_train_ae/oc/coco_logs/2024_08_29-16_18_17-model_ViT-B-32-lr_1e-08-b_100-j_0-p_fp32/checkpoints/epoch_50.pt', model)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.to(device)

    # prepare dataset
    # log
    write_log(f'[{datetime.now().strftime("%H:%M:%S")}] loading dataset...... \n', log_path)
    val_dataset = MSCOCODatasetRetFix(img_dir=MSCOCOCfg.img_dir, ann_file=MSCOCOCfg.annotations_path_test, preprocess=preprocess)
    len_valset = val_dataset.__len__()

    # prepare train dataloader
    write_log(f'[{datetime.now().strftime("%H:%M:%S")}] preparing dataloader......\n', log_path)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_dataset.__len__())



    # accs to print
    acc_imgret_r1 = []
    acc_imgret_r5 = []
    acc_imgret_r10 = []
    acc_capret_r1 = []
    acc_capret_r5 = []
    acc_capret_r10 = []

    for channel_snr in channel_snrs:
        # log
        write_log(f'--------------------------channel_snr: {channel_snr}------------------------------\n',log_path)
        # init Autoencoder model
        assert out_feature_dims is not None, 'please either give the out_feature_dims for the AutoEncoder'
        ae_channel = AutoEncoderChannel(input_embed_dim=512, out_feature_dims=out_feature_dims, channel_snr=channel_snr, dtype=torch.float32, device=device, seeds=seeds)
        optimizer = optim.AdamW(ae_channel.parameters(), lr=1e-5, betas=(0.9, 0.99), eps=1e-6)
        load_checkpoint(ae_channel, optimizer, filename=load_ckpt_path)
        ae_channel.to(device)
        ################################# testing #################################
        val_acc_img = {}
        val_acc_cap = {}
        temp_cap_logit = {}
        for ret_at in ret_ats:
            val_acc_img[f'R@{ret_at}'] = 0
            val_acc_cap[f'R@{ret_at}'] = 0
            temp_cap_logit[f'R@{ret_at}'] = None

        for _, (images, captions) in enumerate(val_dataloader):
            # init
            img_num = len(images)
            images = images.to(torch.float32).to(device)

            for seed in seeds:
                for i in range(0, 5):
                    # collect caption sub batch
                    captions_batch = [captions[i][j] for j in range(0, img_num)]
                    # tokenize each caption
                    captions_tok = tokenizer(captions_batch).to(device)

                    with torch.no_grad():
                        # encode using pretrained open clip
                        image_features, text_features, logit_scale = model(images, captions_tok)

                        # transfer by autoencoder channel
                        x_img, x_cap = ae_channel(image_features, text_features, is_val=True, seed=seed)

                        # get val loss from logits
                        logit_scale = logit_scale.mean()
                        logits_per_image = logit_scale * x_img @ x_cap.T
                        logits_per_text = logits_per_image.T

                        # get val acc
                        groundtruth = torch.arange(img_num).to(device)
                        for ret_at in ret_ats:
                            max_img_index = torch.argsort(logits_per_text, dim=-1)[:, -ret_at:]
                            for i_ret in range(0, ret_at):
                                max_img_index[:, i_ret] = max_img_index[:, i_ret] - groundtruth
                            val_acc_img[f'R@{ret_at}'] += np.any(max_img_index.cpu().detach().numpy() == 0, axis=-1).sum()

                            # cat all cap digits
                            if temp_cap_logit[f'R@{ret_at}'] is None:
                                temp_cap_logit[f'R@{ret_at}'] = logits_per_image
                            else:
                                temp_cap_logit[f'R@{ret_at}'] = torch.cat((temp_cap_logit[f'R@{ret_at}'], logits_per_image), dim=-1)

                for ret_at in ret_ats:
                    temp_max_cap_index = torch.argsort(temp_cap_logit[f'R@{ret_at}'], dim=-1)[:, -ret_at:]
                    for i_ret in range(0, ret_at):
                        temp_max_cap_index[:, i_ret] = temp_max_cap_index[:, i_ret] % img_num - groundtruth
                    val_acc_cap[f'R@{ret_at}'] += np.any(temp_max_cap_index.cpu().detach().numpy() == 0, axis=-1).sum()
                    temp_cap_logit[f'R@{ret_at}'] = None

        acc_imgret_r1.append(val_acc_img['R@1'] / (len_valset * 5) * 100 / 5)
        acc_imgret_r5.append(val_acc_img['R@5'] / (len_valset * 5) * 100 / 5)
        acc_imgret_r10.append(val_acc_img['R@10'] / (len_valset * 5) * 100 / 5)
        acc_capret_r1.append(val_acc_cap['R@1'] / len_valset * 100 / 5)
        acc_capret_r5.append(val_acc_cap['R@5'] / len_valset * 100 / 5)
        acc_capret_r10.append(val_acc_cap['R@10'] / len_valset * 100 / 5)

    # print('[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]'
    #       % (acc_imgret_r1[0], acc_imgret_r1[1], acc_imgret_r1[2], acc_imgret_r1[3], acc_imgret_r1[4], acc_imgret_r1[5], acc_imgret_r1[6], acc_imgret_r1[7], acc_imgret_r1[8],
    #          acc_imgret_r1[9], acc_imgret_r1[10], acc_imgret_r1[11], acc_imgret_r1[12], acc_imgret_r1[13], acc_imgret_r1[14]))
    # print('[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]'
    #       % (acc_imgret_r5[0], acc_imgret_r5[1], acc_imgret_r5[2], acc_imgret_r5[3], acc_imgret_r5[4], acc_imgret_r5[5], acc_imgret_r5[6], acc_imgret_r5[7], acc_imgret_r5[8],
    #          acc_imgret_r5[9], acc_imgret_r5[10], acc_imgret_r5[11], acc_imgret_r5[12], acc_imgret_r5[13], acc_imgret_r5[14]))
    # print('[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]'
    #       % (acc_imgret_r10[0], acc_imgret_r10[1], acc_imgret_r10[2], acc_imgret_r10[3], acc_imgret_r10[4], acc_imgret_r10[5], acc_imgret_r10[6], acc_imgret_r10[7], acc_imgret_r10[8],
    #          acc_imgret_r10[9], acc_imgret_r10[10], acc_imgret_r10[11], acc_imgret_r10[12], acc_imgret_r10[13], acc_imgret_r10[14]))
    # print('[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]'
    #       % (acc_capret_r1[0], acc_capret_r1[1], acc_capret_r1[2], acc_capret_r1[3], acc_capret_r1[4], acc_capret_r1[5], acc_capret_r1[6], acc_capret_r1[7], acc_capret_r1[8],
    #          acc_capret_r1[9], acc_capret_r1[10], acc_capret_r1[11], acc_capret_r1[12], acc_capret_r1[13], acc_capret_r1[14]))
    # print('[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]'
    #       % (acc_capret_r5[0], acc_capret_r5[1], acc_capret_r5[2], acc_capret_r5[3], acc_capret_r5[4], acc_capret_r5[5], acc_capret_r5[6], acc_capret_r5[7], acc_capret_r5[8],
    #          acc_capret_r5[9], acc_capret_r5[10], acc_capret_r5[11], acc_capret_r5[12], acc_capret_r5[13], acc_capret_r5[14]))
    # print('[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]'
    #       % (acc_capret_r10[0], acc_capret_r10[1], acc_capret_r10[2], acc_capret_r10[3], acc_capret_r10[4], acc_capret_r10[5], acc_capret_r10[6], acc_capret_r10[7], acc_capret_r10[8],
    #          acc_capret_r10[9], acc_capret_r10[10], acc_capret_r10[11], acc_capret_r10[12], acc_capret_r10[13], acc_capret_r10[14]))

# ### S1, S2
#     print('[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]'
#           % (acc_imgret_r1[0], acc_imgret_r1[1], acc_imgret_r1[2], acc_imgret_r1[3], acc_imgret_r1[4], acc_imgret_r1[5], acc_imgret_r1[6]))
#     print('[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]'
#           % (acc_imgret_r5[0], acc_imgret_r5[1], acc_imgret_r5[2], acc_imgret_r5[3], acc_imgret_r5[4], acc_imgret_r5[5], acc_imgret_r5[6]))
#     print('[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]'
#           % (acc_imgret_r10[0], acc_imgret_r10[1], acc_imgret_r10[2], acc_imgret_r10[3], acc_imgret_r10[4], acc_imgret_r10[5], acc_imgret_r10[6]))
#     print('[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]'
#           % (acc_capret_r1[0], acc_capret_r1[1], acc_capret_r1[2], acc_capret_r1[3], acc_capret_r1[4], acc_capret_r1[5], acc_capret_r1[6]))
#     print('[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]'
#           % (acc_capret_r5[0], acc_capret_r5[1], acc_capret_r5[2], acc_capret_r5[3], acc_capret_r5[4], acc_capret_r5[5], acc_capret_r5[6]))
#     print('[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]'
#           % (acc_capret_r10[0], acc_capret_r10[1], acc_capret_r10[2], acc_capret_r10[3], acc_capret_r10[4], acc_capret_r10[5], acc_capret_r10[6]))

######## S4
    print('%.2f' % acc_imgret_r1[0])
    print('%.2f' % acc_imgret_r5[0])
    print('%.2f' % acc_imgret_r10[0])
    print('%.2f' % acc_capret_r1[0])
    print('%.2f' % acc_capret_r5[0])
    print('%.2f' % acc_capret_r10[0])

if __name__ == '__main__':
    ######################### set params #########################
    # channel_snrs = [[42, 42], [36, 36], [30, 30], [24, 24], [18, 18], [12, 12], [6, 6], [0, 0], [-6, -6], [-12, -12], [-18, -18], [-24, -24], [-30, -30], [-36, -36], [-42, -42]]
    # R@?
    ret_ats = [1, 5, 10]

    # seeds for generating noise
    seeds = [1234, 2345, 3456, 4567, 5678]

    # path to save the plots
    log_dir='C:/Users/INDA_HIWI/Desktop/plot_oc_flickr_train_ae/4_scenarios_training_coco/val'
    out_feature_dims = [[256], [256]]
    # coco
    # # S1
    # channel_snrs = [[18, 18], [12, 12], [6, 6], [0, 0], [-6, -6], [-12, -12], [-18, -18]]
    #
    # print('training snr None, batch size 100')
    # load_ckpt_path = 'C:/Users/INDA_HIWI/Desktop/logs_fix_val_noise/ckpt_coco/snrNone_None_epoch43.pth'
    # val(ret_ats=ret_ats, channel_snrs=channel_snrs, out_feature_dims=out_feature_dims, log_dir=log_dir, load_ckpt_path=load_ckpt_path, seeds=seeds)
    #
    # # s2
    # channel_snrs = [[18, 18], [12, 12], [6, 6], [0, 0], [-6, -6], [-12, -12], [-18, -18]]
    #
    # print('training with snr 18')
    # load_ckpt_path = 'C:/Users/INDA_HIWI/Desktop/logs_fix_val_noise/ckpt_coco/snr18_18_epoch36.pth'
    # val(ret_ats=ret_ats, channel_snrs=channel_snrs, out_feature_dims=out_feature_dims, log_dir=log_dir, load_ckpt_path=load_ckpt_path, seeds=seeds)
    #
    # print('training with snr 12')
    # load_ckpt_path = 'C:/Users/INDA_HIWI/Desktop/logs_fix_val_noise/ckpt_coco/snr12_12_epoch36.pth'
    # val(ret_ats=ret_ats, channel_snrs=channel_snrs, out_feature_dims=out_feature_dims, log_dir=log_dir, load_ckpt_path=load_ckpt_path, seeds=seeds)
    #
    # print('training with snr 6')
    # load_ckpt_path = 'C:/Users/INDA_HIWI/Desktop/logs_fix_val_noise/ckpt_coco/snr6_6_epoch49.pth'
    # val(ret_ats=ret_ats, channel_snrs=channel_snrs, out_feature_dims=out_feature_dims, log_dir=log_dir, load_ckpt_path=load_ckpt_path, seeds=seeds)
    #
    # print('training with snr 0')
    # load_ckpt_path = 'C:/Users/INDA_HIWI/Desktop/logs_fix_val_noise/ckpt_coco/snr0_0_epoch49.pth'
    # val(ret_ats=ret_ats, channel_snrs=channel_snrs, out_feature_dims=out_feature_dims, log_dir=log_dir, load_ckpt_path=load_ckpt_path, seeds=seeds)
    #
    # print('training with snr -6')
    # load_ckpt_path = 'C:/Users/INDA_HIWI/Desktop/logs_fix_val_noise/ckpt_coco/snr-6_-6_epoch47.pth'
    # val(ret_ats=ret_ats, channel_snrs=channel_snrs, out_feature_dims=out_feature_dims, log_dir=log_dir, load_ckpt_path=load_ckpt_path, seeds=seeds)
    #
    # print('training with snr -12')
    # load_ckpt_path = 'C:/Users/INDA_HIWI/Desktop/logs_fix_val_noise/ckpt_coco/snr-12_-12_epoch34.pth'
    # val(ret_ats=ret_ats, channel_snrs=channel_snrs, out_feature_dims=out_feature_dims, log_dir=log_dir, load_ckpt_path=load_ckpt_path, seeds=seeds)
    #
    # print('training with snr -18')
    # load_ckpt_path = 'C:/Users/INDA_HIWI/Desktop/logs_fix_val_noise/ckpt_coco/snr-18_-18_epoch24.pth'
    # val(ret_ats=ret_ats, channel_snrs=channel_snrs, out_feature_dims=out_feature_dims, log_dir=log_dir, load_ckpt_path=load_ckpt_path, seeds=seeds)

    # s4
    channel_snrs = [[None, None]]

    print('training with snr 18')
    load_ckpt_path = 'C:/Users/INDA_HIWI/Desktop/logs_fix_val_noise/ckpt_coco/snr18_18_epoch36.pth'
    val(ret_ats=ret_ats, channel_snrs=channel_snrs, out_feature_dims=out_feature_dims, log_dir=log_dir, load_ckpt_path=load_ckpt_path, seeds=seeds)

    print('training with snr 12')
    load_ckpt_path = 'C:/Users/INDA_HIWI/Desktop/logs_fix_val_noise/ckpt_coco/snr12_12_epoch36.pth'
    val(ret_ats=ret_ats, channel_snrs=channel_snrs, out_feature_dims=out_feature_dims, log_dir=log_dir, load_ckpt_path=load_ckpt_path, seeds=seeds)

    print('training with snr 6')
    load_ckpt_path = 'C:/Users/INDA_HIWI/Desktop/logs_fix_val_noise/ckpt_coco/snr6_6_epoch49.pth'
    val(ret_ats=ret_ats, channel_snrs=channel_snrs, out_feature_dims=out_feature_dims, log_dir=log_dir, load_ckpt_path=load_ckpt_path, seeds=seeds)

    print('training with snr 0')
    load_ckpt_path = 'C:/Users/INDA_HIWI/Desktop/logs_fix_val_noise/ckpt_coco/snr0_0_epoch49.pth'
    val(ret_ats=ret_ats, channel_snrs=channel_snrs, out_feature_dims=out_feature_dims, log_dir=log_dir, load_ckpt_path=load_ckpt_path, seeds=seeds)

    print('training with snr -6')
    load_ckpt_path = 'C:/Users/INDA_HIWI/Desktop/logs_fix_val_noise/ckpt_coco/snr-6_-6_epoch47.pth'
    val(ret_ats=ret_ats, channel_snrs=channel_snrs, out_feature_dims=out_feature_dims, log_dir=log_dir, load_ckpt_path=load_ckpt_path, seeds=seeds)

    print('training with snr -12')
    load_ckpt_path = 'C:/Users/INDA_HIWI/Desktop/logs_fix_val_noise/ckpt_coco/snr-12_-12_epoch34.pth'
    val(ret_ats=ret_ats, channel_snrs=channel_snrs, out_feature_dims=out_feature_dims, log_dir=log_dir, load_ckpt_path=load_ckpt_path, seeds=seeds)

    print('training with snr -18')
    load_ckpt_path = 'C:/Users/INDA_HIWI/Desktop/logs_fix_val_noise/ckpt_coco/snr-18_-18_epoch24.pth'
    val(ret_ats=ret_ats, channel_snrs=channel_snrs, out_feature_dims=out_feature_dims, log_dir=log_dir, load_ckpt_path=load_ckpt_path, seeds=seeds)




