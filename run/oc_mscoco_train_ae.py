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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

'''training on AutoEncoder'''
def train(ret_ats,
          channel_snrs,
          out_feature_dims=None,
          rand_valset=False,
          epochs=50,
          batch_size=50,
          lr=1e-5,
          device='cuda',
          log_dir='C:/Users/INDA_HIWI/Desktop/',
          break_loss_th=0.1,
          resume=False,
          load_ckpt_path = '',
          resume_at_snr = None,
          save_ckpt_freq = None,
          seeds = None):


    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs, without this setting, the cpu automatically choose tf16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    random_seed(0, 0)


    # time stamper
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    # path to save logs and checkpoints
    plot_dir=log_dir+f'/dim{out_feature_dims[0][0]}_{out_feature_dims[1][0]}_'+date_str
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    log_path=plot_dir+'/log_result'+time_str+'.txt'
    if save_ckpt_freq is not None:
        save_ckpt_path=plot_dir+'/ckpt'+time_str
        if not os.path.exists(save_ckpt_path):
            os.makedirs(save_ckpt_path)

    # log running parameters
    write_log(f'channel_snrs: {channel_snrs}, out_feature_dims: {out_feature_dims}, '
              f'batch_size: {batch_size}, lr: {lr} \n', log_path)

    # init pretrained model
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', precision='fp32')
    model = get_checkpoint_openclip('C:/Users/INDA_HIWI/Desktop/plot_oc_flickr_train_ae/oc/coco_logs/2024_08_29-16_18_17-model_ViT-B-32-lr_1e-08-b_100-j_0-p_fp32/checkpoints/epoch_50.pt', model)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.to(device)

    # prepare dataset
    # log
    write_log(f'[{datetime.now().strftime("%H:%M:%S")}] loading dataset...... \n', log_path)

    val_dataset = MSCOCODatasetRetFix(img_dir=MSCOCOCfg.img_dir, ann_file=MSCOCOCfg.annotations_path_test, preprocess=preprocess)
    train_dataset = MSCOCODatasetRetFix(img_dir=MSCOCOCfg.img_dir, ann_file=MSCOCOCfg.annotations_path_train, preprocess=preprocess)
    len_trainset = train_dataset.__len__()
    len_valset = val_dataset.__len__()

    # prepare train dataloader
    write_log(f'[{datetime.now().strftime("%H:%M:%S")}] preparing dataloader......\n', log_path)
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler_train)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_dataset.__len__())

    for channel_snr in channel_snrs:
        # log
        write_log(f'--------------------------channel_snr: {channel_snr}------------------------------\n',log_path)
        # init Autoencoder model
        assert out_feature_dims is not None, 'please either give the out_feature_dims for the AutoEncoder'
        ae_channel = AutoEncoderChannel(input_embed_dim=512, out_feature_dims=out_feature_dims,
                                        channel_snr=channel_snr, dtype=torch.float32, device=device, seeds=seeds)
        ae_channel.to(device)
        optimizer = optim.AdamW(ae_channel.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-6)
        scheduler = None
        criterion = nn.CrossEntropyLoss()

        # load checkpoint
        resume_flag=False
        if resume:
            assert resume_at_snr is not None, 'resume_at_snr should be given'
            assert load_ckpt_path != '', 'checkpoint path should not be empty string, please give the ckpt_path'
            assert os.path.exists(load_ckpt_path), 'checkpoint path not exist'
            if channel_snr == resume_at_snr:
                epoch_start, train_loss_plot, val_loss_plot, train_accs_plot_img, train_accs_plot_cap, val_accs_plot_img, val_accs_plot_cap = load_checkpoint(ae_channel, optimizer, load_ckpt_path)
                resume_flag = True
        if not resume_flag: # no checkpoint is loaded, init the variables
            epoch_start = 0
            # init list for plot
            train_accs_plot_img, train_accs_plot_cap = {}, {}
            val_accs_plot_img, val_accs_plot_cap = {}, {}
            for ret_at in ret_ats:
                train_accs_plot_img[f'R@{ret_at}'], train_accs_plot_cap[f'R@{ret_at}'] = [], []
                val_accs_plot_img[f'R@{ret_at}'], val_accs_plot_cap[f'R@{ret_at}'] = [], []
            train_loss_plot, val_loss_plot = [], []

        write_log(f'[{datetime.now().strftime("%H:%M:%S")}] start training......\nstart epoch:{epoch_start}\n', log_path)
        # dec_lr_time = 0
        # dec_lr_epoch = 0
        for epoch in range(epoch_start, epochs):
            ################################# training #################################
            # init
            # lr schedule
            if epoch == 4:
                optimizer = optim.AdamW(ae_channel.parameters(), lr=lr / 4, betas=(0.9, 0.99), eps=1e-6)
                scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=1e-6)
            if epoch == 15:
                # optimizer = optim.AdamW(ae_channel.parameters(), lr=1e-7, betas=(0.9, 0.99), eps=1e-6)
                optimizer = optim.AdamW(ae_channel.parameters(), lr=2e-7, betas=(0.9, 0.99), eps=1e-6)
                scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=2e-8)
            train_loss = 0
            train_acc_img, train_acc_cap = {}, {}
            temp_cap_logit = {}
            for ret_at in ret_ats:
                train_acc_img[f'R@{ret_at}'], train_acc_cap[f'R@{ret_at}'] = 0, 0
                temp_cap_logit[f'R@{ret_at}'] = None
            # set model in train mode
            ae_channel.train(True)
            for idx, (images, captions) in enumerate(train_dataloader):
                # init
                img_num = len(images) # img_num != batch_size in the last batch
                images = images.to(torch.float32).to(device)

                # target
                target = torch.from_numpy(np.eye(img_num, dtype=int)).type(torch.float32).to(device)

                for i in range(0, 5):
                    # collect caption sub batch
                    captions_batch=[captions[i][j] for j in range(0, img_num)]

                    # tokenize each caption
                    captions_tok=tokenizer(captions_batch).to(device)

                    # encode using pretrained open clip
                    with torch.no_grad():
                        image_features, text_features, logit_scale = model(images, captions_tok)

                    # transfer by autoencoder channel
                    x_img, x_cap = ae_channel(image_features, text_features)

                    # get loss from logits
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * x_img @ x_cap.T
                    logits_per_text = logits_per_image.T

                    loss = criterion(logits_per_image, target) + criterion(logits_per_text, target)
                    train_loss += loss.item()

                    # update parameters
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


                    # get acc
                    groundtruth = torch.arange(img_num).to(device)
                    for ret_at in ret_ats:
                        max_img_index = torch.argsort(logits_per_text, dim=-1)[:, -ret_at:]
                        for i_r in range(0, ret_at):
                            max_img_index[:, i_r] = max_img_index[:, i_r] - groundtruth
                        train_acc_img[f'R@{ret_at}'] += np.any(max_img_index.cpu().detach().numpy() == 0, axis=-1).sum()

                        # cat all cap digits
                        if temp_cap_logit[f'R@{ret_at}'] is None:
                            temp_cap_logit[f'R@{ret_at}'] = logits_per_image
                        else:
                            temp_cap_logit[f'R@{ret_at}'] = torch.cat((temp_cap_logit[f'R@{ret_at}'], logits_per_image), dim=-1)

                for ret_at in ret_ats:
                    temp_max_cap_index = torch.argsort(temp_cap_logit[f'R@{ret_at}'], dim=-1)[:, -ret_at:]
                    for idx_r in range(0, ret_at):
                        temp_max_cap_index[:, idx_r] = temp_max_cap_index[:, idx_r] % img_num - groundtruth
                    train_acc_cap[f'R@{ret_at}'] += np.any(temp_max_cap_index.cpu().detach().numpy() == 0, axis=-1).sum()
                    temp_cap_logit[f'R@{ret_at}'] = None
            if epoch >= 4:
                scheduler.step()
                current_lr = scheduler.get_last_lr()
            else:
                current_lr = [lr]
            # end for training

            # print and log result

            print('[train epoch %d], [train_loss: %f], [lr: %.4e]' % (epoch, train_loss, current_lr[0]))


            # save data for plot
            for ret_at in ret_ats:
                train_accs_plot_img[f'R@{ret_at}'].append(train_acc_img[f'R@{ret_at}'] / (len_trainset * 5) * 100)
                train_accs_plot_cap[f'R@{ret_at}'].append(train_acc_cap[f'R@{ret_at}'] / len_trainset * 100)
            train_loss_plot.append(train_loss)


            ################################# testing #################################
            # init
            val_loss = 0
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

                # target
                target = torch.from_numpy(np.eye(img_num, dtype=int)).type(torch.float32).to(device)

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
                            x_img, x_cap = ae_channel(image_features, text_features, is_val=True, seed=seed, cap_idx=i)

                            # get val loss from logits
                            logit_scale = logit_scale.mean()
                            logits_per_image = logit_scale * x_img @ x_cap.T
                            logits_per_text = logits_per_image.T

                            loss = criterion(logits_per_image, target) + criterion(logits_per_text, target)
                            val_loss += loss.item() * len_trainset / len_valset

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

            # print and log result
            print('img acc: [R@1:%d/%d, %.2f%%], [R@5:%d/%d, %.2f%%], [R@10:%d/%d, %.2f%%]  '
                  'cap acc sub: [R@1:%d/%d, %.2f%%],[R@5:%d/%d, %.2f%%],[R@10:%d/%d, %.2f%%], '
                  'val loss: %f'
                  % (val_acc_img['R@1'], (len_valset * 5), val_acc_img['R@1'] / (len_valset * 5) * 100 / 5,
                     val_acc_img['R@5'], (len_valset * 5), val_acc_img['R@5'] / (len_valset * 5) * 100 / 5,
                     val_acc_img['R@10'], (len_valset * 5), val_acc_img['R@10'] / (len_valset * 5) * 100 / 5,
                     val_acc_cap['R@1'], len_valset, val_acc_cap['R@1'] / len_valset * 100 / 5,
                     val_acc_cap['R@5'], len_valset, val_acc_cap['R@5'] / len_valset * 100 / 5,
                     val_acc_cap['R@10'], len_valset, val_acc_cap['R@10'] / len_valset * 100 / 5, val_loss))

            write_log('[%s] [train epoch %d]\n'
                      'img acc train: [R@1:%d/%d, %.2f%%], [R@5:%d/%d, %.2f%%], [R@10:%d/%d, %.2f%%] '
                      'cap acc train: [R@1:%d/%d, %.2f%%],[R@5:%d/%d, %.2f%%],[R@10:%d/%d, %.2f%%]\n'
                      'img acc val: [R@1:%d/%d, %.2f%%], [R@5:%d/%d, %.2f%%], [R@10:%d/%d, %.2f%%] '
                      'cap acc val: [R@1:%d/%d, %.2f%%],[R@5:%d/%d, %.2f%%],[R@10:%d/%d, %.2f%%]\n'
                      '[lr: %.4e], [train_loss: %f], val loss: %f\n\n'
                      % (datetime.now().strftime("%H:%M:%S"), epoch,
                         train_acc_img['R@1'], (len_trainset * 5), train_acc_img['R@1'] / (len_trainset * 5) * 100,
                         train_acc_img['R@5'], (len_trainset * 5), train_acc_img['R@5'] / (len_trainset * 5) * 100,
                         train_acc_img['R@10'], (len_trainset * 5), train_acc_img['R@10'] / (len_trainset * 5) * 100,
                         train_acc_cap['R@1'], len_trainset, train_acc_cap['R@1'] / len_trainset * 100,
                         train_acc_cap['R@5'], len_trainset, train_acc_cap['R@5'] / len_trainset * 100,
                         train_acc_cap['R@10'], len_trainset, train_acc_cap['R@10'] / len_trainset * 100,
                         val_acc_img['R@1'], (len_valset * 5 * 5), val_acc_img['R@1'] / (len_valset * 5) * 100 / 5,
                         val_acc_img['R@5'], (len_valset * 5 * 5), val_acc_img['R@5'] / (len_valset * 5) * 100 / 5,
                         val_acc_img['R@10'], (len_valset * 5 * 5), val_acc_img['R@10'] / (len_valset * 5) * 100 / 5,
                         val_acc_cap['R@1'], len_valset * 5, val_acc_cap['R@1'] / len_valset * 100 / 5,
                         val_acc_cap['R@5'], len_valset * 5, val_acc_cap['R@5'] / len_valset * 100 / 5,
                         val_acc_cap['R@10'], len_valset * 5, val_acc_cap['R@10'] / len_valset * 100 / 5,
                         current_lr[0], train_loss, val_loss),
                      log_path)

            # save data for plot
            val_loss_plot.append(val_loss)
            for ret_at in ret_ats:
                val_accs_plot_img[f'R@{ret_at}'].append(val_acc_img[f'R@{ret_at}'] / (len_valset * 5) * 100 / 5)
                val_accs_plot_cap[f'R@{ret_at}'].append(val_acc_cap[f'R@{ret_at}'] / len_valset * 100 / 5)

            # save checkpoints
            # save_checkpoint(ae_channel, optimizer, epoch, train_loss_plot, val_loss_plot, channel_snr, train_accs_plot_img, train_accs_plot_cap, val_accs_plot_img, val_accs_plot_cap, save_ckpt_path)
            # if save_ckpt_freq is not None and (epoch + 1) % save_ckpt_freq == 0:
            # if epoch > 12:
            #     save_checkpoint(ae_channel, optimizer, epoch, train_loss_plot, val_loss_plot, channel_snr, train_accs_plot_img, train_accs_plot_cap, val_accs_plot_img, val_accs_plot_cap, save_ckpt_path)


            if (epoch > 12 and sum(val_loss_plot[-10:]) / 10 - sum(val_loss_plot[-5:]) / 5 < break_loss_th) or (epoch == epochs - 1):
                min_loss = min(val_loss_plot)
                min_loss_idx = val_loss_plot.index(min_loss)
                print(f"dims: {out_feature_dims[0]}, {out_feature_dims[1]}, snr: {channel_snr[0]}, {channel_snr[1]} \n"
                      f"img r1 {val_accs_plot_img['R@1'][min_loss_idx]:.2f}, r5 {val_accs_plot_img['R@5'][min_loss_idx]:.2f}, r10 {val_accs_plot_img['R@10'][min_loss_idx]:.2f}, "
                      f"cap r1 {val_accs_plot_cap['R@1'][min_loss_idx]:.2f}, r5 {val_accs_plot_cap['R@5'][min_loss_idx]:.2f}, r10 {val_accs_plot_cap['R@10'][min_loss_idx]:.2f}")
                write_log(f"dims: {out_feature_dims[0]}, {out_feature_dims[1]}, snr: {channel_snr[0]}, {channel_snr[1]}"
                          f"img r1 {val_accs_plot_img['R@1'][min_loss_idx]:.2f}, r5 {val_accs_plot_img['R@5'][min_loss_idx]:.2f}, r10 {val_accs_plot_img['R@10'][min_loss_idx]:.2f}, "
                          f"cap r1 {val_accs_plot_cap['R@1'][min_loss_idx]:.2f}, r5 {val_accs_plot_cap['R@5'][min_loss_idx]:.2f}, r10 {val_accs_plot_cap['R@10'][min_loss_idx]:.2f}\n",
                          'C:/Users/INDA_HIWI/Desktop/temp_result1.txt')

                fig, ax = plt.subplots(nrows=2, ncols=3)
                fig.set_size_inches(27, 18)
                ax[0, 0].plot(np.arange(1, len(train_accs_plot_img['R@1']) + 1), train_accs_plot_img['R@1'], 'b', label='train img')
                ax[0, 0].plot(np.arange(1, len(val_accs_plot_img['R@1']) + 1), val_accs_plot_img['R@1'], 'r', label='val img')
                ax[0, 0].set_xlabel('Epoch')
                ax[0, 0].set_ylabel('acc/%')
                ax[0, 0].set_title('acc image retrieval R@1')
                ax[0, 0].legend(loc='lower right', fontsize='small')
                ax[0, 0].grid()

                ax[1, 0].plot(np.arange(1, len(val_accs_plot_cap['R@1']) + 1), val_accs_plot_cap['R@1'], 'r', label='val cap')
                ax[1, 0].plot(np.arange(1, len(train_accs_plot_cap['R@1']) + 1), train_accs_plot_cap['R@1'], 'b', label='train cap')
                ax[1, 0].set_xlabel('Epoch')
                ax[1, 0].set_ylabel('acc/%')
                ax[1, 0].set_title('acc caption retrieval R@1')
                ax[1, 0].legend(loc='lower right', fontsize='small')
                ax[1, 0].grid()

                ax[0, 1].plot(np.arange(1, len(train_accs_plot_img['R@5']) + 1), train_accs_plot_img['R@5'], 'b', label='train img')
                ax[0, 1].plot(np.arange(1, len(val_accs_plot_img['R@5']) + 1), val_accs_plot_img['R@5'], 'r', label='val img')
                ax[0, 1].set_xlabel('Epoch')
                ax[0, 1].set_ylabel('acc/%')
                ax[0, 1].set_title('acc image retrieval R@5')
                ax[0, 1].legend(loc='lower right', fontsize='small')
                ax[0, 1].grid()

                ax[1, 1].plot(np.arange(1, len(val_accs_plot_cap['R@5']) + 1), val_accs_plot_cap['R@5'], 'r', label='val cap')
                ax[1, 1].plot(np.arange(1, len(train_accs_plot_cap['R@5']) + 1), train_accs_plot_cap['R@5'], 'b', label='train cap')
                ax[1, 1].set_xlabel('Epoch')
                ax[1, 1].set_ylabel('acc/%')
                ax[1, 1].set_title('acc caption retrieval R@5')
                ax[1, 1].legend(loc='lower right', fontsize='small')
                ax[1, 1].grid()

                ax[0, 2].plot(np.arange(1, len(train_accs_plot_img['R@10']) + 1), train_accs_plot_img['R@10'], 'b', label='train img')
                ax[0, 2].plot(np.arange(1, len(val_accs_plot_img['R@10']) + 1), val_accs_plot_img['R@10'], 'r', label='val img')
                ax[0, 2].set_xlabel('Epoch')
                ax[0, 2].set_ylabel('acc/%')
                ax[0, 2].set_title('acc image retrieval R@10')
                ax[0, 2].legend(loc='lower right', fontsize='small')
                ax[0, 2].grid()

                ax[1, 2].plot(np.arange(1, len(val_accs_plot_cap['R@10']) + 1), val_accs_plot_cap['R@10'], 'r', label='val cap')
                ax[1, 2].plot(np.arange(1, len(train_accs_plot_cap['R@10']) + 1), train_accs_plot_cap['R@10'], 'b', label='train cap')
                ax[1, 2].set_xlabel('Epoch')
                ax[1, 2].set_ylabel('acc/%')
                ax[1, 2].set_title('acc caption retrieval R@10')
                ax[1, 2].legend(loc='lower right', fontsize='small')
                ax[1, 2].grid()
                plt.savefig(plot_dir + f'/accs_snr{channel_snr[0]}_{channel_snr[1]}_{time_str}.jpg')
                plt.close(fig)

                fig_loss, ax_loss = plt.subplots()
                fig_loss.set_size_inches(18, 18)
                plt.plot(np.arange(1, len(val_loss_plot) + 1), val_loss_plot, 'r', label='val loss')
                plt.plot(np.arange(1, len(train_loss_plot) + 1), train_loss_plot, 'b', label='train loss')
                plt.xlabel('Epoch')
                plt.ylabel('loss')
                plt.grid()
                plt.savefig(plot_dir + f'/loss_snr{channel_snr[0]}_{channel_snr[1]}_{time_str}.jpg')
                plt.close(fig_loss)
                break
            # end for epoch
        # end for snr

    return 0

if __name__ == '__main__':
    ######################### set params #########################
    # R@?
    ret_ats = [1, 5, 10]

    # seeds for generating noise
    seeds = [1234, 2345, 3456, 4567, 5678]

    # set snr, SNR=None if not add noise, [snr_img, snr_cap]
    # img_snr = 6
    # cap_snr = 6
    # channel_snrs = [[24, cap_snr], [18, cap_snr], [12, cap_snr], [6, cap_snr], [0, cap_snr], [-6, cap_snr], [-12, cap_snr], [-18, cap_snr], [-24, cap_snr], [-30, cap_snr], [-36, cap_snr], [-42, cap_snr]]
    # channel_snrs = [[img_snr, 24], [img_snr, 18], [img_snr, 12], [img_snr, 6], [img_snr, 0], [img_snr, -6], [img_snr, -12], [img_snr, -18], [img_snr, -24], [img_snr, -30], [img_snr, -36], [img_snr, -42]]
    # channel_snrs =  [[None, None], [24, 24], [18, 18], [12, 12], [6, 6], [0, 0], [-6, -6], [-12, -12], [-18, -18], [-24, -24], [-30, -30], [-36, -36], [-42, -42]]
    # channel_snrs = [[None, None], [24, None], [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None], [-24, None], [-30, None], [-36, None], [-42, None]]
    # channel_snrs = [[None, None], [None, 24], [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18], [None, -24], [None, -30], [None, -36], [None, -42]]


    # random split testing set from the dataset: True or use the testing set from the paper: False
    rand_valset = False

    # path to save the plots
    log_dir='C:/Users/INDA_HIWI/Desktop/logs_fix_val_noise/coco_ft_oc_train_AE'

    # output shape of every AutoEncoder Layers, out_feature_dims = [img_dim, cap_dim]
    # e.g.: out_feature_dims = [[256, 128, 32], [64, 16]],
    # the AutoEncoder structure for image is input_shape->256->128->32->communicational transmission->32->128->256->input_shape,
    # the AutoEncoder structure for caption is input_shape->64->16->communicational transmission->16->64->input_shape
    # out_feature_dims=[[256], [256]]

    # early stop conditions
    break_loss_th = 0.5

    # hyperparam
    epochs=50
    batch_size=100
    assert batch_size >= max(ret_ats), 'batch size must larger than N in R@N'
    lr=4*4e-5

    # if resume
    resume=False
    resume_at_snr=[None, None]
    load_ckpt_path = 'C:/Users/INDA_HIWI/Desktop/plot_oc_flickr_train_ae/pretrained_oc_train_ae/dim256_256_20240604/ckpt1202/snrNone_None_epoch9.pth'
    # if resume:
    #     assert resume_at_snr is not None, 'resume_at_snr should be given'
    #     assert load_ckpt_path != '', 'checkpoint path should not be empty string, please give the ckpt_path'
    #     assert os.path.exists(load_ckpt_path), 'checkpoint path not exist'

    # if not save checkpoints, save_ckpt_freq = None
    save_ckpt_freq = None



    ######################################################################################################################################################################
    # [24, 24], [18, 18], [12, 12], [6, 6], [0, 0], [-6, -6][-18, -18], [-24, -24]#
    # out_feature_dims = [[256], [32]]
    # channel_snrs = [[None, -18]]
    # acc_both = train(ret_ats=ret_ats,
    #                  channel_snrs=channel_snrs,
    #                  out_feature_dims=out_feature_dims,
    #                  rand_valset=rand_valset,
    #                  epochs=epochs,
    #                  batch_size=batch_size,
    #                  lr=lr,
    #                  log_dir=log_dir,
    #                  break_loss_th=break_loss_th,
    #                  resume=resume,
    #                  load_ckpt_path=load_ckpt_path,
    #                  resume_at_snr=resume_at_snr,
    #                  save_ckpt_freq=save_ckpt_freq,
    #                  seeds=seeds)
    out_feature_dims = [[256], [16]]
    channel_snrs = [[None, None],
                    [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None],
                    [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18]]
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq,
                     seeds=seeds)

    out_feature_dims = [[128], [256]]
    channel_snrs = [[None, None],
                    [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None],
                    [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18]]
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq,
                     seeds=seeds)
    out_feature_dims = [[128], [128]]
    channel_snrs = [[None, None],
                    [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None],
                    [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18]]
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq,
                     seeds=seeds)
    out_feature_dims = [[128], [64]]
    channel_snrs = [[None, None],
                    [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None],
                    [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18]]
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq,
                     seeds=seeds)
    out_feature_dims = [[128], [32]]
    channel_snrs = [[None, None],
                    [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None],
                    [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18]]
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq,
                     seeds=seeds)
    out_feature_dims = [[128], [16]]
    channel_snrs = [[None, None],
                    [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None],
                    [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18]]
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq,
                     seeds=seeds)

    out_feature_dims = [[64], [256]]
    channel_snrs = [[None, None],
                    [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None],
                    [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18]]
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq,
                     seeds=seeds)
    out_feature_dims = [[64], [128]]
    channel_snrs = [[None, None],
                    [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None],
                    [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18]]
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq,
                     seeds=seeds)
    out_feature_dims = [[64], [64]]
    channel_snrs = [[None, None],
                    [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None],
                    [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18]]
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq,
                     seeds=seeds)
    out_feature_dims = [[64], [32]]
    channel_snrs = [[None, None],
                    [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None],
                    [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18]]
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq,
                     seeds=seeds)
    out_feature_dims = [[64], [16]]
    channel_snrs = [[None, None],
                    [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None],
                    [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18]]
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq,
                     seeds=seeds)

    out_feature_dims = [[32], [256]]
    channel_snrs = [[None, None],
                    [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None],
                    [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18]]
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq,
                     seeds=seeds)
    out_feature_dims = [[32], [128]]
    channel_snrs = [[None, None],
                    [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None],
                    [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18]]
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq,
                     seeds=seeds)
    out_feature_dims = [[32], [64]]
    channel_snrs = [[None, None],
                    [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None],
                    [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18]]
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq,
                     seeds=seeds)
    out_feature_dims = [[32], [32]]
    channel_snrs = [[None, None],
                    [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None],
                    [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18]]
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq,
                     seeds=seeds)
    out_feature_dims = [[32], [16]]
    channel_snrs = [[None, None],
                    [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None],
                    [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18]]
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq,
                     seeds=seeds)

    out_feature_dims = [[16], [256]]
    channel_snrs = [[None, None],
                    [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None],
                    [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18]]
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq,
                     seeds=seeds)
    out_feature_dims = [[16], [128]]
    channel_snrs = [[None, None],
                    [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None],
                    [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18]]
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq,
                     seeds=seeds)
    out_feature_dims = [[16], [64]]
    channel_snrs = [[None, None],
                    [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None],
                    [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18]]
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq,
                     seeds=seeds)
    out_feature_dims = [[16], [32]]
    channel_snrs = [[None, None],
                    [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None],
                    [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18]]
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq,
                     seeds=seeds)
    out_feature_dims = [[16], [16]]
    channel_snrs = [[None, None],
                    [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None],
                    [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18]]
    acc_both = train(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq,
                     seeds=seeds)


    # channel_snrs = [[30, 30], [30, 24], [30, 18], [30, 12], [30, 6], [30, 0], [30, -6], [30, -12], [30, -18], [30, -24],
    #                 [24, 30], [24, 24], [24, 18], [24, 12], [24, 6], [24, 0], [24, -6], [24, -12], [24, -18], [24, -24],
    #                 [18, 30], [18, 24], [12, 30], [12, 24], [6, 30], [6, 24], [0, 30], [0, 24], [-6, 30], [-6, 24], [-12, 30], [-12, 24], [-18, 30], [-18, 24],
    #                 [-24, 30], [-24, 24], [-24, 18], [-24, 12], [-24, 6], [-24, 0], [-24, -6], [-24, -12], [-24, -18], [-24, -24],
    #                 [18, -24], [12, -24], [6, -24], [0, -24], [-6, -24], [-12, -24], [-18, -24]]
    # channel_snrs = [[30, 30], [30, 24], [30, 18], [30, 12], [30, 6], [30, 0], [30, -6], [30, -12], [30, -18], [30, -24],
    #                 [24, 30], [24, 24], [24, 18], [24, 12], [24, 6], [24, 0], [24, -6], [24, -12], [24, -18], [24, -24],
    #                 [18, 30], [18, 24], [12, 30], [12, 24], [6, 30], [6, 24], [0, 30], [0, 24], [-6, 30], [-6, 24], [-12, 30], [-12, 24], [-18, 30], [-18, 24],
    #                 [-24, 30], [-24, 24], [-24, 18], [-24, 12], [-24, 6], [-24, 0], [-24, -6], [-24, -12], [-24, -18], [-24, -24],
    #                 [18, -24], [12, -24], [6, -24], [0, -24], [-6, -24], [-12, -24], [-18, -24]]
    ######################################################################################################################################################################
# [[256], [256]]
# [[128], [128]]
# [[64], [64]]
# [[32], [32]]
# [[16], [16]]
# channel_snrs = [[18, 18], [18, 12], [18, 6], [18, 0], [18, -6], [18, -12], [18, -18], [12, 18], [12, 12], [12, 6], [12, 0], [12, -6], [12, -12], [12, -18],
#                 [6, 18], [6, 12], [6, 6], [6, 0], [6, -6], [6, -12], [6, -18], [0, 18], [0, 12], [0, 6], [0, 0], [0, -6], [0, -12], [0, -18],
#                 [-6, 18], [-6, 12], [-6, 6], [-6, 0], [-6, -6], [-6, -12], [-6, -18], [-12, 18], [-12, 12], [-12, 6], [-12, 0], [-12, -6], [-12, -12], [-12, -18],
#                 [-18, 18], [-18, 12], [-18, 6], [-18, 0], [-18, -6], [-18, -12], [-18, -18]]
# channel_snrs = [[18, 12], [18, 6], [18, 0], [18, -6], [18, -12], [18, -18], [18, -24], [12, 30], [12, 24], [12, 18], [12, 6], [12, 0], [12, -6], [12, -12], [12, -18], [12, -24],
#                 [6, 30], [6, 24], [6, 18], [6, 12], [6, 0], [6, -6], [6, -12], [6, -18], [6, -24], [0, 30], [0, 24], [0, 18], [0, 12], [0, 6], [0, -6], [0, -12], [0, -18], [0, -24],
#                 [-6, 30], [-6, 24], [-6, 18], [-6, 12], [-6, 6], [-6, 0], [-6, -12], [-6, -18], [-6, -24], [-12, 30], [-12, 24], [-12, 18], [-12, 12], [-12, 6], [-12, 0], [-12, -6], [-12, -18], [-12, -24],
#                 [-18, 30], [-18, 24], [-18, 18], [-18, 12], [-18, 6], [-18, 0], [-18, -6], [-18, -12], [-18, -24], [-24, 30], [-24, 24], [-24, 18], [-24, 12], [-24, 6], [-24, 0], [-24, -6], [-24, -12], [-24, -18]]
# [[256], [128]]
# [[128], [256]]
# [[256], [64]]
# [[64], [256]]
# [[256], [32]]
# [[32], [256]]
# [[256], [16]]
# [[16], [256]]
# [[128], [64]]
# [[64], [128]]
# [[128], [32]]
# [[32], [128]]
# [[128], [16]]
# [[16], [128]]
# [[64], [32]]
# [[32], [64]]
# [[64], [16]]
# [[16], [64]]
# [[32], [16]]
# [[16], [32]]













