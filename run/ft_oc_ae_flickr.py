import torch
import os
import open_clip

import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import nn
from datetime import datetime
from datasets.flickr import Flickr30kDatasetRet, Flickr30kDatasetRetFix
from configs.data import Flickr30kCfg
from utils.model_util import AutoEncoderChannel, save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader, random_split
from utils.utils import write_log


####### grad visualization
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

'''
finetuning for OpenCLIP
'''
def ft_oc_ae(ret_ats,
          channel_snrs,
          out_feature_dims=None,
          rand_valset=False,
          epochs=50,
          batch_size=50,
          lr=1e-5,
          device='cuda',
          log_dir='C:/Users/INDA_HIWI/Desktop/',
          break_acc_th=0.05,
          break_loss_th=0.1,
          resume=False,
          load_ckpt_path = '',
          resume_at_snr = None,
          save_ckpt_freq = None):

    # without the setting here, the cpu automatically choose tf16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


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



    # prepare dataset
    # log
    write_log(f'[{datetime.now().strftime("%H:%M:%S")}] loading dataset...... \n', log_path)
    _, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', precision='fp32')
    if rand_valset:
        valset_len = 1000
        flickr30k_dataset = Flickr30kDatasetRet(is_train=None, preprocess=preprocess)
        train_dataset, val_dataset = random_split(dataset=flickr30k_dataset, lengths=[flickr30k_dataset.__len__() - valset_len, valset_len],
                                      generator=torch.Generator().manual_seed(0))
    else:
        train_dataset = Flickr30kDatasetRetFix(img_dir=Flickr30kCfg.img_dir, ann_file=Flickr30kCfg.annotations_path_train, preprocess=preprocess)
        val_dataset = Flickr30kDatasetRetFix(img_dir=Flickr30kCfg.img_dir, ann_file=Flickr30kCfg.annotations_path_test, preprocess=preprocess)
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
        # todo: init model
        # model = FT_OC_AE_Model(out_feature_dims, channel_snr, device)
        # init pretrained model
        oc_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', precision='fp32')
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        oc_model.to(device)
        # init Autoencoder model
        assert out_feature_dims is not None, 'please either give the out_feature_dims for the AutoEncoder'
        ae_channel = AutoEncoderChannel(input_embed_dim=512, out_feature_dims=out_feature_dims,
                                        channel_snr=channel_snr, dtype=torch.float32, device=device)
        ae_channel.to(device)

        # lr schedule
        optimizer2 = optim.AdamW(ae_channel.parameters(), lr, betas=(0.9, 0.99), eps=1e-6)
        scheduler2 = None
        criterion = nn.CrossEntropyLoss()

        # load checkpoint
        resume_flag=False
        if resume:
            assert resume_at_snr is not None, 'resume_at_snr should be given'
            assert load_ckpt_path != '', 'checkpoint path should not be empty string, please give the ckpt_path'
            assert os.path.exists(load_ckpt_path), 'checkpoint path not exist'
            if channel_snr == resume_at_snr:
                epoch_start, train_loss_plot, val_loss_plot, train_accs_plot_img, train_accs_plot_cap, val_accs_plot_img, val_accs_plot_cap = load_checkpoint(ae_channel, optimizer1, optimizer2, load_ckpt_path)
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


        for epoch in range(epoch_start,epochs):
            ################################# training #################################
            # init
            # lr schedule
            if epoch == 10:
                optimizer1 = optim.AdamW([{"params": oc_model.parameters(), 'lr': lr / 100}], betas=(0.9, 0.99), eps=1e-6)
                scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=5, eta_min=1e-9)
                optimizer2 = optim.AdamW([{"params": ae_channel.parameters(), 'lr': lr}], betas=(0.9, 0.99), eps=1e-6)
                scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=5, eta_min=1e-7)
            train_loss = 0
            train_acc_img = {}
            train_acc_cap = {}
            for ret_at in ret_ats:
                train_acc_img[f'R@{ret_at}'] = 0
                train_acc_cap[f'R@{ret_at}'] = 0

            # set model in train mode
            ae_channel.train(True)
            if epoch >= 10:
                oc_model.train(True)
            for _, (images, captions) in enumerate(train_dataloader):
                # init
                img_num = len(images) # img_num != batch_size in the last batch
                images = images.to(torch.float32).to(device)

                # target
                target_img = torch.from_numpy(np.eye(img_num, dtype=int)).type(torch.float32).to(device)
                target_cap = torch.from_numpy(np.eye(img_num, dtype=int)).type(torch.float32).to(device)

                temp_acc_cap = {}
                temp_acc_cap[f'R@1'], temp_acc_cap[f'R@5'], temp_acc_cap[f'R@10'] = None, None, None
                for i in range(0, 5):
                    # collect caption sub batch
                    captions_batch=[captions[i][j] for j in range(0,img_num)]

                    # tokenize each caption
                    captions_tok = tokenizer(captions_batch).to(device)
                    # logits_per_image = model(images, captions_tok)

                    # encode using pretrained open clip
                    image_features, text_features, logit_scale = oc_model(images, captions_tok)

                    # transfer by autoencoder channel
                    x_img, x_cap = ae_channel(image_features, text_features)

                    # get loss from logits
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * x_img @ x_cap.T
                    logits_per_text = logits_per_image.T

                    loss = criterion(logits_per_image, target_img) + criterion(logits_per_text, target_cap)
                    train_loss += loss.item()

                    # update parameters
                    if scheduler2 is not None:
                        optimizer1.zero_grad()
                        optimizer2.zero_grad()
                        loss.backward()
                        optimizer1.step()
                        optimizer2.step()
                    else:
                        optimizer2.zero_grad()
                        loss.backward()
                        optimizer2.step()

                    # get acc
                    groundtruth = np.arange(img_num)
                    for ret_at in ret_ats:
                        max_img_index = np.argsort(logits_per_image.cpu().detach().numpy(), axis=-1)[:, -ret_at:]
                        max_cap_index = np.argsort(logits_per_text.cpu().detach().numpy(), axis=-1)[:, -ret_at:]
                        for i in range(0, ret_at):
                            max_img_index[:, i] = max_img_index[:, i] - groundtruth
                            max_cap_index[:, i] = max_cap_index[:, i] - groundtruth
                        train_acc_img[f'R@{ret_at}'] += np.any(max_img_index == 0, axis=-1).sum()

                        # in caption retrieval, at least one caption from five captions is correct -> acc+1
                        if temp_acc_cap[f'R@{ret_at}'] is None:
                            temp_acc_cap[f'R@{ret_at}'] = np.any(max_cap_index == 0, axis=-1)
                        else:
                            temp_acc_cap[f'R@{ret_at}'] += np.any(max_cap_index == 0, axis=-1)
                for ret_at in ret_ats:
                    train_acc_cap[f'R@{ret_at}'] += temp_acc_cap[f'R@{ret_at}'].sum()

                # ####### grad visualization
                # # Grad-CAM++ Visualization
                # cam = GradCAMPlusPlus(model=model.visual, target_layers=[model.visual.ln_post])
                # # targets = [ClassifierOutputTarget(class_id) for class_id in torch.arange(len(images)).to(device)]
                # targets=[i for i in np.arange(img_num)]
                #
                # grayscale_cam = cam(input_tensor=images, targets=targets)
                # grayscale_cam = grayscale_cam[0, :]
                #
                # image = images[0].permute(1, 2, 0).cpu().numpy()
                # visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
                #
                # # You can save or display the visualization as needed
                # # For example, to display the image using matplotlib
                # plt.imshow(visualization)
                # plt.show()
                # pass

            # end for training
            if scheduler2 is not None:
                # print and log result
                print('[train epoch %d][lr_oc: %e, lr_ae: %e] img acc: [R@1:%d/%d, %.2f%%], [R@5:%d/%d, %.2f%%], [R@10:%d/%d, %.2f%%]  '
                      'cap acc: [R@1:%d/%d, %.2f%%],[R@5:%d/%d, %.2f%%],[R@10:%d/%d, %.2f%%]'
                      % (epoch, scheduler1.get_last_lr()[0], scheduler2.get_last_lr()[0],
                         train_acc_img['R@1'], (len_trainset * 5), train_acc_img['R@1'] / (len_trainset * 5) * 100,
                         train_acc_img['R@5'], (len_trainset * 5), train_acc_img['R@5'] / (len_trainset * 5) * 100,
                         train_acc_img['R@10'], (len_trainset * 5), train_acc_img['R@10'] / (len_trainset * 5) * 100,
                         train_acc_cap['R@1'], len_trainset, train_acc_cap['R@1'] / len_trainset * 100,
                         train_acc_cap['R@5'], len_trainset, train_acc_cap['R@5'] / len_trainset * 100,
                         train_acc_cap['R@10'], len_trainset, train_acc_cap['R@10'] / len_trainset * 100))
                write_log('[%s] [train epoch %d] [lr_oc: %e, lr_ae: %e]\n'
                          'img acc: [R@1:%d/%d, %.2f%%], [R@5:%d/%d, %.2f%%], [R@10:%d/%d, %.2f%%] '
                          'cap acc: [R@1:%d/%d, %.2f%%],[R@5:%d/%d, %.2f%%],[R@10:%d/%d, %.2f%%]\n'
                          'train_loss: %f\n'
                          % (datetime.now().strftime("%H:%M:%S"), epoch, scheduler1.get_last_lr()[0], scheduler2.get_last_lr()[0],
                             train_acc_img['R@1'], (len_trainset * 5), train_acc_img['R@1'] / (len_trainset * 5) * 100,
                             train_acc_img['R@5'], (len_trainset * 5), train_acc_img['R@5'] / (len_trainset * 5) * 100,
                             train_acc_img['R@10'], (len_trainset * 5), train_acc_img['R@10'] / (len_trainset * 5) * 100,
                             train_acc_cap['R@1'], len_trainset, train_acc_cap['R@1'] / len_trainset * 100,
                             train_acc_cap['R@5'], len_trainset, train_acc_cap['R@5'] / len_trainset * 100,
                             train_acc_cap['R@10'], len_trainset, train_acc_cap['R@10'] / len_trainset * 100,
                             train_loss),
                          log_path)
                scheduler1.step()
                scheduler2.step()

            else:
                # print and log result
                print('[train epoch %d] img acc: [R@1:%d/%d, %.2f%%], [R@5:%d/%d, %.2f%%], [R@10:%d/%d, %.2f%%]  '
                      'cap acc: [R@1:%d/%d, %.2f%%],[R@5:%d/%d, %.2f%%],[R@10:%d/%d, %.2f%%]'
                      % (epoch,
                         train_acc_img['R@1'], (len_trainset * 5), train_acc_img['R@1'] / (len_trainset * 5) * 100,
                         train_acc_img['R@5'], (len_trainset * 5), train_acc_img['R@5'] / (len_trainset * 5) * 100,
                         train_acc_img['R@10'], (len_trainset * 5), train_acc_img['R@10'] / (len_trainset * 5) * 100,
                         train_acc_cap['R@1'], len_trainset, train_acc_cap['R@1'] / len_trainset * 100,
                         train_acc_cap['R@5'], len_trainset, train_acc_cap['R@5'] / len_trainset * 100,
                         train_acc_cap['R@10'], len_trainset, train_acc_cap['R@10'] / len_trainset * 100))
                write_log('[%s] [train epoch %d]\n'
                          'img acc: [R@1:%d/%d, %.2f%%], [R@5:%d/%d, %.2f%%], [R@10:%d/%d, %.2f%%] '
                          'cap acc: [R@1:%d/%d, %.2f%%],[R@5:%d/%d, %.2f%%],[R@10:%d/%d, %.2f%%]\n'
                          'train_loss: %f\n'
                          % (datetime.now().strftime("%H:%M:%S"), epoch,
                             train_acc_img['R@1'], (len_trainset * 5), train_acc_img['R@1'] / (len_trainset * 5) * 100,
                             train_acc_img['R@5'], (len_trainset * 5), train_acc_img['R@5'] / (len_trainset * 5) * 100,
                             train_acc_img['R@10'], (len_trainset * 5), train_acc_img['R@10'] / (len_trainset * 5) * 100,
                             train_acc_cap['R@1'], len_trainset, train_acc_cap['R@1'] / len_trainset * 100,
                             train_acc_cap['R@5'], len_trainset, train_acc_cap['R@5'] / len_trainset * 100,
                             train_acc_cap['R@10'], len_trainset, train_acc_cap['R@10'] / len_trainset * 100,
                             train_loss),
                          log_path)


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
            val_acc_cap_subbatch = {}
            for ret_at in ret_ats:
                val_acc_img[f'R@{ret_at}'] = 0
                val_acc_cap[f'R@{ret_at}'] = 0
                val_acc_cap_subbatch[f'R@{ret_at}'] = 0

            # set model in testing mode
            oc_model.train(False)
            ae_channel.train(False)
            for _, (images, captions) in enumerate(val_dataloader):
                # init
                img_num = len(images)
                images = images.to(torch.float32).to(device)

                # target
                target_img = torch.from_numpy(np.eye(img_num, dtype=int)).type(torch.float32).to(device)
                target_cap = torch.from_numpy(np.eye(img_num, dtype=int)).type(torch.float32).to(device)

                temp_acc_cap={}
                temp_acc_cap[f'R@1'], temp_acc_cap[f'R@5'], temp_acc_cap[f'R@10'] = None, None, None
                for i in range(0, 5):
                    # collect caption sub batch
                    captions_batch = [captions[i][j] for j in range(0, img_num)]
                    # tokenize each caption
                    captions_tok = tokenizer(captions_batch).to(device)


                    with torch.no_grad():
                        # encode using pretrained open clip
                        image_features, text_features, logit_scale = oc_model(images, captions_tok)

                        # transfer by autoencoder channel
                        image_features, text_features = ae_channel(image_features, text_features)

                        # get val loss from logits
                        logit_scale = logit_scale.mean()
                        logits_per_image = logit_scale * image_features @ text_features.T
                        logits_per_text = logits_per_image.T

                        loss = criterion(logits_per_image, target_img) + criterion(logits_per_text, target_cap)
                        val_loss += loss.item() * len_trainset / len_valset

                        # get val acc
                        groundtruth = np.arange(img_num)
                        for ret_at in ret_ats:
                            max_img_index = np.argsort(logits_per_image.cpu().detach().numpy(), axis=-1)[:, -ret_at:]
                            max_cap_index = np.argsort(logits_per_text.cpu().detach().numpy(), axis=-1)[:, -ret_at:]

                            for idx in range(0, ret_at):
                                max_img_index[:, idx] = max_img_index[:, idx] - groundtruth
                                max_cap_index[:, idx] = max_cap_index[:, idx] - groundtruth
                            val_acc_img[f'R@{ret_at}'] += np.any(max_img_index == 0, axis=-1).sum()
                            # # cap ret acc += 1 if correct in one sub-batch
                            # val_acc_cap_subbatch[f'R@{ret_at}'] += np.any(max_cap_index == 0, axis=-1).sum()

                            # acc += 1 if in caption retrieval, at least one caption from five captions is correct
                            if temp_acc_cap[f'R@{ret_at}'] is None:
                                temp_acc_cap[f'R@{ret_at}'] = np.any(max_cap_index == 0, axis=-1)
                            else:
                                temp_acc_cap[f'R@{ret_at}'] += np.any(max_cap_index == 0, axis=-1)
                for ret_at in ret_ats:
                    val_acc_cap[f'R@{ret_at}'] += temp_acc_cap[f'R@{ret_at}'].sum()

            # print and log result
            print('[val epoch %d] img acc: [R@1:%d/%d, %.2f%%], [R@5:%d/%d, %.2f%%], [R@10:%d/%d, %.2f%%]  '
                  'cap acc: [R@1:%d/%d, %.2f%%],[R@5:%d/%d, %.2f%%],[R@10:%d/%d, %.2f%%]'
                  % (epoch,
                     val_acc_img['R@1'], (len_valset * 5), val_acc_img['R@1'] / (len_valset * 5) * 100,
                     val_acc_img['R@5'], (len_valset * 5), val_acc_img['R@5'] / (len_valset * 5) * 100,
                     val_acc_img['R@10'], (len_valset * 5), val_acc_img['R@10'] / (len_valset * 5) * 100,
                     val_acc_cap['R@1'], len_valset, val_acc_cap['R@1'] / len_valset * 100,
                     val_acc_cap['R@5'], len_valset, val_acc_cap['R@5'] / len_valset * 100,
                     val_acc_cap['R@10'], len_valset, val_acc_cap['R@10'] / len_valset * 100))
            print(f'epoch: {epoch}, train loss: {train_loss}, val loss: {val_loss}')
            write_log('[%s] [val epoch %d]\n'
                      'img acc: [R@1:%d/%d, %.2f%%], [R@5:%d/%d, %.2f%%], [R@10:%d/%d, %.2f%%] '
                      'cap acc: [R@1:%d/%d, %.2f%%],[R@5:%d/%d, %.2f%%],[R@10:%d/%d, %.2f%%]\n'
                      'val loss: %f\n\n'
                      % (datetime.now().strftime("%H:%M:%S"), epoch,
                         val_acc_img['R@1'], (len_valset * 5), val_acc_img['R@1'] / (len_valset * 5) * 100,
                         val_acc_img['R@5'], (len_valset * 5), val_acc_img['R@5'] / (len_valset * 5) * 100,
                         val_acc_img['R@10'], (len_valset * 5), val_acc_img['R@10'] / (len_valset * 5) * 100,
                         val_acc_cap['R@1'], len_valset, val_acc_cap['R@1'] / len_valset * 100,
                         val_acc_cap['R@5'], len_valset, val_acc_cap['R@5'] / len_valset * 100,
                         val_acc_cap['R@10'], len_valset, val_acc_cap['R@10'] / len_valset * 100,
                         val_loss),
                      log_path)

            # save data for plot
            val_loss_plot.append(val_loss)
            for ret_at in ret_ats:
                val_accs_plot_img[f'R@{ret_at}'].append(val_acc_img[f'R@{ret_at}'] / (len_valset * 5) * 100)
                val_accs_plot_cap[f'R@{ret_at}'].append(val_acc_cap[f'R@{ret_at}'] / len_valset * 100)

            # save checkpoints
            if save_ckpt_freq is not None and (epoch + 1) % save_ckpt_freq == 0 and epoch >= 10:
                save_checkpoint(ae_channel, optimizer1, optimizer2, epoch, train_loss_plot, val_loss_plot, channel_snr, train_accs_plot_img, train_accs_plot_cap, val_accs_plot_img, val_accs_plot_cap, save_ckpt_path)

            ######################################### plot loss and acc #########################################
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


            # break condition
            if epoch > 20:
                flag_break = 0
                if sum(val_loss_plot[-10:]) / 10 - sum(val_loss_plot[-5:]) / 5 < break_loss_th:
                    flag_break += 1
                for ret_at in ret_ats:
                    if (sum(val_accs_plot_img[f'R@{ret_at}'][-5:]) / 5 - sum(val_accs_plot_img[f'R@{ret_at}'][-10:]) / 10 < break_acc_th
                            and sum(val_accs_plot_cap[f'R@{ret_at}'][-5:]) / 5 - sum(val_accs_plot_cap[f'R@{ret_at}'][-10:]) / 10 < break_acc_th):
                        flag_break += 1
                    else:
                        break
                if flag_break == 4: break
            # end for epoch
        # end for snr

    return 0

if __name__ == '__main__':
    ######################### set params #########################
    # R@?
    ret_ats = [1, 5, 10]

    # set snr, SNR=None if not add noise, [snr_img, snr_cap]
    img_snr = 6
    cap_snr = 6
    # channel_snrs = [[24, cap_snr], [18, cap_snr], [12, cap_snr], [6, cap_snr], [0, cap_snr], [-6, cap_snr], [-12, cap_snr], [-18, cap_snr], [-24, cap_snr], [-30, cap_snr], [-36, cap_snr], [-42, cap_snr]]
    # channel_snrs = [[img_snr, 24], [img_snr, 18], [img_snr, 12], [img_snr, 6], [img_snr, 0], [img_snr, -6], [img_snr, -12], [img_snr, -18], [img_snr, -24], [img_snr, -30], [img_snr, -36], [img_snr, -42]]
    # channel_snrs =  [[None, None], [24, 24], [18, 18], [12, 12], [6, 6], [0, 0], [-6, -6], [-12, -12], [-18, -18], [-24, -24], [-30, -30], [-36, -36], [-42, -42]]
    # channel_snrs = [[None, None], [24, None], [18, None], [12, None], [6, None], [0, None], [-6, None], [-12, None], [-18, None], [-24, None], [-30, None], [-36, None], [-42, None]]
    # channel_snrs = [[None, None], [None, 24], [None, 18], [None, 12], [None, 6], [None, 0], [None, -6], [None, -12], [None, -18], [None, -24], [None, -30], [None, -36], [None, -42]]


    # random split testing set from the dataset: True or use the testing set from the paper: False
    rand_valset = False

    # path to save the plots
    log_dir='C:/Users/INDA_HIWI/Desktop/plot_oc_flickr_train_ae/ft_oc_ae'

    '''
    output shape of every AutoEncoder Layers, out_feature_dims = [img_dim, cap_dim]
    e.g.: out_feature_dims = [[256, 128, 32], [64, 16]],
    the AutoEncoder structure for image is input_shape->256->128->32->communicational transmission->32->128->256->input_shape,
    the AutoEncoder structure for caption is input_shape->64->16->communicational transmission->16->64->input_shape
    '''
    # out_feature_dims=[[256], [256]]

    # early stop conditions
    break_acc_th=0.1
    break_loss_th = 0.5

    # hyperparam
    epochs=100
    batch_size=100
    assert batch_size >= max(ret_ats), 'batch size must larger than N in R@N'
    lr=1e-5

    # if resume
    resume=False
    resume_at_snr=[None, 0]
    load_ckpt_path='C:/Users/INDA_HIWI/Desktop/plot_oc_flickr_train_ae/dim16_256_20240524/ckpt1414/snrNone_0_epoch60.pth'
    # if not save checkpoints, save_ckpt_freq = None
    save_ckpt_freq=None

    ######################################################################################################################################################################


    out_feature_dims = [[256], [256]]
    channel_snrs = [[None, None], [24, 24], [18, 18], [-18, -18], [-24, -24], [24, None], [18, None], [-18, None], [-24, None], [None, 24], [None, 18], [None, -18], [None, -24]]
    acc_both = ft_oc_ae(ret_ats=ret_ats,
                     channel_snrs=channel_snrs,
                     out_feature_dims=out_feature_dims,
                     rand_valset=rand_valset,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     log_dir=log_dir,
                     break_acc_th=break_acc_th,
                     break_loss_th=break_loss_th,
                     resume=resume,
                     load_ckpt_path=load_ckpt_path,
                     resume_at_snr=resume_at_snr,
                     save_ckpt_freq=save_ckpt_freq)
