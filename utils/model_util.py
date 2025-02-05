import math
import os

import numpy as np
import torch
import random
import torch.nn as nn
import matplotlib.pyplot as plt


class AutoEncoderChannel(nn.Module):
    def __init__(self, input_embed_dim: int, out_feature_dims: list, channel_snr: float,  dtype, device, seeds):
        """
        AutoEncoder Block with noise added to the latent
        :param input_embed_dim: dimension of the input img/text_features
        :param out_feature_dims: dimensions of the out_features of each nn.Linear() layers sequentially, from max. dim. to latent dim.
        """
        super().__init__()
        self.channel_snr=channel_snr
        self.device=device
        self.dtype=dtype
        # encoder of the AutoEncoder
        self.encoder_img = nn.Linear(input_embed_dim, out_feature_dims[0][0], dtype=dtype)
        self.encoder_cap = nn.Linear(input_embed_dim, out_feature_dims[1][0], dtype=dtype)
        ######### compare AE dims
        # self.encoder_img = torch.nn.Sequential()
        # self.encoder_img.add_module("linear", torch.nn.Linear(512, 384, dtype=dtype))
        # self.encoder_img.add_module("linear1", torch.nn.Linear(427, 341, dtype=dtype))
        # self.encoder_img.add_module("linear2", torch.nn.Linear(384, 256, dtype=dtype))
        # self.encoder_img.add_module("linear3", torch.nn.Linear(358, 307, dtype=dtype))
        # self.encoder_img.add_module("linear4", torch.nn.Linear(307, 256, dtype=dtype))
        # self.encoder_cap = torch.nn.Sequential()
        # self.encoder_cap.add_module("linear", torch.nn.Linear(512, 384, dtype=dtype))
        # self.encoder_cap.add_module("linear1", torch.nn.Linear(427, 341, dtype=dtype))
        # self.encoder_cap.add_module("linear2", torch.nn.Linear(384, 256, dtype=dtype))
        # self.encoder_cap.add_module("linear3", torch.nn.Linear(358, 307, dtype=dtype))
        # self.encoder_cap.add_module("linear4", torch.nn.Linear(307, 256, dtype=dtype))
        # in_dim=input_embed_dim
        # for out_feature_dim in out_feature_dims:
        #     self.encoder.add_module("linear", torch.nn.Linear(in_dim, out_feature_dim))
        #     # self.encoder.add_module("relu", torch.nn.ReLU())
        #     in_dim = out_feature_dim

        # decoder of the AutoEncoder
        self.decoder_img = nn.Linear(out_feature_dims[0][0], input_embed_dim, dtype=dtype)
        self.decoder_cap = nn.Linear(out_feature_dims[1][0], input_embed_dim, dtype=dtype)
        ######### compare AE dims
        # self.decoder_img = torch.nn.Sequential()
        # self.decoder_img.add_module("linear", torch.nn.Linear(256, 307, dtype=dtype))
        # self.decoder_img.add_module("linear1", torch.nn.Linear(307, 358, dtype=dtype))
        # self.decoder_img.add_module("linear2", torch.nn.Linear(256, 384, dtype=dtype))
        # self.decoder_img.add_module("linear3", torch.nn.Linear(341, 427, dtype=dtype))
        # self.decoder_img.add_module("linear4", torch.nn.Linear(384, 512, dtype=dtype))
        # self.decoder_cap = torch.nn.Sequential()
        # self.decoder_cap.add_module("linear", torch.nn.Linear(256, 307, dtype=dtype))
        # self.decoder_cap.add_module("linear1", torch.nn.Linear(307, 358, dtype=dtype))
        # self.decoder_cap.add_module("linear2", torch.nn.Linear(256, 384, dtype=dtype))
        # self.decoder_cap.add_module("linear3", torch.nn.Linear(341, 427, dtype=dtype))
        # self.decoder_cap.add_module("linear4", torch.nn.Linear(384, 512, dtype=dtype))
        #
        # self.noise_table = {}
        # self.noise_table['channel_snr'] = None
        # for seed in seeds:
        #     self.noise_table[f'img_noise_{seed}'] = None
        #     for i in range(5):
        #         self.noise_table[f'cap_noise_{seed}_{i}'] = None


        # self.decoder = torch.nn.Sequential()
        # for out_feature_dim in reversed(out_feature_dims[1:]):
        #     self.decoder.add_module("linear", torch.nn.Linear(in_dim, out_feature_dim))
        #     self.encoder.add_module("relu", torch.nn.ReLU())
        #     in_dim = out_feature_dim
        # self.decoder.add_module("linear", torch.nn.Linear(in_dim, input_embed_dim))
        # self.encoder.add_module("sigmoid", torch.nn.Sigmoid())

    def add_noise(self, feature, channel_snr, is_val=False, seed=1234, is_img=False, cap_idx=0):
        """
        add noise to the signal according to the channel_snr
        :param feature: Tensor, the signal to which add noise
        :param channel_snr: float
        :return:
        """

        feature_power = signal_power(feature)
        if is_val:
            random.seed(a=seed)
            noise = np.zeros(feature.shape)
            for n in np.nditer(noise, op_flags=['readwrite']):
                n[...] = random.gauss(0, math.sqrt(10 ** (-channel_snr / 10) * feature_power))
            noise = torch.from_numpy(noise)
            noise = noise.to(self.dtype).to(self.device)
        else:
            noise = torch.normal(0, math.sqrt(10 ** (-channel_snr / 10) * feature_power), size=feature.shape).to(self.dtype).to(self.device)
        feature = feature + noise
        return feature


    def forward(self, image_feature, text_feature, is_val=False, seed=1234, cap_idx=0):
        """
        transmit the signal using AutoEncoder
        :param image_feature: Tensor, the output of open_clip.visual_encoder
        :param text_feature: Tensor, the output of open_clip.text_encoder
        :return: x_img, x_cap: Tensor, image_feature and text_feature transmitted by AutoEncoder
        """

        # encoder of the AutoEncoder
        x_img = self.encoder_img(image_feature)
        x_cap = self.encoder_cap(text_feature)

        # add noise
        if self.channel_snr[0] is not None:
            x_img=self.add_noise(x_img, self.channel_snr[0], is_val, seed, is_img=True)
        if self.channel_snr[1] is not None:
            x_cap=self.add_noise(x_cap, self.channel_snr[1], is_val, seed, cap_idx=cap_idx)

        # decoder of the AutoEncoder
        x_img = self.decoder_img(x_img)
        x_cap = self.decoder_cap(x_cap)

        return x_img, x_cap

def signal_power(signal):
    num_elements = signal.shape[0]*signal.shape[1]
    signal_power = torch.sum(signal[:, :] ** 2) / num_elements
    return float(signal_power)

def add_noise(feature, channel_snr, is_val, seed):
    """
    add noise to the signal according to the channel_snr
    :param feature: Tensor, the signal to which add noise
    :param channel_snr: float
    :return:
    """

    feature_power = signal_power(feature)
    if is_val:
        random.seed(a=seed)
        noise = np.zeros(feature.shape)
        for n in np.nditer(noise, op_flags=['readwrite']):
            n[...] = random.gauss(0, math.sqrt(10 ** (-channel_snr / 10) * feature_power))
        noise = torch.from_numpy(noise)
        noise = noise.to(torch.float32).to('cuda')
    else:
        noise = torch.normal(0, math.sqrt(10 ** (-channel_snr / 10) * feature_power), size=feature.shape).to(torch.float32).to('cuda')
        noise_power = signal_power(noise)
        snr = 10*math.log10(feature_power/noise_power)
        print(snr)
    feature = feature + noise
    return feature

def save_checkpoint(model, optimizer, epoch, train_loss_plot, val_loss_plot, channel_snr, train_accs_plot_img, train_accs_plot_cap, val_accs_plot_img, val_accs_plot_cap, ckpt_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss_plot,
        'val_loss': val_loss_plot,
        'accs_train_img': train_accs_plot_img,
        'accs_train_cap': train_accs_plot_cap,
        'accs_val_img': val_accs_plot_img,
        'accs_val_cap': val_accs_plot_cap
    }
    ckpt_filename=os.path.join(ckpt_path,f'snr{channel_snr[0]}_{channel_snr[1]}_epoch{epoch}.pth')
    torch.save(checkpoint, ckpt_filename)

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        val_loss_plot = checkpoint['val_loss']
        train_loss_plot = checkpoint['train_loss']
        train_accs_plot_img = checkpoint['accs_train_img']
        train_accs_plot_cap = checkpoint['accs_train_cap']
        val_accs_plot_img = checkpoint['accs_val_img']
        val_accs_plot_cap = checkpoint['accs_val_cap']

        print(f"Checkpoint loaded: epoch {epoch}\ntrain loss {train_loss_plot}, val loss {val_loss_plot}\naccs_train_img {train_accs_plot_img}\naccs_train_cap {train_accs_plot_cap}\n"
              f"accs_val_img {val_accs_plot_img}\naccs_val_cap {val_accs_plot_cap}\n")
        return epoch, train_loss_plot, val_loss_plot, train_accs_plot_img, train_accs_plot_cap, val_accs_plot_img, val_accs_plot_cap
    else:
        print("No checkpoint found")
        return None, None

# def add_noise(noise_type:str, signal, channel_snr, K=1):
    # """
    # :param noise_type: 'AWGN'/'Rayleigh'/'Rician'
    # :param signal: the signal on which the noise to be added
    # :param channel_snr: SNR of the channel
    # :param K: only for Rician noise, default = 1
    # :return: the signal with noise added
    # """
    # device = signal.device
    # noise_std = 10 ** (-channel_snr / 20)
    # if noise_type == 'AWGN':
    #     noise = torch.normal(0, noise_std / math.sqrt(2), size=signal.shape).to(device)
    #     signal_with_noise = signal + noise
    # elif noise_type == 'Rayleigh':
    #     shape = signal.shape
    #     H_real = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
    #     H_imag = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
    #     H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
    #     signal = torch.matmul(signal.view(shape[0], -1, 2), H)
    #     noise = torch.normal(0, noise_std / math.sqrt(2), size=signal.shape).to(device)
    #     signal_with_noise = signal + noise
    #     signal_with_noise = torch.matmul(signal_with_noise, torch.inverse(H)).view(shape)
    # elif noise_type == 'Rician':
    #     shape = signal.shape
    #     mean = math.sqrt(K / (K + 1))
    #     std = math.sqrt(1 / (K + 1))
    #     H_real = torch.normal(mean, std, size=[1]).to(device)
    #     H_imag = torch.normal(mean, std, size=[1]).to(device)
    #     H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
    #     signal = torch.matmul(signal.view(shape[0], -1, 2), H)
    #     noise = torch.normal(0, noise_std / math.sqrt(2), size=signal.shape).to(device)
    #     signal_with_noise = signal + noise
    #     signal_with_noise = torch.matmul(signal_with_noise, torch.inverse(H)).view(shape)
    # else:
    #     raise NotImplementedError("please choose a noise type from 'AWGN'/'Rayleigh'/'Rician'")
    # return signal_with_noise
