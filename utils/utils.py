
import math
import torch
import random
import numpy as np



def write_log(log:str, log_path:str):
    with open(log_path, "a") as f:
        f.write(log)
        f.close()

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def is_monotonic(x_2D):
    x_2D = x_2D[~np.isnan(x_2D)].reshape((np.max(np.sum(~np.isnan(x_2D), axis=0)), np.max(np.sum(~np.isnan(x_2D), axis=1))))
    if len(x_2D.shape) <=1:
        if np.all(np.diff(x_2D, axis=0) >= 0) or np.all(np.diff(x_2D, axis=0) <= 0):
            return True
    if np.all(np.diff(x_2D, axis=0) >= 0) or np.all(np.diff(x_2D, axis=0) <= 0):
        if np.all(np.diff(x_2D, axis=1) >= 0) or np.all(np.diff(x_2D, axis=1) <= 0):
            return True
    return False

def exp_ploynom_2D(x, y, a0, a1, a2):
    # exp(a0+a1x+a2y+a3xy)
    return np.exp(a0 + a1 * x + a2 * y)

def exp_ploynom_2D_fit(xdata, a0, a1, a2):
    (x, y) = xdata
    return exp_ploynom_2D(x, y, a0, a1, a2).ravel()


def gmm_2D(x, y, A, x0, y0, sigma_x, sigma_y, C,
           A1=None, x1=None, y1=None, sigma_x1=None, sigma_y1=None,
           A2=None, x2=None, y2=None, sigma_x2=None, sigma_y2=None,
           A3=None, x3=None, y3=None, sigma_x3=None, sigma_y3=None,
           A4=None, x4=None, y4=None, sigma_x4=None, sigma_y4=None):
    z = A * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2))) + C
    if A1 is not None:
        z += A1 * np.exp(-(((x - x1) ** 2) / (2 * sigma_x1 ** 2) + ((y - y1) ** 2) / (2 * sigma_y1 ** 2)))
    if A2 is not None:
        z += A2 * np.exp(-(((x - x2) ** 2) / (2 * sigma_x2 ** 2) + ((y - y2) ** 2) / (2 * sigma_y2 ** 2)))
    if A3 is not None:
        z += A3 * np.exp(-(((x - x3) ** 2) / (2 * sigma_x3 ** 2) + ((y - y3) ** 2) / (2 * sigma_y3 ** 2)))
    if A4 is not None:
        z += A4 * np.exp(-(((x - x4) ** 2) / (2 * sigma_x4 ** 2) + ((y - y4) ** 2) / (2 * sigma_y4 ** 2)))
    return z



def gaussian_2d_fit(xdata, A, x0, y0, sigma_x, sigma_y, C,
                    A1=None, x1=None, y1=None, sigma_x1=None, sigma_y1=None,
                    A2=None, x2=None, y2=None, sigma_x2=None, sigma_y2=None,
                    A3=None, x3=None, y3=None, sigma_x3=None, sigma_y3=None,
                    A4=None, x4=None, y4=None, sigma_x4=None, sigma_y4=None):
    (x, y) = xdata
    if A4 is not None:
        return gmm_2D(x, y, A, x0, y0, sigma_x, sigma_y, C,
                      A1, x1, y1, sigma_x1, sigma_y1,
                      A2, x2, y2, sigma_x2, sigma_y2,
                      A3, x3, y3, sigma_x3, sigma_y3,
                      A4, x4, y4, sigma_x4, sigma_y4).ravel()
    elif A3 is not None:
        return gmm_2D(x, y, A, x0, y0, sigma_x, sigma_y, C,
                      A1, x1, y1, sigma_x1, sigma_y1,
                      A2, x2, y2, sigma_x2, sigma_y2,
                      A3, x3, y3, sigma_x3, sigma_y3).ravel()
    elif A2 is not None:
        return gmm_2D(x, y, A, x0, y0, sigma_x, sigma_y, C,
                      A1, x1, y1, sigma_x1, sigma_y1,
                      A2, x2, y2, sigma_x2, sigma_y2).ravel()
    elif A1 is not None:
        return gmm_2D(x, y, A, x0, y0, sigma_x, sigma_y, C,
                      A1, x1, y1, sigma_x1, sigma_y1).ravel()
    else:
        return gmm_2D(x, y, A, x0, y0, sigma_x, sigma_y, C).ravel()

def ls_fun(params, xdata=None, ydata=None):
    (x, y) = xdata
    if len(params)>24:
        (A, x0, y0, sigma_x, sigma_y, C,
         A1, x1, y1, sigma_x1, sigma_y1,
         A2, x2, y2, sigma_x2, sigma_y2,
         A3, x3, y3, sigma_x3, sigma_y3,
         A4, x4, y4, sigma_x4, sigma_y4) = params
        return gmm_2D(x, y, A, x0, y0, sigma_x, sigma_y, C,
                      A1, x1, y1, sigma_x1, sigma_y1,
                      A2, x2, y2, sigma_x2, sigma_y2,
                      A3, x3, y3, sigma_x3, sigma_y3,
                      A4, x4, y4, sigma_x4, sigma_y4).ravel() - ydata
    elif len(params)>19:
        (A, x0, y0, sigma_x, sigma_y, C,
         A1, x1, y1, sigma_x1, sigma_y1,
         A2, x2, y2, sigma_x2, sigma_y2,
         A3, x3, y3, sigma_x3, sigma_y3) = params
        return gmm_2D(x, y, A, x0, y0, sigma_x, sigma_y, C,
                      A1, x1, y1, sigma_x1, sigma_y1,
                      A2, x2, y2, sigma_x2, sigma_y2,
                      A3, x3, y3, sigma_x3, sigma_y3).ravel() - ydata
    elif len(params)>14:
        (A, x0, y0, sigma_x, sigma_y, C,
         A1, x1, y1, sigma_x1, sigma_y1,
         A2, x2, y2, sigma_x2, sigma_y2) = params
        return gmm_2D(x, y, A, x0, y0, sigma_x, sigma_y, C,
                      A1, x1, y1, sigma_x1, sigma_y1,
                      A2, x2, y2, sigma_x2, sigma_y2).ravel() - ydata
    elif len(params)>9:
        (A, x0, y0, sigma_x, sigma_y, C,
         A1, x1, y1, sigma_x1, sigma_y1) = params
        return gmm_2D(x, y, A, x0, y0, sigma_x, sigma_y, C,
                      A1, x1, y1, sigma_x1, sigma_y1).ravel() - ydata
    else:
        (A, x0, y0, sigma_x, sigma_y, C) = params
        return gmm_2D(x, y, A, x0, y0, sigma_x, sigma_y, C).ravel() - ydata


def res_calculator(dim, SNR, channel_SNR):
    """
    calculate the standard resource from dim and snr
    :param channel_SNR: base channel SNR
    :return: standard resource for single img or cap, not in dB
    """
    delta_SNR = 0 if SNR is None else SNR - channel_SNR
    dim_from_SNR = 10 ** (delta_SNR / 20)
    res = 2 ** (math.log2(dim_from_SNR) + math.log2(dim))
    return res


def groups_from_res_imgorcap(res, channel_SNR):
    """
    calculate the possible num_symbol, snr (dB) for either img or cap
    :param res: standard resource for single img or cap, not in dB
    :param channel_SNR: base channel SNR
    :return: a list of groups [num_symbol, snr (dB)]
    """
    assert res >= 1, 'resource must be larger than 0'
    assert math.log2(res) % 1 == 0, 'resource must = 2 ** N, where N is an integer'
    groups = []
    log2res = int(math.log2(res))
    for log2res_snr in range(0, log2res - 3):
        for log2res_numsym in range(4, min(int(log2res - log2res_snr), 8) + 1):
            groups.append([round(2 ** log2res_numsym), round(20 * math.log10(2 ** log2res_snr) / 6) * 6 + channel_SNR])
    # groups = [[round(2 ** (log2res - log2res_snr)), round(20 * math.log10(2 ** log2res_snr) / 6) * 6 + channel_SNR] for log2res_snr in range(0, log2res - 3)]
    return groups
def groups_from_res_imgandcap(res, channel_SNR, equ_num_symbol:bool, equ_res:bool):
    """
    calculate the possible img num_symbol, cap num_symbol, img snr (dB), cap snr (dB)
    :param res: standard resource for img+cap, not in dB
    :param channel_SNR: base channel SNR
    :param equ_num_symbol: if img dim&snr == cap dim&snr
    :param equ_res: if img res == cap res
    :return: a list of groups [img num_symbol, cap num_symbol, img snr (dB), cap snr (dB)]
    """
    assert res >= 2, 'resource must be larger or equal 2'
    assert not (not equ_res and equ_num_symbol), 'if resource for img and cap are not equal, the dim and snr for img and cap cannot be equal'


    if equ_res and equ_num_symbol:
        log2res = int(math.log2(res / 2))
        groups = [[round(2 ** (log2res - log2res_snr)),
                   round(2 ** (log2res - log2res_snr)),
                   round(20 * math.log10(2 ** log2res_snr) / 6) * 6 + channel_SNR,
                   round(20 * math.log10(2 ** log2res_snr) / 6) * 6 + channel_SNR]
                  for log2res_snr in range(0, log2res - 4)]
    elif equ_res and not equ_num_symbol:
        log2res = int(math.log2(res / 2))
        groups = [[round(2 ** (log2res - log2res_snr_img)),
                   round(2 ** (log2res - log2res_snr_cap)),
                   round(20 * math.log10(2 ** log2res_snr_img) / 6) * 6 + channel_SNR,
                   round(20 * math.log10(2 ** log2res_snr_cap) / 6) * 6 + channel_SNR]
                  for log2res_snr_img in range(0, log2res - 4) for log2res_snr_cap in range(0, log2res - 4)]
    else:
        groups = []
        # flag_valid_input = False
        for log2res_img in range(1, int(math.log2(res)) + 1):
            if res == 2 ** log2res_img: continue
            if math.log2(res - 2 ** log2res_img) % 1 == 0:
                # flag_valid_input = True
                log2res_cap = int(math.log2(res - 2 ** log2res_img))
                groups += [[round(2 ** (log2res_img - log2res_snr_img)),
                            round(2 ** (log2res_cap - log2res_snr_cap)),
                            round(20 * math.log10(2 ** log2res_snr_img) / 6) * 6 + channel_SNR,
                            round(20 * math.log10(2 ** log2res_snr_cap) / 6) * 6 + channel_SNR]
                           for log2res_snr_img in range(0, log2res_img + 1) for log2res_snr_cap in range(0, log2res_cap + 1)]
        # if not flag_valid_input:
        #     print(f'the input res = {res} is not vaild, please give a res = 2 ** N + 2 ** M, where N and M are integers')
    return groups

def get_all_groups_imgandcap(res_range: list, equ_num_symbol=False, equ_res=False, channel_SNR=-24, min_num_symbol=16, max_num_symbol=256, max_SNR=30):
    """
    get all possible [num_symbol_img, num_symbol_cap, SNR_img, SNR_cap] in list
    """
    groups_all = None
    for res in res_range:
        # print(res_db)
        groups = np.array(groups_from_res_imgandcap(res, channel_SNR=channel_SNR, equ_num_symbol=equ_num_symbol, equ_res=equ_res))
        if groups.shape[0] > 0:
            groups = groups[groups[:, 0] <= max_num_symbol]
            groups = groups[groups[:, 0] >= min_num_symbol]
            groups = groups[groups[:, 1] <= max_num_symbol]
            groups = groups[groups[:, 1] >= min_num_symbol]
            groups = groups[groups[:, 2] <= max_SNR]
            groups = groups[groups[:, 3] <= max_SNR]
            if groups_all is None:
                groups_all = groups
            else:
                groups_all = np.concatenate([groups_all, groups], axis=0)

    groups_all = np.unique(groups_all, axis=0)
    for i in range(groups_all.shape[0]):
        print(groups_all[i, :])
    return groups_all



def get_all_groups_imgorcap(res_range: list, channel_SNR=-24, min_num_symbol=16, max_num_symbol=256, max_SNR=30):
    """
    get all possible [num_symbol, SNR] in list
    """
    groups_all = None
    for res_bit in res_range:
        res = 2 ** res_bit
        groups = np.array(groups_from_res_imgorcap(res, channel_SNR))
        if groups.shape[0] > 0:
            groups = groups[groups[:, 0] <= max_num_symbol]
            groups = groups[groups[:, 0] >= min_num_symbol]
            groups = groups[groups[:, 1] <= max_SNR]
            if groups_all is None:
                groups_all = groups
            else:
                groups_all = np.concatenate([groups_all, groups], axis=0)
        else:
            print(f'no [num_symbol, SNR] group from res = 2^{res_bit}')

    groups_all = np.unique(groups_all, axis=0)
    for i in range(groups_all.shape[0]):
        print(groups_all[i, :])
    return groups_all

def get_opt_acc_from_imgres_capres(acc, img_res, cap_res, acc_type, SNR_range, max_num_symbol = 256, min_num_symbol = 16, max_SNR = 18, min_SNR = -18):
    img_groups = np.array(groups_from_res_imgorcap(2 ** img_res, -24))
    cap_groups = np.array(groups_from_res_imgorcap(2 ** cap_res, -24))
    if img_groups.shape[0] > 0 and cap_groups.shape[0] > 0:
        img_groups = img_groups[img_groups[:, 0] <= max_num_symbol]
        img_groups = img_groups[img_groups[:, 0] >= min_num_symbol]
        img_groups = img_groups[img_groups[:, 1] <= max_SNR]
        img_groups = img_groups[img_groups[:, 1] >= min_SNR]
        cap_groups = cap_groups[cap_groups[:, 0] <= max_num_symbol]
        cap_groups = cap_groups[cap_groups[:, 0] >= min_num_symbol]
        cap_groups = cap_groups[cap_groups[:, 1] <= max_SNR]
        cap_groups = cap_groups[cap_groups[:, 1] >= min_SNR]
        if img_groups.shape[0] >= 1 and cap_groups.shape[0] >= 1:
            img_groups = list(img_groups)
            cap_groups = list(cap_groups)
            acc_list = [acc[f'symb{img_num_symbol}_{cap_num_symbol}_{acc_type}'][SNR_range.index(img_snr)][SNR_range.index(cap_snr)]
                       for img_num_symbol, img_snr in img_groups for cap_num_symbol, cap_snr in cap_groups]
            opt_acc = max(acc_list)
            return opt_acc
        else: return None
    else: return None




def get_checkpoint_openclip(path: str, model):
    from training.file_utils import pt_load
    checkpoint = pt_load(path, map_location='cpu')
    # loading a bare (model only) checkpoint for fine-tune or evaluation
    model.load_state_dict(checkpoint['state_dict'])
    return model