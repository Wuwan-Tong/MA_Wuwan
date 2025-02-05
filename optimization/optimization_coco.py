import numpy as np
import math
from noise_result.fix_val_noise.result_ft_oc_train_ae_coco import get_acc
from interpolation import fit_polynom_res_res_2D, get_acc_from_polynom_res_res_2D, upsampl_ori_acc_4D, upsampl_acc_res_numsym_4D, fit_gaussian_ninc, get_acc_fit_gaussian, fit_exp_ploynom
from scipy import optimize
from utils.utils import gaussian_2d_fit, gmm_2D, exp_ploynom_2D, is_monotonic
from fg_utils import get_best_g_1d, func_g, derivation_g

def transf_acc4D_snr2res(acc, channel_snr, snr_img, snr_cap, numsym_img, numsym_cap):
    '''
    Transform accuracy matrix: [snr_img, snr_cap, numsym_img, numsym_cap] into [res_img, res_cap, numsym_img, numsym_cap]
    :return: accuracy matrix:[res_img, res_cap, numsym_img, numsym_cap]
    '''
    acc4D_numsym_res = {}
    res_img = [get_res_from_sn(si, ni, channel_snr) for si in snr_img for ni in numsym_img]
    res_img = list(dict.fromkeys(res_img))
    res_img.sort()
    res_img = np.array(res_img)
    res_cap = [get_res_from_sn(sc, nc, channel_snr) for sc in snr_cap for nc in numsym_cap]
    res_cap = list(dict.fromkeys(res_cap))
    res_cap.sort()
    res_cap = np.array(res_cap)
    for ic in ['img', 'cap']:
        for r in [1, 5, 10]:
            acc4D_numsym_res[f'{ic}ret_r{r}'] = np.zeros((len(res_img), len(res_cap), len(numsym_img), len(numsym_cap)))
    for i_ni, ni in enumerate(numsym_img):
        for i_ri, ri in enumerate(res_img):
            for i_nc, nc in enumerate(numsym_cap):
                for i_rc, rc in enumerate(res_cap):
                    if ri < ni or rc < nc:
                        for ic in ['img', 'cap']:
                            for r in [1, 5, 10]:
                                acc4D_numsym_res[f'{ic}ret_r{r}'][i_ri, i_rc, i_ni, i_nc] = np.nan
                        continue
                    si = get_snr_from_res_numsym(channel_snr, ri, ni)
                    if si not in snr_img:
                        for ic in ['img', 'cap']:
                            for r in [1, 5, 10]:
                                acc4D_numsym_res[f'{ic}ret_r{r}'][i_ri, i_rc, i_ni, i_nc] = np.nan
                        continue
                    sc = get_snr_from_res_numsym(channel_snr, rc, nc)
                    if sc not in snr_cap:
                        for ic in ['img', 'cap']:
                            for r in [1, 5, 10]:
                                acc4D_numsym_res[f'{ic}ret_r{r}'][i_ri, i_rc, i_ni, i_nc] = np.nan
                        continue
                    for ic in ['img', 'cap']:
                        for r in [1, 5, 10]:
                            i_si = np.where(snr_img==si)[0]
                            i_sc = np.where(snr_cap==sc)[0]
                            acc4D_numsym_res[f'{ic}ret_r{r}'][i_ri, i_rc, i_ni, i_nc] = acc[f'{ic}ret_r{r}'][i_si, i_sc, i_ni, i_nc]

    return acc4D_numsym_res

def get_snr_from_res_numsym(channel_snr, res, numsym):
    """
    :param channel_snr: environment snr
    :param res: in dB
    :param numsym: in dB
    :return: snr
    """
    assert res >= numsym, 'res must >= num of symbol'
    return (res - numsym) * 6.0 + channel_snr

def get_res_from_siscninc(si, sc, ni, nc, channel_snr):
    """
    :param si: snr of img, in dB
    :param sc:
    :param ni: num of symbol of img, in dB
    :param nc:
    :param channel_snr:
    :return: total res, img res, cap res in dB
    """
    resi = (si - channel_snr) / 6 + ni
    resc = (sc - channel_snr) / 6 + nc
    res = math.log2(2**resi + 2**resc)
    return res, resi, resc
def get_res_from_sn(s, n, channel_snr):
    """
    :param s: snr of img, in dB
    :param n: num of symbol of img, in dB
    :param channel_snr:
    :return: total res, img res, cap res in dB
    """
    res = (s - channel_snr) / 6 + n
    return res

def get_opt_n(acc, res, si, sc, res_total, snr_img, snr_cap):
    """
    fix si, sc, find opt ni, nc
    :param acc: acc[si, sc, ni, nc]
    :param res: res[si, sc, ni, nc]
    :param res_total: constraints resi+resc <= res_total
    :return: opt ni, nc
    """
    si_idx = np.where(snr_img == si)[0]
    sc_idx = np.where(snr_cap == sc)[0]
    acc_2d = acc[si_idx, sc_idx, :, :]
    res_2d = res[si_idx, sc_idx, :, :]
    if res_2d[res_2d <= res_total].shape[0] == 0:
        return None, None, None
    acc_2d[res_2d > res_total] = -1
    acc = np.max(acc_2d)
    temp_loc = np.where(acc_2d == acc)
    ni_idx, nc_idx = temp_loc[1][0], temp_loc[2][0]
    return ni_idx, nc_idx, acc

def get_opt_s(acc, res, ni, nc, res_total, numsym_img, numsym_cap):
    """
    fix ni, nc, find opt si, sc
    :param acc: acc[si, sc, ni, nc]
    :param res: res[si, sc, ni, nc]
    :param res_total: constraints resi+resc <= res_total
    :return: opt si, sc
    """
    ni_idx = np.where(numsym_img == ni)[0]
    nc_idx = np.where(numsym_cap == nc)[0]
    acc_2d = acc[:, :, ni_idx, nc_idx]
    res_2d = res[:, :, ni_idx, nc_idx]
    if res_2d[res_2d <= res_total].shape[0] == 0:
        return None, None, None
    acc_2d[res_2d > res_total] = -1
    acc = np.max(acc_2d)
    temp_loc = np.where(acc_2d == acc)
    si_idx, sc_idx = temp_loc[0][0], temp_loc[1][0]
    return si_idx, sc_idx, acc

def opt_numsym_snr4D(acc, res, snr_img, snr_cap, numsym_img, numsym_cap, si_init, sc_init, res_total):
    """
    get optimal acc, snr_img, snr_cap, numsym_img, numsym_cap on original accuracy matrix without fitting
    """
    si_opt, sc_opt, ni_opt, nc_opt, acc_opt = {}, {}, {}, {}, {}
    for ic in ['img', 'cap']:
        for r in [1, 5, 10]:
            # init si (snr_img), sc (snr_cap), ni (numsym_img), nc (numsym_cap) opt and acc_opt
            # e.g.: si_opt[k] is the optimal snr_img in iter k, k is odd when fix si, sc, k is even when fix ni, nc
            si_opt[f'{ic}ret_r{r}'], sc_opt[f'{ic}ret_r{r}'], ni_opt[f'{ic}ret_r{r}'], nc_opt[f'{ic}ret_r{r}'] = [], [], [], []
            acc_opt[f'{ic}ret_r{r}'] = []

            # iterate optimization
            k = 0
            while len(acc_opt[f'{ic}ret_r{r}']) == 0 or acc_opt[f'{ic}ret_r{r}'][-1] - acc_opt[f'{ic}ret_r{r}'][-2] > acc_th:
                # fix si, sc, opt ni, nc
                if len(acc_opt[f'{ic}ret_r{r}']) == 0:
                    ni_opt_curr, nc_opt_curr, acc_opt_curr = get_opt_n(acc[f'{ic}ret_r{r}'], res, si_init, sc_init, res_total, snr_img, snr_cap)
                    while ni_opt_curr is None:
                        err_flag = True
                        if np.where(snr_img == si_init)[0] + 1 < len(snr_img):
                            si_init = snr_img[np.where(snr_img == si_init)[0] + 1]
                            err_flag = False
                        if np.where(snr_cap == sc_init)[0] + 1 < len(snr_cap):
                            sc_init = snr_cap[np.where(snr_cap == sc_init)[0] + 1]
                            err_flag = False
                        if not err_flag:
                            ni_opt_curr, nc_opt_curr, acc_opt_curr = get_opt_n(acc[f'{ic}ret_r{r}'], res, si_init, sc_init, res_total, snr_img, snr_cap)
                        else:
                            print('all SNRs and num of symbols are larger than res_total, please give a larger res_total')
                            si_opt[f'{ic}ret_r{r}'], sc_opt[f'{ic}ret_r{r}'], ni_opt[f'{ic}ret_r{r}'], nc_opt[f'{ic}ret_r{r}'], acc_opt[f'{ic}ret_r{r}'] = None, None, None, None, None
                            break
                    if ni_opt_curr is None:
                        break
                    si_opt[f'{ic}ret_r{r}'].append(si_init)
                    sc_opt[f'{ic}ret_r{r}'].append(sc_init)
                else:
                    ni_opt_curr, nc_opt_curr, acc_opt_curr = get_opt_n(acc[f'{ic}ret_r{r}'], res, si_opt[f'{ic}ret_r{r}'][-1], sc_opt[f'{ic}ret_r{r}'][-1], res_total, snr_img, snr_cap)
                    si_opt[f'{ic}ret_r{r}'].append(si_opt[f'{ic}ret_r{r}'][-1])
                    sc_opt[f'{ic}ret_r{r}'].append(sc_opt[f'{ic}ret_r{r}'][-1])
                ni_opt[f'{ic}ret_r{r}'].append(numsym_img[ni_opt_curr])
                nc_opt[f'{ic}ret_r{r}'].append(numsym_cap[nc_opt_curr])
                acc_opt[f'{ic}ret_r{r}'].append(acc_opt_curr)
                k += 1

                # check break condition
                if len(acc_opt[f'{ic}ret_r{r}']) >= 2 and acc_opt[f'{ic}ret_r{r}'][-1] - acc_opt[f'{ic}ret_r{r}'][-2] < acc_th:
                    print(f'{ic}ret_r{r}')
                    for i_print in range(len(ni_opt[f'{ic}ret_r{r}'])):
                        print('si: %.2f, sc: %.2f, ni: %.2f, nc: %.2f, acc: %.2f' % (si_opt[f'{ic}ret_r{r}'][i_print], sc_opt[f'{ic}ret_r{r}'][i_print], ni_opt[f'{ic}ret_r{r}'][i_print], nc_opt[f'{ic}ret_r{r}'][i_print], acc_opt[f'{ic}ret_r{r}'][i_print]))
                    break

                # fix ni, nc, opt si, sc
                si_opt_curr, sc_opt_curr, acc_opt_curr = get_opt_s(acc[f'{ic}ret_r{r}'], res, ni_opt[f'{ic}ret_r{r}'][-1], nc_opt[f'{ic}ret_r{r}'][-1], res_total, numsym_img, numsym_cap)
                ni_opt[f'{ic}ret_r{r}'].append(ni_opt[f'{ic}ret_r{r}'][-1])
                nc_opt[f'{ic}ret_r{r}'].append(nc_opt[f'{ic}ret_r{r}'][-1])

                si_opt[f'{ic}ret_r{r}'].append(snr_img[si_opt_curr])
                sc_opt[f'{ic}ret_r{r}'].append(snr_cap[sc_opt_curr])
                acc_opt[f'{ic}ret_r{r}'].append(acc_opt_curr)
                k += 1

                # check break condition
                if acc_opt[f'{ic}ret_r{r}'][-1] - acc_opt[f'{ic}ret_r{r}'][-2] < acc_th:
                    print(f'{ic}ret_r{r}')
                    for i_print in range(len(ni_opt[f'{ic}ret_r{r}'])):
                        print('si: %.2f, sc: %.2f, ni: %.2f, nc: %.2f, acc: %.2f' % (si_opt[f'{ic}ret_r{r}'][i_print], sc_opt[f'{ic}ret_r{r}'][i_print], ni_opt[f'{ic}ret_r{r}'][i_print], nc_opt[f'{ic}ret_r{r}'][i_print], acc_opt[f'{ic}ret_r{r}'][i_print]))
                    break
    return si_opt, sc_opt, ni_opt, nc_opt, acc_opt

def opt_numsym_res4D(acc, snr_img, snr_cap, numsym_img, numsym_cap, ri_init, rc_init, res_total):
    """
    get optimal acc, res_img, res_cap, numsym_img, numsym_cap without fitting
    """
    assert 2 ** ri_init + 2 ** rc_init <= 2 ** res_total, 'init res_img + res_cap must <= res_total'
    acc = transf_acc4D_snr2res(acc, channel_snr, snr_img, snr_cap, numsym_img, numsym_cap)
    ri_opt, rc_opt, ni_opt, nc_opt, acc_opt = {}, {}, {}, {}, {}
    res_img = [get_res_from_sn(si, ni, channel_snr) for si in snr_img for ni in numsym_img]
    res_img = list(dict.fromkeys(res_img))
    res_img.sort()
    res_img = np.array(res_img)
    res_cap = [get_res_from_sn(sc, nc, channel_snr) for sc in snr_cap for nc in numsym_cap]
    res_cap = list(dict.fromkeys(res_cap))
    res_cap.sort()
    res_cap = np.array(res_cap)
    for ic in ['img', 'cap']:
        for r in [1, 5, 10]:
            # init ri (res_img), rc (res_cap), ni (numsym_img), nc (numsym_cap) opt and acc_opt
            # e.g.: si_opt[k] is the optimal snr_img in iter k, k is odd when fix si, sc, k is even when fix ni, nc
            ri_opt[f'{ic}ret_r{r}'], rc_opt[f'{ic}ret_r{r}'], ni_opt[f'{ic}ret_r{r}'], nc_opt[f'{ic}ret_r{r}'] = [], [], [], []
            acc_opt[f'{ic}ret_r{r}'] = []

            # iterate optimization
            k = 0
            while len(acc_opt[f'{ic}ret_r{r}']) == 0 or acc_opt[f'{ic}ret_r{r}'][-1] - acc_opt[f'{ic}ret_r{r}'][-2] > acc_th:
                # fix ri, rc, opt ni, nc
                if len(acc_opt[f'{ic}ret_r{r}']) == 0:
                    # i_ri = np.where(res_img == ri_init)[0]
                    # i_rc = np.where(res_cap == rc_init)[0]
                    i_ri = (np.abs(res_img - ri_init)).argmin()
                    i_rc = (np.abs(res_cap - rc_init)).argmin()
                    temp_acc_2D = acc[f'{ic}ret_r{r}'][i_ri, i_rc, :, :]
                    acc_opt_curr = np.nanmax(temp_acc_2D)
                    n_temp = np.where(temp_acc_2D == acc_opt_curr)
                    i_ni_opt_curr = n_temp[0][0]
                    i_nc_opt_curr = n_temp[1][0]
                    ri_opt[f'{ic}ret_r{r}'].append(res_img[i_ri])
                    rc_opt[f'{ic}ret_r{r}'].append(res_cap[i_rc])
                else:
                    i_ri = np.where(res_img == ri_opt[f'{ic}ret_r{r}'][-1])[0]
                    i_rc = np.where(res_cap == rc_opt[f'{ic}ret_r{r}'][-1])[0]
                    temp_acc_2D = acc[f'{ic}ret_r{r}'][i_ri, i_rc, :, :]
                    acc_opt_curr = np.nanmax(temp_acc_2D)
                    n_temp = np.where(temp_acc_2D == acc_opt_curr)
                    i_ni_opt_curr = n_temp[1][0]
                    i_nc_opt_curr = n_temp[2][0]
                    ri_opt[f'{ic}ret_r{r}'].append(ri_opt[f'{ic}ret_r{r}'][-1])
                    rc_opt[f'{ic}ret_r{r}'].append(rc_opt[f'{ic}ret_r{r}'][-1])

                ni_opt[f'{ic}ret_r{r}'].append(numsym_img[i_ni_opt_curr])
                nc_opt[f'{ic}ret_r{r}'].append(numsym_cap[i_nc_opt_curr])
                acc_opt[f'{ic}ret_r{r}'].append(acc_opt_curr)
                k += 1

                # check break condition
                if len(acc_opt[f'{ic}ret_r{r}']) >= 2 and acc_opt[f'{ic}ret_r{r}'][-1] - acc_opt[f'{ic}ret_r{r}'][-2] < acc_th:
                    print(f'{ic}ret_r{r}')
                    for i_print in range(len(ni_opt[f'{ic}ret_r{r}'])):
                        print('ri: %.2f, rc: %.2f, si: %.2f, sc: %.2f, ni: %.2f, nc: %.2f, acc: %.2f' % (
                            ri_opt[f'{ic}ret_r{r}'][i_print], rc_opt[f'{ic}ret_r{r}'][i_print],
                            get_snr_from_res_numsym(channel_snr, ri_opt[f'{ic}ret_r{r}'][i_print], ni_opt[f'{ic}ret_r{r}'][i_print]),
                            get_snr_from_res_numsym(channel_snr, rc_opt[f'{ic}ret_r{r}'][i_print], nc_opt[f'{ic}ret_r{r}'][i_print]),
                            ni_opt[f'{ic}ret_r{r}'][i_print], nc_opt[f'{ic}ret_r{r}'][i_print], acc_opt[f'{ic}ret_r{r}'][i_print]))
                    break

                # fix ni, nc, opt si, sc
                i_ni = np.where(numsym_img == ni_opt[f'{ic}ret_r{r}'][-1])[0]
                i_nc = np.where(numsym_cap == nc_opt[f'{ic}ret_r{r}'][-1])[0]
                temp_acc_2D = acc[f'{ic}ret_r{r}'][:, :, i_ni, i_nc]
                for i_ri in range(len(res_img)):
                    for i_rc in range(len(res_cap)):
                        if 2 ** res_img[i_ri] + 2 ** res_cap[i_rc] > 2 ** res_total:
                            temp_acc_2D[i_ri, i_rc, :] = np.nan
                while np.isnan(temp_acc_2D).all():
                    err_flag = True
                    if i_ni>0:
                        i_ni -= 1
                        err_flag = False
                    if i_nc>0:
                        i_nc -= 1
                        err_flag = False
                    if not err_flag:
                        temp_acc_2D = acc[f'{ic}ret_r{r}'][:, :, i_ni, i_nc]
                        for i_ri in range(len(res_img)):
                            for i_rc in range(len(res_cap)):
                                if 2 ** res_img[i_ri] + 2 ** res_cap[i_rc] > 2 ** res_total:
                                    temp_acc_2D[i_ri, i_rc, :] = np.nan
                    else:
                        print('all reses and num of symbols are larger than res_total, please give a larger res_total')
                        for ic in ['img', 'cap']:
                            for r in [1, 5, 10]:
                                ri_opt[f'{ic}ret_r{r}'], rc_opt[f'{ic}ret_r{r}'], ni_opt[f'{ic}ret_r{r}'], nc_opt[f'{ic}ret_r{r}'], acc_opt[f'{ic}ret_r{r}'] = None, None, None, None, None
                        return ri_opt, rc_opt, ni_opt, nc_opt, acc_opt
                acc_opt_curr = np.nanmax(temp_acc_2D)
                n_temp = np.where(temp_acc_2D == acc_opt_curr)
                i_ri_opt_curr = n_temp[0][0]
                i_rc_opt_curr = n_temp[1][0]
                ni_opt[f'{ic}ret_r{r}'].append(ni_opt[f'{ic}ret_r{r}'][-1])
                nc_opt[f'{ic}ret_r{r}'].append(nc_opt[f'{ic}ret_r{r}'][-1])

                ri_opt[f'{ic}ret_r{r}'].append(res_img[i_ri_opt_curr])
                rc_opt[f'{ic}ret_r{r}'].append(res_cap[i_rc_opt_curr])
                acc_opt[f'{ic}ret_r{r}'].append(acc_opt_curr)
                k += 1

                # check break condition
                if acc_opt[f'{ic}ret_r{r}'][-1] - acc_opt[f'{ic}ret_r{r}'][-2] < acc_th:
                    print(f'{ic}ret_r{r}')
                    for i_print in range(len(ni_opt[f'{ic}ret_r{r}'])):
                        print('ri: %.2f, rc: %.2f, si: %.2f, sc: %.2f, ni: %.2f, nc: %.2f, acc: %.2f' % (
                        ri_opt[f'{ic}ret_r{r}'][i_print], rc_opt[f'{ic}ret_r{r}'][i_print],
                        get_snr_from_res_numsym(channel_snr, ri_opt[f'{ic}ret_r{r}'][i_print], ni_opt[f'{ic}ret_r{r}'][i_print]),
                        get_snr_from_res_numsym(channel_snr, rc_opt[f'{ic}ret_r{r}'][i_print], nc_opt[f'{ic}ret_r{r}'][i_print]),
                        ni_opt[f'{ic}ret_r{r}'][i_print], nc_opt[f'{ic}ret_r{r}'][i_print], acc_opt[f'{ic}ret_r{r}'][i_print]))
                    break

    return ri_opt, rc_opt, ni_opt, nc_opt, acc_opt


def opt_num_res_intp_4D(acc_ori, snr_img, snr_cap, numsym_img, numsym_cap, step_num_res, step_num_numsym, ri_init, rc_init, res_total, channel_snr, max_n=None, m_gmm=None):
    """
    get optimal acc, res_img, res_cap, numsym_img, numsym_cap with fitting into Gaussian or polynom
    """
    assert 2 ** ri_init + 2 ** rc_init <= 2 ** res_total, 'init res_img + res_cap must <= res_total'
    acc_sr = transf_acc4D_snr2res(acc_ori, channel_snr, snr_img, snr_cap, numsym_img, numsym_cap)

    numsym_img_upsampl = np.linspace(numsym_img[0], numsym_img[-1] + (numsym_img[-1] - numsym_img[0]) / (len(numsym_img)-1) / step_num_numsym * (step_num_numsym - 1), step_num_numsym * (len(numsym_img)))
    numsym_cap_upsampl = np.linspace(numsym_cap[0], numsym_cap[-1] + (numsym_cap[-1] - numsym_cap[0]) / (len(numsym_cap)-1) / step_num_numsym * (step_num_numsym - 1), step_num_numsym * (len(numsym_cap)))

    acc_upsampl_sr = upsampl_acc_res_numsym_4D(acc_sr, step_num_res, step_num_numsym)
    ri_opt, rc_opt, ni_opt, nc_opt, acc_opt, acc_gt, acc_diff = {}, {}, {}, {}, {}, {}, {}
    res_img_min = get_res_from_sn(snr_img[-1], numsym_img[0], channel_snr)
    res_cap_min = get_res_from_sn(snr_cap[-1], numsym_cap[0], channel_snr)
    res_img_max = get_res_from_sn(snr_img[0], numsym_img[-1], channel_snr)
    res_cap_max = get_res_from_sn(snr_cap[0], numsym_cap[-1], channel_snr)

    res_img = np.linspace(res_img_min, res_img_max, int(acc_upsampl_sr['imgret_r1'].shape[0] / step_num_res))
    res_cap = np.linspace(res_cap_min, res_cap_max, int(acc_upsampl_sr['imgret_r1'].shape[1] / step_num_res))
    res_img_upsampl = np.linspace(res_img_min, res_img_max + (res_img_max - res_img_min) / (len(res_img)-1) / step_num_res * (step_num_res - 1), acc_upsampl_sr['imgret_r1'].shape[0])
    res_cap_upsampl = np.linspace(res_cap_min, res_cap_max + (res_cap_max - res_cap_min) / (len(res_cap)-1) / step_num_res * (step_num_res - 1), acc_upsampl_sr['imgret_r1'].shape[1])

    for ic in ['img', 'cap']:
        for r in [1, 5, 10]:
            # init ri (res_img), rc (res_cap), ni (numsym_img), nc (numsym_cap) opt and acc_opt
            # e.g.: si_opt[k] is the optimal snr_img in iter k, k is odd when fix si, sc, k is even when fix ni, nc
            ri_opt[f'{ic}ret_r{r}'], rc_opt[f'{ic}ret_r{r}'], ni_opt[f'{ic}ret_r{r}'], nc_opt[f'{ic}ret_r{r}'] = [], [], [], []
            acc_opt[f'{ic}ret_r{r}'], acc_gt[f'{ic}ret_r{r}'], acc_diff[f'{ic}ret_r{r}'] = [], [], []

            # iterate optimization
            k = 0
            while True: #len(acc_opt[f'{ic}ret_r{r}']) == 0 or np.abs(acc_opt[f'{ic}ret_r{r}'][-1] - max(acc_opt[f'{ic}ret_r{r}'][:-1])) > acc_th:
                # fix ri, rc, opt ni, nc
                if len(acc_opt[f'{ic}ret_r{r}']) == 0:
                    i_ri = (np.abs(res_img_upsampl - ri_init)).argmin()
                    i_rc = (np.abs(res_cap_upsampl - rc_init)).argmin()
                    temp_acc_2D = acc_upsampl_sr[f'{ic}ret_r{r}'][i_ri, i_rc, :, :]
                    while np.isnan(temp_acc_2D).all():
                        err_flag = True
                        if i_ri < len(res_img_upsampl):
                            i_ri += 1
                            err_flag = False
                        if i_rc < len(res_cap_upsampl) and 2 ** res_img_upsampl[i_ri] + 2 ** res_cap_upsampl[i_rc] <= 2 ** res_total:
                            i_rc += 1
                            err_flag = False
                        if not err_flag:
                            temp_acc_2D = acc_upsampl_sr[f'{ic}ret_r{r}'][i_ri, i_rc, :, :]
                        else:
                            print('all res and num of symbols are larger than res_total, please give a larger res_total')
                            for ic in ['img', 'cap']:
                                for r in [1, 5, 10]:
                                    ri_opt[f'{ic}ret_r{r}'], rc_opt[f'{ic}ret_r{r}'], ni_opt[f'{ic}ret_r{r}'], nc_opt[f'{ic}ret_r{r}'], acc_opt[f'{ic}ret_r{r}'] = None, None, None, None, None
                            return ri_opt, rc_opt, ni_opt, nc_opt, acc_opt

                    temp_acc_2D_downs = temp_acc_2D[::step_num_res, ::step_num_res]
                    if temp_acc_2D_downs[~np.isnan(temp_acc_2D_downs)].size == 1:
                        acc_opt_curr = temp_acc_2D_downs[~np.isnan(temp_acc_2D_downs)][0]
                        n_temp = np.where(temp_acc_2D_downs == acc_opt_curr)
                        if len(n_temp) > 1:
                            i_ni_opt_curr = n_temp[0][-1] * step_num_numsym
                            i_nc_opt_curr = n_temp[1][-1] * step_num_numsym
                        else:
                            i_ni_opt_curr = n_temp[0] * step_num_numsym
                            i_nc_opt_curr = n_temp[1] * step_num_numsym
                        ri_opt[f'{ic}ret_r{r}'].append(res_img_upsampl[i_ri])
                        rc_opt[f'{ic}ret_r{r}'].append(res_cap_upsampl[i_rc])
                    else:
                        # if is_monotonic(temp_acc_2D[0, ::step_num_numsym, ::step_num_numsym]):
                        #     acc_opt_curr = np.nanmax(temp_acc_2D[0, ::step_num_numsym, ::step_num_numsym])
                        #     n_temp = np.where(temp_acc_2D[0, :, :] == acc_opt_curr)
                        if is_monotonic(temp_acc_2D_downs):
                            acc_opt_curr = np.nanmax(temp_acc_2D_downs)
                            n_temp = np.where(temp_acc_2D_downs == acc_opt_curr)
                            if len(n_temp) > 1:
                                i_ni_opt_curr = n_temp[0][-1] * step_num_numsym
                                i_nc_opt_curr = n_temp[1][-1] * step_num_numsym
                            else:
                                i_ni_opt_curr = n_temp[0] * step_num_numsym
                                i_nc_opt_curr = n_temp[1] * step_num_numsym
                        else:
                            # gaussian
                            popt = fit_gaussian_ninc(temp_acc_2D[::step_num_numsym, ::step_num_numsym], numsym_img, numsym_cap, m_gmm)
                            temp_acc_2D_intp = get_acc_fit_gaussian(numsym_img_upsampl, numsym_cap_upsampl, popt.x, m_gmm)
                            # polynom
                            # coeff = fit_polynom_res_res_2D(temp_acc_2D[::step_num_numsym, ::step_num_numsym], numsym_img, numsym_cap, max_n)
                            # temp_acc_2D_intp = get_acc_from_polynom_res_res_2D(coeff, numsym_img_upsampl, numsym_cap_upsampl, max_n)
                            temp_acc_2D_intp[np.isnan(temp_acc_2D[:, :])] = np.nan
                            acc_opt_curr = np.nanmax(temp_acc_2D_intp[:-step_num_numsym+1,:-step_num_numsym+1])
                            n_temp = np.where(temp_acc_2D_intp == acc_opt_curr)
                            i_ni_opt_curr = n_temp[0][-1]
                            i_nc_opt_curr = n_temp[1][-1]

                        ri_opt[f'{ic}ret_r{r}'].append(res_img_upsampl[i_ri])
                        rc_opt[f'{ic}ret_r{r}'].append(res_cap_upsampl[i_rc])
                else:
                    i_ri = np.where(res_img_upsampl == ri_opt[f'{ic}ret_r{r}'][-1])[0]
                    i_rc = np.where(res_cap_upsampl == rc_opt[f'{ic}ret_r{r}'][-1])[0]
                    temp_acc_2D = acc_upsampl_sr[f'{ic}ret_r{r}'][i_ri, i_rc, :, :]
                    # # find best g
                    # ttnn = temp_acc_2D[0, ::step_num_numsym, ::step_num_numsym]
                    # acc_g = get_best_g(ttnn, numsym_img, numsym_cap)
                    # print(ttnn)
                    # print(acc_g)
                    if is_monotonic(temp_acc_2D[0, ::step_num_numsym, ::step_num_numsym]):
                        acc_opt_curr = np.nanmax(temp_acc_2D[0, ::step_num_numsym, ::step_num_numsym])
                        n_temp = np.where(temp_acc_2D[0, ::step_num_numsym, ::step_num_numsym] == acc_opt_curr)
                        i_ni_opt_curr = n_temp[0][-1] * step_num_numsym
                        i_nc_opt_curr = n_temp[1][-1] * step_num_numsym
                    else:
                        # gaussian
                        # popt = fit_gaussian_ninc(temp_acc_2D[0, ::step_num_numsym, ::step_num_numsym], numsym_img, numsym_cap, m_gmm)
                        # temp_acc_2D_intp = get_acc_fit_gaussian(numsym_img_upsampl, numsym_cap_upsampl, popt.x, m_gmm)
                        # polynom
                        coeff = fit_polynom_res_res_2D(temp_acc_2D[0, ::step_num_numsym, ::step_num_numsym], numsym_img, numsym_cap, max_n)
                        temp_acc_2D_intp = get_acc_from_polynom_res_res_2D(coeff, numsym_img_upsampl, numsym_cap_upsampl, max_n)
                        temp_acc_2D_intp[np.isnan(temp_acc_2D[0, :, :])] = np.nan
                        acc_opt_curr = np.nanmax(temp_acc_2D_intp[:-step_num_numsym+1,:-step_num_numsym+1])
                        n_temp = np.where(temp_acc_2D_intp == acc_opt_curr)
                        i_ni_opt_curr = n_temp[0][-1]
                        i_nc_opt_curr = n_temp[1][-1]

                    ri_opt[f'{ic}ret_r{r}'].append(ri_opt[f'{ic}ret_r{r}'][-1])
                    rc_opt[f'{ic}ret_r{r}'].append(rc_opt[f'{ic}ret_r{r}'][-1])

                ni_opt[f'{ic}ret_r{r}'].append(numsym_img_upsampl[i_ni_opt_curr])
                nc_opt[f'{ic}ret_r{r}'].append(numsym_cap_upsampl[i_nc_opt_curr])
                acc_opt[f'{ic}ret_r{r}'].append(acc_opt_curr)
                k += 1

                # check break condition
                if (len(acc_opt[f'{ic}ret_r{r}']) >= 2 and np.abs(acc_opt[f'{ic}ret_r{r}'][-1] - max(acc_opt[f'{ic}ret_r{r}'][:-1])) < acc_th) or k > 20: # or (np.abs(acc_opt[f'{ic}ret_r{r}'][-1]-acc_opt[f'{ic}ret_r{r}'][-3])<0.001 and np.abs(acc_opt[f'{ic}ret_r{r}'][-2]-acc_opt[f'{ic}ret_r{r}'][-4])<0.001):
                    print(f'{ic}ret_r{r}')
                    flag_res_total = False
                    for i_print in range(len(ni_opt[f'{ic}ret_r{r}'])):
                        i_ri_temp = (np.abs(res_img - ri_opt[f'{ic}ret_r{r}'][i_print])).argmin()
                        i_rc_temp = (np.abs(res_cap - rc_opt[f'{ic}ret_r{r}'][i_print])).argmin()
                        i_ni_temp = (np.abs(numsym_img - ni_opt[f'{ic}ret_r{r}'][i_print])).argmin()
                        i_nc_temp = (np.abs(numsym_cap - nc_opt[f'{ic}ret_r{r}'][i_print])).argmin()
                        flag_res_total = False
                        if ri_opt[f'{ic}ret_r{r}'][i_print] < res_img[i_ri_temp]:
                            i_ri = [i_ri_temp - 1, i_ri_temp]
                        elif ri_opt[f'{ic}ret_r{r}'][i_print] == res_img[i_ri_temp] or i_ri_temp == len(res_img) - 1:
                            i_ri = [i_ri_temp, i_ri_temp]
                        else:
                            if 2 ** (i_ri_temp + 1) + 2 ** i_rc_temp > res_total:
                                i_ri = [i_ri_temp, i_ri_temp]
                            else:
                                i_ri = [i_ri_temp, i_ri_temp + 1]
                                flag_res_total = True

                        if rc_opt[f'{ic}ret_r{r}'][i_print] < res_cap[i_rc_temp]:
                            i_rc = [i_rc_temp - 1, i_rc_temp]
                        elif rc_opt[f'{ic}ret_r{r}'][i_print] == res_cap[i_rc_temp] or i_rc_temp == len(res_cap) - 1:
                            i_rc = [i_rc_temp, i_rc_temp]
                        else:
                            if 2 ** (i_rc_temp + 1) + 2 ** i_ri_temp > res_total:
                                i_rc = [i_rc_temp, i_rc_temp]
                            else:
                                i_rc = [i_rc_temp, i_rc_temp + 1]
                                flag_res_total = True

                        if flag_res_total and 2 ** (i_rc_temp + 1) + 2 ** (i_ri_temp + 1) > res_total:
                            flag_res_total = True
                        else:
                            flag_res_total = False

                        if ni_opt[f'{ic}ret_r{r}'][i_print] < numsym_img[i_ni_temp]:
                            i_ni = [i_ni_temp - 1, i_ni_temp]
                        elif ni_opt[f'{ic}ret_r{r}'][i_print] == numsym_img[i_ni_temp] or i_ni_temp == len(numsym_img) - 1:
                            i_ni = [i_ni_temp, i_ni_temp]
                        else:
                            i_ni = [i_ni_temp, i_ni_temp + 1]

                        if nc_opt[f'{ic}ret_r{r}'][i_print] < numsym_cap[i_nc_temp]:
                            i_nc = [i_nc_temp - 1, i_nc_temp]
                        elif nc_opt[f'{ic}ret_r{r}'][i_print] == numsym_cap[i_nc_temp] or i_nc_temp == len(numsym_cap) - 1:
                            i_nc = [i_nc_temp, i_nc_temp]
                        else:
                            i_nc = [i_nc_temp, i_nc_temp + 1]

                        rircninc = [[i_ri[0], i_rc[0], i_ni[0], i_nc[0]], [i_ri[1], i_rc[0], i_ni[0], i_nc[0]],
                                    [i_ri[0], i_rc[1], i_ni[0], i_nc[0]], [i_ri[1], i_rc[1], i_ni[0], i_nc[0]],
                                    [i_ri[0], i_rc[0], i_ni[1], i_nc[0]], [i_ri[1], i_rc[0], i_ni[1], i_nc[0]],
                                    [i_ri[0], i_rc[0], i_ni[0], i_nc[1]], [i_ri[1], i_rc[0], i_ni[0], i_nc[1]],
                                    [i_ri[0], i_rc[1], i_ni[1], i_nc[0]], [i_ri[1], i_rc[1], i_ni[1], i_nc[0]],
                                    [i_ri[0], i_rc[1], i_ni[0], i_nc[1]], [i_ri[1], i_rc[1], i_ni[0], i_nc[1]],
                                    [i_ri[0], i_rc[0], i_ni[1], i_nc[1]], [i_ri[1], i_rc[0], i_ni[1], i_nc[1]],
                                    [i_ri[0], i_rc[1], i_ni[1], i_nc[1]], [i_ri[1], i_rc[1], i_ni[1], i_nc[1]]]
                        acc_candidate = np.array([acc_sr[f'{ic}ret_r{r}'][iri, irc, ini, inc] for iri, irc, ini, inc in rircninc])
                        if flag_res_total:
                            acc_candidate[[-1, -3, -4, -7]] = np.nan
                        temp_acc_gt = np.nanmax(acc_candidate)
                        (iri, irc, ini, inc) = rircninc[np.where(acc_candidate == temp_acc_gt)[0][0]]
                        ri, rc, ni, nc = res_img[iri], res_cap[irc], numsym_img[ini], numsym_cap[inc]
                        print('ri: %.2f, rc: %.2f, si: %.2f, sc: %.2f, ni: %.2f, nc: %.2f, \n acc: %.2f, temp_acc_gt: %.2f, acc_diff: %.2f, ri: %.2f, rc: %.2f, ni: %.2f, nc: %.2f' % (
                            ri_opt[f'{ic}ret_r{r}'][i_print], rc_opt[f'{ic}ret_r{r}'][i_print],
                            get_snr_from_res_numsym(channel_snr, ri_opt[f'{ic}ret_r{r}'][i_print], ni_opt[f'{ic}ret_r{r}'][i_print]),
                            get_snr_from_res_numsym(channel_snr, rc_opt[f'{ic}ret_r{r}'][i_print], nc_opt[f'{ic}ret_r{r}'][i_print]),
                            ni_opt[f'{ic}ret_r{r}'][i_print], nc_opt[f'{ic}ret_r{r}'][i_print], acc_opt[f'{ic}ret_r{r}'][i_print], temp_acc_gt, np.abs(acc_opt[f'{ic}ret_r{r}'][i_print] - temp_acc_gt),
                            ri, rc, ni, nc))
                        acc_gt[f'{ic}ret_r{r}'].append(temp_acc_gt)
                        acc_diff[f'{ic}ret_r{r}'].append(np.abs(acc_opt[f'{ic}ret_r{r}'][i_print] - temp_acc_gt))
                    break

                # fix ni, nc, opt si, sc
                i_ni = np.where(numsym_img_upsampl == ni_opt[f'{ic}ret_r{r}'][-1])[0]
                i_nc = np.where(numsym_cap_upsampl == nc_opt[f'{ic}ret_r{r}'][-1])[0]
                temp_acc_2D = acc_upsampl_sr[f'{ic}ret_r{r}'][:, :, i_ni, i_nc]
                for i_ri in range(len(res_img_upsampl)):
                    for i_rc in range(len(res_cap_upsampl)):
                        if 2 ** res_img_upsampl[i_ri] + 2 ** res_cap_upsampl[i_rc] > 2 ** res_total:
                            # if not np.isnan(temp_acc_2D[i_ri, i_rc, :]):
                            #     a=1
                            temp_acc_2D[i_ri, i_rc, :] = np.nan

                while np.isnan(temp_acc_2D).all():
                    err_flag = True
                    if i_ni > 0:
                        i_ni -= 1
                        err_flag = False
                    if i_nc > 0:
                        i_nc -= 1
                        err_flag = False
                    if not err_flag:
                        temp_acc_2D = acc_upsampl_sr[f'{ic}ret_r{r}'][:, :, i_ni, i_nc]
                        for i_ri in range(len(res_img_upsampl)):
                            for i_rc in range(len(res_cap_upsampl)):
                                if 2 ** res_img_upsampl[i_ri] + 2 ** res_cap_upsampl[i_rc] > 2 ** res_total:
                                    temp_acc_2D[i_ri, i_rc, :] = np.nan
                    else:
                        print('all res and num of symbols are larger than res_total, please give a larger res_total')
                        for ic in ['img', 'cap']:
                            for r in [1, 5, 10]:
                                ri_opt[f'{ic}ret_r{r}'], rc_opt[f'{ic}ret_r{r}'], ni_opt[f'{ic}ret_r{r}'], nc_opt[f'{ic}ret_r{r}'], acc_opt[f'{ic}ret_r{r}'] = None, None, None, None, None
                        return ri_opt, rc_opt, ni_opt, nc_opt, acc_opt
                # todo: change intp func ri, rc
                ttrr = temp_acc_2D[::step_num_res, ::step_num_res, 0]
                ##### no intp no opt
                acc_opt_curr = np.nanmax(temp_acc_2D[::step_num_res, ::step_num_res, 0])
                n_temp = np.where(temp_acc_2D[::step_num_res, ::step_num_res, 0] == acc_opt_curr)
                # if is_monotonic(temp_acc_2D[::step_num_res, ::step_num_res, 0]):
                #     acc_opt_curr = np.nanmax(temp_acc_2D[::step_num_res, ::step_num_res, 0])
                #     n_temp = np.where(temp_acc_2D[:, :, 0] == acc_opt_curr)
                # else:
                #     # gaussian
                #     # popt = fit_gaussian_ninc(temp_acc_2D[::step_num_res, ::step_num_res, 0], res_img, res_cap, m_gmm)
                #     # temp_acc_2D_intp = get_acc_fit_gaussian(res_img_upsampl, res_cap_upsampl, popt.x, m_gmm)
                #     # polynom
                #     coeff = fit_polynom_res_res_2D(temp_acc_2D[::step_num_res, ::step_num_res, 0], res_img, res_cap, max_n)
                #     temp_acc_2D_intp = get_acc_from_polynom_res_res_2D(coeff, res_img_upsampl, res_cap_upsampl, max_n)
                #     temp_acc_2D_intp[np.isnan(temp_acc_2D[:, :, 0])] = np.nan
                #     acc_opt_curr = np.nanmax(temp_acc_2D_intp)
                #     n_temp = np.where(temp_acc_2D_intp == acc_opt_curr)

                i_ri_opt_curr = n_temp[0][-1] * step_num_res
                i_rc_opt_curr = n_temp[1][-1] * step_num_res
                ni_opt[f'{ic}ret_r{r}'].append(ni_opt[f'{ic}ret_r{r}'][-1])
                nc_opt[f'{ic}ret_r{r}'].append(nc_opt[f'{ic}ret_r{r}'][-1])

                ri_opt[f'{ic}ret_r{r}'].append(res_img_upsampl[i_ri_opt_curr])
                rc_opt[f'{ic}ret_r{r}'].append(res_cap_upsampl[i_rc_opt_curr])
                acc_opt[f'{ic}ret_r{r}'].append(acc_opt_curr)
                k += 1

                # check break condition
                if np.abs(acc_opt[f'{ic}ret_r{r}'][-1] - max(acc_opt[f'{ic}ret_r{r}'][:-1])) < acc_th or k > 20: # or (np.abs(acc_opt[f'{ic}ret_r{r}'][-1]-acc_opt[f'{ic}ret_r{r}'][-3])<0.001 and np.abs(acc_opt[f'{ic}ret_r{r}'][-2]-acc_opt[f'{ic}ret_r{r}'][-4])<0.001):
                    print(f'{ic}ret_r{r}')
                    for i_print in range(len(ni_opt[f'{ic}ret_r{r}'])):
                        i_ri_temp = (np.abs(res_img - ri_opt[f'{ic}ret_r{r}'][i_print])).argmin()
                        i_rc_temp = (np.abs(res_cap - rc_opt[f'{ic}ret_r{r}'][i_print])).argmin()
                        i_ni_temp = (np.abs(numsym_img - ni_opt[f'{ic}ret_r{r}'][i_print])).argmin()
                        i_nc_temp = (np.abs(numsym_cap - nc_opt[f'{ic}ret_r{r}'][i_print])).argmin()
                        flag_res_total = False
                        if ri_opt[f'{ic}ret_r{r}'][i_print] < res_img[i_ri_temp]:
                            i_ri = [i_ri_temp - 1, i_ri_temp]
                        elif ri_opt[f'{ic}ret_r{r}'][i_print] == res_img[i_ri_temp] or i_ri_temp == len(res_img) - 1:
                            i_ri = [i_ri_temp, i_ri_temp]
                        else:
                            if 2 ** (i_ri_temp + 1) + 2 ** i_rc_temp > res_total:
                                i_ri = [i_ri_temp, i_ri_temp]
                            else:
                                i_ri = [i_ri_temp, i_ri_temp + 1]
                                flag_res_total = True

                        if rc_opt[f'{ic}ret_r{r}'][i_print] < res_cap[i_rc_temp]:
                            i_rc = [i_rc_temp - 1, i_rc_temp]
                        elif rc_opt[f'{ic}ret_r{r}'][i_print] == res_cap[i_rc_temp] or i_rc_temp == len(res_cap) - 1:
                            i_rc = [i_rc_temp, i_rc_temp]
                        else:
                            if 2 ** (i_rc_temp + 1) + 2 ** i_ri_temp > res_total:
                                i_rc = [i_rc_temp, i_rc_temp]
                            else:
                                i_rc = [i_rc_temp, i_rc_temp + 1]
                                flag_res_total = True

                        if flag_res_total and 2 ** (i_rc_temp + 1) + 2 ** (i_ri_temp + 1) > res_total:
                            flag_res_total = True
                        else:
                            flag_res_total = False

                        if ni_opt[f'{ic}ret_r{r}'][i_print] < numsym_img[i_ni_temp]:
                            i_ni = [i_ni_temp - 1, i_ni_temp]
                        elif ni_opt[f'{ic}ret_r{r}'][i_print] == numsym_img[i_ni_temp] or i_ni_temp == len(numsym_img) - 1:
                            i_ni = [i_ni_temp, i_ni_temp]
                        else:
                            i_ni = [i_ni_temp, i_ni_temp + 1]

                        if nc_opt[f'{ic}ret_r{r}'][i_print] < numsym_cap[i_nc_temp]:
                            i_nc = [i_nc_temp - 1, i_nc_temp]
                        elif nc_opt[f'{ic}ret_r{r}'][i_print] == numsym_cap[i_nc_temp] or i_nc_temp == len(numsym_cap) - 1:
                            i_nc = [i_nc_temp, i_nc_temp]
                        else:
                            i_nc = [i_nc_temp, i_nc_temp + 1]

                        rircninc = [[i_ri[0], i_rc[0], i_ni[0], i_nc[0]], [i_ri[1], i_rc[0], i_ni[0], i_nc[0]],
                                    [i_ri[0], i_rc[1], i_ni[0], i_nc[0]], [i_ri[1], i_rc[1], i_ni[0], i_nc[0]],
                                    [i_ri[0], i_rc[0], i_ni[1], i_nc[0]], [i_ri[1], i_rc[0], i_ni[1], i_nc[0]],
                                    [i_ri[0], i_rc[0], i_ni[0], i_nc[1]], [i_ri[1], i_rc[0], i_ni[0], i_nc[1]],
                                    [i_ri[0], i_rc[1], i_ni[1], i_nc[0]], [i_ri[1], i_rc[1], i_ni[1], i_nc[0]],
                                    [i_ri[0], i_rc[1], i_ni[0], i_nc[1]], [i_ri[1], i_rc[1], i_ni[0], i_nc[1]],
                                    [i_ri[0], i_rc[0], i_ni[1], i_nc[1]], [i_ri[1], i_rc[0], i_ni[1], i_nc[1]],
                                    [i_ri[0], i_rc[1], i_ni[1], i_nc[1]], [i_ri[1], i_rc[1], i_ni[1], i_nc[1]]]
                        acc_candidate = np.array([acc_sr[f'{ic}ret_r{r}'][iri, irc, ini, inc] for iri, irc, ini, inc in rircninc])
                        if flag_res_total:
                            acc_candidate[[-1, -3, -4, -7]] = np.nan
                        temp_acc_gt = np.nanmax(acc_candidate)
                        (iri, irc, ini, inc) = rircninc[np.where(acc_candidate == temp_acc_gt)[0][0]]
                        ri, rc, ni, nc = res_img[iri], res_cap[irc], numsym_img[ini], numsym_cap[inc]
                        print('ri: %.2f, rc: %.2f, si: %.2f, sc: %.2f, ni: %.2f, nc: %.2f, \n acc: %.2f, acc_gt: %.2f, acc_diff: %.2f,  ri: %.2f, rc: %.2f, ni: %.2f, nc: %.2f' % (
                            ri_opt[f'{ic}ret_r{r}'][i_print], rc_opt[f'{ic}ret_r{r}'][i_print],
                            get_snr_from_res_numsym(channel_snr, ri_opt[f'{ic}ret_r{r}'][i_print], ni_opt[f'{ic}ret_r{r}'][i_print]),
                            get_snr_from_res_numsym(channel_snr, rc_opt[f'{ic}ret_r{r}'][i_print], nc_opt[f'{ic}ret_r{r}'][i_print]),
                            ni_opt[f'{ic}ret_r{r}'][i_print], nc_opt[f'{ic}ret_r{r}'][i_print], acc_opt[f'{ic}ret_r{r}'][i_print], temp_acc_gt, np.abs(acc_opt[f'{ic}ret_r{r}'][i_print] - temp_acc_gt),
                            ri, rc, ni, nc))
                        acc_gt[f'{ic}ret_r{r}'].append(temp_acc_gt)
                        acc_diff[f'{ic}ret_r{r}'].append(np.abs(acc_opt[f'{ic}ret_r{r}'][i_print] - temp_acc_gt))
                    break


    return ri_opt, rc_opt, ni_opt, nc_opt, acc_opt, acc_gt, acc_diff

def get_neighbors(x, arr):
    if x < arr[0]:
        return 0, 0
    elif x < arr[1]:
        return 0, 1
    elif x > arr[-1]:
        return -1, -1
    elif x > arr[-2]:
        return -2, -1
    else:
        idx = (np.abs(arr - x)).argmin()
        if x > arr[idx]:
            return idx, idx + 1
        else:
            return idx - 1, idx
def opt_num_res_g_4D(acc_ori, snr_img, snr_cap, numsym_img, numsym_cap, step_num_res, step_num_numsym, ri_init, ni_init, res_total, channel_snr):
    """
    get optimal acc, res_img, res_cap, numsym_img, numsym_cap with fitting into function g(x)
    """
    acc_sr = transf_acc4D_snr2res(acc_ori, channel_snr, snr_img, snr_cap, numsym_img, numsym_cap)

    numsym_img_upsampl = np.linspace(numsym_img[0], numsym_img[-1] + (numsym_img[-1] - numsym_img[0]) / (len(numsym_img)-1) / step_num_numsym * (step_num_numsym - 1), step_num_numsym * (len(numsym_img)))
    numsym_cap_upsampl = np.linspace(numsym_cap[0], numsym_cap[-1] + (numsym_cap[-1] - numsym_cap[0]) / (len(numsym_cap)-1) / step_num_numsym * (step_num_numsym - 1), step_num_numsym * (len(numsym_cap)))

    acc_upsampl_sr = upsampl_acc_res_numsym_4D(acc_sr, step_num_res, step_num_numsym)
    ri_opt, rc_opt, ni_opt, nc_opt, acc_opt, acc_gt, acc_diff = {}, {}, {}, {}, {}, {}, {}
    res_img_min = get_res_from_sn(snr_img[-1], numsym_img[0], channel_snr)
    res_cap_min = get_res_from_sn(snr_cap[-1], numsym_cap[0], channel_snr)
    res_img_max = get_res_from_sn(snr_img[0], numsym_img[-1], channel_snr)
    res_cap_max = get_res_from_sn(snr_cap[0], numsym_cap[-1], channel_snr)

    res_img = np.linspace(res_img_min, res_img_max, int(acc_upsampl_sr['imgret_r1'].shape[0] / step_num_res))
    res_cap = np.linspace(res_cap_min, res_cap_max, int(acc_upsampl_sr['imgret_r1'].shape[1] / step_num_res))
    res_img_upsampl = np.linspace(res_img_min, res_img_max + (res_img_max - res_img_min) / (len(res_img)-1) / step_num_res * (step_num_res - 1), acc_upsampl_sr['imgret_r1'].shape[0])
    res_cap_upsampl = np.linspace(res_cap_min, res_cap_max + (res_cap_max - res_cap_min) / (len(res_cap)-1) / step_num_res * (step_num_res - 1), acc_upsampl_sr['imgret_r1'].shape[1])

    for ic in ['img', 'cap']:
        for r in [1, 5, 10]:
            # init ri (res_img), rc (res_cap), ni (numsym_img), nc (numsym_cap) opt and acc_opt
            # e.g.: si_opt[k] is the optimal snr_img in iter k, k is odd when fix si, sc, k is even when fix ni, nc
            ri_opt[f'{ic}ret_r{r}'], rc_opt[f'{ic}ret_r{r}'], ni_opt[f'{ic}ret_r{r}'], nc_opt[f'{ic}ret_r{r}'] = [], [], [], []
            acc_opt[f'{ic}ret_r{r}'], acc_gt[f'{ic}ret_r{r}'], acc_diff[f'{ic}ret_r{r}'] = [], [], []

            # iterate optimization
            k = 0
            while True:
                # fix ri, ni, opt rc, nc
                if len(acc_opt[f'{ic}ret_r{r}']) == 0:
                    i_ri = (np.abs(res_img_upsampl - ri_init)).argmin()
                    i_ni = (np.abs(numsym_img_upsampl - ni_init)).argmin()
                    ri_opt[f'{ic}ret_r{r}'].append(res_img_upsampl[i_ri])
                    ni_opt[f'{ic}ret_r{r}'].append(numsym_img_upsampl[i_ni])
                    i_rtotal = (np.abs(res_img_upsampl - np.log2(2 ** res_total - 2 ** 5))).argmin()
                    if res_img_upsampl[i_rtotal] > np.log2(2 ** res_total - 2 ** 5):
                        i_rtotal -= 1
                    tt = acc_upsampl_sr[f'{ic}ret_r{r}'][i_rtotal, :, :, ::step_num_numsym]
                    if len(tt[~np.isnan(tt)]) == 0:
                        print('too small res_total')
                        break
                else:
                    i_ri = np.where(res_img_upsampl == ri_opt[f'{ic}ret_r{r}'][-1])[0][0]
                    i_ni = np.where(numsym_img_upsampl == ni_opt[f'{ic}ret_r{r}'][-1])[0][0]
                    ni_opt[f'{ic}ret_r{r}'].append(ni_opt[f'{ic}ret_r{r}'][-1])
                    ri_opt[f'{ic}ret_r{r}'].append(ri_opt[f'{ic}ret_r{r}'][-1])



                temp_acc_2D = acc_upsampl_sr[f'{ic}ret_r{r}'][i_ri, :, i_ni, ::step_num_numsym]
                while len(temp_acc_2D[~np.isnan(temp_acc_2D)]) == 0:# and i_ri <= len(res_img_upsampl)-1:# and i_ni < len(numsym_img_upsampl)-1:
                    if i_ri < len(res_img_upsampl) - 1 and i_ri <= i_rtotal - 1:
                        i_ri += 1
                    else:
                        i_ni = (i_ni + 1) % len(numsym_img_upsampl)
                    temp_acc_2D = acc_upsampl_sr[f'{ic}ret_r{r}'][i_ri, :, i_ni, ::step_num_numsym]

                max_rc = np.log2(2**res_total-2**res_img_upsampl[i_ri])
                max_i_rc = (np.abs(res_cap_upsampl - max_rc)).argmin()
                if res_cap_upsampl[max_i_rc] > max_rc + 1e-5:
                    max_i_rc -= 1
                acc_max = 0
                for i_row in range(min(temp_acc_2D.shape[0], max_i_rc) + 1):
                    acc = temp_acc_2D[i_row, :]
                    flag_gfunc = False
                    if len(acc[~np.isnan(acc)]) == 0:
                        continue
                    elif len(acc[~np.isnan(acc)]) == 1:
                        acc_g = acc[~np.isnan(acc)]
                    elif len(acc[~np.isnan(acc)]) == 2:
                        acc_g = np.max(acc[~np.isnan(acc)])
                    else:
                        dg_opt, a0, a1, a2, a3, sig, c = get_best_g_1d(acc, numsym_cap)
                        # dg = derivation_g(numsym_cap_upsampl, a1, a2, a3, a0, sig, c)
                        acc_g = func_g(numsym_cap_upsampl, a1, a2, a3, a0, sig, c)
                        # acc_g = acc_g + acc[~np.isnan(acc)][0] - acc_g[0]
                        ####
                        max_acc_g = np.max(acc_g)
                        idx_max_acc_g = np.where(acc_g == max_acc_g)[0][0]
                        idx_acc = int(idx_max_acc_g/step_num_numsym)
                        if np.isnan(acc[idx_acc]):
                            acc_g = acc_g + np.max(acc[~np.isnan(acc)]) - max_acc_g
                        else:
                            acc_g = acc_g + acc[idx_acc] - max_acc_g
                        ####
                        flag_gfunc = True
                    if acc_max < np.max(acc_g):
                        temp_idxn = np.where(acc_g == np.max(acc_g))[0][0]
                        if not flag_gfunc:
                            temp_idxn = np.where(acc == np.max(acc_g))[0][0] * 5
                        tt = acc_upsampl_sr[f'{ic}ret_r{r}'][:, i_row, ::step_num_numsym, temp_idxn]
                        if len(tt[~np.isnan(tt)]) == 0:
                            continue
                        acc_max = np.max(acc_g)
                        acc_opt_curr = acc_max
                        i_rc_opt_curr = i_row
                        i_nc_opt_curr = np.where(acc_g == acc_max)[0][0]
                        if not flag_gfunc:
                            i_nc_opt_curr = np.where(acc == acc_max)[0][0]
                            if res_total < 9:
                                i_nc_opt_curr = i_nc_opt_curr * step_num_numsym
                            else:
                                i_nc_opt_curr = i_nc_opt_curr * step_num_numsym + 4

                nc_opt[f'{ic}ret_r{r}'].append(numsym_cap_upsampl[i_nc_opt_curr])
                rc_opt[f'{ic}ret_r{r}'].append(res_cap_upsampl[i_rc_opt_curr])
                acc_opt[f'{ic}ret_r{r}'].append(acc_opt_curr)
                k += 1
                # check break condition
                if k > 10 or (k > 1 and numsym_cap_upsampl[i_nc_opt_curr] == nc_opt[f'{ic}ret_r{r}'][-2] and res_cap_upsampl[i_rc_opt_curr] == rc_opt[f'{ic}ret_r{r}'][-2]):
                    print(k)
                    print('intp: ri: %d, rc: %d, ni: %d, nc: %d, acc: %f' % (ri_opt[f'{ic}ret_r{r}'][-1], rc_opt[f'{ic}ret_r{r}'][-1], ni_opt[f'{ic}ret_r{r}'][-1], nc_opt[f'{ic}ret_r{r}'][-1], acc_opt[f'{ic}ret_r{r}'][-1]))
                    i_ni1, i_ni2 = get_neighbors(ni_opt[f'{ic}ret_r{r}'][-1], numsym_img)
                    i_nc1, i_nc2 = get_neighbors(nc_opt[f'{ic}ret_r{r}'][-1], numsym_cap)
                    i_ri1, i_ri2 = get_neighbors(ri_opt[f'{ic}ret_r{r}'][-1], res_img)
                    i_rc1, i_rc2 = get_neighbors(rc_opt[f'{ic}ret_r{r}'][-1], res_cap)
                    if res_img[i_ri2] >= res_total:
                        i_ri2 = i_ri1
                    if res_cap[i_rc2] >= res_total:
                        i_rc2 = i_rc1
                    rircninc = [[i_ri1, i_rc1, i_ni1, i_nc1], [i_ri1, i_rc2, i_ni1, i_nc1], [i_ri2, i_rc1, i_ni1, i_nc1], [i_ri2, i_rc2, i_ni1, i_nc1],
                                [i_ri1, i_rc1, i_ni1, i_nc2], [i_ri1, i_rc2, i_ni1, i_nc2], [i_ri2, i_rc1, i_ni1, i_nc2], [i_ri2, i_rc2, i_ni1, i_nc2],
                                [i_ri1, i_rc1, i_ni2, i_nc1], [i_ri1, i_rc2, i_ni2, i_nc1], [i_ri2, i_rc1, i_ni2, i_nc1], [i_ri2, i_rc2, i_ni2, i_nc1],
                                [i_ri1, i_rc1, i_ni2, i_nc2], [i_ri1, i_rc2, i_ni2, i_nc2], [i_ri2, i_rc1, i_ni2, i_nc2], [i_ri2, i_rc2, i_ni2, i_nc2]]
                    acc_candidate = np.array([acc_sr[f'{ic}ret_r{r}'][iri, irc, ini, inc] for iri, irc, ini, inc in rircninc])
                    if 2 ** res_img[i_ri2] + 2 ** res_cap[i_rc2] > 2 ** res_total:
                        acc_candidate[[3, 7, 11, 15]] = np.nan
                    temp_acc_gt = np.nanmax(acc_candidate)
                    (iri, irc, ini, inc) = rircninc[np.where(acc_candidate == temp_acc_gt)[0][0]]
                    ri, rc, ni, nc = res_img[iri], res_cap[irc], numsym_img[ini], numsym_cap[inc]
                    print('disc: ri: %d, rc: %d, ni: %d, nc: %d, acc: %f' % (ri, rc, ni, nc, temp_acc_gt))
                    break

                # fix ni, nc, opt si, sc
                i_rc = np.where(res_cap_upsampl == rc_opt[f'{ic}ret_r{r}'][-1])[0]
                i_nc = np.where(numsym_cap_upsampl == nc_opt[f'{ic}ret_r{r}'][-1])[0]

                temp_acc_2D = acc_upsampl_sr[f'{ic}ret_r{r}'][:, i_rc, ::step_num_numsym, i_nc][0, :, :]

                max_ri = np.log2(2 ** res_total - 2 ** res_cap_upsampl[i_rc])
                max_i_ri = (np.abs(res_img_upsampl - max_ri)).argmin()
                if res_img_upsampl[max_i_ri] > max_ri + 1e-5:
                    max_i_ri -= 1
                acc_max = 0
                for i_row in range(min(temp_acc_2D.shape[0], max_i_ri) + 1):
                    acc = temp_acc_2D[i_row, :]
                    flag_gfunc = False
                    if len(acc[~np.isnan(acc)]) == 0:
                        continue
                    elif len(acc[~np.isnan(acc)]) == 1:
                        acc_g = acc[~np.isnan(acc)]
                    elif len(acc[~np.isnan(acc)]) == 2:
                        acc_g = np.max(acc[~np.isnan(acc)])
                    else:
                        _, a0, a1, a2, a3, sig, c = get_best_g_1d(acc, numsym_cap)
                        acc_g = func_g(numsym_cap_upsampl, a1, a2, a3, a0, sig, c)
                        # acc_g = acc_g + acc[~np.isnan(acc)][0] - acc_g[0]
                        ####
                        max_acc_g = np.max(acc_g)
                        idx_max_acc_g = np.where(acc_g == max_acc_g)[0][0]
                        idx_acc = int(idx_max_acc_g/step_num_numsym)
                        if np.isnan(acc[idx_acc]):
                            acc_g = acc_g + np.max(acc[~np.isnan(acc)]) - max_acc_g
                        else:
                            acc_g = acc_g + acc[idx_acc] - max_acc_g
                        # max_acc_real = np.max(acc[~np.isnan(acc)])
                        # idx_max_acc_real = np.where(acc[~np.isnan(acc)] == max_acc_real)[0][0]
                        # acc_g = acc_g + max_acc_real - acc_g[idx_max_acc_real * step_num_numsym]
                        ####
                        flag_gfunc = True
                    if acc_max < np.max(acc_g):
                        temp_idxn = np.where(acc_g == np.max(acc_g))[0][0]
                        if not flag_gfunc:
                            temp_idxn = np.where(acc == np.max(acc_g))[0][0] * 5
                        tt = acc_upsampl_sr[f'{ic}ret_r{r}'][i_row, :, temp_idxn, :step_num_numsym]
                        if len(tt[~np.isnan(tt)]) == 0:
                            continue
                        acc_opt_curr = np.max(acc_g)
                        acc_max = acc_opt_curr
                        i_ri_opt_curr = i_row
                        i_ni_opt_curr = np.where(acc_g == acc_max)[0][0]
                        if not flag_gfunc:
                            i_ni_opt_curr = np.where(acc == acc_max)[0][0]
                            if res_total < 9:
                                i_ni_opt_curr = i_ni_opt_curr * step_num_numsym
                            else:
                                i_ni_opt_curr = i_ni_opt_curr * step_num_numsym + 3

                ni_opt[f'{ic}ret_r{r}'].append(numsym_img_upsampl[i_ni_opt_curr])
                ri_opt[f'{ic}ret_r{r}'].append(res_img_upsampl[i_ri_opt_curr])
                nc_opt[f'{ic}ret_r{r}'].append(nc_opt[f'{ic}ret_r{r}'][-1])
                rc_opt[f'{ic}ret_r{r}'].append(rc_opt[f'{ic}ret_r{r}'][-1])
                acc_opt[f'{ic}ret_r{r}'].append(acc_opt_curr)
                k += 1

                # check break condition
                if k > 10 or (numsym_img_upsampl[i_ni_opt_curr] == ni_opt[f'{ic}ret_r{r}'][-2] and res_img_upsampl[i_ri_opt_curr] == ri_opt[f'{ic}ret_r{r}'][-2]):
                    print(k)
                    print('intp: ri: %d, rc: %d, ni: %d, nc: %d, acc: %f' % (ri_opt[f'{ic}ret_r{r}'][-1], rc_opt[f'{ic}ret_r{r}'][-1], ni_opt[f'{ic}ret_r{r}'][-1], nc_opt[f'{ic}ret_r{r}'][-1], acc_opt[f'{ic}ret_r{r}'][-1]))
                    i_ni1, i_ni2 = get_neighbors(ni_opt[f'{ic}ret_r{r}'][-1], numsym_img)
                    i_nc1, i_nc2 = get_neighbors(nc_opt[f'{ic}ret_r{r}'][-1], numsym_cap)
                    i_ri1, i_ri2 = get_neighbors(ri_opt[f'{ic}ret_r{r}'][-1], res_img)
                    i_rc1, i_rc2 = get_neighbors(rc_opt[f'{ic}ret_r{r}'][-1], res_cap)
                    if res_img[i_ri2] >= res_total:
                        i_ri2 = i_ri1
                    if res_cap[i_rc2] >= res_total:
                        i_rc2 = i_rc1
                    rircninc = [[i_ri1, i_rc1, i_ni1, i_nc1], [i_ri1, i_rc2, i_ni1, i_nc1], [i_ri2, i_rc1, i_ni1, i_nc1], [i_ri2, i_rc2, i_ni1, i_nc1],
                                [i_ri1, i_rc1, i_ni1, i_nc2], [i_ri1, i_rc2, i_ni1, i_nc2], [i_ri2, i_rc1, i_ni1, i_nc2], [i_ri2, i_rc2, i_ni1, i_nc2],
                                [i_ri1, i_rc1, i_ni2, i_nc1], [i_ri1, i_rc2, i_ni2, i_nc1], [i_ri2, i_rc1, i_ni2, i_nc1], [i_ri2, i_rc2, i_ni2, i_nc1],
                                [i_ri1, i_rc1, i_ni2, i_nc2], [i_ri1, i_rc2, i_ni2, i_nc2], [i_ri2, i_rc1, i_ni2, i_nc2], [i_ri2, i_rc2, i_ni2, i_nc2]]
                    acc_candidate = np.array([acc_sr[f'{ic}ret_r{r}'][iri, irc, ini, inc] for iri, irc, ini, inc in rircninc])
                    if 2 ** res_img[i_ri2] + 2 ** res_cap[i_rc2] > 2 ** res_total:
                        acc_candidate[[3, 7, 11, 15]] = np.nan
                    temp_acc_gt = np.nanmax(acc_candidate)
                    (iri, irc, ini, inc) = rircninc[np.where(acc_candidate == temp_acc_gt)[0][0]]
                    ri, rc, ni, nc = res_img[iri], res_cap[irc], numsym_img[ini], numsym_cap[inc]
                    print('disc: ri: %d, rc: %d, ni: %d, nc: %d, acc: %f' % (ri, rc, ni, nc, temp_acc_gt))
                    break


    return ri_opt, rc_opt, ni_opt, nc_opt, acc_opt, acc_gt, acc_diff


if __name__=='__main__':
    # init params
    acc_ori = get_acc()
    max_n = 17
    channel_snr = -24  # dB
    acc_th = 0.01
    si_init = 0
    sc_init = 0
    ri_init = 4
    rc_init = 4
    res_total = 15  # dB
    # define wanted snr_img, snr_cap, numsym_img(dB), numsym_cap(dB)
    snr_img = np.arange(18, -19, -6)  # img snr
    snr_cap = np.arange(18, -19, -6)  # cap snr
    numsym_img = np.arange(4, 9, 1)  # number of symbol img
    numsym_cap = np.arange(4, 9, 1)  # number of symbol cap
    ############################################ opt 4D numsym res with intp in each iter
    total_res_range = np.arange(6, 17, 1)
    opt_acc_totalres, acc_gt_totalres, acc_diff_totalres = {}, {}, {}
    step_num_snr = 5
    step_num_numsym = 5
    # acc_upsampl = upsampl_ori_acc_4D(acc_ori, step_num_snr, step_num_numsym)
    for ic in ['img', 'cap']:
        for r in [1, 5, 10]:
            opt_acc_totalres[f'{ic}ret_r{r}'] = np.zeros(total_res_range.shape)
            acc_gt_totalres[f'{ic}ret_r{r}'] = np.zeros(total_res_range.shape)
            acc_diff_totalres[f'{ic}ret_r{r}'] = np.zeros(total_res_range.shape)
    for i_res_total, res_total in enumerate(total_res_range):
        # using g func
        ri_init = res_total - 1
        ni_init = 5
        if res_total >= 13:
            ni_init = 7
        print(f'res_total = {res_total}')
        ri_opt, rc_opt, ni_opt, nc_opt, acc_opt, acc_gt, acc_diff = opt_num_res_g_4D(acc_ori, snr_img, snr_cap, numsym_img, numsym_cap, step_num_snr, step_num_numsym, ri_init, ni_init, res_total, channel_snr)

        #### interpolation
        # if res_total >= 11:
        #     max_n = 3
        # ri_init = res_total - 1
        # rc_init = res_total - 1
        # print('res_total = %d' % res_total)
        # print('opt value each steps:')
        # ri_opt, rc_opt, ni_opt, nc_opt, acc_opt, acc_gt, acc_diff = opt_num_res_intp_4D(acc_ori, snr_img, snr_cap, numsym_img, numsym_cap, step_num_snr, step_num_numsym, ri_init, rc_init, res_total, channel_snr, max_n, m_gmm)
    #     print('opt value final:')
    #     for ic in ['img', 'cap']:
    #         for r in [1, 5, 10]:
    #             print(f'{ic}ret_r{r}')
    #             if ri_opt[f'{ic}ret_r{r}'] is not None:
    #                 np_acc_opt = np.array(acc_opt[f'{ic}ret_r{r}'])
    #                 opt_idx = np.where(np_acc_opt == np.max(np_acc_opt[-2:]))[0][0]
    #                 opt_acc_totalres[f'{ic}ret_r{r}'][i_res_total] = np.max(np_acc_opt[-2:])
    #                 acc_gt_totalres[f'{ic}ret_r{r}'][i_res_total] = acc_gt[f'{ic}ret_r{r}'][opt_idx]
    #                 acc_diff_totalres[f'{ic}ret_r{r}'][i_res_total] = acc_diff[f'{ic}ret_r{r}'][opt_idx]
    #                 print('ri: %.2f, rc: %.2f, si: %.2f, sc: %.2f, ni: %.2f, nc: %.2f, acc: %.2f, acc_gt: %.2f, acc_diff: %.2f' % (
    #                     ri_opt[f'{ic}ret_r{r}'][opt_idx], rc_opt[f'{ic}ret_r{r}'][opt_idx],
    #                     get_snr_from_res_numsym(channel_snr, ri_opt[f'{ic}ret_r{r}'][opt_idx], ni_opt[f'{ic}ret_r{r}'][opt_idx]),
    #                     get_snr_from_res_numsym(channel_snr, rc_opt[f'{ic}ret_r{r}'][opt_idx], nc_opt[f'{ic}ret_r{r}'][opt_idx]),
    #                     ni_opt[f'{ic}ret_r{r}'][opt_idx], nc_opt[f'{ic}ret_r{r}'][opt_idx], np.max(np_acc_opt[-2:]), acc_gt[f'{ic}ret_r{r}'][opt_idx], acc_diff[f'{ic}ret_r{r}'][opt_idx]))
    #             else:
    #                 print('too small res_total, no acc satisfies the condition')
    #
    # for ic in ['img', 'cap']:
    #     for r in [1, 5, 10]:
    #         print(opt_acc_totalres[f'{ic}ret_r{r}'])
    # for ic in ['img', 'cap']:
    #     for r in [1, 5, 10]:
    #         print(acc_gt_totalres[f'{ic}ret_r{r}'])
    # for ic in ['img', 'cap']:
    #     for r in [1, 5, 10]:
    #         print(acc_diff_totalres[f'{ic}ret_r{r}'])

    # # fit polynomial function
    # coeff = fit_polynom_4D(acc_ori, max_n=max_n)
    # # get interpolated acc: acc_intp[f'{ic}ret_r{r}'][s_i, s_c, n_i, n_c]
    # acc_intp = get_acc_from_polynom_4D(coeff, snr_img, snr_cap, numsym_img, numsym_cap, max_n=max_n)
    # ######################## temp start
    # acc_intp = acc_ori
    # ######################## temp end
    #
    # # get res in shape acc_intp
    # res_intp = np.zeros(acc_intp[f'imgret_r1'].shape)
    # for idx_si, si in enumerate(snr_img):
    #     for idx_sc, sc in enumerate(snr_cap):
    #         for idx_ni, ni in enumerate(numsym_img):
    #             for idx_nc, nc in enumerate(numsym_cap):
    #                 res_intp[idx_si, idx_sc, idx_ni, idx_nc], _, _ = get_res_from_siscninc(si, sc, ni, nc, channel_snr)


    # ############################################ opt 4D numsym snr
    # for res_total in np.arange(12, 13, 1):
    #     print('res_total = %d' % res_total)
    #     print('opt value each steps:')
    #     si_opt, sc_opt, ni_opt, nc_opt, acc_opt = opt_numsym_snr4D(acc_intp, res_intp, snr_img, snr_cap, numsym_img, numsym_cap, si_init, sc_init, res_total)
    #     print('opt value final:')
    #     for ic in ['img', 'cap']:
    #         for r in [1, 5, 10]:
    #             print(f'{ic}ret_r{r}')
    #             if si_opt[f'{ic}ret_r{r}'] is not None:
    #                 print('si: %.2f, sc: %.2f, ni: %.2f, nc: %.2f, acc: %.2f' % (
    #                 si_opt[f'{ic}ret_r{r}'][-1], sc_opt[f'{ic}ret_r{r}'][-1], ni_opt[f'{ic}ret_r{r}'][-1], nc_opt[f'{ic}ret_r{r}'][-1], acc_opt[f'{ic}ret_r{r}'][-1]))
    #             else:
    #                 print('too small res_total, no acc satisfies the condition')

    # ############################################ opt 4D numsym res
    # total_res_range = np.arange(6, 17, 1)
    # opt_acc_totalres = {}
    # for ic in ['img', 'cap']:
    #     for r in [1, 5, 10]:
    #         opt_acc_totalres[f'{ic}ret_r{r}'] = np.zeros(total_res_range.shape)
    # for i_res_total, res_total in enumerate(total_res_range):
    #     print('res_total = %d' % res_total)
    #     print('opt value each steps:')
    #     ri_opt, rc_opt, ni_opt, nc_opt, acc_opt = opt_numsym_res4D(acc_ori, snr_img, snr_cap, numsym_img, numsym_cap, ri_init, rc_init, res_total)
    #     print('opt value final:')
    #     for ic in ['img', 'cap']:
    #         for r in [1, 5, 10]:
    #             print(f'{ic}ret_r{r}')
    #             if ri_opt[f'{ic}ret_r{r}'] is not None:
    #                 opt_acc_totalres[f'{ic}ret_r{r}'][i_res_total] = acc_opt[f'{ic}ret_r{r}'][-1]
    #                 print('ri: %.2f, rc: %.2f, si: %.2f, sc: %.2f, ni: %.2f, nc: %.2f, acc: %.2f' % (
    #                     ri_opt[f'{ic}ret_r{r}'][-1], rc_opt[f'{ic}ret_r{r}'][-1],
    #                     get_snr_from_res_numsym(channel_snr, ri_opt[f'{ic}ret_r{r}'][-1], ni_opt[f'{ic}ret_r{r}'][-1]),
    #                     get_snr_from_res_numsym(channel_snr, rc_opt[f'{ic}ret_r{r}'][-1], nc_opt[f'{ic}ret_r{r}'][-1]),
    #                     ni_opt[f'{ic}ret_r{r}'][-1], nc_opt[f'{ic}ret_r{r}'][-1], acc_opt[f'{ic}ret_r{r}'][-1]))
    #             else:
    #                 print('too small res_total, no acc satisfies the condition')
    #
    # for ic in ['img', 'cap']:
    #     for r in [1, 5, 10]:
    #         print(opt_acc_totalres[f'{ic}ret_r{r}'])
    #
