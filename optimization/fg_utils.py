import math
import numpy as np
from scipy import special as sp
from noise_result.fix_val_noise.result_ft_oc_train_ae import get_acc
from optimization.optimization_4D import transf_acc4D_snr2res, get_res_from_sn
from scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt

def qfunc(x):
    return 0.5-0.5*sp.erf(x/math.sqrt(2))

def gausssian_func(x, mu, sigma, A):
    return A / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-np.power((x - mu) / sigma, 2.0) / 2)

def derivation_gaussian(x, mu, sigma, A):
    return -gausssian_func(x, mu, sigma, A) * (x - mu) / sigma ** 2


def func_g(x, a1, a2, a3, a0, sig, c):
    '''
    :return: g(x)=a3 / 4 * Rayleigh(x)**4 + a2 / 3 * Rayleigh(x)**3 + a1 / 2 * Rayleigh(x) ** 2 + a0 * Rayleigh(x)
    '''
    x = x + c
    x = np.exp(-x ** 2 / 2 / sig ** 2) * x / sig ** 2
    return a3 / 4 * x**4 + a2 / 3 * x**3 + a1 / 2 * x ** 2 + a0 * x
    # return gausssian_func(x, mu1, sigma1, A1) + gausssian_func(x, mu2, sigma2, A2)
    # return A * (1 - qfunc((x - mu) / sigma))


def derivation_g(x, a1, a2, a3, a0, sig, c):
    '''
    :return: g'(x), where g(x)=a3 / 4 * Rayleigh(x)**4 + a2 / 3 * Rayleigh(x)**3 + a1 / 2 * Rayleigh(x) ** 2 + a0 * Rayleigh(x)
    '''
    x = x + c
    drayl = np.exp(-x ** 2 / 2 / sig ** 2) * (1 - x ** 2 / sig ** 2) / sig ** 2
    rayl = np.exp(-x ** 2 / 2 / sig ** 2) * x / sig ** 2
    return (a1 * rayl + a2 * rayl**2 + a3 * rayl**3 + a0) * drayl
    # return drayl*a1+c
    # return derivation_gaussian(x, mu1, sigma1, A1) + derivation_gaussian(x, mu2, sigma2, A2) + derivation_gaussian(x, mu, sigma, A) + derivation_gaussian(x, mu3, sigma3, A3) + derivation_gaussian(x, mu4, sigma4, A4) + derivation_gaussian(x, mu5, sigma5, A5)
    # return -A / math.sqrt(2 * math.pi) / sigma * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def ls_fun(params, xdata=None, ydata=None):
    (a1, a2, a3, a0, sig, c) = params
    return derivation_g(xdata, a1, a2, a3, a0, sig, c) - ydata
def get_best_g_1d(acc, x, plot_idx=None):
    '''
    fit 1D g(x)
    '''
    mask = ~np.isnan(acc)
    acc_len = np.sum(mask)
    if acc_len <= 2:
        return None
    x_real = x[mask]
    acc_real = acc[mask]
    dacc = np.gradient(acc_real)
    initial_guess = (1, 1, 1, 1, 100, 1)
    popt = least_squares(ls_fun, initial_guess, args=(x_real, dacc))
    (a1, a2, a3, a0, sig, c) = popt.x
    dg_opt = derivation_g(x_real, a1, a2, a3, a0, sig, c)
    # ## plot
    # # x_intp = np.arange(x_real[0], x_real[-1]+0.1, 0.1)
    # x_intp = x_real
    # # dg_intp = derivation_g(x_intp, a1, a2, a3, a0, sig, c)
    # g_intp = func_g(x_intp, a1, a2, a3, a0, sig, c)
    # g_intp = g_intp + acc_real[0] - g_intp[0]
    # fig, ax = plt.subplots()
    # fig.set_size_inches(27, 18)
    # # plt.plot(np.arange(0, len(dacc), 1), dacc, 'g', label='ori data')
    # # plt.plot(dg_intp, 'b:', label='func g')
    # plt.plot(np.arange(0, 1 * len(acc_real), 1), acc_real, 'g', label='ori data')
    # plt.plot(g_intp, 'b:', label='func g')
    # ax.legend(loc='lower right', fontsize='large')
    # ax.set_title('first derivation of ori. acc. and func. g()')
    # plt.savefig(f'C:/Users/INDA_HIWI/Desktop/temp_figs4/{plot_idx}.jpg')
    # plt.close(fig)
    return dg_opt, a0, a1, a2, a3, sig, c

def get_func_g_params():
    channel_snr = -24
    snr_cap = np.arange(18, -19, -6)  # cap snr
    snr_img = np.arange(18, -19, -6)  # img snr
    numsym_img = np.arange(4, 9, 1)  # img num symbol, 256 = 2**8
    numsym_cap = np.arange(4, 9, 1)  # cap num symbol, 256 = 2**8
    res_img = [get_res_from_sn(si, ni, channel_snr) for si in snr_img for ni in numsym_img]
    res_img = list(dict.fromkeys(res_img))
    res_img.sort()
    res_img = np.array(res_img)
    res_cap = [get_res_from_sn(sc, nc, channel_snr) for sc in snr_cap for nc in numsym_cap]
    res_cap = list(dict.fromkeys(res_cap))
    res_cap.sort()
    res_cap = np.array(res_cap)

    acc_4D = get_acc()
    acc_4D = transf_acc4D_snr2res(acc_4D, channel_snr, snr_img, snr_cap, numsym_img, numsym_cap)  # [i_ri, i_rc, i_ni, i_nc]

    save_params_numi = []
    save_params_numc = []

    for i_ri in range(len(res_img)):
        for i_rc in range(len(res_cap)):
            for i_nc in range(len(numsym_cap)):
                plot_idx = f'numi_{i_ri}_{i_rc}_{i_nc}'
                acc = acc_4D['imgret_r1'][i_ri, i_rc, :, i_nc]
                params = get_best_g_1d(acc, numsym_img, plot_idx)
                if params is None:
                    a0, a1, a2, a3, sig, c = None, None, None, None, None, None
                else:
                    (dg_opt, a0, a1, a2, a3, sig, c) = params
                    save_params_numi.append([res_img[i_ri], res_cap[i_rc], numsym_cap[i_nc], a1, a2, a3, a0, sig])


    for i_ri in range(len(res_img)):
        for i_rc in range(len(res_cap)):
            for i_ni in range(len(numsym_img)):
                plot_idx = f'numc_{i_ri}_{i_rc}_{i_ni}'
                acc = acc_4D['imgret_r1'][i_ri, i_rc, i_ni, :]
                params = get_best_g_1d(acc, numsym_cap, plot_idx)
                if params is None:
                    a0, a1, a2, a3, sig, c = None, None, None, None, None, None
                else:
                    (dg_opt, a0, a1, a2, a3, sig, c) = params
                    save_params_numc.append([res_img[i_ri], res_cap[i_rc], numsym_img[i_ni], a1, a2, a3, a0, sig])

    # for item in save_params_numc:
    #     print(item)
    return save_params_numi, save_params_numc

def fit_params_mse(params, xdata=None, ydata=None):
    y = 0 * xdata
    for i, ai in enumerate(params):
        y += ai * xdata ** i
    return y - ydata



def find_polynom_k(x, y):
    '''
    find best polynom order
    '''
    K = 10
    diff_opt = 1e16
    k_opt = 0
    coeff_opt = 0
    for k in np.arange(2, K + 1, 1):
        initial_guess = (1,) * k
        popt = least_squares(fit_params_mse, initial_guess, args=(x, y))
        coeff = popt.x
        y_polyn = 0 * x
        for i in np.arange(0, k, 1):
            y_polyn += coeff[i] * x ** i
        diff = np.sum(np.abs(y_polyn - y)) / len(y)
        if diff < diff_opt:
            diff_opt = diff
            k_opt = k
            coeff_opt = coeff

    return k_opt, coeff_opt, diff_opt




if __name__ == "__main__":
    save_params_numi, save_params_numc = get_func_g_params()
    # fit_params(save_params_numi)



    # def func_g(X, Y, mu_x, mu_y, C, sigma_x, sigma_y, sigma_xy):
#     '''
#     :return: g(x, y; mu_x, mu_y, C, sigma_x, sigma_y)
#     '''
#     return C * qfunc(-np.sqrt((X-mu_x) ** 2 / sigma_x ** 2 + (Y-mu_y) ** 2 / sigma_y ** 2 + (X-mu_x) * (Y-mu_y) / sigma_xy ** 2))
#
# def derivation_g(X, Y, mu_x, mu_y, C, sigma_x, sigma_y, sigma_xy):
#     '''
#     :return: first order derivation of function g
#     '''
#     w = -np.sqrt((X - mu_x) ** 2 / sigma_x ** 2 + (Y - mu_y) ** 2 / sigma_y ** 2 + (X-mu_x) * (Y-mu_y) / sigma_xy ** 2)
#
#     dg = -C / math.sqrt(2 * math.pi) * np.exp(-w / 2)
#     dgdx = -dg * ((X - mu_x) / sigma_x ** 2 + (Y - mu_y) / sigma_xy ** 2) / w
#     dgdy = -dg * ((Y - mu_y) / sigma_y ** 2 + (X - mu_x) / sigma_xy ** 2) / w
#     return dgdx, dgdy

# def get_best_g(acc, x, y):
#     X, Y = np.meshgrid(x, y)
#     mask = np.isnan(acc)
#     x_len = np.max(np.sum(~mask, axis=1))
#     y_len = np.max(np.sum(~mask, axis=0))
#     if x_len == 1 and y_len == 1:
#         return acc
#
#     X_real = np.reshape(X[~mask], (y_len, x_len))
#     Y_real = np.reshape(Y[~mask], (y_len, x_len))
#
#     if x_len == 1 or y_len == 1:
#         acc_real = np.reshape(acc[~mask], max(y_len, x_len))
#         daccdx = daccdy = np.gradient(acc_real)
#     else:
#         acc_real = np.reshape(acc[~mask], (y_len, x_len))
#         daccdy = np.gradient(acc_real, axis=0)
#         daccdx = np.gradient(acc_real, axis=1)
#     mu_x_range = np.arange(-5, 15, 0.5)
#     sigma_x_range = np.arange(0.02, 2, 0.1)
#     sigma_xy_range = np.arange(-2, 2, 0.1)
#     C = 1
#     loss_min = 1e10
#     for mu_x in mu_x_range:
#         mu_y = mu_x
#         # for mu_y in mu_x + np.arange(-1, 1, 0.5):
#         for sigma_x in sigma_x_range:
#             sigma_y = sigma_x
#             for sigma_xy in sigma_xy_range:
#                 dgdx, dgdy = derivation_g(X_real, Y_real, mu_x, mu_y, C, sigma_x, sigma_y, sigma_xy)
#                 c_tmp = np.sum(acc_real) / np.sum(func_g(X_real, Y_real, mu_x, mu_y, C, sigma_x, sigma_y, sigma_xy))
#                 if math. isinf(c_tmp):
#                     continue
#                 # if np.abs(c_tmp) > 5 * (np.max(daccdx) + np.max(daccdy)):
#                 #     c_tmp = c_tmp / np.abs(c_tmp) * 5 * (np.max(daccdx) + np.max(daccdy))
#                 # if np.abs(c_tmp) > 100:
#                 #     c_tmp = 100
#                 # if np.abs(c_tmp) < 0.01:
#                 #     c_tmp = 0.01
#                 dgdx *= c_tmp
#                 dgdy *= c_tmp
#                 if dgdx.shape != daccdx.shape or dgdy.shape != daccdy.shape:
#                     dgdx = np.reshape(dgdx, daccdx.shape)
#                     dgdy = np.reshape(dgdy, daccdy.shape)
#                 loss = 10 * (np.sum(np.abs(dgdx-daccdx)) + np.sum(np.abs(dgdy-daccdy))) + np.sum(dgdx * daccdx < 0) + np.sum(dgdy * daccdy < 0)
#                 if loss_min > loss:
#                     loss_min = loss
#                     mu_x_opt = mu_x
#                     mu_y_opt = mu_y
#                     sigma_x_opt = sigma_x
#                     sigma_y_opt = sigma_y
#                     sigma_xy_opt = sigma_xy
#                     C_opt = c_tmp
#                     dgdx_opt = dgdx
#                     dgdy_opt = dgdy
#     acc_g = np.zeros(acc.shape)
#     acc_g[~mask] = np.reshape(func_g(X_real, Y_real, mu_x_opt, mu_y_opt, C_opt, sigma_x_opt, sigma_y_opt, sigma_xy_opt), y_len * x_len)
#     acc_g[mask] = np.nan
#     print(dgdx_opt)
#     print(daccdx)
#     print(dgdy_opt)
#     print(daccdy)
#     return acc_g, mu_x_opt, mu_y_opt, C_opt, sigma_x_opt, sigma_y_opt


