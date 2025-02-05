import numpy as np
import itertools
from scipy.optimize import curve_fit, least_squares
from utils.utils import gaussian_2d_fit, gmm_2D, exp_ploynom_2D_fit, exp_ploynom_2D, ls_fun

def get_acc_from_polynom_4D(coeff, snr_img, snr_cap, numsym_img, numsym_cap, max_n):
    """
    get acc from polynomial function 4D
    :param coeff: coefficients of polynom
    :param max_n: max. power n of polynom
    :return: acc from polynomial function
    """

    # meshgrid and flatten
    SNR_cap, SNR_img, NUMsym_img, NUMsym_cap = np.meshgrid(snr_cap, snr_img, numsym_img, numsym_cap, copy=False)
    acc_shape = SNR_cap.shape
    SNR_cap = SNR_cap.flatten()
    SNR_img = SNR_img.flatten()
    NUMsym_img = NUMsym_img.flatten()
    NUMsym_cap = NUMsym_cap.flatten()

    # A: list of polynomials, e.g., 1, X, Y, X**2, Y**2, X*Y, X**3, X**2*Y, X*Y**2, Y**3, ....
    A = []
    A.append(SNR_cap * 0 + 1)

    for L in range(1, max_n + 1, 1):
        for subset in itertools.combinations_with_replacement([SNR_cap, SNR_img, NUMsym_img, NUMsym_cap], L):
            a = 1.0 + np.zeros(subset[0].shape)
            for idx_a in range(0, L, 1):
                a *= subset[idx_a]
            A.append(a)

    A = np.array(A).T * 1.0

    # init acc
    acc = {}

    # acc = sum(coeff * A)
    for ic in ['img', 'cap']:
        for r in [1, 5, 10]:
            for i in range(len(SNR_cap)):
                A[i, :] = A[i, :] * coeff[f'{ic}ret_r{r}']

            acc[f'{ic}ret_r{r}'] = np.reshape(np.sum(A, axis=1), acc_shape)

    return acc

def get_acc_from_polynom_2D(coeff, snr1, snr_cap=None, numsym=None, max_n=8):
    """
    get acc from polynomial function 2D, (not all img/cap R@1/5/10 but one from them)
    :param coeff: coefficients of polynom
    :param max_n: max. power n of polynom
    :return: acc from polynomial function
    """
    assert snr_cap is not None or numsym is not None, 'either snr2 or numsym must be not None'
    # meshgrid and flatten
    if snr_cap is not None:
        X, Y = np.meshgrid(snr_cap, snr1, copy=True)
    else:
        X, Y = np.meshgrid(snr1, numsym, copy=True)

    X = X.flatten()
    Y = Y.flatten()

    # A: list of polynomials, e.g., 1, X, Y, X**2, Y**2, X*Y, X**3, X**2*Y, X*Y**2, Y**3, ....
    A = []
    A.append(X * 0 + 1)
    for L in range(1, max_n + 1, 1):
        for subset in itertools.combinations_with_replacement([X, Y], L):
            a = 1.0 + np.zeros(subset[0].shape)
            for idx_a in range(0, L, 1):
                a *= subset[idx_a]
            A.append(a)
    A = np.array(A).T * 1.0

    # get acc from polynomial function
    for i in range(len(X)):
        A[i, :] = A[i, :] * coeff

    acc = np.reshape(np.sum(A, axis=1), X.shape)
    return acc
def get_acc_from_polynom_res_res_2D(coeff, res_img, res_cap, max_n=8):
    """
    get acc from polynomial function 2D, (not all img/cap R@1/5/10 but one from them)
    :param coeff: coefficients of polynom
    :param max_n: max. power n of polynom
    :return: acc from polynomial function
    """
    # meshgrid and flatten
    X, Y = np.meshgrid(res_img, res_cap, copy=True)
    Z_shape = X.shape

    X = X.flatten()
    Y = Y.flatten()

    # A: list of polynomials, e.g., 1, X, Y, X**2, Y**2, X*Y, X**3, X**2*Y, X*Y**2, Y**3, ....
    A = []
    A.append(X * 0 + 1)
    for L in range(1, max_n + 1, 1):
        for subset in itertools.combinations_with_replacement([X, Y], L):
            a = 1.0 + np.zeros(subset[0].shape)
            for idx_a in range(0, L, 1):
                a *= subset[idx_a]
            A.append(a)
    A = np.array(A).T * 1.0

    # get acc from polynomial function
    for i in range(len(X)):
        A[i, :] = A[i, :] * coeff

    acc = np.reshape(np.sum(A, axis=1), Z_shape)
    return acc
def fit_polynom_4D(acc, max_n = 13):
    """
    interpolation: fit a polynomial function, the input is a 4D numpy array, each dimension as snr_cap, snr_img, numsym_img, numsym_cap
    :param max_n: max. power n of polynom
    :return: coefficients of polynom
    """
    # init variables
    snr_cap = np.arange(18, -19, -6)  # cap snr
    snr_img = np.arange(18, -19, -6)  # img snr
    numsym_img = np.arange(4, 9, 1)  # img num symbol, 256 = 2**8
    numsym_cap = np.arange(4, 9, 1)  # cap num symbol, 256 = 2**8

    # meshgrid and flatten
    SNR_cap, SNR_img, NUMsym_img, NUMsym_cap = np.meshgrid(snr_cap, snr_img, numsym_img, numsym_cap, copy=False)

    SNR_cap = SNR_cap.flatten()
    SNR_img = SNR_img.flatten()
    NUMsym_img = NUMsym_img.flatten()
    NUMsym_cap = NUMsym_cap.flatten()

    # A: list of polynomials, e.g., 1, X, Y, X**2, Y**2, X*Y, X**3, X**2*Y, X*Y**2, Y**3, ....
    A = []
    A.append(SNR_cap * 0 + 1)

    for L in range(1, max_n + 1, 1):
        for subset in itertools.combinations_with_replacement([SNR_cap, SNR_img, NUMsym_img, NUMsym_cap], L):
            a = 1.0 + np.zeros(subset[0].shape)
            for idx_a in range(0, L, 1):
                a *= subset[idx_a]
            A.append(a)

    A = np.array(A).T * 1.0

    # init acc and coefficients
    # Z_acc = {}
    coeff = {}
    Z_intp = {}
    # for ic in ['img', 'cap']:
    #     for r in [1, 5, 10]:
    #         Z_acc[f'{ic}ret_r{r}'] = np.zeros((len(snr_img), len(snr_cap), len(numsym_img), len(numsym_cap)))

    for ic in ['img', 'cap']:
        for r in [1, 5, 10]:
            # for i_ni, ni in enumerate(numsym_img):
            #     for i_nc, nc in enumerate(numsym_cap):
            #         Z_acc[f'{ic}ret_r{r}'][:, :, i_ni, i_nc] = acc[f'symb{2**ni}_{2**nc}_{ic}ret_r{r}']

            # flat acc
            B = acc[f'{ic}ret_r{r}'].flatten()

            # fit polynomial
            coeff[f'{ic}ret_r{r}'], _, _, _ = np.linalg.lstsq(A, B)

            # get acc from polynomial function, (for comparing with ori acc)
            for i in range(len(SNR_cap)):
                A[i, :] = A[i, :] * coeff[f'{ic}ret_r{r}']

            Z_intp[f'{ic}ret_r{r}'] = np.reshape(np.sum(A, axis=1), acc[f'{ic}ret_r{r}'].shape)
            diff = abs(Z_intp[f'{ic}ret_r{r}'] - acc[f'{ic}ret_r{r}'])
            max_diff = np.max(diff)
            sum_diff = diff.sum()
            print(max_diff)
            print(sum_diff)

    return coeff

def fit_polynom_snr_numsym_2D(acc, max_n = 8):
    """
    fit polynomial function from single 2D matrix, (not all img/cap R@1/5/10 but one from them)
    :param acc: 2D acc with dim snr, numsym, there are Nones in acc
    :param max_n: max. power n of polynom
    :return: coefficients of polynom
    """
    # init variables
    snr = np.arange(18, -19, -6)
    numsym = np.arange(4, 9, 1)
    # meshgrid and flatten
    X, Y = np.meshgrid(snr, numsym, copy=True)
    X[acc == None] = 100  # mark Nones
    Y[acc == None] = 100  # mark Nones
    X = X.flatten()
    Y = Y.flatten()
    X = X[X != 100]  # delete Nones
    Y = Y[Y != 100]  # delete Nones

    # A: list of polynomials, e.g., 1, X, Y, X**2, Y**2, X*Y, X**3, X**2*Y, X*Y**2, Y**3, ....
    A = []
    A.append(X * 0 + 1)
    for L in range(1, max_n + 1, 1):
        for subset in itertools.combinations_with_replacement([X, Y], L):
            a = 1.0 + np.zeros(subset[0].shape)
            for idx_a in range(0, L, 1):
                a *= subset[idx_a]
            A.append(a)
    A = np.array(A).T * 1.0

    # init acc
    Z = np.array(acc, dtype=float)
    # flat acc
    B = Z.flatten()
    # positions of not None accs (for Z_intp)
    pos = np.arange(len(B))
    pos = pos[~np.isnan(B)]

    # delete Nones
    B = B[~np.isnan(B)]

    coeff, _ = np.linalg.lstsq(A, B)

    # # get acc from polynomial function, (for comparing with ori acc)
    # for i in range(len(B)):
    #     A[i, :] = A[i, :] * coeff
    #
    # Z_flat = np.sum(A, axis=1)
    # Z_intp = np.zeros(len(snr) * len(numsym))
    # for i in range(len(pos)):
    #     Z_intp[pos[i]] = Z_flat[i]
    # Z_intp = np.reshape(Z_intp, Z.shape)

    return coeff

def fit_polynom_res_res_2D(acc, res_img, res_cap, max_n = 8):
    """
    fit polynomial function from single 2D matrix, (not all img/cap R@1/5/10 but one from them)
    :param acc: 2D acc with dim snr, numsym, there are Nones in acc
    :param max_n: max. power n of polynom
    :return: coefficients of polynom
    """

    # meshgrid and flatten
    X, Y = np.meshgrid(res_img, res_cap, copy=True)
    X[np.isnan(acc)] = 100  # mark Nones
    Y[np.isnan(acc)] = 100  # mark Nones
    X = X.flatten()
    Y = Y.flatten()
    X = X[X != 100]  # delete Nones
    Y = Y[Y != 100]  # delete Nones

    # A: list of polynomials, e.g., 1, X, Y, X**2, Y**2, X*Y, X**3, X**2*Y, X*Y**2, Y**3, ....
    A = []
    A.append(X * 0 + 1)
    for L in range(1, max_n + 1, 1):
        for subset in itertools.combinations_with_replacement([X, Y], L):
            a = 1.0 + np.zeros(subset[0].shape)
            for idx_a in range(0, L, 1):
                a *= subset[idx_a]
            A.append(a)
    A = np.array(A).T * 1.0

    # init acc
    Z = np.array(acc, dtype=float)
    # flat acc
    B = Z.flatten()
    # positions of not None accs (for Z_intp)
    pos = np.arange(len(B))
    pos = pos[~np.isnan(B)]

    # delete Nones
    B = B[~np.isnan(B)]

    coeff, _, _, _ = np.linalg.lstsq(A, B)

    # get acc from polynomial function, (for comparing with ori acc)
    for i in range(len(B)):
        A[i, :] = A[i, :] * coeff

    Z_flat = np.sum(A, axis=1)
    Z_intp = np.zeros(len(res_img) * len(res_cap))
    for i in range(len(pos)):
        Z_intp[pos[i]] = Z_flat[i]
    Z_intp = np.reshape(Z_intp, Z.shape)

    return coeff

def fit_polynom_snr_snr_2D(acc, max_n=8):
    """
    fit polynomial function from single 2D matrix, (not all img/cap R@1/5/10 but one from them)
    :param acc: 2D acc with dim snr_img, snr_cap
    :param max_n: max. power n of polynom
    :return: coefficients of polynom
    """
    # init variables
    snr_cap = np.arange(18, -19, -6)  # cap snr
    snr_img = np.arange(18, -19, -6)  # img snr
    # meshgrid and flatten
    X, Y = np.meshgrid(snr_cap, snr_img, copy=True)
    X = X.flatten()
    Y = Y.flatten()

    # A: list of polynomials, e.g., 1, X, Y, X**2, Y**2, X*Y, X**3, X**2*Y, X*Y**2, Y**3, ....
    A = []
    A.append(X * 0 + 1)
    for L in range(1, max_n + 1, 1):
        for subset in itertools.combinations_with_replacement([X, Y], L):
            a = 1.0 + np.zeros(subset[0].shape)
            for idx_a in range(0, L, 1):
                a *= subset[idx_a]
            A.append(a)
    A = np.array(A).T * 1.0

    # init acc
    Z = np.array(acc)
    # flat acc
    B = Z.flatten()
    coeff, _ = np.linalg.lstsq(A, B)

    # # get acc from polynomial function, (for comparing with ori acc)
    # for i in range(49):
    #     A[i, :] = A[i, :] * coeff
    #
    # Z_intp = np.reshape(np.sum(A, axis=1), Z.shape)
    return coeff

def upsampl_ori_acc_4D(acc_ori, step_num_snr, step_num_numsym):
    '''
    interpolate the accuracy matrix for original accuracy matrix [snr_img, snr_cap, numsym_img, numsym_cap]
    '''
    # acc_ori[f'{ic}ret_r{r}'][s_i, s_c, n_i, n_c]
    acc_upsampl = {}
    for ic in ['img', 'cap']:
        for r in [1, 5, 10]:
            acc_upsampl[f'{ic}ret_r{r}'] = acc_ori[f'{ic}ret_r{r}'].repeat(step_num_snr, axis=0).astype(np.float16)
            for i0 in range(acc_ori[f'{ic}ret_r{r}'].shape[0]-1):
                for i1 in range(acc_ori[f'{ic}ret_r{r}'].shape[1]-1):
                    for i2 in range(acc_ori[f'{ic}ret_r{r}'].shape[2]-1):
                        for i3 in range(acc_ori[f'{ic}ret_r{r}'].shape[3]-1):
                            acc_upsampl[f'{ic}ret_r{r}'][i0 * step_num_snr:(i0 + 1) * step_num_snr, i1, i2, i3] += np.linspace(0, acc_upsampl[f'{ic}ret_r{r}'][(i0+1)*step_num_snr, i1, i2, i3] - acc_upsampl[f'{ic}ret_r{r}'][i0*step_num_snr, i1, i2, i3], step_num_snr+1)[:-1]
            acc_upsampl[f'{ic}ret_r{r}'] = acc_upsampl[f'{ic}ret_r{r}'].repeat(step_num_snr, axis=1)

            for i0 in range(acc_upsampl[f'{ic}ret_r{r}'].shape[0] - step_num_snr):
                for i1 in range(acc_ori[f'{ic}ret_r{r}'].shape[1]-1):
                    for i2 in range(acc_ori[f'{ic}ret_r{r}'].shape[2]-1):
                        for i3 in range(acc_ori[f'{ic}ret_r{r}'].shape[3]-1):
                            acc_upsampl[f'{ic}ret_r{r}'][i0, i1 * step_num_snr:(i1 + 1) * step_num_snr, i2, i3] += np.linspace(0, acc_upsampl[f'{ic}ret_r{r}'][i0, (i1+1)*step_num_snr, i2, i3] - acc_upsampl[f'{ic}ret_r{r}'][i0, i1*step_num_snr, i2, i3], step_num_snr+1)[:-1]
            acc_upsampl[f'{ic}ret_r{r}'] = acc_upsampl[f'{ic}ret_r{r}'].repeat(step_num_numsym, axis=2)


            for i0 in range(acc_upsampl[f'{ic}ret_r{r}'].shape[0] - step_num_snr):
                for i1 in range(acc_upsampl[f'{ic}ret_r{r}'].shape[1] - step_num_snr):
                    for i2 in range(acc_ori[f'{ic}ret_r{r}'].shape[2]-1):
                        for i3 in range(acc_ori[f'{ic}ret_r{r}'].shape[3]-1):
                            acc_upsampl[f'{ic}ret_r{r}'][i0, i1, i2 * step_num_numsym:(i2 + 1) * step_num_numsym, i3] += np.linspace(0, acc_upsampl[f'{ic}ret_r{r}'][i0, i1, (i2+1)*step_num_numsym, i3] - acc_upsampl[f'{ic}ret_r{r}'][i0, i1, i2*step_num_numsym, i3], step_num_numsym+1)[:-1]
            acc_upsampl[f'{ic}ret_r{r}'] = acc_upsampl[f'{ic}ret_r{r}'].repeat(step_num_numsym, axis=3)

            for i0 in range(acc_upsampl[f'{ic}ret_r{r}'].shape[0] - step_num_snr):
                for i1 in range(acc_upsampl[f'{ic}ret_r{r}'].shape[1] - step_num_snr):
                    for i2 in range(acc_upsampl[f'{ic}ret_r{r}'].shape[2]-step_num_numsym):
                        for i3 in range(acc_ori[f'{ic}ret_r{r}'].shape[3]-1):
                            acc_upsampl[f'{ic}ret_r{r}'][i0, i1, i2, i3 * step_num_numsym:(i3 + 1) * step_num_numsym] += np.linspace(0, acc_upsampl[f'{ic}ret_r{r}'][i0, i1, i2, (i3+1)*step_num_numsym] - acc_upsampl[f'{ic}ret_r{r}'][i0, i1, i2, i3*step_num_numsym], step_num_numsym+1)[:-1]
            acc_upsampl[f'{ic}ret_r{r}'] = acc_upsampl[f'{ic}ret_r{r}'][:-step_num_snr, :-step_num_snr, :-step_num_numsym, :-step_num_numsym]
    return acc_upsampl

def upsampl_acc_res_numsym_4D(acc_sr, step_num_res, step_num_numsym):
    '''
    interpolate the accuracy matrix for  accuracy matrix [res_img, res_cap, numsym_img, numsym_cap]
    '''
    # acc_ori[f'{ic}ret_r{r}'][s_i, s_c, n_i, n_c]
    acc_upsampl = {}
    for ic in ['img', 'cap']:
        for r in [1, 5, 10]:
            acc_upsampl[f'{ic}ret_r{r}'] = acc_sr[f'{ic}ret_r{r}'].repeat(step_num_res, axis=0).astype(np.float16)
            for i0 in range(acc_sr[f'{ic}ret_r{r}'].shape[0]-1):
                for i1 in range(acc_sr[f'{ic}ret_r{r}'].shape[1]):
                    for i2 in range(acc_sr[f'{ic}ret_r{r}'].shape[2]):
                        for i3 in range(acc_sr[f'{ic}ret_r{r}'].shape[3]):
                            if np.isnan(acc_sr[f'{ic}ret_r{r}'][i0+1, i1, i2, i3]):
                                if np.isnan(acc_sr[f'{ic}ret_r{r}'][i0, i1, i2, i3]):
                                    acc_upsampl[f'{ic}ret_r{r}'][i0 * step_num_res:(i0 + 1) * step_num_res, i1, i2, i3] = np.nan
                            else:
                                acc_upsampl[f'{ic}ret_r{r}'][i0 * step_num_res:(i0 + 1) * step_num_res, i1, i2, i3] += np.linspace(0, acc_upsampl[f'{ic}ret_r{r}'][(i0+1)*step_num_res, i1, i2, i3] - acc_upsampl[f'{ic}ret_r{r}'][i0*step_num_res, i1, i2, i3], step_num_res+1)[:-1]
            acc_upsampl[f'{ic}ret_r{r}'] = acc_upsampl[f'{ic}ret_r{r}'].repeat(step_num_res, axis=1)

            for i0 in range(acc_upsampl[f'{ic}ret_r{r}'].shape[0]):
                for i1 in range(acc_sr[f'{ic}ret_r{r}'].shape[1]-1):
                    for i2 in range(acc_sr[f'{ic}ret_r{r}'].shape[2]):
                        for i3 in range(acc_sr[f'{ic}ret_r{r}'].shape[3]):
                            if np.isnan(acc_upsampl[f'{ic}ret_r{r}'][i0, (i1 + 1) * step_num_res, i2, i3]):
                                if np.isnan(acc_upsampl[f'{ic}ret_r{r}'][i0, i1 * step_num_res, i2, i3]):
                                    acc_upsampl[f'{ic}ret_r{r}'][i0, i1 * step_num_res:(i1 + 1) * step_num_res, i2, i3] = np.nan
                            else:
                                acc_upsampl[f'{ic}ret_r{r}'][i0, i1 * step_num_res:(i1 + 1) * step_num_res, i2, i3] += np.linspace(0, acc_upsampl[f'{ic}ret_r{r}'][i0, (i1+1)*step_num_res, i2, i3] - acc_upsampl[f'{ic}ret_r{r}'][i0, i1*step_num_res, i2, i3], step_num_res+1)[:-1]
            acc_upsampl[f'{ic}ret_r{r}'] = acc_upsampl[f'{ic}ret_r{r}'].repeat(step_num_numsym, axis=2)


            for i0 in range(acc_upsampl[f'{ic}ret_r{r}'].shape[0]):
                for i1 in range(acc_upsampl[f'{ic}ret_r{r}'].shape[1]):
                    for i2 in range(acc_sr[f'{ic}ret_r{r}'].shape[2]-1):
                        for i3 in range(acc_sr[f'{ic}ret_r{r}'].shape[3]):
                            if np.isnan(acc_upsampl[f'{ic}ret_r{r}'][i0, i1, (i2 + 1) * step_num_numsym, i3]):
                                if np.isnan(acc_upsampl[f'{ic}ret_r{r}'][i0, i1, i2 * step_num_numsym, i3]):
                                    acc_upsampl[f'{ic}ret_r{r}'][i0, i1, i2 * step_num_numsym:(i2 + 1) * step_num_numsym, i3] = np.nan
                            else:
                                acc_upsampl[f'{ic}ret_r{r}'][i0, i1, i2 * step_num_numsym:(i2 + 1) * step_num_numsym, i3] += np.linspace(0, acc_upsampl[f'{ic}ret_r{r}'][i0, i1, (i2+1)*step_num_numsym, i3] - acc_upsampl[f'{ic}ret_r{r}'][i0, i1, i2*step_num_numsym, i3], step_num_numsym+1)[:-1]
            acc_upsampl[f'{ic}ret_r{r}'] = acc_upsampl[f'{ic}ret_r{r}'].repeat(step_num_numsym, axis=3)

            for i0 in range(acc_upsampl[f'{ic}ret_r{r}'].shape[0]):
                for i1 in range(acc_upsampl[f'{ic}ret_r{r}'].shape[1]):
                    for i2 in range(acc_upsampl[f'{ic}ret_r{r}'].shape[2]):
                        for i3 in range(acc_sr[f'{ic}ret_r{r}'].shape[3]-1):
                            if np.isnan(acc_upsampl[f'{ic}ret_r{r}'][i0, i1, i2, (i3 + 1) * step_num_numsym]):
                                if np.isnan(acc_upsampl[f'{ic}ret_r{r}'][i0, i1, i2, i3 * step_num_numsym]):
                                    acc_upsampl[f'{ic}ret_r{r}'][i0, i1, i2, i3 * step_num_numsym:(i3 + 1) * step_num_numsym] = np.nan
                            else:
                                acc_upsampl[f'{ic}ret_r{r}'][i0, i1, i2, i3 * step_num_numsym:(i3 + 1) * step_num_numsym] += np.linspace(0, acc_upsampl[f'{ic}ret_r{r}'][i0, i1, i2, (i3+1)*step_num_numsym] - acc_upsampl[f'{ic}ret_r{r}'][i0, i1, i2, i3*step_num_numsym], step_num_numsym+1)[:-1]
            acc_upsampl[f'{ic}ret_r{r}'] = acc_upsampl[f'{ic}ret_r{r}']
    return acc_upsampl

def fit_gaussian_ninc(acc_2D, ni, nc, m_gmm):
    '''
    fit 2D gaussian function
    '''
    Ni, Nc = np.meshgrid(ni, nc)
    acc_flatten = acc_2D.ravel()
    Ni_flatten = Ni.ravel()
    Nc_flatten = Nc.ravel()
    # acc_fit = np.zeros(acc_flatten.shape)
    pos_mask = ~np.isnan(acc_flatten)

    Ni_flatten = Ni_flatten[pos_mask]
    Nc_flatten = Nc_flatten[pos_mask]
    acc_flatten = acc_flatten[pos_mask]

    xdata = np.vstack((Ni_flatten, Nc_flatten))
    ydata = acc_flatten

    # Initial guess for the parameters [A, x0, y0, sigma_x, sigma_y, C]
    if m_gmm == 0:
        initial_guess = (1, 1, 1, 1, 1, 0)
    elif m_gmm == 1:
        initial_guess = (1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1)
    elif m_gmm == 2:
        initial_guess = (1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    elif m_gmm == 3:
        initial_guess = (1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    elif m_gmm == 4:
        initial_guess = (1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1)
    else:
        print('m_gmm = %d, please give a m_gmm <= 4' % m_gmm)


    # Fit the Gaussian
    # popt, pcov = curve_fit(gaussian_2d_fit, xdata, ydata, p0=initial_guess)

    popt = least_squares(ls_fun, initial_guess, args=(xdata, ydata))

    # Extract the fitted parameters
    # A, x0, y0, sigma_x, sigma_y, C = popt

    # # Generate the fitted data
    # Z_fit = gmm_2D(Ni_flatten, Nc_flatten, A, x0, y0, sigma_x, sigma_y, C)
    # # for i in range(len(pos_mask)):
    # acc_fit[pos_mask] = Z_fit
    # acc_fit = np.reshape(acc_fit, acc_2D.shape)

    # diff1 = np.abs(acc_2D.ravel() - acc_fit)
    # diff2 = np.abs(acc_2D - np.reshape(acc_fit, acc_2D.shape))
    # sum1 = diff1.sum()
    # sum2 = diff2.sum()

    return popt # A, x0, y0, sigma_x, sigma_y, C

def fit_exp_ploynom(acc_2D, ni, nc):
    Ni, Nc = np.meshgrid(ni, nc)
    acc_flatten = acc_2D.ravel()
    Ni_flatten = Ni.ravel()
    Nc_flatten = Nc.ravel()
    acc_fit = np.zeros(acc_flatten.shape)
    pos_mask = ~np.isnan(acc_flatten)

    Ni_flatten = Ni_flatten[pos_mask]
    Nc_flatten = Nc_flatten[pos_mask]
    acc_flatten = acc_flatten[pos_mask]

    xdata = np.vstack((Ni_flatten, Nc_flatten))
    ydata = acc_flatten

    initial_guess = (1, 1, 1)

    popt, pcov = curve_fit(exp_ploynom_2D_fit, xdata, ydata, p0=initial_guess)
    # test
    a0, a1, a2 = popt

    # Generate the fitted data
    Z_fit = exp_ploynom_2D(Ni_flatten, Nc_flatten, a0, a1, a2)
    acc_fit[pos_mask] = Z_fit
    diff2 = np.abs(acc_2D - np.reshape(acc_fit, acc_2D.shape))
    sum2 = diff2.sum()
    print(sum2)
    return popt
def get_acc_fit_gaussian(ni, nc, popt, m_gmm):

    Ni, Nc = np.meshgrid(ni, nc)

    if m_gmm == 4:
        (A, x0, y0, sigma_x, sigma_y, C,
         A1, x1, y1, sigma_x1, sigma_y1,
         A2, x2, y2, sigma_x2, sigma_y2,
         A3, x3, y3, sigma_x3, sigma_y3,
         A4, x4, y4, sigma_x4, sigma_y4) = popt
        Z_fit = gmm_2D(Ni, Nc, A, x0, y0, sigma_x, sigma_y, C,
                       A1, x1, y1, sigma_x1, sigma_y1,
                       A2, x2, y2, sigma_x2, sigma_y2,
                       A3, x3, y3, sigma_x3, sigma_y3,
                       A4, x4, y4, sigma_x4, sigma_y4)
    elif m_gmm == 3:
        (A, x0, y0, sigma_x, sigma_y, C,
         A1, x1, y1, sigma_x1, sigma_y1,
         A2, x2, y2, sigma_x2, sigma_y2,
         A3, x3, y3, sigma_x3, sigma_y3) = popt
        Z_fit = gmm_2D(Ni, Nc, A, x0, y0, sigma_x, sigma_y, C,
                       A1, x1, y1, sigma_x1, sigma_y1,
                       A2, x2, y2, sigma_x2, sigma_y2,
                       A3, x3, y3, sigma_x3, sigma_y3)
    elif m_gmm == 2:
        (A, x0, y0, sigma_x, sigma_y, C,
         A1, x1, y1, sigma_x1, sigma_y1,
         A2, x2, y2, sigma_x2, sigma_y2) = popt
        Z_fit = gmm_2D(Ni, Nc, A, x0, y0, sigma_x, sigma_y, C,
                       A1, x1, y1, sigma_x1, sigma_y1,
                       A2, x2, y2, sigma_x2, sigma_y2)
    elif m_gmm == 1:
        (A, x0, y0, sigma_x, sigma_y, C,
         A1, x1, y1, sigma_x1, sigma_y1) = popt
        Z_fit = gmm_2D(Ni, Nc, A, x0, y0, sigma_x, sigma_y, C,
                       A1, x1, y1, sigma_x1, sigma_y1)
    else:
        (A, x0, y0, sigma_x, sigma_y, C) = popt
        Z_fit = gmm_2D(Ni, Nc, A, x0, y0, sigma_x, sigma_y, C)

    return Z_fit


############################################# polynomal interpolation 4D ---- snr_img, snr_cap, , numsym_img, numsym_cap
if __name__ == '__main__':
    pass
