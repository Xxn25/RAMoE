import torch
import numpy as np
from skimage.metrics import structural_similarity
import torch
from scipy.ndimage import uniform_filter
from skimage.transform import resize


def calc_ergas(img_tgt, img_fus, r):
    img_tgt = torch.squeeze(img_tgt)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = torch.squeeze(img_fus)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)

    rmse = torch.mean((img_tgt - img_fus) ** 2)
    rmse = rmse ** 0.5
    mean = torch.mean(img_tgt)

    ergas = torch.mean((rmse / mean) ** 2)
    ergas = 100 / r * ergas ** 0.5

    return ergas.item()

def calc_psnr(img_tgt, img_fus):
    img_tgt = torch.squeeze(img_tgt)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = torch.squeeze(img_fus)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    mse = torch.mean(torch.square(img_tgt-img_fus))
    img_max = torch.max(img_tgt)
    # img_max = 1.0
    psnr = 10.0 * torch.log10(img_max**2/mse)

    return psnr.item()

def calc_rmse(img_tgt, img_fus):

    img_tgt = torch.squeeze(img_tgt)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = torch.squeeze(img_fus)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    rmse = torch.sqrt(torch.mean((img_tgt-img_fus)**2))

    return rmse.item()

def calc_sam(img_tgt, img_fus):
    img_tgt = torch.squeeze(img_tgt)
    img_tgt = img_tgt.reshape(img_tgt.shape[1], -1)
    img_fus = torch.squeeze(img_fus)
    img_fus = img_fus.reshape(img_fus.shape[1], -1)
    img_tgt = img_tgt / torch.max(img_tgt)
    img_fus = img_fus / torch.max(img_fus)

    A = torch.sqrt(torch.sum(img_tgt**2))
    B = torch.sqrt(torch.sum(img_fus**2))
    AB = torch.sum(img_tgt*img_fus)

    sam = AB/(A*B)

    sam = torch.arccos(sam)
    sam = torch.mean(sam)*180/torch.pi

    return sam.item()

def calc_ssim(img_tgt, img_fus):
    '''
    :param reference:
    :param target:
    :return:
    '''

    img_tgt = torch.squeeze(img_tgt)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = torch.squeeze(img_fus)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    img_tgt = img_tgt.cpu().numpy()
    img_fus = img_fus.cpu().numpy()

    ssim = structural_similarity(img_tgt, img_fus, data_range=1)

    return ssim


def uqi(x, y):
    """
    计算 UIQI (Universal Image Quality Index)
    x, y: 两幅同尺寸图像 (H×W)
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    cov_xy = np.mean((x - mean_x) * (y - mean_y))

    numerator = 4 * mean_x * mean_y * cov_xy
    denominator = (mean_x**2 + mean_y**2) * (var_x**2 + var_y**2)

    if denominator == 0:
        return 1.0
    return numerator / denominator


def d_lambda(ms, fused, p=1):
    """计算光谱失真指数（D_lambda）。

    :param ms: 低分辨率多光谱图像。
    :param fused: 高分辨率融合图像。
    :param p: 用于强调较大光谱差异的参数（默认值为1）。

    :returns:  float -- D_lambda。
    """
    L = ms.shape[2]

    M1 = np.zeros((L, L))
    M2 = np.zeros((L, L))

    for l in range(L):
        for r in range(l, L):
            M1[l, r] = M1[r, l] = uqi(fused[:, :, l], fused[:, :, r])
            M2[l, r] = M2[r, l] = uqi(ms[:, :, l], ms[:, :, r])

    diff = np.abs(M1 - M2) ** p
    return (1. / (L * (L - 1)) * np.sum(diff)) ** (1. / p)


def d_s(pan, ms, fused, q=1, r=4, ws=7):
    """计算空间失真指数（D_S）。

    :param pan: 高分辨率全色图像。
    :param ms: 低分辨率多光谱图像。
    :param fused: 高分辨率融合图像。
    :param q: 用于强调较大空间差异的参数（默认值为1）。
    :param r: 高分辨率与低分辨率的比例（默认值为4）。
    :param ws: 滑动窗口大小（默认值为7）。

    :returns:  float -- D_S。
    """
    pan = pan.astype(np.float64)
    fused = fused.astype(np.float64)

    pan_degraded = uniform_filter(pan.astype(np.float64), size=ws) / (ws ** 2)
    pan_degraded = resize(pan_degraded, (pan.shape[0] // r, pan.shape[1] // r))

    L = ms.shape[2]

    M1 = np.zeros(L)
    M2 = np.zeros(L)

    for l in range(L):
        M1[l] = uqi(fused[:, :, l], pan)
        M2[l] = uqi(ms[:, :, l], pan_degraded)

    diff = np.abs(M1 - M2) ** q
    return ((1. / L) * (np.sum(diff))) ** (1. / q)


def qnr(pan, ms, fused, alpha=1, beta=1, p=1, q=1, r=4, ws=7):
    """计算无参考质量指标（QNR）。

    :param pan: 高分辨率全色图像。
    :param ms: 低分辨率多光谱图像。
    :param fused: 高分辨率融合图像。
    :param alpha: 强调光谱失真对整体的影响。
    :param beta: 强调空间失真对整体的影响。
    :param p: 用于强调较大光谱差异的参数（默认值为1）。
    :param q: 用于强调较大空间差异的参数（默认值为1）。
    :param r: 高分辨率与低分辨率的比例（默认值为4）。
    :param ws: 滑动窗口大小（默认值为7）。

    :returns:  float -- QNR。
    """
    a1 = d_lambda(ms, fused, p=p)
    b1 = d_s(pan, ms, fused, q=q, ws=ws, r=r)
    a = (1 - a1) ** alpha
    b = (1 - b1) ** beta
    return a1, b1, a * b

