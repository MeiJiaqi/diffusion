from skimage.metrics import peak_signal_noise_ratio,structural_similarity
import numpy as np

def get_psnr_ssim(pre,gt):
    mask1 = np.unique(np.nonzero(gt)[0])
    mask = np.nonzero(gt)
    gt = gt[mask]
    pre = pre[mask]
    gt1 = gt[mask1]
    pre1 = pre[mask1]
    # pre = pre[:,np.newaxis,...]
    # gt = gt[:,np.newaxis,...]
    # for i in range(len(gt)):
    #     s_pre = pre[i]
    #     s_gt = gt[i]
    #     if s_gt.max()>0.1:
    #         psnr.append(peak_signal_noise_ratio(s_gt,s_pre,data_range=1))
    #         ssim.append(structural_similarity(s_gt,s_pre,data_range=1))
    psnr=(peak_signal_noise_ratio(gt, pre, data_range=1))
    ssim=(structural_similarity(gt1,pre1,data_range=1))
    return psnr, ssim