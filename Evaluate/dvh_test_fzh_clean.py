import csv

import numpy as np

from evaluate_openKBP import get_DVH_metrics, plot_DVH, get_3D_Dose_score, get_SDE
import SimpleITK  as sit
import os
import torch
import torch.nn as nn
from psnr_n_ssim import get_psnr_ssim
#from record import comparison,ablation
import sys
sys.path.append("..")
from networks.unet import UNet
from networks.diffusion import GaussianDiffusion

import utils


# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

def get_dose_diff(prediction,target,roi):

    dose_diff = []
    for key in roi.keys():
        dose_diff.append(get_3D_Dose_score(prediction,target,roi[key]))
    return np.array(dose_diff)
def get_shift_dvh_err(prediction,target,roi):

    SDE = []
    for key in roi.keys():
        SDE.append(get_SDE(prediction,target,roi[key]))
    return np.array(SDE)
def test_one(prediction,target,roi,name,dvh_path):
    prediction = prediction*1.1*50.4
    target = target*1.1*50.4
    pre_metric = dict()
    tar_metric = dict()

    dose_score = get_3D_Dose_score(prediction,target)
    pre_metric['dose_score'] = dose_score
    tar_metric['dose_score'] = dose_score

    for key in roi.keys():
        if key=='PTV':
            pre_metric['PTV'] = get_DVH_metrics(prediction,roi['PTV'],'target')
            tar_metric['PTV'] = get_DVH_metrics(target, roi['PTV'], 'target')
        else:
            pre_metric[key] = get_DVH_metrics(prediction, roi[key], 'OAR')
            tar_metric[key] = get_DVH_metrics(target, roi[key], 'OAR')
    ref = {'real_dose': target, 'ROI': roi}
    dose_diff = get_dose_diff(prediction, target, roi)
    SD_err = get_shift_dvh_err(prediction/1.1/50.4, target/1.1/50.4, roi)
    plot_DVH(prediction, ref, os.path.join(dvh_path,"{}.png".format(name)))
    return pre_metric,tar_metric,dose_diff,SD_err
def dict2table(d):
    """
    d:{
        dose_score,
        ptv{
            D98,D50,D2,Dmean,HI,CI
            },
        oar{
            D2.V40,V50
            }
        }
    :param d:
    :return:
    """

    dose_score=d['dose_score']

    ptv_idx = d['PTV']
    ptv = [ptv_idx['D98'],ptv_idx['D50'],ptv_idx['D2'],ptv_idx['Dmean'],ptv_idx['HI'],ptv_idx['CI']]

    oars = []
    oars_name = ['ST','FHL','FHR','BLD']
    for o in oars_name:
        oar_idx = d[o]
        oar = [oar_idx['D2'],oar_idx['Dmean'],oar_idx['V40'],oar_idx['V50']]
        oars.append(oar)
    return dose_score,ptv,oars



net1 = UNet(in_channel=2, out_channel=4, inner_channel=32, norm_groups=16, channel_mults=(1, 2, 4, 8, 16),
                   attn_res=[],
                   res_blocks=1, dropout=0, with_time_emb=False, with_feature_emb=False,image_size=160)


net2 = UNet(in_channel=3, out_channel=1, inner_channel=32, norm_groups=16, channel_mults=(1, 2, 4, 8, 16),
                   attn_res=[],
                   res_blocks=1, dropout=0, with_time_emb=True, with_feature_emb=True,image_size=160)

# # net1 = UNet(in_channel=1, out_channel=1, inner_channel=32, norm_groups=16, channel_mults=(1, 2, 4, 8, 16),
# #                    attn_res=[],
# #                    res_blocks=1, dropout=0, with_time_emb=True, with_feature_emb=False,with_ptv=True,image_size=160)
#
#
# net2 = UNet(in_channel=2, out_channel=1, inner_channel=32, norm_groups=16, channel_mults=(1, 2, 4, 8, 16),
#                    attn_res=[],
#                    res_blocks=1, dropout=0, with_time_emb=False, with_feature_emb=False,with_ptv=False,image_size=160)

print("Num params: ", sum(p.numel() for p in net2.parameters()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
    print(f"Let's use {torch.cuda.device_count()} GPUs!")

net1 = nn.DataParallel(net1)
net2 = nn.DataParallel(net2)
#net1=net1.to(device)
# net1=0
net1, net2, iter, epoch = utils.load_model(net1, net2, 700)
net2=net2.to(device)
net1=net1.to(device)
#net2=net2.to(device)
timesteps=1000
linear_start=1e-4
linear_end=1e-2
schedule_opt={'schedule':'linear','n_timestep':timesteps,'linear_start':linear_start,'linear_end':linear_end}
#gaussian_diffusion=GaussianDiffusion(model=net2,image_size=160,channels=1,loss_type='l1',conditional=True,schedule_opt=schedule_opt,device=device)
gaussian_diffusion=GaussianDiffusion(model=net2,image_size=160,channels=1,loss_type='l1',conditional=True,schedule_opt=schedule_opt,device=device)
if __name__ == '__main__':




    names = []
    # pre_path = r"../ablation/mha/{}".format(ablation['baseline+enc+cat']['mha'])
    # save_path = r"../results_all_ablation/index/{}".format(ablation['baseline+enc+cat']['mha'])
    # pre_path = r"D:\workfile\dlfile\2024mic\diffusion\sample"
    # save_path = r"D:\workfile\dlfile\2024mic\diffusion\sample\mha-mul-unet-200"
    # data_path = r"D:\workfile\dlfile\2024mic\diffusion\data\rectum333_npz\test"
    pre_path = r'/home/scusw1/mic/pycharm_project_433/sample'
    save_path = r'/home/scusw1/mic/pycharm_project_433/sample/mha-mul-diff-v11-700'
    data_path = r'/home/scusw1/mic/pycharm_project_433/data/rectum333_npz/test'

    # pre_path = r'/data/shuangjun.du/diffusion/sample'
    # save_path = r'/data/shuangjun.du/diffusion/sample/mha-mul-diff-600'
    # data_path = r'/data/shuangjun.du/diffusion/data/rectum333_npz/test'


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for root,_,fnames in sorted(os.walk(data_path)):
        for fname in fnames:
            name = fname.split('.')[0]

            #names = os.path.join(root,name)
            names.append(name)

    ptv_tars = []
    ptv_pres = []
    oar_tars = []
    oar_pres = []
    dss = []
    psnr = []
    ssim = []
    dose_diffs = []
    sd_errs = []
    i=0
    for name in names:
        print(name)
        # i=i+1
        # if i < 6 :
        #     continue
        inputs = np.load(os.path.join(data_path,name+'.npz'))['arr_0']
        rd=np.load(os.path.join(data_path,name+'.npz'))['arr_1']
        ptv= inputs[:,-2]
        small = inputs[:,1]
        fl = inputs[:,2]
        fr = inputs[:,3]
        bladder = inputs[:,-1]
        realdose=np.squeeze(rd, axis=1)
        ct=inputs[:,0]
        ct=ct*2-1

        # 打印 ptv 的形状
        # print("rd 的形状:", rd.shape)
        # # 打印 ptv 的最大值和最小值
        # print("rd 的最大值:", np.max(rd))
        # print("rd 的最小值:", np.min(rd))
        batch_size = 40  # 你可以根据需要设置 batch size
        total_nums, height, width = ct.shape
        # 将 ct 转换为 PyTorch 张量
        ct_tensor = torch.from_numpy(ct).float().to(device)
        ptv_tensor = torch.from_numpy(ptv).float().to(device)
        predose_list = []
        # print("ct_tensor 的最大值:", torch.max(ct_tensor))
        # print("ct_tensor 的最小值:", torch.min(ct_tensor))
        for i in range(0, total_nums, batch_size):
            # # 选择当前 batch 的数据
            # current_batch = ct_tensor[i:i + batch_size]
            #
            # # 在第二维上添加一个维度，将形状变为 [batch_size, 1, height, width]
            # current_batch = current_batch.unsqueeze(1)
            # output, features = net1(current_batch)
            # pred_rd = gaussian_diffusion.p_sample_loop(current_batch, False, features)
            # #pred_rd=output
            # # print(torch.max(output))
            # # print(torch.min(output))
            # # print(torch.max(pred_rd))
            # # print(torch.min(pred_rd))
            # pred_rd=((pred_rd + 1) / 2).squeeze(dim=1).cpu().detach().numpy()
            # # print(pred_rd.shape)
            # # print("pred_rd 的最大值:", np.max(pred_rd))
            # # print("pred_rd 的最小值:", np.min(pred_rd))
            current_batch = ct_tensor[i:i + batch_size]
            cerrent_ptv = ptv_tensor[i:i + batch_size]
            current_batch = current_batch.unsqueeze(1)
            cerrent_ptv = cerrent_ptv.unsqueeze(1)
            input = torch.cat([current_batch, cerrent_ptv], dim=1)
            #pred_rd = gaussian_diffusion.p_sample_loop(current_batch, False, condition=conditon)
            output, feature = net1(input)

            #pred_rd = net2(current_batch,feature)
            pred_rd = gaussian_diffusion.p_sample_loop(current_batch,ptv=input, continous=False,condition=feature)
            # print(pred_rd.shape)
            pred_rd=((pred_rd + 1) / 2).squeeze(dim=1).cpu().detach().numpy()
            predose_list.append(pred_rd)
            # print("pred_rd 的最大值:", np.max(pred_rd))
            # print("pred_rd 的最小值:", np.min(pred_rd))
        predose = np.concatenate(predose_list, axis=0)
        print("合并后的 predose 形状:", predose.shape)
        utils.save_mha(predose,name,save_path)
        #
        # try:
        #     #realdose= sit.GetArrayFromImage(sit.ReadImage(os.path.join(pre_path,"{}_tar.mha".format(name))))
        #     predose = sit.GetArrayFromImage(sit.ReadImage(os.path.join(save_path,"{}.mha".format(name))))
        #     print(predose.shape)
        # except RuntimeError:
        #     continue




        pad = ptv.shape[-1] // 2 - 80

        roi = {'PTV':ptv[:,pad:pad+160,pad:pad+160],'ST':small[:,pad:pad+160,pad:pad+160],'FHL':fl[:,pad:pad+160,pad:pad+160],
               'FHR':fr[:,pad:pad+160,pad:pad+160],'BLD':bladder[:,pad:pad+160,pad:pad+160]}

        pre_idx, tar_idx, dose_diff,sd_err = test_one(predose,realdose,roi,name,save_path)

        dose_diffs.append(dose_diff)
        sd_errs.append(sd_err)

        ds, ptv_pre,oars_pre = dict2table(pre_idx)
        _, ptv_tar, oars_tar = dict2table(tar_idx)

        dss.append(ds)
        ptv_pres.append(ptv_pre)
        ptv_tars.append(ptv_tar)
        oar_pres.append(oars_pre)   #[patient,oar,metric]
        oar_tars.append(oars_tar)
        pp,ss = get_psnr_ssim(predose,realdose)
        psnr.append(pp)
        ssim.append(ss)

    ptv_h = ['D98','D50','D2','Dmean','HI','CI']
    oars_h = ['D2','Dmean','V40','V50']

    ptv_res = list(np.mean(abs(np.array(ptv_pres)-np.array(ptv_tars)),axis=0))
    ptv_std = list(np.std(abs(np.array(ptv_pres)-np.array(ptv_tars)),axis=0))

    oar_res = list(np.mean(abs(np.array(oar_pres).reshape([22,-1])-np.array(oar_tars).reshape([22,-1])),axis=0))
    oar_std = list(np.std(abs(np.array(oar_pres).reshape([22,-1])-np.array(oar_tars).reshape([22,-1])),axis=0))

    dss_res = [np.mean(abs(np.array(dss)))]
    dss_std = [np.std(abs(np.array(dss)))]

    oar_pres = list(np.array(oar_pres).reshape([22,-1]))
    oar_tars = list(np.array(oar_tars).reshape([22,-1]))

    # oar_res = list(np.mean(abs(np.array(oar_pres).reshape([5, -1]) - np.array(oar_tars).reshape([5, -1])), axis=0))
    # oar_std = list(np.std(abs(np.array(oar_pres).reshape([5,-1])-np.array(oar_tars).reshape([5,-1])),axis=0))
    #
    # dss_res = [np.mean(abs(np.array(dss)))]
    # dss_std = [np.std(abs(np.array(dss)))]
    #
    # oar_pres = list(np.array(oar_pres).reshape([5,-1]))
    # oar_tars = list(np.array(oar_tars).reshape([5,-1]))

    psnr_all = (np.array(psnr).reshape(-1)).mean()
    psnr_std = (np.array(psnr).reshape(-1)).std()

    ssim_all = (np.array(ssim).reshape(-1)).mean()
    ssim_std = (np.array(ssim).reshape(-1)).std()
    with open(os.path.join(save_path,'index.csv'), 'w', newline='') as file_obj:
        writer = csv.writer(file_obj)
        # 写表头
        writer.writerow(ptv_h)
        # 遍历，将每一行的数据写入csv
        for p in ptv_pres:
            writer.writerow(p)
        writer.writerow([])
        for p in ptv_tars:
            writer.writerow(p)
        writer.writerow(['avg'])
        writer.writerow(ptv_res)
        writer.writerow(ptv_std)
        writer.writerow([])

        writer.writerow(oars_h)
        # 遍历，将每一行的数据写入csv

        for p in oar_pres:
            writer.writerow(p)
        writer.writerow([])
        for p in oar_tars:
            writer.writerow(p)
        writer.writerow(['avg'])
        writer.writerow(oar_res)
        writer.writerow(oar_std)

        writer.writerow([])
        writer.writerow('dose score')

        writer.writerow(dss)
        writer.writerow(['avg'])
        writer.writerow(dss_res)
        writer.writerow(dss_std)

        writer.writerow(['psnr'])
        writer.writerow(psnr)
        writer.writerow([psnr_all])
        writer.writerow([psnr_std])
        writer.writerow(['ssim'])
        writer.writerow(ssim)
        writer.writerow([ssim_all])
        writer.writerow([ssim_std])

        writer.writerow(['dose_diff'])
        writer.writerow(['ptv', 'st', 'fhl', 'fhr', 'bld'])
        for p in dose_diffs:
            writer.writerow(p)

        writer.writerow(['AVG'])
        writer.writerow(list(np.array(dose_diffs).mean(axis=0)))
        writer.writerow(list(np.array(dose_diffs).std(axis=0)))

        writer.writerow([])
        writer.writerow(['shifted_DVH_error'])
        writer.writerow(['ptv', 'st', 'fhl', 'fhr', 'bld'])
        for p in sd_errs:
            writer.writerow(p)
        writer.writerow(['AVG'])
        writer.writerow(list(np.array(sd_errs).mean(axis=0)))
        writer.writerow(list(np.array(sd_errs).std(axis=0)))






