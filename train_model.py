"""Train the model"""

import argparse
import logging
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import datasets_npz
import utils
from networks.unet import UNet
from networks.diffusion import GaussianDiffusion

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

# net1 = UNet(in_channel=1, out_channel=1, inner_channel=32, norm_groups=16, channel_mults=(1, 2, 4, 8, 16),
#                    attn_res=[],
#                    res_blocks=1, dropout=0, with_time_emb=False, with_feature_emb=False, image_size=160)
#
# net2 = UNet(in_channel=1, out_channel=1, inner_channel=32, norm_groups=16, channel_mults=(1, 2, 4, 8, 16),
#                    attn_res=[],
#                    res_blocks=1, dropout=0, with_time_emb=True, with_feature_emb=True, image_size=160)
# print("Num params: ", sum(p.numel() for p in net1.parameters()))
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if torch.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
#     print(f"Let's use {torch.cuda.device_count()} GPUs!")
# # 定义要使用的GPU数量
# num_gpus = torch.cuda.device_count()
#
#
#
# iter = 0
# loss1= 0
# loss2= 0
# epoch=0
#
# # 使用DataParallel包装模型
# net1 = nn.DataParallel(net1)
# net2 = nn.DataParallel(net2)
# net1, net2, iter, epoch = utils.load_model(net1, net2, 550)
#
# net1=net1.to(device)
# net2=net2.to(device)
# epochs = 1500
# timesteps=1000
# linear_start=1e-4
# linear_end=1e-2
# # learning_rate =1e-4
# learning_rate =5e-5      #550epoch后
#
# batch_size=30
# schedule_opt={'schedule':'linear','n_timestep':timesteps,'linear_start':linear_start,'linear_end':linear_end}
# gaussian_diffusion=GaussianDiffusion(model=net2,image_size=160,channels=1,loss_type='l1',conditional=True,schedule_opt=schedule_opt,device=device)
# optimizer1=torch.optim.Adam(net1.parameters(),lr=learning_rate)
# optimizer2=torch.optim.Adam(net2.parameters(),lr=learning_rate)
# criterion = torch.nn.BCEWithLogitsLoss()
# l1_loss=nn.L1Loss().to(device)
# best_val_dice=0  # 初始化为一个很大的值
#
# if __name__ == '__main__':
#
#     # val_model()
#     # utils.save_model(net1, 0, iter, 1)
#     # utils.save_model(net2, 0, iter, 2)
#
#     for epo in range(epoch,epochs):
#         # # val
#         # with torch.no_grad():
#         #     val = datasets_npz.make_Valdataset()
#         #     bach = 35
#         #     print('-----------------------------------')
#         #     print('val model')
#         #     dice_values_total = []
#         #     hd95_values_total = []
#         #     for ii, batch_sample in enumerate(val):
#         #         dataloader = DataLoader(dataset=datasets_npz.MyDataset(batch_sample), batch_size=bach, shuffle=True,
#         #                                 drop_last=True)
#         #         dice_values_i = []
#         #         hd95_values_i = []
#         #         for i, (ct, ptv, rd) in enumerate(tqdm(dataloader)):
#         #             ct = ct.to(device)
#         #             inputs = ct.to(device)
#         #             output, feature = net1(inputs)
#         #             ptv = ptv.to(device)
#         #             rd = rd.to(device)
#         #             pred_rd = rd
#         #             # 计算指标并累加
#         #             dice, hd95, mae, dd95 = utils.test_metric(torch.sigmoid(output), ptv, pred_rd, rd, ct)
#         #             dice_values_i.append(dice)
#         #             dice_values_total.append(dice)
#         #
#         #             # 计算平均值
#         #             # 计算平均值和方差
#         #         avg_dice_i = torch.tensor(dice_values_i, dtype=torch.float32).mean().item()
#         #         var_dice_i = torch.tensor(dice_values_i, dtype=torch.float32).var().item()
#         #
#         #         print(f'[val]:volume{ii}: Average Dice: {avg_dice_i}, Variance: {var_dice_i}')
#         #
#         #     avg_dice_total = torch.tensor(dice_values_total, dtype=torch.float32).mean().item()
#         #     var_dice_total = torch.tensor(dice_values_total, dtype=torch.float32).var().item()
#         #
#         #     print('-----------------------------------------------------------------')
#         #     print(f'Total: Average Dice: {avg_dice_total}, Variance: {var_dice_total}')
#         #
#         #     if avg_dice_total > best_val_dice:
#         #         best_val_dice = avg_dice_total
#         #         utils.save_model(net1, epo, iter, 1, True)
#
#         #train
#         tra=datasets_npz.make_datasetS()
#         print(f'epoch:{epo}')
#         for ii, batch_sample in enumerate(tra):
#             name = batch_sample['name']
#             print(f'{ii}，{name}')
#             dataloader = DataLoader(dataset=datasets_npz.MyDataset(batch_sample), batch_size=batch_size, shuffle=True, drop_last=True)
#             for i, (ct, ptv,rd) in enumerate(tqdm(dataloader)):
#                 #optimizer1.zero_grad()
#                 optimizer2.zero_grad()
#                 ct=ct.to(device)
#                 inputs = ct.to(device)
#                 output, feature = net1(inputs)
#                 #output,_=net1(inputs)
#                 ptv = ptv.to(device)
#                 rd=rd.to(device)
#                 # print(ct.shape)
#                 # print("Minimum value in ptv:", torch.min(ptv).item())
#                 # print("Maximum value in ptv:", torch.max(ptv).item())
#                 # #
#                 # print("Minimum value in output:", torch.min(output).item())
#                 # print("Maximum value in output:", torch.max(output).item())
#                 #loss1=criterion(output,ptv)
#                 #loss1=l1_loss(output,ptv)
#                 loss2 = gaussian_diffusion.p_losses(rd,condition=feature)
#                 iter=iter+1
#
#                 #loss1.backward(retain_graph=True)
#                 loss2.backward()
#                 #optimizer1.step()
#                 optimizer2.step()
#                 # 更新tqdm的描述字符串
#                 #tqdm.write(f'Iteration: {iter},[train]: SegLoss: {loss1.item()}, DoseLoss: {loss2.item()}')
#                 tqdm.write(f'[train]: epoch:{epo} Iteration: {iter}, DoseLoss: {loss2.item()}')
#         if epo%50 == 0:
#             #utils.save_model(net1,epo,iter,1)
#             utils.save_model(net2,epo,iter,2)

net1 = UNet(in_channel=2, out_channel=4, inner_channel=32, norm_groups=16, channel_mults=(1, 2, 4, 8, 16),
                   attn_res=[],
                   res_blocks=1, dropout=0, with_time_emb=False, with_feature_emb=False,image_size=160)


net2 = UNet(in_channel=3, out_channel=1, inner_channel=32, norm_groups=16, channel_mults=(1, 2, 4, 8, 16),
                   attn_res=[],
                   res_blocks=1, dropout=0, with_time_emb=True, with_feature_emb=True,image_size=160)

print("Num params: ", sum(p.numel() for p in net2.parameters()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
# 定义要使用的GPU数量
num_gpus = torch.cuda.device_count()



iter = 0
loss1= 0
loss2= 0
loss_rec=0
epoch=0

# 使用DataParallel包装模型
net1 = nn.DataParallel(net1)
net2 = nn.DataParallel(net2)
net1, net2, iter, epoch = utils.load_model(net1, net2, 400)
net1=net1.to(device)
net2=net2.to(device)
epochs = 1500
timesteps=1000
linear_start=1e-4
linear_end=1e-2
learning_rate =1e-4
# learning_rate =5e-5      #550epoch后

batch_size=40
schedule_opt={'schedule':'linear','n_timestep':timesteps,'linear_start':linear_start,'linear_end':linear_end}
gaussian_diffusion=GaussianDiffusion(model=net2,image_size=160,channels=1,loss_type='l1',conditional=True,schedule_opt=schedule_opt,device=device)
# optimizer1=torch.optim.Adam(net1.parameters(),lr=learning_rate)
optimizer2=torch.optim.Adam(net2.parameters(),lr=learning_rate)
criterion = torch.nn.BCEWithLogitsLoss()
# criterion = torch.nn.CrossEntropyLoss()
l1_loss=nn.L1Loss().to(device)
best_val_dice=0  # 初始化为一个很大的值

if __name__ == '__main__':


    # utils.save_model(net1, 0, iter, 1)
    utils.save_model(net2, 0, iter, 2)
    for epo in range(epoch,epochs):

        #train
        tra=datasets_npz.make_datasetS()
        print(f'epoch:{epo}')
        for ii, batch_sample in enumerate(tra):
            name = batch_sample['name']
            print(f'{ii}，{name}')
            dataloader = DataLoader(dataset=datasets_npz.MyDataset(batch_sample), batch_size=batch_size, shuffle=True, drop_last=True)
            for i, (ct, ptv,rd,oars) in enumerate(tqdm(dataloader)):
                # optimizer1.zero_grad()
                optimizer2.zero_grad()
                # print(oars.shape)
                ct=ct.to(device)
                oars=oars.to(device)
                ptv = ptv.to(device)
                rd=rd.to(device)
                input=torch.cat([ct,ptv],dim=1)
                # print(input.shape)
                # print(ct.shape)
                # print("Minimum value in ptv:", torch.min(oars).item())
                # print("Maximum value in ptv:", torch.max(oars).item())
                # non_ptv_rd = rd * (1 - ptv)
                # print(non_ptv_rd.shape)
                # print("Minimum value in non_ptv_rd:", torch.min(non_ptv_rd).item())
                # print("Maximum value in non_ptv_rd:", torch.max(non_ptv_rd).item())
                output,feature = net1(input)
                # print("Minimum value in output:", torch.min(output).item())
                # print("Maximum value in output:", torch.max(output).item())
                # pred_rd = net2(ct,feature)
                # loss2 = l1_loss(pred_rd,rd)
                # loss1=criterion(output,oars)
                loss2 = gaussian_diffusion.p_losses(x_start=rd, ptv=input,condition=feature)
                # loss_rec = gaussian_diffusion.p_losses(x_start=non_ptv_rd, ptv=input,condition=feature)  #修正损失，让模型更关注ptv外面区域
                # if epo <=500 :
                #     loss1.backward(retain_graph=True)
                loss2.backward()
                # loss_rec.backward()
                # optimizer1.step()
                optimizer2.step()
                iter=iter+1
                # 更新tqdm的描述字符串
                # tqdm.write(f'[train]: epoch:{epo},Iteration: {iter}, SegLoss: {loss1.item()}, DoseLoss: {loss2.item()}, RecLoss: {loss_rec.item()}')
                # tqdm.write(f'[train]: epoch:{epo} Iteration: {iter}, DoseLoss: {loss2.item()} , RecLoss: {loss_rec.item()}')
                tqdm.write(f'[train]: epoch:{epo} Iteration: {iter}, DoseLoss: {loss2.item()} ')
        if epo%50 == 0:
            # if epo <= 500:
            #     utils.save_model(net1,epo,iter,1)
            utils.save_model(net2,epo,iter,2)









def testimg():
    # 读取图像
    image_path = 'leijinqiong_rd_99.jpg'
    ct = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 检查图像是否成功读取
    if ct is not None:
        # 将图像转换为 PyTorch 张量
        transform = transforms.ToTensor()
        ct = transform(ct).unsqueeze(0)
    else:
        print("Error: Unable to read the image.")
    ct=ct.to(device)
    # 选择要展示的步骤
    display_steps = [0, 100, 200, 300, 400, 500,600,700,800,900]
    noisy_images = []
    for t in range(timesteps):
        # 生成加噪图像

        noisy_image = gaussian_diffusion.q_sample(x_start=ct,
                                                  t=torch.tensor([t], dtype=torch.long, device='cuda'))

        # 将 PyTorch 张量转换为 NumPy 数组，并将通道调整到正确的范围（0-1）
        if t in display_steps:
            noisy_image_np = noisy_image.squeeze(dim=0).cpu().numpy().transpose(1, 2, 0)
            noisy_image_np = np.clip(noisy_image_np, 0, 1)
            # 对于灰度图，可能需要处理通道数
            if len(noisy_image_np.shape) == 2:
                noisy_image_np = np.expand_dims(noisy_image_np, axis=-1)  # 添加通道维度

            noisy_image_np = np.clip(noisy_image_np, 0, 1)
            # 保存或显示加噪图像
            noisy_images.append(noisy_image_np)
            plt.imshow(noisy_image_np, cmap='gray')
            plt.title(f'Step {t}')
            plt.show()

    # 保存图片
    for i, img in enumerate(noisy_images):
        img_path = f'noisy_image_{display_steps[i]}.png'
        cv2.imwrite(img_path, (img * 255).astype(np.uint8))
        print("保存成功")
def val_model():
    val = datasets_npz.make_Valdataset()
    bach=35
    dice_values_total = []
    hd95_values_total = []
    mae_values_total = []
    dd95_values_total = []
    for ii, batch_sample in enumerate(val):
        dataloader = DataLoader(dataset=datasets_npz.MyDataset(batch_sample), batch_size=bach, shuffle=True,drop_last=True)
        dice_values_i = []
        hd95_values_i = []
        mae_values_i = []
        dd95_values_i = []
        for i, (ct, ptv, rd) in enumerate(tqdm(dataloader)):
            ct = ct.to(device)
            inputs = ct.to(device)
            output, feature = net1(inputs)
            ptv = ptv.to(device)
            rd = rd.to(device)
            #pred_rd = gaussian_diffusion.p_sample_loop(ct, False, feature)
            pred_rd = rd
            # 计算指标并累加
            dice, hd95, mae, dd95 = utils.test_metric(torch.sigmoid(output), ptv, pred_rd, rd, ct)
            print(dice, hd95, mae, dd95)
            dice_values_i.append(dice)
            hd95_values_i.append(hd95)
            mae_values_i.append(mae)
            dice_values_total.append(dice)
            hd95_values_total.append(hd95)
            mae_values_total.append(mae)
            dd95_values_i.append(dd95)
            dd95_values_total.append(dd95)

            # 计算平均值
            # 计算平均值和方差
        avg_dice_i = torch.tensor(dice_values_i).mean().item()
        avg_hd95_i = torch.tensor(hd95_values_i).mean().item()
        avg_mae_i = torch.tensor(mae_values_i).mean().item()
        avg_dd95_i = torch.tensor(dd95_values_i).mean().item()

        var_dice_i = torch.tensor(dice_values_i).var().item()
        var_hd95_i = torch.tensor(hd95_values_i).var().item()
        var_mae_i = torch.tensor(mae_values_i).var().item()
        var_dd95_i = torch.tensor(dd95_values_i).var().item()
        print(f'volume{ii}: Average Dice: {avg_dice_i}, Variance: {var_dice_i}')
        print(f'volume{ii}: Average HD95: {avg_hd95_i}, Variance: {var_hd95_i}')
        print(f'volume{ii}: Average MAE: {avg_mae_i}, Variance: {var_mae_i}')
        print(f'volume{ii}: Average DD95: {avg_dd95_i}, Variance: {var_dd95_i}')

    avg_dice_total = torch.tensor(dice_values_total).mean().item()
    avg_hd95_total = torch.tensor(hd95_values_total).mean().item()
    avg_mae_total = torch.tensor(mae_values_total).mean().item()
    avg_dd95_total = torch.tensor(dd95_values_total).mean().item()

    var_dice_total = torch.tensor(dice_values_total).var().item()
    var_hd95_total = torch.tensor(hd95_values_total).var().item()
    var_mae_total = torch.tensor(mae_values_total).var().item()
    var_dd95_total = torch.tensor(dd95_values_total).var().item()
    print('-----------------------------------------------------------------')
    print(f'Total: Average Dice: {avg_dice_total}, Variance: {var_dice_total}')
    print(f'Total: Average HD95: {avg_hd95_total}, Variance: {var_hd95_total}')
    print(f'Total: Average MAE: {avg_mae_total}, Variance: {var_mae_total}')
    print(f'Total: Average DD95: {avg_dd95_total}, Variance: {var_dd95_total}')
def show_pred_dose(imgs):
    print(imgs.shape)
    b, h, w = imgs.shape
    imgs=imgs.cpu().detach().numpy()
    # min_value=np.min(imgs)
    # max_value=np.max(imgs)
    # imgs=(imgs - min_value) / (max_value - min_value)   #归一化

    # 设置画布大小
    plt.figure(figsize=(15, 5))

    # 遍历并展示每张照片
    for i in range(b):
        plt.subplot(1, b, i + 1)
        plt.imshow(imgs[i, ...], cmap='gray')
        if i ==0:
            plt.title(f'label')
        else:
            plt.title(f'step: {i*100}')
        plt.axis('off')

    plt.show()



