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
import SimpleITK  as sit


import datasets_npz
import utils
from networks.unet import UNet
from networks.diffusion import GaussianDiffusion

net1 = UNet(in_channel=1, out_channel=1, inner_channel=32, norm_groups=16, channel_mults=(1, 2, 4, 8, 16),
                   attn_res=[],
                   res_blocks=1, dropout=0, with_time_emb=False, with_feature_emb=False, image_size=160)

net2 = UNet(in_channel=1, out_channel=1, inner_channel=32, norm_groups=16, channel_mults=(1, 2, 4, 8, 16),
                   attn_res=[],
                   res_blocks=1, dropout=0, with_time_emb=True, with_feature_emb=True, image_size=160)
print("Num params: ", sum(p.numel() for p in net1.parameters()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
iter = 0
loss1= 0
loss2= 0
epoch=0

# 使用DataParallel包装模型
net1 = nn.DataParallel(net1)
net2 = nn.DataParallel(net2)
# net1, net2, iter, epoch = utils.load_model(net1, net2, 301)

net1=net1.to(device)
net2=net2.to(device)
epochs = 1500
timesteps=1000
linear_start=1e-4
linear_end=1e-2
learning_rate = 1e-4
batch_size=300
schedule_opt={'schedule':'linear','n_timestep':timesteps,'linear_start':linear_start,'linear_end':linear_end}
gaussian_diffusion=GaussianDiffusion(model=net2,image_size=160,channels=1,loss_type='l2',conditional=True,schedule_opt=schedule_opt,device=device)
optimizer1=torch.optim.Adam(net1.parameters(),lr=learning_rate)
optimizer2=torch.optim.Adam(net2.parameters(),lr=learning_rate)
criterion = torch.nn.BCEWithLogitsLoss()

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
    bach=1
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
            pred_rd = gaussian_diffusion.p_sample_loop(ct, False, feature)
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


if __name__ == '__main__':

    #val_model()
    # utils.save_model(net1, 0, iter, 1)
    # utils.save_model(net2, 0, iter, 2)

    for epo in range(epoch,epochs):
        tra=datasets_npz.make_datasetS()
        print(f'epoch:{epo}')

        for ii, batch_sample in enumerate(tra):
            print(ii)
            name=batch_sample['name']
            print(name)
            dataloader = DataLoader(dataset=datasets_npz.MyDataset(batch_sample), batch_size=batch_size, shuffle=False, drop_last=False)
            for i, (ct, ptv,rd,oars) in enumerate(tqdm(dataloader)):
                # optimizer1.zero_grad()
                # optimizer2.zero_grad()
                ct=ct.to(device)
                print(oars.shape)
                inputs = ct.to(device)
                ptv = ptv.to(device)
                rd=rd.to(device)
                print(ct.shape)
                #utils.save_mha(ct,name,'D:/workfile/dlfile/2024mic/diffusion/sample')
                sys.exit(0)
                print("Minimum value in ct:", torch.min(ct).item())
                print("Maximum value in ct:", torch.max(ct).item())
                # #
                # print("Minimum value in output:", torch.min(output).item())
                # print("Maximum value in output:", torch.max(output).item())
                if(torch.max(ptv).item()>0):
                    inputs = ct.to(device)
                    output, feature = net1(inputs)
                    output=((output + 1) / 2).squeeze().cpu().detach().numpy()
                    print("Minimum value in output:", np.min(output).item())
                    print("Maximum value in output:", np.max(output).item())
                    print(output.shape)
                    rd=((rd+ 1) / 2).squeeze().cpu().detach().numpy()
                    plt.figure(figsize=(15, 5))

                    plt.subplot(1, 2, 1)
                    plt.imshow(output, cmap='gray')
                    plt.title('pred dose')
                    plt.subplot(1, 2, 2)
                    plt.imshow(rd, cmap='gray')
                    plt.title('dose')
                    plt.show()

                    # pred_rd=gaussian_diffusion.p_sample_loop(rd,True,feature)
                    # print("Minimum value in pred_rd:", torch.min(pred_rd[-1]).item())
                    # print("Maximum value in pred_rd:", torch.max(pred_rd[-1]).item())
                    # pred_rd=(pred_rd - min_value) / (max_value - min_value)
                    # #
                    # print("Minimum value in pred_rd:", torch.min(pred_rd).item())
                    # print("Maximum value in pred_rd:", torch.max(pred_rd).item())
                    # print(pred_rd.shape)
                    # utils.test_metric(torch.sigmoid(output), ptv,pred_rd[-1], rd,ct)
                    # show_pred_dose(pred_rd.squeeze())




