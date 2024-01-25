import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torchvision import transforms


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes



def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 0, 0
    else:
        return 1, 0



def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

def save_mha(data,name,savepath,voxel_spacing=(3, 3, 3)):

    os.makedirs(savepath, exist_ok=True)
    data_array=data

    # Create a SimpleITK image from the numpy array
    image = sitk.GetImageFromArray(data_array)
    image.SetSpacing(voxel_spacing)
    # Save the image with the specified name and path
    sitk.WriteImage(image, f"{savepath}/{name}.mha")
    print("mha保存成功")

def testimg():
    # 读取图像
    image_path = 'leijinqiong_rd_99.jpg'
    # ct = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # 检查图像是否成功读取
    # if ct is not None:
    #     # 将图像转换为 PyTorch 张量
    #     transform = transforms.ToTensor()
    #     ct = transform(ct).unsqueeze(0)
    # else:
    #     print("Error: Unable to read the image.")
    # ct=ct.to(device)
    # # 选择要展示的步骤
    # display_steps = [0, 100, 200, 300, 400, 500,600,700,800,900]
    # noisy_images = []
    #
    # for t in range(timesteps):
    #     # 生成加噪图像
    #
    #     noisy_image = gaussian_diffusion.q_sample(x_start=ct,
    #                                               t=torch.tensor([t], dtype=torch.long, device='cuda'))
    #
    #     # 将 PyTorch 张量转换为 NumPy 数组，并将通道调整到正确的范围（0-1）
    #     if t in display_steps:
    #         noisy_image_np = noisy_image.squeeze(dim=0).cpu().numpy().transpose(1, 2, 0)
    #         noisy_image_np = np.clip(noisy_image_np, 0, 1)
    #         # 对于灰度图，可能需要处理通道数
    #         if len(noisy_image_np.shape) == 2:
    #             noisy_image_np = np.expand_dims(noisy_image_np, axis=-1)  # 添加通道维度
    #
    #         noisy_image_np = np.clip(noisy_image_np, 0, 1)
    #         # 保存或显示加噪图像
    #         noisy_images.append(noisy_image_np)
    #         plt.imshow(noisy_image_np, cmap='gray')
    #         plt.title(f'Step {t}')
    #         plt.show()
    #
    # # 保存图片
    # for i, img in enumerate(noisy_images):
    #     img_path = f'noisy_image_{display_steps[i]}.png'
    #     cv2.imwrite(img_path, (img * 255).astype(np.uint8))
    #     print("保存成功")
def save_model(model,epoch,iter, index, best=False, max_saved=8):
    base_path = '/data/shuangjun.du/diffusion/checkpoints/'  # 杜双军服务器
    # base_path = '/home/scusw1/mic/pycharm_project_433/'  # 川大实验室服务器
    # base_path = 'D:/workfile/dlfile/2024mic/diffusion/'
    save_dir = os.path.join(base_path, 'v9')
    os.makedirs(save_dir, exist_ok=True)
    if best == True :
        file_path = os.path.join(save_dir, f'Unet{index}_Best_9999.pth')
        torch.save({
            'iter': iter,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, file_path)
        print("Best Model saved successfully.")
        return
    # 构建文件路径
    file_path = os.path.join(save_dir, f'Unet{index}_epoch_{epoch}.pth')
    # 保存模型和优化器状态字典
    torch.save({
        'iter': iter,
        'epoch':epoch,
        'model_state_dict': model.state_dict(),
    }, file_path)

    print("Model saved successfully.")

    # 删除超过最大保存数量的最旧模型文件
    saved_files = sorted(os.listdir(save_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    while len(saved_files) > max_saved:
        file_to_delete = os.path.join(save_dir, saved_files.pop(0))
        os.remove(file_to_delete)
        print(f"Deleted old checkpoint: {file_to_delete}")

def load_model(model1,model2, epochnum):

    net1path = f'/data/shuangjun.du/diffusion/checkpoint/v8/Unet1_epoch_500.pth'    #杜双军服务器地址
    net2path = f'/data/shuangjun.du/diffusion/checkpoint/v8/Unet2_epoch_{epochnum}.pth'

    #net1path = f'/home/scusw1/mic/pycharm_project_433/checkpoints/v8/Unet1_epoch_{epochnum}.pth'    #川大服务器地址
    # net2path = f'/home/scusw1/mic/pycharm_project_433/checkpoints/v8/Unet2_epoch_{epochnum}.pth'

    # net1path = f'D:\\workfile\\dlfile\\2024mic\\diffusion\\checkpoint\\Unet1_epoch_560.pth'
    # net2path = f'D:\\workfile\\dlfile\\2024mic\\diffusion\\checkpoint\\net2_epoch_{epochnum}.pth'

    checkpoint1 = torch.load(net1path)
    checkpoint2 = torch.load(net2path)
    # 加载模型和优化器的状态字典
    model1.load_state_dict(checkpoint1['model_state_dict'])
    model2.load_state_dict(checkpoint2['model_state_dict'])
    # 获取迭代次数和损失
    iter_num = checkpoint2['iter']
    epoch_num = checkpoint2['epoch']

    print(f"Model loaded successfully from iteration {iter_num}")

    return model1, model2, iter_num,epoch_num


def dose_matrix_metrics(pred, gt, ptv):
    """
    评估剂量预测的各种指标。

    参数:
    - pred: 预测的剂量矩阵
    - gt: 实际的剂量矩阵
    - ptv: PTV（靶区）的掩模

    返回:
    - mse: 均方误差
    - rmse: 均方根误差
    - mae: 平均绝对误差
    - ddi: 剂量偏差指数
    - ci: 一致性指数
    - dci: 剂量覆盖指数
    - dd95: 95%处的剂量差异
    - vd: 体积百分比剂量
    """
    # 使用 PTV 掩模将剂量矩阵展平
    pred_flat = pred[ptv]
    gt_flat = gt[ptv]
    if len(gt_flat) > 0 and len(pred_flat) > 0:
        # 计算指标
        mse = mean_squared_error(gt_flat, pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(gt_flat, pred_flat)

        # 剂量偏差指数 (DDI)
        ddi = np.abs(gt_flat.mean() - pred_flat.mean()) / gt_flat.mean()

        # 一致性指数 (CI)
        ci = np.sum(np.minimum(gt[ptv], pred[ptv])) / np.sum(gt[ptv])

        # 剂量覆盖指数 (DCI)
        dci = np.sum(np.minimum(gt[ptv], pred[ptv])) / np.sum(pred[ptv])

        # 95%处的剂量差异 (DD95)
        dd95 = np.percentile(np.abs(gt_flat - pred_flat), 95)

        # 体积百分比剂量 (VD) - 例如，95%处的剂量
        vd = np.sum(pred_flat >= 0.95 * gt_flat) / len(pred_flat)

        return mse, rmse, mae, ddi, ci, dci, dd95, vd
    else:
        print("Error: Empty arrays.")



def test_metric(pred_ptv,ptv,pred_dose,dose,ct):
    b, _, h, w = ct.shape
    dice_list = []
    hd95_list = []
    mae_list = []
    dd95_list = []

    for i in range(b):
        current_pred_ptv = pred_ptv[i, ...].squeeze().cpu().detach().numpy()
        current_ptv = ptv[i, ...].squeeze().cpu().numpy()
        # current_ct = ct[i, ...].squeeze().cpu().numpy()
        # current_pred_dose = ((pred_dose[i, ...] + 1) / 2).squeeze().cpu().detach().numpy()
        # current_dose = ((dose[i, ...] + 1) / 2).squeeze().cpu().numpy()
        # print(current_ct.shape)
        # print(current_pred_dose.shape)
        # print(current_pred_ptv.shape)
        # print(current_ptv.shape)
        current_mae = 0
        current_dd95 = 0

        # if np.max(current_ptv) > 0:
        #     mse, rmse, current_mae, ddi, ci, dci, current_dd95, vd = dose_matrix_metrics(
        #         current_pred_dose, current_dose, (current_ptv > 0.5))
        #     print(f'Mean Squared Error (MSE): {mse:.4f}')
        #     print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
        #     print(f'Mean Absolute Error (MAE): {current_mae:.4f}')
        #     print(f'Dose Deviation Index (DDI): {ddi:.4f}')
        #     print(f'Conformity Index (CI): {ci:.4f}')
        #     print(f'Dose Conformity Index (DCI): {dci:.4f}')
        #     print(f'Dose Difference at 95% (DD95): {current_dd95:.4f}')
        #     print(f'Volume at Dose (VD): {vd:.4f}')

        current_pred_ptv = (current_pred_ptv > 0.5).astype(float)
        current_ptv = (current_ptv > 0.5).astype(float)
        current_dice, current_hd95 = calculate_metric_percase(current_pred_ptv, current_ptv)

        #print(f'dice:{current_dice},hd95:{current_hd95}')

        dice_list.append(current_dice)
        hd95_list.append(current_hd95)
        # mae_list.append(current_mae)
        # dd95_list.append(current_dd95)
        # # 当 dice 大于 0.9 时展示预测结果
        # if current_dice > 0.8 and current_dice < 1 :
        #     plt.figure(figsize=(15, 5))
        #
        #     plt.subplot(1, 5, 1)
        #     plt.imshow(current_ct, cmap='gray')
        #     plt.title('CT Image')
        #
        #     plt.subplot(1, 5, 2)
        #     plt.imshow(current_pred_ptv, cmap='gray')
        #     plt.title('Predicted PTV')
        #
        #     plt.subplot(1, 5, 3)
        #     plt.imshow(current_ptv, cmap='gray')
        #     plt.title('Ground Truth PTV')
        #
        #     plt.subplot(1, 5, 4)
        #     plt.imshow(current_pred_dose, cmap='gray')  # 使用 'jet' colormap
        #     plt.title('Predicted Dose')
        #
        #     plt.subplot(1, 5, 5)
        #     plt.imshow(current_dose, cmap='gray')  # 使用 'jet' colormap
        #     plt.title('Ground Truth Dose')
        #
        #     plt.show()
    return dice_list, hd95_list, mae_list, dd95_list
    # print("Minimum value in rd:", np.min(dose))
    # print("Maximum value in rd:", np.max(dose))
    # print("Minimum value in pred_rd:", np.min(pred_dose))
    # print("Maximum value in pred_rd:", np.max(pred_dose))
    # print("Minimum value in ptv:", np.min(ptv))
    # print("Maximum value in ptv:", np.max(ptv))
    # print("Minimum value in pred_ptv:", np.min(pred_ptv))
    # print("Maximum value in pred_ptv:", np.max(pred_ptv))

