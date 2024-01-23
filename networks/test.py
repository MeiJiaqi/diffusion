import torch
import datasets_npz
import unet
from torch import nn
import time

if __name__ == '__main__':
    model1=unet.UNet(in_channel=5,out_channel=1,inner_channel=32,norm_groups=16,channel_mults=(1,2,4,8,16),attn_res=[],
                    res_blocks=1,dropout=0,with_time_emb=False,with_feature_emb=False,image_size=160)

    model2 = unet.UNet(in_channel=1, out_channel=1, inner_channel=32, norm_groups=16, channel_mults=(1, 2, 4, 8, 16),
                      attn_res=[],
                      res_blocks=1, dropout=0, with_time_emb=True, with_feature_emb=True, image_size=160)
    print("Num params: ", sum(p.numel() for p in model1.parameters()))
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    timesteps=1000
    batch_size=20


    inputs=torch.randn([batch_size,5,160,160]).to(device)
    print(device)
    model1.to(device)
    model2.to(device)
    tra = datasets_npz.make_datasetS()
    for ii, batch_sample in enumerate(tra):
        inputs, target, c, name = batch_sample['inputs'], batch_sample['ptv'], batch_sample['channel'], batch_sample[
            'name']
        print(inputs.shape)
        print(target.shape)
    # t1=time.time()
    # output,features=model1(inputs)
    # t = torch.randint(0, timesteps, (batch_size,), device=device).long()
    # print(model2(output,features,t).shape)
    # t2=time.time()
    # print('程序运行时间:%s毫秒' % ((t2 - t1) * 1000))
