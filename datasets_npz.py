import os.path
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torch
import numpy as np
import SimpleITK as sit

from PIL import Image
import torch.nn.functional as F

# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir,phase):
    paths = os.path.join(dir,phase)
    #assert os.path.isdir(paths), '%s is not a valid directory' % dir
    rd = ""
    origin = ""
    rs = ""
    for root,files, _ in sorted(os.walk(paths)):
        for file in files:
            if "origin" in file:
                origin = os.path.join(root, file)
            elif "rd" in file:
                rd = os.path.join(root, file)
            # elif "rs" in file:
            elif "rs" in file:
                rs = os.path.join(root, file)
    return origin,rs,rd

def make_files(origin,rs,rd):
    names = []
    images = {}
    allRs = []
    for root,_,fnames in sorted(os.walk(rs)):
        for fname in fnames:
            pathrs =os.path.join(root,fname)
            allRs.append(pathrs)
    for root,_,fnames in sorted(os.walk(origin)):
        for fname in fnames:
            name = fname.split('_')[0]
            opath = os.path.join(root,fname)
            #names = os.path.join(root,name)
            names.append(name)
            #if name not in images.keys():
            images[name] = []
            images[name].append(opath)
            for item in allRs:
                #print(item)
                t = item.find(name+'_')
                if t > 0:
                    images[name].append(item)
            rdname = str(name) + '_rd.mha'
            pathrd = os.path.join(rd,rdname)
            images[name].append(pathrd)
    names_ = [] #为什么不用name而增加一个name_用于返回呢 我认为只是为了增加下面 “ == 7:” 这一行方便，其实if里面continue并且最后return name也行
    images_ = {}
    # if len(names)>15:
    #     names = names[18:]
    # else:
    #     names = names[:8]
    # names=names[:20]
    for i in range(len(names)):
        if len(images[names[i]]) == 7:  #由于rd和origin一样 但是有病人的危机器官信息可能少 所以去掉问题数据
            names_.append(names[i])
            images_[names[i]] = images[names[i]]

    # names_ = names_[:18]

    return names_,images_



class TrainDataset(Dataset):
    def __init__(self,dir):
        super(TrainDataset, self).__init__()
        self.dir = dir
        self.transform = torch.from_numpy

        self.files =[]
        self.names=[]
        for root,_,fnames in sorted(os.walk(dir)):
            for fname in fnames:
                self.files.append(os.path.join(self.dir,fname))
                self.names.append(fname.split(".")[0])

        print("patients: ",len(self.names))


    def __getitem__(self, index):
        file = np.load(self.files[index])

        inputs = self.transform(file["arr_0"]).type(torch.FloatTensor) #torch.Size([ 185, 6, 512, 512])

        rd = self.transform(file["arr_1"]).type(torch.FloatTensor) #torch.Size([ 185, 1, 512, 512])
        # inputs = F.interpolate(inputs,size=[128,128],mode='bilinear')
        # rd= F.interpolate(rd, size=[128, 128], mode='bilinear')
        c = len(rd)
        # inputs = F.interpolate(inputs,size=[128,128],mode='nearest')
        # rd = F.interpolate(rd,size=[128,128],mode='bilinear')
        # print(rd.shape)




        #return {'original': orig,'rd':rd,'B':Bladder_numpy,'FHL':FemoralHeadL_numpy,'FHR':FemoralHeadR_numpy,'P':PCTV_numpy,'S':Smallintestine_numpy,'channel':c}
        #归一化
        inputs=inputs*2-1
        ptv=inputs[:,4,:,:]
        ptv = ptv.unsqueeze(1)
        ct = inputs[:, 0, :, :]
        ct=ct.unsqueeze(1)
        inputs = torch.cat((inputs[:, :3, :, :], inputs[:, 4:, :, :]), dim=1)
        oars = torch.cat((inputs[:, 0:3, :, :], inputs[:, 4:, :, :]), dim=1)
        return {'inputs': inputs, 'rd': rd*2-1, 'channel': c, "name": self.names[index],'ptv':(ptv+1)/2,'ct':ct,'oars':(oars+1)/2}
    def __len__(self):
        return len(self.names)

   # def name(self):
     #   return str(self.kind)+'Dataset'

class MyDataset(Dataset):
    def __init__(self,person_sample):
        super(MyDataset, self).__init__()
        self.person_sample = person_sample
        self.channal=person_sample['channel']
    def __getitem__(self, index):

        ct = self.person_sample['ct'].squeeze(dim=0)         #torch.Size([ 185, 1, 160, 160])
        ptv=self.person_sample['ptv'].squeeze(dim=0)              #torch.Size([ 185, 1, 160, 160])
        rd=self.person_sample['rd'].squeeze(dim=0)
        oars = self.person_sample['oars'].squeeze(dim=0)
        return ct[index,:,:,],ptv[index,:,:,],rd[index,:,:,],oars[index,:,:,]

    def __len__(self):
        return self.channal




def make_datasetS():
    dir = r'/data/shuangjun.du/diffusion/data/rectum333_npz/train'   #杜双军服务器
    # dir = r'/home/scusw1/mic/pycharm_project_433/data/rectum333_npz/train'   #实验室服务器
    #dir = r'D:\workfile\dlfile\数据集\rectum333_npz\test'
    batch_size = 1
    # Syn_train = TrainDataset(dir,"keshihuaForComparison_DVH")
    Syn_train = TrainDataset(dir)

    SynData_train = DataLoader(dataset=(Syn_train),batch_size=batch_size,shuffle=False,drop_last=True,num_workers=0)

    return SynData_train

def make_Valdataset():
    dir = r'/data/shuangjun.du/diffusion/data/rectum333_npz/test'   #杜双军服务器
    #dir = r'D:\workfile\dlfile\数据集\rectum333_npz\test'
    # dir = r'/home/scusw1/mic/pycharm_project_433/data/rectum333_npz/val'  # 实验室服务器

    batch_size = 1
    # Syn_train = TrainDataset(dir,"keshihuaForComparison_DVH")
    Syn_train = TrainDataset(dir)

    SynData_train = DataLoader(dataset=(Syn_train),batch_size=batch_size,shuffle=True,drop_last=True,num_workers=0)

    return SynData_train

# if __name__ == "__main__":
#     tra = make_datasetS()
#     channels = []
#     batch_size = 50
#     iter=0
#



