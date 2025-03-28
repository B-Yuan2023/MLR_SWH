# import glob
# import random
# import os
import numpy as np

import torch
from torch.utils.data import Dataset
# from PIL import Image
# import torchvision.transforms as transforms
from funs_prepost import nc_normalize_vars


# rtra = 0.75 # ratio of training dataset
# rval = 0.1 # ratio of validation dataset
# ntpd = 24 # number of time steps in an nc file

# def lr_transform(hr_height, hr_width, up_factor):
#     return transforms.Compose(
#         [
#             transforms.Resize((hr_height // up_factor, hr_width // up_factor), Image.BICUBIC), # transforms.InterpolationMode.BICUBIC
#             transforms.ToTensor(),
#             # transforms.Normalize(mean, std),
#         ]
#     )

# def hr_transform(hr_height,hr_width):
#     return transforms.Compose(
#         [
#             transforms.Resize((hr_height, hr_width), Image.BICUBIC), # 
#             transforms.ToTensor(),
#             # transforms.Normalize(mean, std),
#         ]
    # )


def my_loss(output,target,nlm=2,wtlim=None):  
    # output, target: dimensionless tensor[N,C,H,W]
    # nlm: key for loss function, originally used as norm order
    # wtlim[1,2]: weight upper and lower limit dimensionless
    if nlm == 1:    # norm1
        loss = torch.norm(output-target,1)
    elif nlm == 2:      # norm2
        loss = torch.norm(output-target,2)
    elif nlm == 3:      # norm3
        loss = torch.norm(output-target,3)
    elif nlm == 4:      # mae
        loss = torch.mean(torch.abs(output - target))
    elif nlm == 5:      # mse
        loss = torch.mean((output - target)**2)
    elif nlm == 6:      # weighted using target directly 
        wt = torch.abs(target)
        loss = torch.mean(wt*(output - target)**2)
    elif nlm == 7:      # weighted e^abs(target)
        wt = torch.exp(torch.abs(target))
        loss = torch.mean(wt*(output - target)**2)  
    elif nlm == 8:      # weighted using limit
        wt = torch.abs(target)
        if wtlim is None:
            wtlim = [0.05, 0.95]
        wt[torch.abs(target)<wtlim[0]] = wtlim[0]
        wt[torch.abs(target)>wtlim[1]] = wtlim[1]
        loss = torch.mean(wt*(output - target)**2)               
    elif nlm == 9:      # weighted 1+abs(target)
        wt = torch.ones_like(target)+torch.abs(target)
        loss = torch.mean(wt*(output - target)**2)        
    return loss

class myDataset(Dataset):
    def __init__(self, files_lr,files_hr,indt_lr,indt_hr,hr_shape,up_factor=4,
                 mode='train', rtra=0.75,var_lr=['elevation'],var_hr=['elevation'],
                 varm_lr=None,varm_hr=None,ll_lr=None,ll_hr=None,kintp=[0,0]):
        # files_lr, files_lr: nc files of low, high resolution data
        # ind_lr,ind_hr: list of index of used time step in nc files
        # hr_shape (H,W)
        # rtra: ratio of training set in total dataset
        # var_lr,var_hr: variable names in nc files
        # varm_lr,varm_hr: var range for normalization
        
        hr_height, hr_width = hr_shape
        self.mode = mode
        
        self.files_lr = files_lr
        self.files_hr = files_hr
        self.indt_lr = indt_lr  # time index list, length is total no. of samples
        self.indt_hr = indt_hr  # time index list, length is total no. of samples
        
        self.var_lr = var_lr
        self.var_hr = var_hr
        self.varm_lr = varm_lr
        self.varm_hr = varm_hr
        
        # self.files_lr = sorted(glob.glob(dir_lr + "/*.nc"))
        # self.files_hr = sorted(glob.glob(dir_hr + "/*.nc"))
        # assert len(self.files_lr) == len(self.files_hr),'lr & hr samples not match!'
        nsample= len(self.files_hr)
        
        if isinstance(rtra, (int, float)): # if only input one number, no validation
            rtra = [rtra,0]
        # ind_train = np.arange(15*24,int((nsample-24)*rtra+15*24)) # only for schism
        ind_train = np.arange(0,int(nsample*rtra[0]))
        ind_valid= np.arange(int(nsample*rtra[0]),int(nsample*sum(rtra)))
        ind_test= np.delete(np.arange(0,nsample),np.arange(0,int(nsample*sum(rtra))))
        # self.ind_train = ind_train
        # self.ind_valid = ind_valid
        self.ll_lr = ll_lr
        self.ll_hr = ll_hr
        self.kintp = kintp
        
        self.mode = mode
        if self.mode == "train":
            self.files_hr = [files_hr[i] for i in ind_train] # list can not directly use array as index 
            self.files_lr = [files_lr[i] for i in ind_train]
        elif self.mode == "valid":
            self.files_hr = [files_hr[i] for i in ind_valid] #
            self.files_lr = [files_lr[i] for i in ind_valid]
        elif self.mode == "test":
            self.files_hr = [files_hr[i] for i in ind_test] # getitem does not change self
            self.files_lr = [files_lr[i] for i in ind_test]

    def __getitem__(self, index):
        # from datetime import datetime,timedelta
        
        nc_f = self.files_hr[index]
        indt = self.indt_hr[index] # time index in nc_f
        data = nc_normalize_vars(nc_f,self.var_hr,indt,self.varm_hr,
                                 self.ll_hr[0],self.ll_hr[1],self.kintp[1])  #(H,W,C)
        x = np.transpose(data,(2,0,1)) #(C,H,W)
        dat_hr = torch.from_numpy(x)
        
        # for low resolution
        # schism output is 2 hours earlier than cmems, i.e, t=0 in schism->t=2 h in cmems
        nc_f = self.files_lr[index]
        indt = self.indt_lr[index] # time index in nc_f
        data = nc_normalize_vars(nc_f,self.var_lr,indt,self.varm_lr,
                                 self.ll_lr[0],self.ll_lr[1],self.kintp[0])  #(H,W,C)
        x = np.transpose(data,(2,0,1)) #(C,H,W)
        dat_lr = torch.from_numpy(x)
        
        return {"lr": dat_lr, "hr": dat_hr}

    def __len__(self):
        return len(self.files_hr)
