#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:46:30 2024

@author: g260218
"""

import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from datasets import myDataset
from funs_prepost import nc_load_vars,var_denormalize,plot_distri
# import pandas as pd

import sys
import importlib
path_par = "../"  # the path of parameter files, also used for output path
sys.path.append(path_par)  # add path of parameter files to system temporarily for module import
mod_name= sys.argv[1]          #'par01_md0' # sys.argv[1]
mod_para=importlib.import_module(mod_name)
# from mod_para import * 
kmask = 1

if __name__ == '__main__':
    
    opt = mod_para.opt
    suf = mod_name # mod_para.suf+mod_name
    files_lr = mod_para.files_lr
    files_hr = mod_para.files_hr
    indt_lr = mod_para.indt_lr # 
    indt_hr = mod_para.indt_hr # 

    rtra = mod_para.rtra
    var_lr = mod_para.var_lr
    var_hr = mod_para.var_hr
    ivar_hr = mod_para.ivar_hr
    ivar_lr = mod_para.ivar_lr
    varm_hr = mod_para.varm_hr
    varm_lr = mod_para.varm_lr
    nchl_i = len(var_lr)
    nchl_o = len(var_hr)
    
    if isinstance(rtra, (int, float)): # if only input one number, no validation
        rtra = [rtra,0]
    
    # create nested list of files and indt
    if len(files_hr[0])!=nchl_o:
        files_hr = [[ele for _ in range(nchl_o)] for ele in files_hr]
        indt_hr = [[ele for _ in range(nchl_o)] for ele in indt_hr]
    if len(files_lr[0])!=nchl_i:
        files_lr = [[ele for _ in range(nchl_i)] for ele in files_lr]
        indt_lr = [[ele for _ in range(nchl_i)] for ele in indt_lr]
        
    if hasattr(mod_para, 'll_lr'):
        ll_lr = mod_para.ll_lr # user domain latitude
    else:
        ll_lr = [None]*2
    if hasattr(mod_para, 'll_hr'):
        ll_hr = mod_para.ll_hr # user domain longitude
    else:
        ll_hr = [None]*2
    if hasattr(mod_para, 'kintp'):
        kintp = mod_para.kintp # 1, griddata, 2 RBFInterpolator
    else:
        kintp = [0,0] # no interpolation for lr and hr
    
    nrep = mod_para.nrep
    rep = list(range(0,nrep))
    # rep = [0]
    # # suf = '_res' + str(opt.residual_blocks) + '_max_suv' # + '_nb' + str(opt.batch_size)
    # print(f'parname: {mod_name}')
    # print('--------------------------------')

    nchl = nchl_o
    
    hr_shape = (opt.hr_height, opt.hr_width)

    train_set = myDataset(files_lr,files_hr,indt_lr,indt_hr,hr_shape, opt.up_factor,
                          mode='train',rtra = rtra,var_lr=var_lr,var_hr=var_hr,
                          varm_lr=varm_lr,varm_hr=varm_hr,ll_lr=ll_lr,ll_hr=ll_hr,kintp=kintp)
    
    test_set = myDataset(files_lr,files_hr,indt_lr,indt_hr,hr_shape, opt.up_factor,
                          mode='test',rtra = rtra,var_lr=var_lr,var_hr=var_hr,
                          varm_lr=varm_lr,varm_hr=varm_hr,ll_lr=ll_lr,ll_hr=ll_hr,kintp=kintp)
    
    data_train = DataLoader(
        train_set,
        batch_size=opt.batch_size,
        num_workers=opt.n_cpu,
    )

    data_test = DataLoader(
        test_set,
        batch_size=opt.batch_size, 
        num_workers=opt.n_cpu,
    ) 

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    
    
    unit_suv = ['ssh (m)','u (m/s)','v (m/s)','uw (m/s)','vw (m/s)','swh (m)','pwp (s)','swh_ww (m)',]

    opath_st = path_par+'stat_' + suf +'/'
    os.makedirs(opath_st, exist_ok=True)
        
    # get all training data 
    lr_all_train = []
    hr_all_train = []
    for i, dat in enumerate(data_train):                
        dat_lr = Variable(dat["lr"].type(Tensor))
        dat_hr = Variable(dat["hr"].type(Tensor))
        hr_norm0 = var_denormalize(dat_hr.detach().cpu().numpy(),varm_hr)
        lr_norm0 = var_denormalize(dat_lr.detach().cpu().numpy(),varm_lr)
        
        # hr_norm0[mask] = np.nan
        hr_all_train.append(hr_norm0)
        # lr_norm0[mask_lr] = np.nan
        lr_all_train.append(lr_norm0)

    hr_all_train = np.concatenate(hr_all_train, axis=0)
    lr_all_train = np.concatenate(lr_all_train, axis=0)
    
    # get all test data 
    lr_all_test = []
    hr_all_test = []
    for i, dat in enumerate(data_test):                
        dat_lr = Variable(dat["lr"].type(Tensor))
        dat_hr = Variable(dat["hr"].type(Tensor))
        hr_norm0 = var_denormalize(dat_hr.detach().cpu().numpy(),varm_hr)
        lr_norm0 = var_denormalize(dat_lr.detach().cpu().numpy(),varm_lr) 
        
        # hr_norm0[mask] = np.nan
        hr_all_test.append(hr_norm0)
        # lr_norm0[mask_lr] = np.nan
        lr_all_test.append(lr_norm0)

    hr_all_test = np.concatenate(hr_all_test, axis=0)  #(Nt,C,H,W))
    lr_all_test = np.concatenate(lr_all_test, axis=0)

    # estimate errors excluding masked area. 
    Nsample_train= len(hr_all_train)
    Nsample_test = len(hr_all_test)

    if kmask == 1: 
        for it in range(Nsample_test):
            for ichl in range(nchl):
                nc_f = test_set.files_hr[it][ichl]
                indt = test_set.indt_hr[it][ichl]  # the time index in a ncfile
                mask = nc_load_vars(nc_f,var_hr[ichl],[indt],ll_hr[0],ll_hr[1])[4] # mask at 1 time in a batch
                hr_all_test[it,mask] = np.nan
                nc_f = test_set.files_lr[it][ichl]
                indt = test_set.indt_lr[it][ichl]  # the time index in a ncfile
                mask = nc_load_vars(nc_f,var_lr[ichl],[indt],ll_lr[0],ll_lr[1])[4] # mask at 1 time in a batch
                lr_all_test[it,mask] = np.nan
        for it in range(Nsample_train):
            for ichl in range(nchl):
                nc_f = train_set.files_hr[it][ichl]
                indt = train_set.indt_hr[it][ichl]  # the time index in a ncfile
                mask = nc_load_vars(nc_f,var_hr[ichl],[indt],ll_hr[0],ll_hr[1])[4] # mask at 1 time in a batch
                hr_all_train[it,mask] = np.nan
                nc_f = train_set.files_lr[it][ichl]
                indt = train_set.indt_lr[it][ichl]  # the time index in a ncfile
                mask = nc_load_vars(nc_f,var_lr[ichl],[indt],ll_lr[0],ll_lr[1])[4] # mask at 1 time in a batch
                lr_all_train[it,mask] = np.nan

    varm_hr_train = np.zeros((nchl,2))  # store max/min of dataset
    varm_hr_test = np.zeros((nchl,2))

    # show distribution of training/testing dataset
    for i in range(nchl):
        ichl = ivar_hr[i]

        var1 = hr_all_train[:,i,:,:].flatten()
        var2 = lr_all_train[:,i,:,:].flatten()
        var3 = hr_all_test[:,i,:,:].flatten()
        var4 = lr_all_test[:,i,:,:].flatten()
        
        max_hr_train,min_hr_train = np.nanmax(var1), np.nanmin(var1)
        max_lr_train,min_lr_train = np.nanmax(var2), np.nanmin(var2)
        max_hr_test,min_hr_test = np.nanmax(var3), np.nanmin(var3)
        max_lr_test,min_lr_test = np.nanmax(var4), np.nanmin(var4)
        
        varm_hr_train[i,:] = np.array([max_hr_train,min_hr_train])
        varm_hr_test[i,:] = np.array([max_hr_test,min_hr_test])
        
        ofname = 'c%d'%ichl+'_maxmin'+'.csv'
        combined_ind= np.array([[max_hr_train,min_hr_train,max_hr_test,min_hr_test],
                       [max_lr_train,min_lr_train,max_lr_test,min_lr_test]])
        np.savetxt(opath_st + ofname, combined_ind,fmt='%f,') # ,delimiter=","
        
        unit_var = unit_suv[ichl]
        # plot distribution of reconstructed vs target, all data, histogram
        axlab = (unit_var,'Frequency','')
        leg = ['hr_train'+'('+'%4.2f,%4.2f'%(min_hr_train,max_hr_train)+')',
               'lr_train'+'('+'%4.2f,%4.2f'%(min_lr_train,max_lr_train)+')',
               'hr_test'+'('+'%4.2f,%4.2f'%(min_hr_test,max_hr_test)+')',
               'lr_test'+'('+'%4.2f,%4.2f'%(min_lr_test,max_lr_test)+')'] #,'nearest'
        var = [var1[~np.isnan(var1)],var2[~np.isnan(var2)],
               var3[~np.isnan(var3)],var4[~np.isnan(var4)],
               ]
        figname = opath_st+"c%d" % (ichl) +'_dist_train_test'+'.png'
        plot_distri(var,figname,bins=20,axlab=axlab,leg=leg,
                       figsize=(10, 5), fontsize=16,capt='')

    
    for i in range(nchl):
        vmax = varm_hr_train[i,0]
        vmin = varm_hr_train[i,1]
        temp = hr_all_train[:,i,:,:]
        hr_all_train[:,i,:,:] = (temp - vmin)/(vmax-vmin) # convert to [0,1]
        vmax = varm_hr_test[i,0]
        vmin = varm_hr_test[i,1]
        temp = hr_all_test[:,i,:,:]
        hr_all_test[:,i,:,:] = (temp - vmin)/(vmax-vmin) # convert to [0,1]
    
    # estimate spatial gradient in the x and y direction for hr data
    hr_train_grdx = np.gradient(hr_all_train,axis=3)
    hr_train_grdy = np.gradient(hr_all_train,axis=2)
    hr_test_grdx = np.gradient(hr_all_test,axis=3)
    hr_test_grdy = np.gradient(hr_all_test,axis=2)
    
    # show distribution of gradients of training/testing dataset (fail on local node, may due to memory)
    for i in range(nchl):
        ichl = ivar_hr[i]
        unit_var = unit_suv[ichl]
    
        # plot distribution of reconstructed vs target, all data, histogram
        axlab = ('grad '+ unit_var,'Frequency','')
        leg = ['hr_train_gdx','hr_train_gdy','hr_test_gdx','hr_test_gdy'] #,'nearest'
        var1 = hr_train_grdx[:,i,:,:].flatten()
        var2 = hr_train_grdy[:,i,:,:].flatten()
        var3 = hr_test_grdx[:,i,:,:].flatten()
        var4 = hr_test_grdy[:,i,:,:].flatten()
        var = [var1[~np.isnan(var1)],var2[~np.isnan(var2)],
               var3[~np.isnan(var3)],var4[~np.isnan(var4)],
               ]
        figname = opath_st+"c%d" % (ichl) +'_dist_train_test_gd'+'.png'
        plot_distri(var,figname,bins=40,axlab=axlab,leg=leg,
                       figsize=(10, 5), fontsize=16,capt='')