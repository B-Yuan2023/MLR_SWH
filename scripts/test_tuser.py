#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:46:30 2024

@author: g260218
"""

import os
import numpy as np
import torch

from funs_prepost import (make_list_file_t,nc_load_vars,nc_normalize_vars,
                          var_denormalize)
from datetime import datetime, timedelta # , date

import sys
import importlib
path_par = "../"  # the path of parameter files, also used for output path
sys.path.append(path_par)  # add path of parameter files to system temporarily for module import
mod_name= 'par55e_s40_md0' # 'par55e_md0' # sys.argv[1]
mod_para=importlib.import_module(mod_name)
# from mod_para import * 
kmask = 1

# https://scikit-learn.org/stable/modules/multiclass.html#multioutput-regression
# from sklearn.linear_model import LinearRegression, RidgeCV, Ridge
# from sklearn.neighbors import KNeighborsRegressor #,RadiusNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
# from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
# from sklearn.utils.validation import check_random_state
# from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import joblib
import pickle
# import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    opt = mod_para.opt
    suf = mod_name # mod_para.suf+mod_name
    dir_lr = opt.dir_lr
    dir_hr = opt.dir_hr
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
    
    if hasattr(mod_para, 'kmod'): # ml model: 0 linReg, 1 RidgeCV, 2 Ridge, 3 KNeighbors,4 DT, 5 ExtTree, 6 RandomForest, 7 ExtTrees
        kmod = mod_para.kmod
    else:
        kmod = 0    
    
    nrep = mod_para.nrep
    rep = list(range(0,nrep))
    # rep = [0]
    # # suf = '_res' + str(opt.residual_blocks) + '_max_suv' # + '_nb' + str(opt.batch_size)
    # print(f'parname: {mod_name}')
    # print('--------------------------------')

    nchl = nchl_o
    
    hr_shape = (opt.hr_height, opt.hr_width)
    
    # select a range of data for testing 
    # tlim = [datetime(2021,11,29),datetime(2021,12,1)]
    tlim = [datetime(2021,11,29),datetime(2021,12,2)]    
    # tlim = [datetime(2021,1,26),datetime(2021,1,28)]
    # tlim = [datetime(2021,1,16),datetime(2021,1,18)]
    dt = 3
    
    # tlim = [datetime(2021,11,29,3),datetime(2021,11,30,3)] # just for 2d time series
    # dt = 6
    
    tstr = '_'+tlim[0].strftime('%Y%m%d')+'_'+tlim[1].strftime('%Y%m%d') + '_t%d'%dt
    Nt = int((tlim[1]-tlim[0]).total_seconds()/(dt*3600)) ## total time steps
    tuser0 = [(tlim[0] + timedelta(hours=x*dt)) for x in range(0,Nt)]
    tshift = 0 # in hour
    tuser = [(tlim[0] + timedelta(hours=x*dt)) for x in range(tshift,Nt+tshift)] # time shift for numerical model
    # iday0 = (tlim[0] - datetime(2017,1,2)).days+1 # schism out2d_interp_001.nc corresponds to 2017.1.2
    # iday1 = (tlim[1] - datetime(2017,1,2)).days+1
    # id_test = np.arange(iday0,iday1)
    # files_lr = [dir_lr + '/out2d_interp_' + str(i).zfill(3) + ".nc" for i in id_test]  # schism output
    # files_hr = [dir_hr + '/out2d_interp_' + str(i).zfill(3) + ".nc" for i in id_test]
    
    files_hr, indt_hr = make_list_file_t(dir_hr,tlim,dt,tshift=0,t0_h=0,dt_h=1,ntpf=24)
    files_lr, indt_lr = make_list_file_t(dir_lr,tlim,dt,tshift=0,t0_h=0,dt_h=1,ntpf=24)
    # Nt = len(files_lr)
    
    # create nested list of files and indt, 
    # no. of inner list == no. of channels, no. outer list == no. of samples
    if len(files_hr[0])!=nchl_o:
        files_hr = [[ele for _ in range(nchl_o)] for ele in files_hr]
        indt_hr = [[ele for _ in range(nchl_o)] for ele in indt_hr]
    if len(files_lr[0])!=nchl_i:
        files_lr = [[ele for _ in range(nchl_i)] for ele in files_lr]
        indt_lr = [[ele for _ in range(nchl_i)] for ele in indt_lr]
    
    # get logitude and latitude of data 
    nc_f = files_hr[0][0]
    lon = nc_load_vars(nc_f,var_hr[0],[0],lats=ll_hr[0],lons=ll_hr[1])[1]
    lat = nc_load_vars(nc_f,var_hr[0],[0],lats=ll_hr[0],lons=ll_hr[1])[2]
    
    # # get all hr data. Note: only for a short period, otherwise could be too large
    # hr_all_test = np.zeros((Nt,nchl_o,len(lat),len(lon))) 
    # for i in range(Nt):
    #     for ichl in range(nchl_o): 
    #         nc_f = files_hr[i][ichl]
    #         indt = indt_hr[i][ichl]
    #         dat_hr =  nc_load_vars(nc_f,var_hr[0],[indt],lats=ll_hr[0],lons=ll_hr[1])[3]
    #         # mask =  nc_load_vars(files_hr[i],var_hr[0],[indt_hr[i]],lats=ll_hr[0],lons=ll_hr[1])[4]
    #         # dat_hr[mask] = np.nan
    #         hr_all_test[i,ichl,:,:] = dat_hr
    
    # load hr_all using the way for NN model 
    hr_all = []  # check if after normalize/denormalize, the same as hr_all_test
    lr_all = []
    mask_all = []
    for i in range(Nt):
        nc_f = files_hr[i]
        indt = indt_hr[i]
        data = nc_normalize_vars(nc_f,var_hr,indt,varm_hr,
                                 ll_hr[0],ll_hr[1],kintp[1])  #(H,W,C)
        x = np.transpose(data,(2,0,1)) #(C,H,W)
        hr = torch.from_numpy(x)
        mask = nc_load_vars(nc_f[0],var_hr[0],indt,lats=ll_hr[0],lons=ll_hr[1])[4] #(1,H,W)
        mask = np.squeeze(mask)
        
        nc_f = files_lr[i]
        indt = indt_lr[i] # time index in nc_f
        data = nc_normalize_vars(nc_f,var_lr,indt,varm_lr,
                                  ll_lr[0],ll_lr[1],kintp[0])  #(H,W,C)
        x = np.transpose(data,(2,0,1)) #(C,H,W)
        lr = torch.from_numpy(x)
        # mask_lr = nc_load_vars(nc_f[0],var_lr[0],indt,lats=ll_lr[0],lons=ll_lr[1])[4] #(1,H,W)
        # mask_lr = np.squeeze(mask_lr)
                        
        lr = lr.reshape(1,lr.shape[0],lr.shape[1],lr.shape[2]) # 3d to 4d
        hr = hr.reshape(1,hr.shape[0],hr.shape[1],hr.shape[2]) # 3d to 4d
        lr_norm0 = var_denormalize(lr.detach().numpy(),varm_lr)
        hr_norm0 = var_denormalize(hr.detach().numpy(),varm_hr)
        
        if kmask == 1: 
            hr_norm0[:,:,mask] = np.nan
        hr_all.append(hr_norm0)
        lr_all.append(lr_norm0)
        mask_all.append(mask.reshape(1,1,mask.shape[0],mask.shape[1]))
    lr_all = np.concatenate(lr_all, axis=0)
    hr_all = np.concatenate(hr_all, axis=0)
    mask_all = np.concatenate(mask_all, axis=0)
    # np.allclose(hr_all, hr_all_test, equal_nan=True)
    
    out_path0 = path_par+'results_pnt/'+'S'+str(opt.up_factor)+ '_'+ suf +'/'
    os.makedirs(out_path0, exist_ok=True)    
    
    # save original hr for the selected time range
    filename = out_path0 + 'hr'+tstr+'.npz'
    # if not os.path.isfile(filename): 
    np.savez(filename,hr_all=hr_all,lat=lat,lon=lon,t=tuser0)

    # reconstuct use direct interpolation 
    # interpolation for cases if low and high variable are the same
    if ivar_hr==ivar_lr:
        hr_re1_all = []
        hr_re2_all = []
        hr_re3_all = []
        for i in range(Nt):
            nc_f = files_hr[i] 
            indt = indt_hr[i] # time index in nc_f
            mask = nc_load_vars(nc_f[0],var_hr[0],indt,lats=ll_hr[0],lons=ll_hr[1])[4] #(1,H,W)
            mask = np.squeeze(mask)
    
            nc_f = files_lr[i]
            indt = indt_lr[i] # time index in nc_f
            data = nc_normalize_vars(nc_f,var_lr,indt,varm_lr,
                                     ll_lr[0],ll_lr[1],kintp[0])  #(H,W,C)
            x = np.transpose(data,(2,0,1)) #(C,H,W)
            lr = torch.from_numpy(x)
            lr = lr.reshape(1,lr.shape[0],lr.shape[1],lr.shape[2]) # 3d to 4d
            
            # nearest, linear (3D-only), bilinear, bicubic (4D-only), trilinear (5D-only), area, nearest-exact
            hr_restore1 = torch.nn.functional.interpolate(lr, scale_factor=opt.up_factor,mode='bicubic') # default nearest;bicubic; input 4D/5D
            hr_restore2 = torch.nn.functional.interpolate(lr, scale_factor=opt.up_factor,mode='bilinear') # default nearest;
            hr_restore3 = torch.nn.functional.interpolate(lr, scale_factor=opt.up_factor,mode='nearest') # default nearest;
            
            hr_restore1_norm0  = var_denormalize(hr_restore1.detach().numpy(),varm_hr)
            hr_restore2_norm0  = var_denormalize(hr_restore2.detach().numpy(),varm_hr)
            hr_restore3_norm0  = var_denormalize(hr_restore3.detach().numpy(),varm_hr)
            
            if kmask == 1: 
                hr_norm0[:,:,mask] = np.nan
                hr_restore1_norm0[:,:,mask] = np.nan
                hr_restore2_norm0[:,:,mask] = np.nan
                hr_restore3_norm0[:,:,mask] = np.nan
            
            hr_re1_all.append(hr_restore1_norm0)
            hr_re2_all.append(hr_restore2_norm0)
            hr_re3_all.append(hr_restore3_norm0)
        
        hr_re1_all = np.concatenate(hr_re1_all, axis=0)
        hr_re2_all = np.concatenate(hr_re2_all, axis=0)
        hr_re3_all = np.concatenate(hr_re3_all, axis=0)
        
        filename = out_path0 + 'hr'+tstr+'_interp'+'.npz'
        # if not os.path.isfile(filename):
        np.savez(filename,hr_re1_all=hr_re1_all,hr_re2_all=hr_re2_all,hr_re3_all=hr_re3_all,lat=lat,lon=lon,t=tuser0)
    
    # # reconstuct use direct interpolation 
    # hr_all_test = []
    # lr_all_test = []
    # mask_all = []
    # hr_re1_all = []
    # hr_re2_all = []
    # hr_re3_all = []
    # for i in range(Nt):
    #     nc_f = files_hr[i] 
    #     indt = indt_hr[i] # time index in nc_f
    #     data = nc_normalize_vars([nc_f],var_hr,[indt],varm_hr,
    #                                      ll_hr[0],ll_hr[1],kintp[1])  #(H,W,C)
    #     x = np.transpose(data,(2,0,1)) #(C,H,W)
    #     hr = torch.from_numpy(x)
    #     mask = nc_load_vars(nc_f,var_hr[0],[indt_hr[i]],lats=ll_hr[0],lons=ll_hr[1])[4] #(1,H,W)
    #     mask = np.squeeze(mask)
    
    #     nc_f = files_lr[i]
    #     indt = indt_lr[i] # time index in nc_f
    #     data = nc_normalize_vars([nc_f],var_lr,[indt],varm_lr,
    #                              ll_lr[0],ll_lr[1],kintp[0])  #(H,W,C)
    #     x = np.transpose(data,(2,0,1)) #(C,H,W)
    #     lr = torch.from_numpy(x)
    #     # mask_lr = nc_load_vars(nc_f,var_lr[0],[indt_lr[i]],lats=ll_lr[0],lons=ll_lr[1])[4] #(1,H,W)
    #     # mask_lr = np.squeeze(mask_lr)
                        
    #     lr = lr.reshape(1,lr.shape[0],lr.shape[1],lr.shape[2]) # 3d to 4d
    #     hr = hr.reshape(1,hr.shape[0],hr.shape[1],hr.shape[2]) # 3d to 4d
        
    #     # nearest, linear (3D-only), bilinear, bicubic (4D-only), trilinear (5D-only), area, nearest-exact
    #     hr_restore1 = torch.nn.functional.interpolate(lr, scale_factor=opt.up_factor,mode='bicubic') # default nearest;bicubic; input 4D/5D
    #     hr_restore2 = torch.nn.functional.interpolate(lr, scale_factor=opt.up_factor,mode='bilinear') # default nearest;
    #     hr_restore3 = torch.nn.functional.interpolate(lr, scale_factor=opt.up_factor,mode='nearest') # default nearest;
        
    #     hr_norm0 = var_denormalize(hr.detach().numpy(),varm_hr)
    #     lr_norm0 = var_denormalize(lr.detach().numpy(),varm_lr)
    #     hr_restore1_norm0  = var_denormalize(hr_restore1.detach().numpy(),varm_hr)
    #     hr_restore2_norm0  = var_denormalize(hr_restore2.detach().numpy(),varm_hr)
    #     hr_restore3_norm0  = var_denormalize(hr_restore3.detach().numpy(),varm_hr)
        
    #     if kmask == 1: 
    #         hr_norm0[:,:,mask] = np.nan
    #         # lr_norm0[:,:,mask_lr] = np.nan  # for reconstruction, nan not allowed
    #         hr_restore1_norm0[:,:,mask] = np.nan
    #         hr_restore2_norm0[:,:,mask] = np.nan
    #         hr_restore3_norm0[:,:,mask] = np.nan
        
    #     mask_all.append(mask.reshape(1,1,mask.shape[0],mask.shape[1]))
    #     hr_all_test.append(hr_norm0)
    #     lr_all_test.append(lr_norm0)
    #     hr_re1_all.append(hr_restore1_norm0)
    #     hr_re2_all.append(hr_restore2_norm0)
    #     hr_re3_all.append(hr_restore3_norm0)
    
    # mask_all = np.concatenate(mask_all, axis=0)
    # hr_all_test = np.concatenate(hr_all_test, axis=0)
    # lr_all_test = np.concatenate(lr_all_test, axis=0)
    # hr_re1_all = np.concatenate(hr_re1_all, axis=0)
    # hr_re2_all = np.concatenate(hr_re2_all, axis=0)
    # hr_re3_all = np.concatenate(hr_re3_all, axis=0)
    
    # out_path0 = path_par+'results_pnt/'+'S'+str(opt.up_factor)+ '_'+ suf +'/'
    # os.makedirs(out_path0, exist_ok=True)    
    
    # # save original hr and interpolated hr for the selected time range
    # filename = out_path0 + 'hr'+tstr+'.npz'
    # if not os.path.isfile(filename): 
    #     np.savez(filename,hr_all=hr_all_test,lat=lat,lon=lon,t=tuser0)
    
    # filename = out_path0 + 'hr'+tstr+'_interp'+'.npz'
    # if not os.path.isfile(filename):
    #     np.savez(filename,hr_re1_all=hr_re1_all,hr_re2_all=hr_re2_all,hr_re3_all=hr_re3_all,lat=lat,lon=lon,t=tuser0)
        
    opath_mod = path_par+'mod_' + str(opt.up_factor) + '_'+ suf +'/' # 
    # if not os.path.exists(opath_mod):
    #     os.makedirs(opath_mod)
    
    Nsample_test = len(hr_all)
    hr_test = hr_all.reshape((Nsample_test,-1)) # (Nt,C*H*W)
    lr_test = lr_all.reshape((Nsample_test,-1))
    
    X_test = lr_test
    y_test = hr_test
    
    # repeated runs
    for irep in rep:        
        print(f'Repeat {irep}')
        print('--------------------------------')
        
        out_path = path_par+'results_test/'+'S'+str(opt.up_factor)+'_'+suf+'_re'+ str(irep)+'_mk'+str(kmask)+'/'
        os.makedirs(out_path, exist_ok=True)

        # load the model from disk
        filename = opath_mod + 'mod'+'%d_rp%d'%(kmod,irep)+'.pkl' 
        if os.path.isfile(filename):
            print("pickle version:", pickle.format_version)
            estimator = pickle.load(open(filename,'rb'))
            if kmod in [0,1,2]:
                coefs = estimator.coef_
        else:
            # load the model from disk, joblib versions mismatch
            filename = opath_mod + 'mod'+'%d_rp%d'%(kmod,irep)+'.jbl' 
            print("Joblib version:", joblib.__version__)
            estimator = joblib.load(filename)
            if kmod in [0,1,2]:
                coefs = estimator.coef_
        
        y_pred= estimator.predict(X_test)
        sr_all = y_pred.reshape((Nsample_test,1,hr_shape[0],hr_shape[1]))
        
        if kmask == 1: 
            sr_all[mask_all] = np.nan
                    
        # save super-resolution hr for the selected time range
        filename = out_path0+'sr'+tstr+'_re%d' % (irep) +'.npz'
        # if not os.path.isfile(filename): 
        np.savez(filename,sr_all=sr_all,hr_all=hr_all,lat=lat,lon=lon,t=tuser0)
    
        