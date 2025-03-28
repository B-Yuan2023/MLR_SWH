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
from funs_prepost import nc_load_vars,var_denormalize,plt_sub,plt_pcolor_list
import pandas as pd

import sys
import importlib
path_par = "../"  # the path of parameter files, also used for output path
sys.path.append(path_par)  # add path of parameter files to system temporarily for module import
mod_name= sys.argv[1]          #'par55e_md0_tuse1' # sys.argv[1]
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
import matplotlib.pyplot as plt

# inerpolation 
from scipy.interpolate import griddata, interp2d, RBFInterpolator
def interpolate_4Dtensor(tensor, scale_factor, interp_func=griddata, method='linear', **kwargs):
    """
    Interpolates the last two dimensions of a tensor using the specified interpolation function.
    
    Parameters:
        tensor (ndarray): Input tensor of shape (N, C, H, W).
        scale_factor (float): Scale factor for the last two dimensions. The new dimensions will be
                              original_dimensions * scale_factor.
        interp_func (function, optional): Interpolation function to use.
                                                     Default is griddata from scipy.interpolate.
        method (str, optional): Interpolation method to use ('linear', 'nearest', 'cubic').
                                 Default is 'linear'.
        kwargs: Additional keyword arguments to be passed to the interpolation function.
        
    Returns:
        ndarray: Interpolated tensor of shape (N, C, new_H, new_W).
    """
    N, C, H, W = tensor.shape
    new_H, new_W = int(H * scale_factor), int(W * scale_factor)
    new_tensor = np.zeros((N, C, new_H, new_W))
    
    # Create 2D grids for interpolation
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    X, Y = np.meshgrid(x, y)
    
    # Create new 2D grids for interpolated domain
    new_x = np.linspace(0, W - 1, new_W)
    new_y = np.linspace(0, H - 1, new_H)
    new_X, new_Y = np.meshgrid(new_x, new_y)
    
    for n in range(N):
        for c in range(C):
            # Flatten the original grid
            points = np.column_stack((X.flatten(), Y.flatten()))
            values = tensor[n, c].flatten()
            
            # Interpolate using the specified interpolation function
            interpolated = interp_func(points, values, (new_X, new_Y), method=method, **kwargs)
            new_tensor[n, c] = interpolated
    new_tensor = torch.from_numpy(new_tensor).type(torch.float)
    return new_tensor

if __name__ == '__main__':
    
    opt = mod_para.opt
    suf = mod_name # mod_para.suf+mod_name
    str_suff = '_tuse' # suffix string in the additional testing file with time specified
    assert str_suff in suf, 'check tesing para file name!'
    # to origianl training file (used for saving nn), 
    pos = suf.find(str_suff)
    suf0 = suf[:pos]  # back to original training para file name, for loading nn model
    
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
    rep = [0]
    # # suf = '_res' + str(opt.residual_blocks) + '_max_suv' # + '_nb' + str(opt.batch_size)
    # print(f'parname: {mod_name}')
    # print('--------------------------------')

    nchl = nchl_o
    
    hr_shape = (opt.hr_height, opt.hr_width)

    test_set = myDataset(files_lr,files_hr,indt_lr,indt_hr,hr_shape, opt.up_factor,
                          mode='test',rtra = rtra,var_lr=var_lr,var_hr=var_hr,
                          varm_lr=varm_lr,varm_hr=varm_hr,ll_lr=ll_lr,ll_hr=ll_hr,kintp=kintp)

    data_test = DataLoader(
        test_set,
        batch_size=opt.batch_size, 
        num_workers=opt.n_cpu,
    )        
    
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    
    # get logitude and latitude of data 
    nc_f = test_set.files_hr[0]
    lon = nc_load_vars(nc_f[0],var_hr[0],[0],lats=ll_hr[0],lons=ll_hr[1])[1]
    lat = nc_load_vars(nc_f[0],var_hr[0],[0],lats=ll_hr[0],lons=ll_hr[1])[2]
        
    clim = [[[1.3,3.3],[1.3,3.3],[-0.2,0.2]],  # ssh
            [[0.2,1.8],[0.2,1.8],[-0.3,0.3]],  # u
            [[0.2,1.8],[0.2,1.8],[-0.3,0.3]],  # v
            [[12,15],[12,15],[-1.0,1.0]],  # uw
            [[12,15],[12,15],[-1.0,1.0]],  # vw
            [[2.0,5.0],[2.0,5.0],[-0.5,0.5]],  # swh
            [[5.0,15],[5.0,15],[-2.0,2.0]],  # pwp
            [[2.0,5.0],[2.0,5.0],[-0.5,0.5]],]  # swh_ww
    clim_a = [[-3.0,3.0],[-1.2,1.2],[-1.5,1.5],[12,15],[12,15],[0.0,4.0],[0.0,15.]]  # ssh,u,v,uw,vw,swh,pwp
    unit_suv = ['ssh (m)','u (m/s)','v (m/s)','uw (m/s)','vw (m/s)','swh (m)','pwp (s)','swh_ww (m)',]
    
    # select a range of data for testing 
    # tlim = [datetime(2022,1,1),datetime(2022,12,31)]
    tlim = mod_para.tlim
    dt = mod_para.dt
    
    tstr = '_'+tlim[0].strftime('%Y%m')+'_'+tlim[1].strftime('%Y%m') # + '_t%d'%dt
    # Nt = int((tlim[1]-tlim[0]).total_seconds()/(dt*3600)) ## total time steps

    
    opath_st = path_par + suf +tstr+'/'
    os.makedirs(opath_st, exist_ok=True)
    
    ipath_mod = path_par+'mod_' + str(opt.up_factor) + '_'+ suf0 +'/' # 
    # if not os.path.exists(ipath_mod):
    #     os.makedirs(ipath_mod)
    
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
    
    Nsample_test = len(hr_all_test)
    hr_test = hr_all_test.reshape((Nsample_test,-1)) # (Nt,C*H*W)
    lr_test = lr_all_test.reshape((Nsample_test,-1))
    
    X_test = lr_test
    y_test = hr_test
    
    # select certain samples for plotting 
    nsub_test = 5
    # rng = check_random_state(4)
    # sub_ids = rng.randint(lr_all_test.shape[0], size=(nsub_test,))
    # X_test_sub = lr_all_test[sub_ids,:]
    # y_test_sub = hr_all_test[sub_ids,:]
    nsubpfig = 6 # subfigure per figure
    sub_ids = np.arange(0,Nsample_test-nsubpfig*2,int(Nsample_test/nsub_test))
                          
    # repeated runs
    metrics = {'rp':[], 'mae': [], 'rmse': [], 'mae_99': [],'rmse_99': [],
               'mae_01': [],'rmse_01': [],'mae_m': [],'rmse_m': [],'mae_t': [],'rmse_t': [],} # 
    metrics_chl = {}
   
    for irep in rep:        
        print(f'Repeat {irep}')
        print('--------------------------------')
        
        out_path = path_par+'results_test/'+suf+'_re'+ str(irep)+tstr+'/'
        os.makedirs(out_path, exist_ok=True)

        # load the model from disk
        filename = ipath_mod + 'mod'+'%d_rp%d'%(kmod,irep)+'.pkl' 
        if os.path.isfile(filename):
            print("pickle version:", pickle.format_version)
            estimator = pickle.load(open(filename,'rb'))
            if kmod in [0,1,2]: # only for linear regression
                coefs = estimator.coef_
        else:
            # load the model from disk, joblib versions mismatch
            filename = ipath_mod + 'mod'+'%d_rp%d'%(kmod,irep)+'.jbl' 
            print("Joblib version:", joblib.__version__)
            estimator = joblib.load(filename)
            if kmod in [0,1,2]:
                coefs = estimator.coef_
        
        y_pred= estimator.predict(X_test)
        sr_all = y_pred.reshape((Nsample_test,1,hr_shape[0],hr_shape[1]))
        
        for i in sub_ids:  # plot several time steps dimensional field
            lr_norm0 = torch.from_numpy(lr_all_test[i:i+nsubpfig*2,:,:,:])
            lr_nr_norm0 = torch.nn.functional.interpolate(lr_norm0, scale_factor=opt.up_factor)# default nearest;bicubic; input 4D/5D
            lr_nr_norm0 = lr_nr_norm0.cpu().numpy()
            # lr_norm0 = lr_all_test[i:i+nsubpfig*2,:,:,:]
            # lr_nr_norm0 = interpolate_4Dtensor(lr_norm0, opt.up_factor, interp_func=griddata, method='nearest')
            hr_norm0 = hr_all_test[i:i+nsubpfig*2,:,:,:]
            sr_norm0 = sr_all[i:i+nsubpfig*2,:,:]

            # Save image grid with upsampled inputs and SR outputs
            cat_dim = 2 # concatenate for dimension H:2, W:-1 or 3. 
            if nchl_i == nchl_o ==1: # or cat_dim == 2: # same vars or 1 var to 1 var
                img_grid_nm0 = np.concatenate((np.flip(lr_nr_norm0,2),
                                               np.flip(hr_norm0,2), np.flip(sr_norm0,2)), cat_dim) 
            else:
                img_grid_nm0 = np.concatenate((np.flip(hr_norm0,2), np.flip(sr_norm0,2)), cat_dim)

            # nsubpfig = 6 # subfigure per figure
            nfig = int(-(-len(img_grid_nm0) // nsubpfig))
            for j in np.arange(nfig):
                ne = min((j+1)*nsubpfig,len(img_grid_nm0)) # index of the last sample in a plot
                ind = np.arange(j*nsubpfig,ne) # index of the samples in a plot
                image_nm0 = img_grid_nm0[ind,...] #*(C,H,W)
                ncol = 2
                if cat_dim == 2: # if cat hr, sr in vertical direction, cat samples in W direction
                    temp_nm0 = image_nm0[0,:]
                    for ij in range(1,len(image_nm0)):
                        temp_nm0 = np.concatenate((temp_nm0,image_nm0[ij,...]), -1)
                    image_nm0 = temp_nm0.reshape(1,temp_nm0.shape[0],temp_nm0.shape[1],temp_nm0.shape[2])
                    ncol=1
                for k in range(nchl_o):
                    ichl = ivar_hr[k]
                    figname = out_path+"c%d_dt%d_id%d_nm0.png" % (ichl,i,j)
                    clim_v = clim_a[ichl]
                    plt_sub(image_nm0,ncol,figname,k,clim=clim_v,cmp='bwr') # 'bwr','coolwarm',,contl=[-0.05,]
                    figname = out_path+"c%d_dt%d_id%d.png" % (ichl,i,j)
                    # image = (image_nm0-varm_hr[k][1])/(varm_hr[k][0]-varm_hr[k][1])
                    # # contl = [-0.01-varm_hr[k][1]/(varm_hr[k][0]-varm_hr[k][1])]
                    # plt_sub(image,ncol,figname,cmp='bwr') # 'bwr','coolwarm',contl=contl
        
        # estimate errors excluding masked area. 
        if kmask == 1: 
            for it in range(Nsample_test):
                for ichl in range(nchl):
                    nc_f = test_set.files_hr[it][ichl]
                    indt = test_set.indt_hr[it][ichl]  # the time index in a ncfile
                    mask = nc_load_vars(nc_f,var_hr[ichl],[indt],ll_hr[0],ll_hr[1])[4] # mask at 1 time in a batch
                    hr_all_test[it,mask] = np.nan
                    sr_all[it,mask] = np.nan
        
        # save sr_all for comparison
        for k in range(nchl_o): # nchl_o, save for all time steps. 
            ichl = ivar_hr[k]
            filename = out_path + "c%d_sr_all" % (ichl)+'.npz'
            if not os.path.isfile(filename): 
                var_all  = sr_all[:,k,:,:]
                np.savez(filename,v0=var_all) 
        
        # mae = mean_absolute_error(y_test, y_pred)
        # mse = mean_squared_error(y_test, y_pred)
        # rmse = mse**0.5
        # r2 = r2_score(y_test, y_pred)
        rmse = np.nanmean((sr_all - hr_all_test) ** 2, axis = (0,2,3))**(0.5)
        mae = np.nanmean(abs(sr_all - hr_all_test), axis = (0,2,3))
        
        sr_99per = np.nanpercentile(sr_all, 99, axis = (0,))  # (C,H,W)
        sr_01per = np.nanpercentile(sr_all, 1, axis = (0,))
        sr_mean = np.nanmean(sr_all, axis = (0,))
        
        filename99 = out_path + 'hr_99per'+'_train%4.2f'%(rtra[0])+'.npz'
        if not os.path.isfile(filename99):
            hr_99per = np.nanpercentile(hr_all_test, 99, axis = (0,))
            hr_01per = np.nanpercentile(hr_all_test, 1, axis = (0,))
            hr_mean = np.nanmean(hr_all_test, axis = (0,))
            np.savez(filename99,v0=hr_99per,v1=hr_01per,v2=hr_mean) 
        else:
            datald = np.load(filename99) # load
            hr_99per= datald['v0']
            hr_01per= datald['v1']
            hr_mean= datald['v2']
        
        # estimate and save sr_99per in this script
        rmse_99 = np.nanmean((sr_99per - hr_99per) ** 2,axis=(1,2))**(0.5)
        mae_99 = np.nanmean(abs(sr_99per - hr_99per),axis=(1,2))
        # save 99th data for each epoch 
        filename99 = out_path + "sr_99th_" +'re'+ str(irep)+'.npz'
        np.savez(filename99,v0=sr_99per,v1=hr_99per,v2=rmse_99,v3=mae_99) 
        
        # estimate and save sr_01per in this script
        rmse_01 = np.nanmean((sr_01per - hr_01per) ** 2,axis=(1,2))**(0.5)
        mae_01 = np.nanmean(abs(sr_01per - hr_01per),axis=(1,2))
        # save 99th data for each epoch 
        filename01 = out_path + "sr_01th_" +'re'+ str(irep)+'.npz'
        np.savez(filename01,v0=sr_01per,v1=hr_01per,v2=rmse_01,v3=mae_01) 

        # estimate and save sr_mean in this script
        rmse_m = np.nanmean((sr_mean - hr_mean) ** 2,axis=(1,2))**(0.5) # spatial rmse of temporal mean 
        mae_m = np.nanmean(abs(sr_mean - hr_mean),axis=(1,2))
        # save 99th data for each epoch 
        filename_m = out_path + "sr_mean_" +'re'+ str(irep)+'.npz'
        np.savez(filename_m,v0=sr_mean,v1=hr_mean,v2=rmse_m,v3=mae_m) 
        
        # estimate and save sr_rmse sr_mae 
        sr_rmse = np.nanmean((sr_all - hr_all_test) ** 2,axis=(0))**(0.5) # temporal rmse per point per channel (C,H,W)
        sr_mae = np.nanmean(abs(sr_all - hr_all_test),axis=(0))
        rmse_t = np.nanmean((sr_rmse),axis=(1,2))  # spatial average of rmse (C)
        mae_t = np.nanmean(sr_mae,axis=(1,2))
        # save 99th data for each epoch 
        filename_m = out_path + "sr_tave_" +'re'+ str(irep)+'.npz'
        np.savez(filename_m,v0=sr_rmse,v1=sr_mae,v2=rmse_t,v3=mae_t) 

        metrics['mae'].append(mae)
        metrics['rmse'].append(rmse)
        metrics['mae_99'].append(mae_99)
        metrics['rmse_99'].append(rmse_99)
        metrics['mae_01'].append(mae_01)
        metrics['rmse_01'].append(rmse_01)
        metrics['mae_m'].append(mae_m)
        metrics['rmse_m'].append(rmse_m)
        metrics['mae_t'].append(mae_t)
        metrics['rmse_t'].append(rmse_t)
        metrics['rp'].append([irep]*nchl)
        
        # if epoch % opt.sample_epoch == 0:
        for i in range(nchl_o):
            ichl = ivar_hr[i]
            clim_chl = clim[ichl]                
            sample  = [hr_99per[i,:,:],sr_99per[i,:,:],sr_99per[i,:,:]-hr_99per[i,:,:]]
            unit = [unit_suv[ichl]]*len(sample)
            title = ['hr_99','sr_99','sr-hr'+'(%5.3f'%mae_99[i]+',%5.3f'%rmse_99[i]+')']
            figname = out_path+"99th_c%d_ax0.png" % (ivar_hr[i])
            plt_pcolor_list(lon,lat,sample,figname,cmap = 'coolwarm',clim=clim_chl,unit=unit,title=title,nrow=2,axoff=1) 

        # output metrics for all epochs to csv
        # data_frame = pd.DataFrame.from_dict(metrics, orient='index').transpose()
        # ofname = "srf_%d_c%d_mask" % (opt.up_factor,ivar_hr[i]) + '_test_metrics.csv'
        # data_frame.to_csv(opath_st + os.sep + ofname, index_label='rp')
        
        # output metrics for all repeated runs to csv
        for i in range(nchl):
            for key, value in metrics.items():
                metrics_chl[key] = [value[j][i] for j in range(len(value))]
            data_frame = pd.DataFrame.from_dict(metrics_chl, orient='index').transpose()
            ofname = "srf_%d_c%d_mask" % (opt.up_factor,ivar_hr[i]) + '_test_metrics.csv'
            data_frame.to_csv(opath_st + os.sep + ofname, index_label='rp')
        