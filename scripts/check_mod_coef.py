#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:46:30 2024

@author: g260218
"""

import os
import numpy as np
import torch

from funs_prepost import (make_list_file_t,nc_load_vars,nc_normalize_vars,var_denormalize,
                          plt_sub,plt_pcolor_list,plt_pcolorbar_list)
# from funs_sites import select_sta
from datetime import datetime, timedelta # , date

import sys
import importlib
path_par = "../"  # the path of parameter files, also used for output path
sys.path.append(path_par)  # add path of parameter files to system temporarily for module import
mod_name= 'par55e_s80_md0'          # # sys.argv[1]
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
    lr_shape = (int(opt.hr_height/opt.up_factor), int(opt.hr_width/opt.up_factor))

    # select a range of data for testing 
    # tlim = [datetime(2021,11,29),datetime(2021,12,1)]
    tlim = [datetime(2021,11,29),datetime(2021,12,2)]    
    # tlim = [datetime(2021,1,26),datetime(2021,1,28)]
    # tlim = [datetime(2021,1,16),datetime(2021,1,18)]
    dt = 3
    
    tlim = [datetime(2021,11,29,3),datetime(2021,11,29,9)] # just for 2d time series
    dt = 6
    
    tstr = '_'+tlim[0].strftime('%Y%m%d%H')+'_'+tlim[1].strftime('%Y%m%d%H') + '_t%d'%dt
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
    lon_hr = nc_load_vars(nc_f,var_hr[0],[0],lats=ll_hr[0],lons=ll_hr[1])[1]
    lat_hr = nc_load_vars(nc_f,var_hr[0],[0],lats=ll_hr[0],lons=ll_hr[1])[2]
    nc_f = files_lr[0][0]
    lon = nc_load_vars(nc_f,var_lr[0],[0],lats=ll_lr[0],lons=ll_lr[1])[1]
    lat = nc_load_vars(nc_f,var_lr[0],[0],lats=ll_lr[0],lons=ll_lr[1])[2]
    
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
    mask_all_lr = []
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
        mask_lr = nc_load_vars(nc_f[0],var_lr[0],indt,lats=ll_lr[0],lons=ll_lr[1])[4] #(1,H,W)
        mask_lr = np.squeeze(mask_lr)
                        
        lr = lr.reshape(1,lr.shape[0],lr.shape[1],lr.shape[2]) # 3d to 4d
        hr = hr.reshape(1,hr.shape[0],hr.shape[1],hr.shape[2]) # 3d to 4d
        lr_norm0 = var_denormalize(lr.detach().numpy(),varm_lr)
        hr_norm0 = var_denormalize(hr.detach().numpy(),varm_hr)
        
        if kmask == 1: 
            hr_norm0[:,:,mask] = np.nan
        hr_all.append(hr_norm0)
        lr_all.append(lr_norm0)
        mask_all.append(mask.reshape(1,1,mask.shape[0],mask.shape[1]))
        mask_all_lr.append(mask_lr.reshape(1,1,mask_lr.shape[0],mask_lr.shape[1]))
    lr_all = np.concatenate(lr_all, axis=0)
    hr_all = np.concatenate(hr_all, axis=0)
    mask_all = np.concatenate(mask_all, axis=0)
    mask_all_lr = np.concatenate(mask_all_lr, axis=0)
    # np.allclose(hr_all, hr_all_test, equal_nan=True)
    
    Nsample_test = len(hr_all)
    hr_test = hr_all.reshape((Nsample_test,-1)) # (Nt,C*H*W)
    lr_test = lr_all.reshape((Nsample_test,-1))
    
    X_test = lr_test
    y_test = hr_test
    
    in_path = path_par+'results_test/'+'SRF_'+str(opt.up_factor)+'/' # hr_all, etc
    
    unit_suv = ['ssh (m)','u (m/s)','v (m/s)','uw (m/s)','vw (m/s)','swh (m)','pwp (s)','swh_ww (m)',]
    
    opath_st = path_par+'stat_' + suf +'/'
    os.makedirs(opath_st, exist_ok=True)
    
    opath_mod = path_par+'mod_' + str(opt.up_factor) + '_'+ suf +'/' # 
    # if not os.path.exists(opath_mod):
    #     os.makedirs(opath_mod)
    
    #  make a list for figure captions
    alpha = list(map(chr, range(ord('a'), ord('z')+1)))
    alpha_l = alpha + ['a'+i for i in alpha]
    capt_all = ['('+alpha_l[i]+')' for i in range(len(alpha_l))]
    
    # use selected stations, 3 near buoys at 10, 20, 40 m, 1 at maximam SWH
    index = np.array([[104, 22],[76, 6+8],[83, 20],[88, 239]])
    # ll_sta = np.array([[27.950,43.200],[27.550,42.500],[27.900,42.675],[33.375,42.800]])
    ll_sta = np.array([[43.200,27.950],[42.500,27.550+8*0.025],[42.675,27.900],[42.800,33.375]]) # 1st lat 2nd lon in select_sta
    sta_user = ['P'+str(ip+1) for ip in range(len(index))]

# overlapped points 
    # index = np.array([[76, 14],[76, 34],[76, 54],[76, 74]])
    # # ll_sta = np.array([[27.950,43.200],[27.550,42.500],[27.900,42.675],[33.375,42.800]])
    # ll_sta = np.array([[42.500,27.750],[42.500,28.250],[42.500,28.750],[42.500,29.250]]) # 1st lat 2nd lon in select_sta
    # sta_user = ['P'+str(ip+1) for ip in range(len(index))]
    
    
    # index,sta_user,ll_sta,varm_hr_test,ind_varm = select_sta(hr_all,ivar_hr,lon,lat,nskp)
    nsta = len(index)
    
    opath_st = path_par+'stat_' + suf +'/'
   
    for irep in rep:        
        print(f'Repeat {irep}')
        print('--------------------------------')
        
        # out_path = path_par+'results_test/'+'S'+str(opt.up_factor)+'_'+suf+'_re'+ str(irep)+'_mk'+str(kmask)+'/'
        # os.makedirs(out_path, exist_ok=True)
        
        # load the model from disk
        filename = opath_mod + 'mod'+'%d_rp%d'%(kmod,irep)+'.pkl' 
        if os.path.isfile(filename):
            print("pickle version:", pickle.format_version)
            estimator = pickle.load(open(filename,'rb'))
            if kmod in [0,1,2]:
                coefs = estimator.coef_         # (Nout,Nin)
                coef0 = estimator.intercept_    # (Nout,1)
        else:
            # load the model from disk, joblib versions mismatch
            filename = opath_mod + 'mod'+'%d_rp%d'%(kmod,irep)+'.jbl' 
            print("Joblib version:", joblib.__version__)
            estimator = joblib.load(filename)
            if kmod in [0,1,2]:
                coefs = estimator.coef_         # (Nout,Nin)
                coef0 = estimator.intercept_    # (Nout,1)
        
        y_pred0= estimator.predict(X_test)
        y_pred = np.dot(X_test,coefs.T)+coef0.reshape((1,-1))
        np.allclose(y_pred, y_pred0, equal_nan=True)    # checked True
        
        icheck = np.arange(998,1006,1)  # 998~[4,38] check certain locations
        ycheck = np.dot(X_test,(coefs.T)[:,icheck])+coef0.reshape((1,-1))[:,icheck]
        
        
        # lr_all1 = X_test.reshape((Nsample_test,1,lr_shape[0],lr_shape[1]))
        # np.allclose(lr_all1, lr_all, equal_nan=True)    # checked True
        
        lr_flip = np.flip(lr_all,axis=2)
        plt_sub(lr_flip,1)

        # hr_flip = np.flip(hr_all,axis=2)
        # plt_sub(hr_flip,1)
        
        sr_all = y_pred.reshape((Nsample_test,1,hr_shape[0],hr_shape[1]))
        if kmask == 1: 
            sr_all[mask_all] = np.nan
        
        # sr_flip = np.flip(sr_all,axis=2)
        # plt_sub(sr_flip,1)
        
        # dim of coefs is (H_hr*W_hr,H_lr*W_lr)
        coefs_reshape = np.reshape(coefs,(hr_shape[0],hr_shape[1],lr_shape[0],lr_shape[1]))
        if kmask == 1: 
            coefs_reshape[:,:,mask_lr] = np.nan 
        
        coefs_sta = []
        for i in range(nsta):
            coefs_sta.append(coefs_reshape[index[i,0],index[i,1],:,:])            
        
        tstr0 = '_'+tlim[0].strftime('%Y%m%d%H')
        iy0 = 4 # int(hr_shape[0]/2)
        ix0 = 38 # hr_shape[1]-1 # int(hr_shape[1]/2)
        sample = coefs_reshape[iy0:iy0+16,ix0:ix0+1,:,:]
        sample = np.flip(sample,axis=2)
        figname = opath_st+"check_coe_ix%d_iy%d"%(ix0,iy0)+tstr0+".png"
        plt_sub(sample,4,figname)
        
        # coefs_reshape = np.reshape(coefs,(-1,lr_shape[0],lr_shape[1]))
        # coefs_sta = []
        # for i in range(nsta):
        #     id = index[i,0]*hr_shape[1]+index[i,1]
        #     coefs_sta.append(coefs_reshape[id,:,:])
        
        # show 2D spatial coefficients with target location
        for i in range(nchl_o):
            ichl = ivar_hr[i]
            clim_chl = None # [clim[ichl]]*len(sample)
            unit = None # [unit_suv[ichl]]*len(sample)
            # subsize = [2.5,2]
            subsize = [2.0,1.6]
            kax = 1
            kbar= 0
            sample = coefs_sta
            title = sta_user
            txt = ['o']* len(sta_user)
            loc_txt = np.flip(ll_sta,1)
            figname = opath_st+"c%d_re%d"% (ichl,irep)+"_coe_ax%d_kb%d"%(kax,kbar)+tstr0+".png"
            plt_pcolorbar_list(lon,lat,sample,figname,subsize = subsize,cmap = 'coolwarm',
                               clim=clim_chl,kbar=kbar,unit=unit,title=title,
                               nrow=2,axoff=kax,capt=capt_all,txt=txt,loc_txt=loc_txt ) # 

