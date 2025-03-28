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
mod_name= 'par534e_md0'         #'par04' # sys.argv[1]
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
import time

if __name__ == '__main__':
    
    start = time.time()
    
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
    
    in_path = path_par+'results_test/'+'SRF_'+str(opt.up_factor)+'/' # hr_all, etc
    
    # opath_st = path_par+'stat_' + suf +'/'
    # os.makedirs(opath_st, exist_ok=True)
    
    opath_mod = path_par+'mod_' + str(opt.up_factor) + '_'+ suf +'/' # 
    # if not os.path.exists(opath_mod):
    #     os.makedirs(opath_mod)
    
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
        
        out_path = path_par+'results_test/'+'S'+str(opt.up_factor)+'_'+suf+'_re'+ str(irep)+'_mk'+str(kmask)+'/'
        os.makedirs(out_path, exist_ok=True)

        # load the model from disk
        filename = opath_mod + 'mod'+'%d_rp%d'%(kmod,irep)+'.pkl' 
        if os.path.isfile(filename):
            print("pickle version:", pickle.format_version)
            estimator = pickle.load(open(filename,'rb'))
            if kmod in [0,1,2]: # only for linear regression
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
        
        end = time.time()
        elapsed = (end - start)
        print('rep%d cost %s s'%(irep,elapsed))
   