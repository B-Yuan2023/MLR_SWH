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
from funs_prepost import nc_load_vars,var_denormalize
# import pandas as pd

import sys
import importlib
path_par = "../"  # the path of parameter files, also used for output path
sys.path.append(path_par)  # add path of parameter files to system temporarily for module import
mod_name= sys.argv[1]         #'par55e_md0' # sys.argv[1]
mod_para=importlib.import_module(mod_name)
# from mod_para import * 
kmask = 1

# https://scikit-learn.org/stable/modules/multiclass.html#multioutput-regression
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge
from sklearn.neighbors import KNeighborsRegressor #,RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
# from sklearn.utils.validation import check_random_state
# from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import joblib
import pickle
# import matplotlib.pyplot as plt

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
    
    if hasattr(mod_para, 'kmod'): # ml model: 0 linReg, 1 RidgeCV, 2 Ridge, 3 KNeighbors,4 DT, 5 ExtTree, 6 RandomForest, 7 ExtTrees
        kmod = mod_para.kmod
    else:
        kmod = 0
    
    if hasattr(mod_para, 'n_est'):  # number of estimators, for ensemble 6,7
        n_est = mod_para.n_est
    else:
        n_est = 10
    if hasattr(mod_para, 'n_jobs'):  # parallel jobs, for ensemble 6,7
        n_jobs = mod_para.n_jobs
    else:
        n_jobs = None
    if hasattr(mod_para, 'max_dep'):  # max_depth of trees, for tree type 4-7
        max_dep = mod_para.max_dep
    else:
        max_dep = None
    if hasattr(mod_para, 'srand'): # rand_state for tree type 4-8
        srand = mod_para.srand
    else:
        srand = None
    
    if hasattr(mod_para, 'hidden_sizes'):
        hidden_sizes = mod_para.hidden_sizes # hidden layer sizes
    else:
        hidden_sizes = (100,)
    if hasattr(mod_para, 'kfun_act'):
        kfun_act = mod_para.kfun_act # key for actication function for kmod==8 MLP
    else:
        kfun_act = 'relu'  # activation function for MLP kmod==8, {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}

    
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

    data_train = DataLoader(
        train_set,
        batch_size=opt.batch_size,
        num_workers=opt.n_cpu,
    )

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
            
    # opath_st = path_par+'stat' + suf +'/'
    # os.makedirs(opath_st, exist_ok=True)
    
    opath_mod = path_par+'mod_' + str(opt.up_factor) + '_'+ suf +'/' # 
    if not os.path.exists(opath_mod):
        os.makedirs(opath_mod)
        
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
    
    
    Nsample_train = len(hr_all_train)
    hr_train = hr_all_train.reshape((Nsample_train,-1))  # (N,H,W)->(N,H*W)
    lr_train = lr_all_train.reshape((Nsample_train,-1))

    X_train = lr_train
    y_train = hr_train
    
    # repeated runs
    for irep in rep:        
        print(f'Repeat {irep}')
        print('--------------------------------')
        
        # Initialize generator 
        if kmod ==0:
            estimator = LinearRegression()
        elif kmod == 1:
            estimator = RidgeCV()
        elif kmod == 2:
            estimator = Ridge()
        elif kmod == 3:
            estimator = KNeighborsRegressor()
        elif kmod == 4:        
            estimator = DecisionTreeRegressor(max_depth=max_dep,random_state=srand)
        elif kmod == 5:        
            estimator = ExtraTreeRegressor(max_depth=max_dep,random_state=srand)            
        elif kmod == 6:
            estimator = RandomForestRegressor(n_estimators=n_est, max_depth=max_dep,
                                              n_jobs=n_jobs,random_state=srand)
        elif kmod == 7:
            estimator = ExtraTreesRegressor(n_estimators=n_est, max_depth=max_dep,
                                            n_jobs=n_jobs,random_state=srand) # ,max_features=32
        elif kmod == 8:
            niter = -(-Nsample_train//opt.batch_size) * opt.N_epochs
            estimator = MLPRegressor(hidden_layer_sizes=hidden_sizes, activation=kfun_act, 
                                     solver='adam', batch_size=opt.batch_size, learning_rate_init=opt.lr, 
                                     max_iter=niter, shuffle=True, random_state=srand) 

        estimator.fit(X_train, y_train)
        # save the model to disk, joblib versions mismatch on node and local machine
        # filename = opath_mod + 'mod'+'%d_rp%d'%(kmod,irep)+'.jbl'
        # joblib.dump(estimator, filename)
        # print("Joblib version used to save the model:", joblib.__version__)

        # save the model to disk, use pickle as local and node versions match        
        filename = opath_mod + 'mod'+'%d_rp%d'%(kmod,irep)+'.pkl'
        pickle.dump(estimator, open(filename,'wb'))  
        print("pickle version used to save the model:", pickle.format_version)        
       