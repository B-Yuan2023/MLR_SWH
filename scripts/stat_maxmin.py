#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:27:02 2024

sort the files with the maximum of variables 
@author: g260218
"""

import numpy as np
import os
# import glob
from funs_prepost import nc_load_vars
import pandas as pd
# from datetime import datetime # , date, timedelta

import sys
import importlib
path_par = "../"  # the path of parameter files, also used for output path
sys.path.append(path_par)  # add path of parameter files to system temporarily for module import
mod_name=sys.argv[1]          #'par55e_exp' # sys.argv[1]
mod_para=importlib.import_module(mod_name)


def lst_flatten(xss):
    return [x for xs in xss for x in xs]

# sorted var per file
def find_max_global_file(files,varname,indt,lats=None,lons=None,kdesc=False):
    # files: list of files
    # varname: list of var names
    # indt: list of time step index in ncfiles, same dim as files
    nfile = len(files)
    nvar = len(varname)
    ind_sort = [[]]*nvar
    var_sort = [[]]*nvar
    for i in range(nvar):
        var_comb = []
        for indf in range(nfile):
            nc_f = files[indf]
            var = nc_load_vars(nc_f,varname[i],indt[indf],lats,lons)[3] # (NT,H,W) one channel
            var_max = var.max() # maximum in time dim 0 and space dim 1,2
            var_comb.append(var_max)
        ind_sort[i] = sorted(range(len(var_comb)), key=lambda k: var_comb[k], reverse=kdesc) # reverse, False ascending,True descending
        var_sort[i] = [var_comb[k] for k in ind_sort[i]]
    return var_sort,ind_sort 


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
    
    # if isinstance(rtra, (int, float)): # if only input one number, no validation
    #     rtra = [rtra,0]
    # # create nested list of files and indt
    # if len(files_hr[0])!=nchl_o:
    #     files_hr = [[ele for _ in range(nchl_o)] for ele in files_hr]
    #     indt_hr = [[ele for _ in range(nchl_o)] for ele in indt_hr]
    # if len(files_lr[0])!=nchl_i:
    #     files_lr = [[ele for _ in range(nchl_i)] for ele in files_lr]
    #     indt_lr = [[ele for _ in range(nchl_i)] for ele in indt_lr]
        
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
        
    # opath_st = 'statistics' + suf +'/'
    opath_st_hr = path_par +'statistics_hr'+'_%d_%d'%(opt.hr_height, opt.hr_width)+'/'+suf +'/'
    if not os.path.exists(opath_st_hr):
        os.makedirs(opath_st_hr)
    

    nfile = len(files_hr)
    ichlo = ivar_hr[0] # only work for one variable
    
    # estimate maximum var in file in all dataset, test for single var
    filename = opath_st_hr+'var%d'%ichlo+'_sorted_file'+'.npz'
    if not os.path.isfile(filename):
        var_sort,ind_sort = find_max_global_file(files_hr,var_hr,indt_hr,ll_hr[0],ll_hr[1]) # find maximum index 
        np.savez(filename, v1=var_sort,v2=ind_sort) 
        ofname = 'var%d'%ichlo+'_sorted_file'+'.csv'
        combined_ind= np.column_stack((np.array(var_sort[0]), np.array(ind_sort[0])))
        np.savetxt(opath_st_hr + ofname, combined_ind,fmt='%f,%d') # ,delimiter=","
    else:
        datald = np.load(filename) # load
        nfl = datald['v1'].size
        var_sort = datald['v1']
        ind_sort = datald['v2']

    
    rt_use = 0.01 # use top largest values of testing data for testing 
    tstr = '_rk%4.2f'%rt_use
    
    kfsort = 0 # to sort the file first in desending order, next sort in tran/test
    if kfsort==1: 
        files_lr = [files_lr[i] for i in ind_sort[0]] 
        files_hr = [files_hr[i] for i in ind_sort[0]]  # files with var from small to large
        indt_lr = [indt_lr[i] for i in ind_sort[0]]
        indt_hr = [indt_hr[i] for i in ind_sort[0]]
        suf = '_fs'
    else:
        suf = ''
    ind_train = np.arange(0,int(nfile*rtra))    # 
    ind_valid= np.delete(np.arange(0,nfile),ind_train) # note: only train&test
    nt_test = len(ind_valid) 
    nt_train = len(ind_train)
    # files_lr_test = [files_lr[i] for i in ind_valid]
    # files_hr_test = [files_hr[i] for i in ind_valid]
    # indt_lr_test = [indt_lr[i] for i in ind_valid]
    # indt_hr_test = [indt_hr[i] for i in ind_valid]

    # estimate maximum var of all samples in hr training set
    filestr = 'var%d'%ichlo+'_sorted_train'+'_rt%4.2f'%(rtra)+suf
    filename = opath_st_hr+filestr+'.npz'
    if not os.path.isfile(filename):
        files = [files_hr[i] for i in ind_train]
        indt = [indt_hr[i] for i in ind_train]
        var_sort_train,ind_sort_train = find_max_global_file(files,var_hr,indt,ll_hr[0],ll_hr[1],kdesc=True) # find maximum index 
        np.savez(filename, v1=var_sort_train,v2=ind_sort_train) 
        ofname = filestr+'.csv'
        # # combined_ind= np.column_stack((np.array(var_sort_train[0][0:int(nt_train*rt_use)]), np.array(ind_sort_train[0][0:int(nt_train*rt_use)])))
        # combined_ind= np.column_stack((np.array(var_sort_train[0]), np.array(ind_sort_train[0])))
        # np.savetxt(opath_st_hr + ofname, combined_ind,fmt='%f,%d') # ,delimiter=","
        # dictionary of lists
        files_sort = [files[i] for i in ind_sort_train[0]]  # files with var from small to large
        indt_s = [indt[i] for i in ind_sort_train[0]]
        dict = {'var':var_sort_train[0],'ind':ind_sort_train[0],'files': files_sort, 'indt': indt_s}
        df = pd.DataFrame(dict)
        # saving the dataframe
        df.to_csv(opath_st_hr + ofname)
    
    # estimate maximum var of all samples in lr training set
    filestr = 'var%d'%ichlo+'_sorted_train'+'_lr%d_rt%4.2f'%(opt.up_factor,rtra)+suf
    filename = opath_st_hr+filestr+'.npz'
    if not os.path.isfile(filename):
        files = [files_lr[i] for i in ind_train]
        indt = [indt_lr[i] for i in ind_train]
        var_sort_train,ind_sort_train = find_max_global_file(files,var_lr,indt,ll_lr[0],ll_lr[1],kdesc=True) # find maximum index 
        np.savez(filename, v1=var_sort_train,v2=ind_sort_train) 
        ofname = filestr+'.csv'
        # combined_ind= np.column_stack((np.array(var_sort_train[0][0:int(nt_train*rt_use)]), np.array(ind_sort_train[0][0:int(nt_train*rt_use)])))
        combined_ind= np.column_stack((np.array(var_sort_train[0]), np.array(ind_sort_train[0])))
        np.savetxt(opath_st_hr + ofname, combined_ind,fmt='%f,%d') # ,delimiter=","
    
    # estimate maximum var of all samples in hr testing set
    filestr = 'var%d'%ichlo+'_sorted_test'+'_rt%4.2f'%(rtra)+suf
    filename = opath_st_hr+filestr+'.npz'
    if not os.path.isfile(filename):
        files = [files_hr[i] for i in ind_valid]
        indt = [indt_hr[i] for i in ind_valid]
        var_sort,ind_sort = find_max_global_file(files,var_hr,indt,ll_hr[0],ll_hr[1],kdesc=True) # find maximum index 
        np.savez(filename, v1=var_sort,v2=ind_sort)
        ofname = filestr+'.csv'
        # # combined_ind= np.column_stack((np.array(var_sort[0][0:int(nt_test*rt_use)]), np.array(ind_sort[0][0:int(nt_test*rt_use)])))
        # combined_ind= np.column_stack((np.array(var_sort[0]), np.array(ind_sort[0])))
        # np.savetxt(opath_st_hr + ofname, combined_ind,fmt='%f,%d') # ,delimiter=","
        files_sort = [files[i] for i in ind_sort[0]]  # files with var from small to large
        indt_s = [indt[i] for i in ind_sort[0]]
        dict = {'var':var_sort[0],'ind':ind_sort[0],'files': files_sort, 'indt': indt_s}
        df = pd.DataFrame(dict)
        # saving the dataframe
        df.to_csv(opath_st_hr + ofname)
        
    
    # estimate maximum var of all samples in lr testing set
    filestr = 'var%d'%ichlo+'_sorted_test'+'_lr%d_rt%4.2f'%(opt.up_factor,rtra)+suf
    filename = opath_st_hr+filestr+'.npz'
    if not os.path.isfile(filename):
        files = [files_lr[i] for i in ind_valid]
        indt = [indt_lr[i] for i in ind_valid]
        var_sort,ind_sort = find_max_global_file(files,var_lr,indt,ll_lr[0],ll_lr[1],kdesc=True) # find maximum index 
        np.savez(filename, v1=var_sort,v2=ind_sort) 
        ofname = filestr+'.csv'
        # combined_ind= np.column_stack((np.array(var_sort[0][0:int(nt_test*rt_use)]), np.array(ind_sort[0][0:int(nt_test*rt_use)])))
        combined_ind= np.column_stack((np.array(var_sort[0]), np.array(ind_sort[0])))
        np.savetxt(opath_st_hr + ofname, combined_ind,fmt='%f,%d') # ,delimiter=","

        
        