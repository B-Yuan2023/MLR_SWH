#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 10:16:41 2024
parameters 

@author: g260218
"""
import sys
import argparse
import numpy as np
# from funs_prepost import find_maxmin_global
import glob
from datetime import datetime, timedelta # , date

# def create_parser():
parser = argparse.ArgumentParser()
parser.add_argument("--dir_lr", type=str, default="/work/gg0028/g260218/Data/cds_wave_BlackSea/wave_27.2_42.0_40.5_47.0_h0/swh", help="directory of lr dataset")
parser.add_argument("--dir_hr", type=str, default="/work/gg0028/g260218/Data/cmems_wave_BlackSea/wave_27.2_42.0_40.5_47.0_h0/swh", help="directory of hr dataset")
parser.add_argument("--up_factor", type=int, default=20, help="upscale factor") # match with dirs
parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=240, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=240, help="high res. image width")
# parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="batch interval between saving image")
parser.add_argument("--residual_blocks", type=int, default=6, help="number of residual blocks in the generator")
opt = parser.parse_args(sys.argv[2:])
print(opt)
    # return opt

krand = 1
nrep = 1
tshift = 0	# time shift in hour of low resolution data
kmod = 0

# actual time period
tlim = [datetime(2018,1,1,0,0,0),datetime(2022,1,1,0,0,0)]
#dt = 12 # usd time step in hour, should be integer times of delta_t in ncfiles
dt = 3 # usd time step in hour, should be integer times of delta_t in ncfiles. For Testing using same files!!!
nt = int((tlim[1]-tlim[0]).total_seconds()/(dt*3600)) ## total time steps 

# get files of hr, update t0_h, dt_h and ntpf according to ncfile
t0_h,dt_h = 0,1  # initial time and delta time in hour in ncfile
ntpf = 24  # number of time steps in each ncfile
t_hr = [(tlim[0] + timedelta(hours=x)) for x in range(0,nt)]
files_hr0 = sorted(glob.glob(opt.dir_hr + "/*.nc"))
files_hr0_ext = [ele for ele in files_hr0 for i in range(ntpf)]
indt_ = [i for i in range(ntpf)]
indt_hr0_ext = [ele for i in range(len(files_hr0)) for ele in indt_ ]
# files_hr = files_hr0_ext # files with repeat, length=no. samples
# indt_hr = indt_hr0_ext 
ymd0 = t_hr[0].strftime("%Y%m%d")
indt0 = int((t_hr[0].hour-t0_h)/dt_h)  # index of first time step in ncfile
indf0 =[i for i, item in enumerate(files_hr0) if ymd0 in item][0] # file index of first time step
ind_ext0 = (indf0*ntpf+indt0) + np.arange(0, nt) * (dt/dt_h)
ind_ext = [int(i) for i in ind_ext0]
files_hr = [files_hr0_ext[i] for i in ind_ext]  # files with repeat, length=no. samples
indt_hr = [indt_hr0_ext[i] for i in ind_ext]  # files with repeat, length=no. samples

# get files of lr, update t0_h, dt_h and ntpf according to ncfile
t0_h,dt_h = 0,1  # initial time and delta time in hour in ncfile
ntpf = 24  # number of time steps in each ncfile
t_lr = [(tlim[0] + timedelta(hours=x+tshift)) for x in range(0,nt)] # add time shift to lr data
files_lr0 = sorted(glob.glob(opt.dir_lr + "/*.nc"))
files_lr0_ext = [ele for ele in files_lr0 for i in range(ntpf)]
indt_ = [i for i in range(ntpf)]
indt_lr0_ext = [ele for i in range(len(files_lr0)) for ele in indt_ ]
ymd0 = t_lr[0].strftime("%Y%m%d")
indt0 = int((t_lr[0].hour-t0_h)/dt_h)  # index of first time step in ncfile
indf0 =[i for i, item in enumerate(files_lr0) if ymd0 in item][0] # file index of first time step
ind_ext0 = (indf0*ntpf+indt0) + np.arange(0, nt) * (dt/dt_h)
ind_ext = [int(i) for i in ind_ext0]
files_lr = [files_lr0_ext[i] for i in ind_ext]  # files with repeat, length=no. samples
indt_lr = [indt_lr0_ext[i] for i in ind_ext]  # files with repeat, length=no. samples


# cmems 
xmin=27.4    # lon 27.4-34 part of black sea 27.4:0.2:42
ymin=40.6    # lat 40.6:0.2:47
# low resolution coordinate
dx = 0.5 # original step in longitude lr 
dy = 0.5 # original step in latitude
nx = int(opt.hr_width/opt.up_factor) # no. of x in low resolution data
ny = int(opt.hr_height/opt.up_factor) # no. of y in low resolution data
lons_lr = xmin + np.arange(0, nx) * dx
lats_lr = ymin + np.arange(0, ny) * dy
ll_lr =[lats_lr,lons_lr]
# high resolution coordinate
dx = 0.025 # original step in longitude lr 
dy = 0.025 # original step in latitude
nx = int(opt.hr_width) # no. of x in low resolution data
ny = int(opt.hr_height) # no. of y in low resolution data
lons_hr = xmin + np.arange(0, nx) * dx
lats_hr = ymin + np.arange(0, ny) * dy
ll_hr =[lats_hr,lons_hr]

# kintp = 0  # key to interpolate to user domain, 0 no, 1 griddata, 2 RBFInterpolator

suf = '_res' + str(opt.residual_blocks) + '_v1'
suf0 = '_res' + str(opt.residual_blocks) + '_v1'

rtra = 0.75 # ratio of training dataset

var_lr = ['swh']
var_hr = ['VHM0']
# files_lr = sorted(glob.glob(opt.dir_lr + "/*.nc"))
# varm_lr,_,_ = find_maxmin_global(files_lr, ivar_lr)
# files_hr = sorted(glob.glob(opt.dir_hr + "/*.nc"))
# varm_hr,_,_ = find_maxmin_global(files_hr, ivar_hr)
# estimated based on whole dataset

varm_hr0 = np.array([[ 4.0, -4.0],
                     [ 2.6, -2.6],
                     [ 2.6, -2.6],
                     [ 22, -22],
                     [ 22, -22],
                     [ 10, 0],
                     [ 24,0]])

varm_hr = np.zeros((len(var_hr),2))
varm_lr = np.zeros((len(var_lr),2))
ivar_hr = [None]*len(var_lr)
ivar_lr = [None]*len(var_hr)
for i in range(len(var_hr)):
    if var_hr[i] in ['elevation','zos'] : # ssh 
        varm_hr[i,:] = varm_hr0[0,:]
        ivar_hr[i] = 0
    elif var_hr[i] == 'depthAverageVelX': # u
        varm_hr[i,:] = varm_hr0[1,:]
        ivar_hr[i] = 1
    elif var_hr[i] == 'depthAverageVelY': # v
        varm_hr[i,:] = varm_hr0[2,:]
        ivar_hr[i] = 2
    elif var_hr[i] == 'windSpeedX': # uw
        varm_hr[i,:] = varm_hr0[3,:]
        ivar_hr[i] = 3
    elif var_hr[i] == 'windSpeedY': # vw
        varm_hr[i,:] = varm_hr0[4,:]
        ivar_hr[i] = 4
    elif var_hr[i] in ['sigWaveHeight','VHM0','swh'] : # swh
        varm_hr[i,:] = varm_hr0[5,:]
        ivar_hr[i] = 5
    elif var_hr[i] == 'peakPeriod': # pwp
        varm_hr[i,:] = varm_hr0[6,:]
        ivar_hr[i] = 6
for i in range(len(var_lr)):
    if var_lr[i] in ['elevation','zos']: # ssh 
        varm_lr[i,:] = varm_hr0[0,:]
        ivar_lr[i] = 0
    elif var_lr[i] == 'depthAverageVelX': # u
        varm_lr[i,:] = varm_hr0[1,:]
        ivar_lr[i] = 1
    elif var_lr[i] == 'depthAverageVelY': # v
        varm_lr[i,:] = varm_hr0[2,:]
        ivar_lr[i] = 2
    elif var_lr[i] == 'windSpeedX': # uw
        varm_lr[i,:] = varm_hr0[3,:]
        ivar_lr[i] = 3
    elif var_lr[i] == 'windSpeedY': # vw
        varm_lr[i,:] = varm_hr0[4,:]
        ivar_lr[i] = 4
    elif var_lr[i] in ['sigWaveHeight','VHM0','swh']: # swh
        varm_lr[i,:] = varm_hr0[5,:]
        ivar_lr[i] = 5
    elif var_lr[i] == 'peakPeriod': # pwp
        varm_lr[i,:] = varm_hr0[6,:]
        ivar_lr[i] = 6
# varm = varm_hr

# "VMDR","VTM02" wave direction and mean period
