#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 07:15:24 2024

pre-post functions for data

@author: g260218
"""

import numpy as np
from matplotlib import pyplot as plt
import netCDF4 

# make lists for files and the corresponding used time step, with repeated file names
import glob
from datetime import datetime, timedelta # , date
def make_list_file_t(dir_file,tlim,dt,tshift=0,t0_h=0,dt_h=1,ntpf=24):
    nt = int((tlim[1]-tlim[0]).total_seconds()/(dt*3600)) ## total time steps 
    # dir_file: directory of the data files
    # tlim: used time range, e.g., [datetime(2018,1,1,0,0,0),datetime(2022,1,1,0,0,0)]
    # dt: usd time step in hour, should be integer times of original time step in ncfiles
    # t0_h,dt_h = 0,1  # initial time and delta time (time step) in hour in ncfile
    # ntpf = 24  # number of time steps in each ncfile
    t_s = [(tlim[0] + timedelta(hours=x+tshift)) for x in range(0,nt)] # time list with time shift
    files0 = sorted(glob.glob(dir_file + "/*.nc"))  # all files in dir
    files0_ext = [ele for ele in files0 for i in range(ntpf)]  # list of files with repeat base on ntpf
    indt_ = [i for i in range(ntpf)]  # list of time steps in one file 
    indt0_ext = [ele for i in range(len(files0)) for ele in indt_ ]  # list of time steps in all files 
    ymd0 = t_s[0].strftime("%Y%m%d")
    indt0 = int((t_s[0].hour-t0_h)/dt_h)  # index of used inital time in an ncfile
    indf0 =[i for i, item in enumerate(files0) if ymd0 in item][0] # file index of first time step, file name e.g. 20200102.nc
    ind_ext0 = (indf0*ntpf+indt0) + np.arange(0, nt) * (dt/dt_h)  # index of used times in indt0_ext or files0_ext
    ind_ext = [int(i) for i in ind_ext0]  # change array to list 
    files = [files0_ext[i] for i in ind_ext]  # files with repeat, length=no. samples
    indt = [indt0_ext[i] for i in ind_ext]  # selected time indexs in all files, length=no. samples
    return files, indt


# read one var per file per time
def nc_load_vars(nc_f,varname,indt=None,lats=None,lons=None,kintp=0,method='linear'):
    # nc_f: string, nc file name
    # varname: string, variable name in ncfile, 1 var
    # indt: list, time steps in nc_f
    # import glob
    # nc_f = sorted(glob.glob(dir_sub + "/*"+ymd+"*.nc"))[0] # use the file contain string ymd
    # print(f'filename:{nc_f}')
    nc_fid = netCDF4.Dataset(nc_f, 'r')  # class Dataset: open ncfile, create ncCDF4 class
    # nc_fid.variables.keys() # 
    ncvar = list(nc_fid.variables)
    lon_var = [i for i in ncvar if 'lon' in i][0] # only work if only one var contain lon
    lat_var = [i for i in ncvar if 'lat' in i][0]
    # print(nc_fid)
    # Extract data from NetCDF file
    lon = nc_fid.variables[lon_var][:]  # extract/copy the data
    lat = nc_fid.variables[lat_var][:]
    time = nc_fid.variables['time'][:]
    # dt_h = (time[1]-time[0])/3600  #  change time step in second to hour 
    if indt is None:
        indt = np.arange(0,len(time))  # read all times
    # else:
    #     indt = [int(nc_t[i]/dt_h) for i in range(len(nc_t))]  # indt is a list here, gives var[nt,Ny,nx] even nt=1

    var = nc_fid.variables[varname][indt,:]  # shape is time, Ny*Nx
    nc_fid.close()
    mask = np.ma.getmask(var)
    FillValue=0.0 # np.nan
    data = var.filled(fill_value=FillValue)
    data = np.ma.getdata(data) # data of masked array
    if mask.size == 1: # in case all data are available
        mask = data!=data

    # use user domain when lats,lons are specified
    if lats is not None and lons is not None:
        if kintp==0: # no interpolation, only select user domain, use original coordinate
            # Find indices of x_s and y_s in x and y arrays
            ind_x = np.array([np.argmin(np.abs(lon-lons[i])) for i in range(len(lons))])# np.searchsorted(lon, lons)
            ind_y = np.array([np.argmin(np.abs(lat-lats[i])) for i in range(len(lats))])
            if len(data.shape)==2: # only one time step
                data = data[ind_y[:, np.newaxis],ind_x[np.newaxis, :]]
                mask = mask[ind_y[:, np.newaxis],ind_x[np.newaxis, :]]
            else:
                data = data[:, ind_y[:, np.newaxis], ind_x[np.newaxis, :]]
                mask = mask[:, ind_y[:, np.newaxis], ind_x[np.newaxis, :]]
            lon,lat = lon[ind_x],lat[ind_y]
        else:  # interpolated to user domain, use new coordinate 
            data[mask] = np.nan 
            data = interpolate_array(data,lon,lat,lons,lats, kintp=kintp, method=method)
            mask = np.isnan(data)
            data = np.nan_to_num(data,nan=0)
            lon,lat = lons,lats
    return time,lon,lat,data,mask


# normalize the data from ncfile 
def nc_normalize_vars(nc_f,varname,indt,varmaxmin=None,lats=None,lons=None,kintp=0,method='linear'):
    # output: (H,W,C)
    # nc_f: list of nc file name, length = no. of varname
    # varname: list, variable name in ncfile
    # indt: list, time index in nc_f, length = no. of varname
    nvar = len(varname)
    Nx = len(nc_load_vars(nc_f[0],varname[0],indt[0],lats,lons)[1])
    Ny = len(nc_load_vars(nc_f[0],varname[0],indt[0],lats,lons)[2])
    data = np.zeros(shape=(Ny,Nx,nvar))
    for i in range(nvar):
        # if len(nc_f) == 1:
        #     nc_fi,indti = nc_f[0],indt[0]
        # else:
        #     nc_fi,indti = nc_f[i],indt[i]

        var = nc_load_vars(nc_f[i],varname[i],[indt[i]],lats,lons,kintp,method)[3] # (NT,H,W) one channel
        
        # data = np.squeeze(data[indt,:,:])  # (Ny,Nx), lat,lon
        temp = np.flip(var,axis=1) # original data first row -> lowest latitude
        # convert data to [0,1]
        if varmaxmin is None:
            vmax = temp.max()
            vmin = temp.min()
        else:
            vmax = varmaxmin[i,0]
            vmin = varmaxmin[i,1]
        data[:,:,i] = (temp - vmin)/(vmax-vmin) # convert to [0,1]
    return data 


# denormalize 
def var_denormalize(var,varmaxmin):
    # var(N,C,H,W)
    nc = var.shape[1]
    var = np.flip(var,2) # flip the dimenson of height as in narmalize
    data = np.zeros(shape=var.shape)
    for i in range(nc):
        vmax = varmaxmin[i,0]
        vmin = varmaxmin[i,1]
        data[:,i,:,:] = var[:,i,:,:]*(vmax-vmin) + vmin 
    return data 


def nc_load_depth(nc_f,lats=None,lons=None,kintp=0,method='linear'):
    nc_fid = netCDF4.Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file
    # nc_fid.variables.keys() # list(nc_fid.variables)
    ncvar = list(nc_fid.variables) # 'depth' in schism, 'deptho' in cmems
    dep_var = [i for i in ncvar if 'dep' in i][0] # only work if only one var contain dep
    
    lon = nc_fid.variables['longitude'][:]  # extract/copy the data
    lat = nc_fid.variables['latitude'][:]
    depth = nc_fid.variables[dep_var] # shape is Ny*Nx
    mask = np.ma.getmask(depth)
    FillValue=0.0 # np.nan
    # depth = depth.filled(fill_value=FillValue)
    lon = np.ma.getdata(lon) # data of masked array
    lat = np.ma.getdata(lat) # data of masked array
    temp = np.ma.getdata(depth) # data of masked array
    vmax = temp.max()
    vmin = temp.min()
    data = np.ma.getdata(temp) # strange, why need twice to get data
    data[np.isnan(data)] = FillValue
    if mask.size == 1: # in case all data are available
        mask = data!=data
    
    # use user domain when lats,lons are specified
    if lats is not None and lons is not None:
        if kintp==0: # no interpolation, only select user domain, use original coordinate
            # Find indices of x_s and y_s in x and y arrays
            ind_x = np.array([np.argmin(np.abs(lon-lons[i])) for i in range(len(lons))])# np.searchsorted(lon, lons)
            ind_y = np.array([np.argmin(np.abs(lat-lats[i])) for i in range(len(lats))])
            if len(data.shape)==2: # only one time step
                data = data[ind_y[:, np.newaxis],ind_x[np.newaxis, :]]
                mask = mask[ind_y[:, np.newaxis],ind_x[np.newaxis, :]]
            else:
                data = data[:, ind_y[:, np.newaxis], ind_x[np.newaxis, :]]
                mask = mask[:, ind_y[:, np.newaxis], ind_x[np.newaxis, :]]
            lon,lat = lon[ind_x],lat[ind_y]
        else:  # interpolated to user domain, use new coordinate 
            data[mask] = np.nan 
            data = interpolate_array(data,lon,lat,lons,lats, kintp=kintp, method=method)
            mask = np.isnan(data)
            data = np.nan_to_num(data,nan=0)
            lon,lat = lons,lats
    
    depth_nm = (data - vmin)/(vmax-vmin)
    depth_nm = np.flipud(depth_nm) # flipped normalized depth refer to mean sea level
    return depth,lon,lat,mask,depth_nm


# functions for read variables in nc files from schism output
def nc_load_all(nc_f,indt=None):
    nc_fid = netCDF4.Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file
     # and create an instance of the ncCDF4 class
    # nc_fid.variables.keys() # list(nc_fid.variables)
    # print(nc_fid)
    # Extract data from NetCDF file
    lon = nc_fid.variables['longitude'][:]  # extract/copy the data
    lat = nc_fid.variables['latitude'][:]
    time = nc_fid.variables['time'][:]
    if indt is None:
        indt = np.arange(0,len(time))
    ssh = nc_fid.variables['elevation'][indt,:]  # shape is time, Ny*Nx
    uwind = nc_fid.variables['windSpeedX'][indt,:]  # shape is time, Ny*Nx
    vwind = nc_fid.variables['windSpeedY'][indt,:]  # shape is time, Ny*Nx
    swh = nc_fid.variables['sigWaveHeight'][indt,:]  # shape is time, Ny*Nx
    pwp = nc_fid.variables['peakPeriod'][indt,:]  # shape is time, Ny*Nx
    ud = nc_fid.variables['depthAverageVelX'][indt,:]  # shape is time, Ny*Nx
    vd = nc_fid.variables['depthAverageVelY'][indt,:]  # shape is time, Ny*Nx
    nc_fid.close()
    
    mask = np.ma.getmask(ssh)
    
    FillValue=0.0 # np.nan
    ssh = ssh.filled(fill_value=FillValue)
    uwind = uwind.filled(fill_value=FillValue)
    vwind = vwind.filled(fill_value=FillValue)
    swh = swh.filled(fill_value=FillValue)
    pwp = pwp.filled(fill_value=FillValue)
    ud = ud.filled(fill_value=FillValue)
    vd = vd.filled(fill_value=FillValue)
    
    ssh = np.ma.getdata(ssh) # data of masked array
    uw = np.ma.getdata(uwind) # data of masked array
    vw = np.ma.getdata(vwind) # data of masked array
    swh = np.ma.getdata(swh) # data of masked array
    pwp = np.ma.getdata(pwp) # data of masked array
    ud = np.ma.getdata(ud) # data of masked array
    vd = np.ma.getdata(vd) # data of masked array
    return time,lon,lat,ssh,ud,vd,uw,vw,swh,pwp,mask


# instance or dataset normalization of schims output
# ivar = [3,4,5] # ssh, ud, vd
def nc_var_normalize(nc_f,indt,ivar,varmaxmin=None):
    nvar = len(ivar)
    Nx = len(nc_load_all(nc_f,indt)[1])
    Ny = len(nc_load_all(nc_f,indt)[2])
    data = np.zeros(shape=(Ny,Nx,nvar))
    for i in range(nvar):
        var = nc_load_all(nc_f,indt)[ivar[i]]
        # data = np.squeeze(data[indt,:,:])  # (Ny,Nx), lat,lon
        temp = np.flipud(var) # original data first row -> lowest latitude
        # convert data to [0,1]
        if varmaxmin is None:
            vmax = temp.max()
            vmin = temp.min()
        else:
            vmax = varmaxmin[i,0]
            vmin = varmaxmin[i,1]
                
        data[:,:,i] = (temp - vmin)/(vmax-vmin) # convert to [0,1]
        #data = np.array(data).reshape(data.shape[0],data.shape[1],1) # height, width, channel (top to bot)
    # data = np.dstack(data)
    # if nvar==1:
    #     data = np.repeat(data[..., np.newaxis], 3, -1)  # make 1 channel to 3 channels for later interpolation and trained model like vgg19
    return data 


# inerpolation 
import torch
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


from scipy.interpolate import griddata, interp2d, RBFInterpolator
# griddata: no extropolation; RBFInterpolator: with extropolation; interp2d not suggested
def interpolate_array(array_in,x_in,y_in,x_out,y_out, kintp=1, method='linear', **kwargs):
    """
    Interpolates the last two dimensions of an array using the specified interpolation function.
    
    Parameters:
        array_in (ndarray): Input array of shape (C, H, W).
        x_in,y_in,x_out,y_out: input and output coordinates of x,y
        kintp (function, optional): Interpolation function to use.
              1 Default is griddata from scipy.interpolate, 2 RBFInterpolator
        method (str, optional): Interpolation method to use ('linear', 'nearest', 'cubic').for griddata
                                 Default is 'linear'.
        kwargs: Additional keyword arguments to be passed to the interpolation function.
        
    Returns:
        ndarray: Interpolated array of shape (C, new_H, new_W).
    """
    C, H, W = array_in.shape
    new_H, new_W = len(y_out),len(x_out)
    array_out = np.zeros((C, new_H, new_W))
    
    # Create 2D grids for input data
    X, Y = np.meshgrid(x_in, y_in)

    # Create new 2D grids for interpolated domain
    new_X, new_Y = np.meshgrid(x_out, y_out)
    
    for c in range(C):
        # Flatten the original grid
        points = np.column_stack((X.flatten(), Y.flatten()))
        values = array_in[c].flatten()
        
        # Interpolate using the specified interpolation function
        if kintp==1:
            array_out[c] = griddata(points, values, (new_X, new_Y), method=method, **kwargs)
        elif kintp==2:
            values[np.isnan(values)] = 0
            new_points = np.stack([new_X.ravel(), new_Y.ravel()], -1)  # shape (N, 2) in 2d
            array_out[c] = RBFInterpolator(points, values, kernel=method, **kwargs)(new_points).reshape(new_X.shape)
    return array_out


def interpolate_array_scale(array_in,scale_factor,kintp=griddata, method='linear', **kwargs):
    """
    Interpolates the last two dimensions of an array using the specified interpolation function.
    
    Parameters:
        array_in (ndarray): Input array of shape (C, H, W).
        scale_factor (float): Scale factor for the last two dimensions. The new dimensions will be
                              original_dimensions * scale_factor.
        kintp (function, optional): Interpolation function to use.
                                                     Default is griddata from scipy.interpolate.
        method (str, optional): Interpolation method to use ('linear', 'nearest', 'cubic').
                                 Default is 'linear'.
        kwargs: Additional keyword arguments to be passed to the interpolation function.
        
    Returns:
        ndarray: Interpolated array of shape (N, C, new_H, new_W).
    """
    C, H, W = array_in.shape
    new_H, new_W = int(H * scale_factor), int(W * scale_factor)
    array_out = np.zeros((C, new_H, new_W))
    
    # Create 2D grids for interpolation
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    X, Y = np.meshgrid(x, y)
    
    # Create new 2D grids for interpolated domain
    new_x = np.linspace(0, W - 1, new_W)
    new_y = np.linspace(0, H - 1, new_H)
    new_X, new_Y = np.meshgrid(new_x, new_y)
    
    for c in range(C):
        # Flatten the original grid
        points = np.column_stack((X.flatten(), Y.flatten()))
        values = array_in[c].flatten()
        
        # Interpolate using the specified interpolation function
        interpolated = kintp(points, values, (new_X, new_Y), method=method, **kwargs)
        array_out[c] = interpolated
    return array_out


# functions for read data in nc files from cmems an interpolate when necessary
# ivar = [3,4,5] # ssh, ud, vd
# varname = ["zos","uo","vo"] # varname from cmems
# ymd: e.g.'20170101', year month day string of the file to read
def nc_load_cmems(dir_sub,ymd,varname,indt=None,lats=None,lons=None,kintp=0,method='linear'):
    import glob
    nc_f = sorted(glob.glob(dir_sub + "/*"+ymd+"*.nc"))[0] # use the file contain string ymd
    # print(f'filename:{nc_f}')
    nc_fid = netCDF4.Dataset(nc_f, 'r')  # class Dataset: open ncfile, create ncCDF4 class
    # nc_fid.variables.keys() # list(nc_fid.variables)
    # print(nc_fid)
    # Extract data from NetCDF file
    lon = nc_fid.variables['longitude'][:]  # extract/copy the data
    lat = nc_fid.variables['latitude'][:]
    time = nc_fid.variables['time'][:]
    if indt is None:
        indt = np.arange(0,len(time))  # read all times
    var = nc_fid.variables[varname][indt,:]  # shape is time, Ny*Nx
    nc_fid.close()
    mask = np.ma.getmask(var)
    FillValue=0.0 # np.nan
    data = var.filled(fill_value=FillValue)
    data = np.ma.getdata(data) # data of masked array

    # use user domain when lats,lons are specified
    if lats is not None and lons is not None:
        if kintp==0: # no interpolation, only select user domain, use original coordinate
            # Find indices of x_s and y_s in x and y arrays
            ind_x = np.array([np.argmin(np.abs(lon-lons[i])) for i in range(len(lons))])# np.searchsorted(lon, lons)
            ind_y = np.array([np.argmin(np.abs(lat-lats[i])) for i in range(len(lats))])
            if len(data.shape)==2: # only one time step
                data = data[ind_y[:, np.newaxis],ind_x[np.newaxis, :]]
                mask = mask[ind_y[:, np.newaxis],ind_x[np.newaxis, :]]
            else:
                data = data[:, ind_y[:, np.newaxis], ind_x[np.newaxis, :]]
                mask = mask[:, ind_y[:, np.newaxis], ind_x[np.newaxis, :]]
            lon,lat = lon[ind_x],lat[ind_y]
        else:  # interpolated to user domain, use new coordinate 
            data[mask] = np.nan 
            data = interpolate_array(data,lon,lat,lons,lats, kintp=kintp, method=method)
            mask = np.isnan(data)
            data = np.nan_to_num(data,nan=0)
            lon,lat = lons,lats
    return time,lon,lat,data,mask

# functions for read data in nc files from cmems
# ivar = [3,4,5] # ssh, ud, vd
# varname = ["zos","uo","vo"] # varname from cmems
# ymd: e.g.'20170101', year month day string of the file to read
def nc_load_cmems0(dir_sub,ymd,varname,indt=None):
    import glob
    # subdir = ["ssh","u","v"] # subdir to for vars from cmems
    # indf: index of the file to be read
    # varname = ["zos","uo","vo"] # varname from cmems
    # index of the time in a file to be read
    # files = sorted(glob.glob(dir_sub + "/*.nc")) # in this way, each file links to one file from schism
    # nc_f = files[indf]
    # nfiles = len(files)
    # print(f'dir_sub:{dir_sub},indf:{indf},nfile:{nfiles}\n')
    nc_f = sorted(glob.glob(dir_sub + "/*"+ymd+"*.nc"))[0] # use the file contain string ymd
    nc_fid = netCDF4.Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file
     # and create an instance of the ncCDF4 class
    # nc_fid.variables.keys() # list(nc_fid.variables)
    # print(nc_fid)
    # Extract data from NetCDF file
    lon = nc_fid.variables['longitude'][:]  # extract/copy the data
    lat = nc_fid.variables['latitude'][:]
    time = nc_fid.variables['time'][:]
    if indt is None:
        indt = np.arange(0,len(time))
    var = nc_fid.variables[varname][indt,:]  # shape is time, Ny*Nx
    nc_fid.close()
    
    mask = np.ma.getmask(var)
    FillValue=0.0 # np.nan
    data = var.filled(fill_value=FillValue)
    data = np.ma.getdata(data) # data of masked array
    return time,lon,lat,data,mask

# normalize the data from cmems
# ivar = [3,4,5] # ssh, ud, vd
def nc_var_normalize_cmems(dir_fl,ymd,ivar,indt,varmaxmin=None,lats=None,lons=None,kintp=0,method='linear'):
    # output: (H,W,C)
    # indt: index of time, should have length of 1 when call this function
    
    varname = ["zos","uo","vo"] # varname from cmems
    subdir = ["ssh","u","v"] # subdir to save each var
    nvar = len(ivar)
    dir_sub = dir_fl + '/'+ subdir[0]
    Nx = len(nc_load_cmems(dir_sub,ymd,varname[0],indt,lats,lons)[1])
    Ny = len(nc_load_cmems(dir_sub,ymd,varname[0],indt,lats,lons)[2])
    data = np.zeros(shape=(Ny,Nx,nvar))
    for i in range(nvar):
        ichl = ivar[i]-3
        dir_sub = dir_fl + '/'+ subdir[ichl]
        var = nc_load_cmems(dir_sub,ymd,varname[ichl],indt,lats,lons,kintp,method)[3] # (NT,H,W) one channel
        
        # data = np.squeeze(data[indt,:,:])  # (Ny,Nx), lat,lon
        temp = np.flipud(var) # original data first row -> lowest latitude
        # convert data to [0,1]
        if varmaxmin is None:
            vmax = temp.max()
            vmin = temp.min()
        else:
            vmax = varmaxmin[i,0]
            vmin = varmaxmin[i,1]
        data[:,:,i] = (temp - vmin)/(vmax-vmin) # convert to [0,1]
    return data 


# find max and min of variable in the files
def find_maxmin_global(files, ivar=[3,3,3]):
    # files = sorted(glob.glob(dirname + "/*.nc"))
    # nfile = len(files)
    # files = files[:int(nfile*rtra)]
    file_varm = [] 
    ind_varm = np.ones((len(ivar),2),dtype= np.int64)
    varmaxmin = np.ones((len(ivar),2))
    varmaxmin[:,0] *= -10e6 # maximum 
    varmaxmin[:,1] *= 10e6 # minimum 
    for i in range(len(ivar)):
        for indf in range(len(files)):
            nc_f = files[indf]
            var = nc_load_all(nc_f)[ivar[i]]
            if varmaxmin[i,0]<var.max():
                varmaxmin[i,0] = var.max()
                ind_varm[i,0] = np.argmax(var)
                file_max = nc_f
            if varmaxmin[i,1]>var.min():
                varmaxmin[i,1] = var.min()
                ind_varm[i,1] = np.argmin(var)
                file_min = nc_f
            # varmaxmin[i,0] = max(varmaxmin[i,0],var.max())
            # varmaxmin[i,1] = min(varmaxmin[i,1],var.min())
        file_varm.append([file_max,file_min])
    return varmaxmin,ind_varm,file_varm


# sorted var in hour
ntpd = 24 # number of time steps in an nc file
def find_max_global(files, ivar=[3]):
    nfile = len(files)
    nvar = len(ivar)
    ind_sort = [[]]*nvar
    var_sort = [[]]*nvar
    # ind_file = [[]]*nvar
    # ind_it = [[]]*nvar
    for i in range(nvar):
        var_comb = []
        # var_file = []
        # var_it = []
        for indf in range(nfile):
            nc_f = files[indf]
            var = nc_load_all(nc_f)[ivar[i]]
            var_max = var.max(axis=(1,2)) # maximum in 2d space, note during ebb sl can be <0
            for indt in range(ntpd):
                var_comb.append(var_max[indt])
                # var_file.append(indf) # the indf th file, not file name index
                # var_it.append(indt)
        ind_sort[i] = sorted(range(len(var_comb)), key=lambda k: var_comb[k], reverse=True)
        var_sort[i] = [var_comb[k] for k in ind_sort[i]]
        # var_sort[i] = sorted(var_comb)
        # ind_file[i] = [var_file[k] for k in ind_sort[i]]
        # ind_it[i] = [var_it[k] for k in ind_sort[i]]
    return var_sort,ind_sort #,ind_file,ind_it


def plt_sub(sample,ncol=1,figname=None,ichan=0,clim=None,cmp='bwr',contl=None):  
    # sample: array, normalized sample(nk,1,nx,ny)
    nsub = len(sample)
    columns = ncol
    rows = int(-(-(nsub/columns))//1)
    fig = plt.figure()
    for i in range(0,nsub):
        fig.add_subplot(rows, columns, i+1)        
        plt.imshow(sample[i,ichan,:,:],cmap=cmp) # bwr,coolwarm
        plt.axis('off')
        if clim:
            plt.clim(clim[0],clim[1]) 
        plt.tight_layout()
        #plt.title("First")
        if contl is not None: # add 0 contour
            plt.contour(sample[i,ichan,:,:], levels=contl, colors='black', linewidths=1)
    if figname:
        plt.savefig(figname,bbox_inches='tight',dpi=300) #,dpi=100    
        plt.close(fig)
    else:
        plt.show()
    
# def plt_contour_list(lat,lon,sample,figname,lev=11,cmap='bwr',clim=None,unit=None,title=None):  # sample is a list with k array[C,nx,ny]
#     nsub = len(sample)
#     fig, axes = plt.subplots(1, nsub, figsize=(5 * nsub, 5))
#     for i in range(0,nsub):
#         ax = axes[i] if nsub > 1 else axes
#         # ax.set_facecolor('xkcd:gray')
#         if clim:
#             vmin, vmax = clim[i]
#             cf = ax.contourf(lat,lon,sample[i],levels=np.linspace(vmin, vmax, lev),cmap=cmap)
#             # cf = ax.contourf(lat,lon,sample[i],levels=np.linspace(vmin, vmax, lev),cmap=cmap)
#         else:
#             cf = ax.contourf(lat,lon,sample[i],levels=lev,cmap=cmap) # bwr,coolwarm
#         cbar = fig.colorbar(cf, ax=ax)
#         ax.set_title(title[i] if title else f'Array {i + 1}')
#         if unit:
#             cbar.set_label(unit[i])
#         ax.set_xlabel('lon',fontsize=16)
#         ax.set_ylabel('lat',fontsize=16)
#         plt.tight_layout()
#     plt.savefig(figname,dpi=100) #,dpi=100    
#     plt.close(fig)
    
    
def plt_contour_list(lat,lon,sample,figname,lev=20,subsize = [5,4],cmap='bwr',clim=None,unit=None,
                    title=None,nrow=1,axoff=0,capt=None,txt=None,loc_txt=None):  # sample is a list with k array[C,nx,ny]
    import matplotlib.transforms as mtransforms    
    nsub = len(sample)
    ncol = int(nsub/nrow+0.5)
    cm = 1/2.54  # centimeters in inches
    fig, axes = plt.subplots(nrow, ncol, figsize=(subsize[0]*ncol, subsize[1]*nrow)) # default unit inch 2.54 cm
    size_tick = 16
    size_label = 18
    size_title = 18
    axes = axes.flatten()
    irm_ax = np.delete(np.arange(nrow*ncol),np.arange(nsub))
    if irm_ax is not None: # remove empty axis
        for i in range(len(irm_ax)):
            fig.delaxes(axes[irm_ax[i]])
    for i in range(0,nsub):
        ax = axes[i] if nsub > 1 else axes
        # ax.set_facecolor('xkcd:gray')
        if clim:
            vmin, vmax = clim[i]
            cf = ax.contourf(lat,lon,sample[i],levels=np.linspace(vmin, vmax, lev),cmap=cmap)
        else:
            cf = ax.contourf(lat,lon,sample[i],levels=lev,cmap=cmap) # bwr,coolwarm
        cbar = fig.colorbar(cf, ax=ax)
        cbar.ax.tick_params(labelsize=size_tick)
        ax.set_title(title[i] if title else f'Array {i + 1}',fontsize=size_title)
        if unit:
            cbar.set_label(unit[i],fontsize=size_tick+1)
        if not axoff: # keep axes or not 
            ax.set_xlabel('lon',fontsize=size_label)
            ax.set_ylabel('lat',fontsize=size_label)
            ax.tick_params(axis="both", labelsize=size_tick-1) 
        # plt.xticks(fontsize=size_tick)
        # plt.yticks(fontsize=size_tick)
        else:
            ax.axis('off')
        if txt is not None: 
            plt.text(loc_txt[0],loc_txt[1], txt[i],fontsize=size_tick,ha='left', va='top', transform=ax.transAxes) #add text
        if capt is not None: 
            trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans) # add shift in txt
            plt.text(0.00, 1.00, capt[i],fontsize=size_label,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption
        plt.tight_layout()

    plt.savefig(figname,bbox_inches='tight',dpi=300) #,dpi=100    
    plt.close(fig)    
    
    
def plt_pcolor_list(lat,lon,sample,figname,subsize = [5,4],cmap='bwr',clim=None,unit=None,
                    title=None,nrow=1,axoff=0,capt=None,txt=None,loc_txt=None,xlim=None,ylim=None):  
    # sample is a list with k array[nx,ny]
    import matplotlib.transforms as mtransforms    
    nsub = len(sample)
    ncol = int(nsub/nrow+0.5)
    cm = 1/2.54  # centimeters in inches
    fig, axes = plt.subplots(nrow, ncol, figsize=(subsize[0]*ncol, subsize[1]*nrow)) # default unit inch 2.54 cm
    size_tick = 16
    size_label = 18
    size_title = 18
    axes = axes.flatten()
    irm_ax = np.delete(np.arange(nrow*ncol),np.arange(nsub))
    if irm_ax is not None: # remove empty axis
        for i in range(len(irm_ax)):
            fig.delaxes(axes[irm_ax[i]])
    for i in range(0,nsub):
        ax = axes[i] if nsub > 1 else axes
        # ax.set_facecolor('xkcd:gray')
        if clim:
            vmin, vmax = clim[i]
            cf = ax.pcolor(lat, lon, sample[i], cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            cf = ax.pcolor(lat, lon, sample[i], cmap=cmap)
        cbar = fig.colorbar(cf, ax=ax)
        cbar.ax.tick_params(labelsize=size_tick)
        ax.set_title(title[i] if title else f'Array {i + 1}',fontsize=size_title)
        if unit:
            cbar.set_label(unit[i],fontsize=size_tick+1)
        if not axoff: # keep axes or not 
            ax.set_xlabel('lon',fontsize=size_label)
            ax.set_ylabel('lat',fontsize=size_label)
            ax.tick_params(axis="both", labelsize=size_tick-1) 
        # plt.xticks(fontsize=size_tick)
        # plt.yticks(fontsize=size_tick)
        else:
            ax.axis('off')
        if txt is not None: 
            plt.text(loc_txt[0],loc_txt[1], txt[i],fontsize=size_tick,ha='left', va='top', transform=ax.transAxes) #add text
        if capt is not None: 
            trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
            plt.text(0.00, 1.00, capt[i],fontsize=size_label,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        plt.tight_layout()

    plt.savefig(figname,bbox_inches='tight',dpi=300) #,dpi=100    
    plt.close(fig)


def plt_pcolorbar_list(lat,lon,sample,figname,subsize = [2.5,2],fontsize=12,
                       cmap='bwr',clim=None,kbar=0,unit=None,title=None,nrow=1,
                       axoff=0,capt=None,txt=None,loc_txt=None,xlim=None,ylim=None):  
    # sample is a list with k array[nx,ny], or array [Nt,H,W]
    # kbar: control colorbar appearance
    import matplotlib.transforms as mtransforms    
    nsub = len(sample)
    ncol = int(nsub/nrow+0.5)
    cm = 1/2.54  # centimeters in inches
    # figure layout,layout="constrained"
    fig, axs = plt.subplots(nrow, ncol,layout="constrained", figsize=(subsize[0]*ncol, subsize[1]*nrow)) # default unit inch 2.54 cm
    
    # size_tick,size_label,size_title = 16,18,18 # subsize = [5,4]
    size_tick,size_label,size_title = 10,12,12 # subsize = [2.5,2]
    size_tick,size_label,size_title = fontsize-2,fontsize,fontsize

    axs1 = axs.flatten()
    irm_ax = np.delete(np.arange(nrow*ncol),np.arange(nsub))
    cf_a = []
    if irm_ax is not None: # remove empty axis
        for i in range(len(irm_ax)):
            fig.delaxes(axs1[irm_ax[i]])
    for i in range(0,nsub):
        ax = axs1[i] if nsub > 1 else axs
        # ax.set_facecolor('xkcd:gray')
        if clim:
            vmin, vmax = clim[i]
            cf = ax.pcolormesh(lat, lon, sample[i], cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            cf = ax.pcolormesh(lat, lon, sample[i], cmap=cmap)
        cf_a.append(cf)
        ax.set_title(title[i] if title else f'Array {i + 1}',fontsize=size_title)

        if not axoff: # keep axes or not 
            ax.set_xlabel('lon',fontsize=size_label)
            ax.set_ylabel('lat',fontsize=size_label)
            ax.tick_params(axis="both", labelsize=size_tick-1) 
        # plt.xticks(fontsize=size_tick)
        # plt.yticks(fontsize=size_tick)
        else:
            ax.axis('off')
        if txt is not None:  # adding text, Dont use plt.text that modify subfigs!
            if len(txt)==1:
                ax.text(loc_txt[0],loc_txt[1], txt[i],fontsize=size_tick,ha='left', va='top', transform=ax.transAxes) #add text
            else:
                ax.text(loc_txt[i][0],loc_txt[i][1], txt[i],fontsize=size_tick,ha='center', va='center') 
        if capt is not None: 
            # trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
            # ax.text(0.06, 1.00, capt[i],fontsize=size_label,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption
            ax.text(0.01, 1.05, capt[i],fontsize=size_label,ha='left', va='top', transform=ax.transAxes) #add fig caption
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        # plt.tight_layout()  # conflict with layout constrained

    # plot colorbar 
    if kbar == 0:  # plot for every subplot
        for i in range(0,nsub):
            cbar = fig.colorbar(cf_a[i], ax=axs1[i])
            cbar.ax.tick_params(labelsize=size_tick)
            if unit:
                cbar.set_label(unit[i],fontsize=size_tick+1)
    elif kbar==1:  # plot 1 colorbar for each row, on the right, if no constrained layout, subfig size differs
        for i in range(0,nrow):
            cbar = fig.colorbar(cf_a[i*nrow+ncol-1], ax=[axs[i,-1]],location='right') # , shrink=0.6
            cbar.ax.tick_params(labelsize=size_tick)
            if unit:
                cbar.set_label(unit[i],fontsize=size_tick+1)
    elif kbar==2:  # plot 1 colorbar for each colume, on the bottom 
        for i in range(0,ncol):
            cbar = fig.colorbar(cf_a[i], ax=[axs[-1,i]],location='bottom')
            cbar.ax.tick_params(labelsize=size_tick)
            if unit:
                cbar.set_label(unit[i],fontsize=size_tick+1)
    elif kbar==3:  # plot 1 colorbar for all rows, on the right, implicit
        cbar = fig.colorbar(cf, ax=axs[:,-1],location='right', shrink=0.6) # , shrink=0.6
    elif kbar==4:  # plot 1 colorbar for all columes, on the bottom 
        cbar = fig.colorbar(cf, ax=axs[-1,:],location='bottom', shrink=0.6)        
    elif kbar==5:  # plot 1 colorbar for all rows, on the right, explicit 
        fig.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.93,
                            wspace=0.02, hspace=0.06)
        # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with width 0.02 and height 0.8
        cb_ax = fig.add_axes([0.95, 0.1, 0.01, 0.8])
        cbar = fig.colorbar(cf, cax=cb_ax, shrink=0.6)
    elif kbar==6:  # plot 4 colorbar for all columes, on the bottom, explicit 
        fig.subplots_adjust(bottom=0.1, top=0.8, left=0.1, right=0.9,
                            wspace=0.02, hspace=0.02)
        # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with width 0.02 and height 0.8
        cb_ax = fig.add_axes([0.1, 0.83, 0.8, 0.02])
        cbar = fig.colorbar(cf, cax=cb_ax)
    if kbar in [3,4,5,6]:
        cbar.ax.tick_params(labelsize=size_tick)
        if unit:
            cbar.set_label(unit[i],fontsize=size_tick+1)          
            
    plt.savefig(figname,bbox_inches='tight',dpi=300) #,dpi=100    
    # plt.savefig(figname,dpi=300) #,dpi=100    
    plt.close(fig)


def plot_line_list(time_lst,dat_lst,tlim=None,figname='Fig',axlab=None,leg=None,
                   leg_col=1, legloc=None,line_sty=None,style='default',capt=''):
    import matplotlib.transforms as mtransforms    
    
    size_tick = 14
    size_label = 16
    # size_title = 18
    fig = plt.figure()
    ndat = len(time_lst)
    # line_sty=['k','b','r','m','g','c']
    with plt.style.context(style):
        for i in range(ndat): 
            if line_sty is not None and len(line_sty)>=ndat:
                plt.plot(time_lst[i],dat_lst[i],line_sty[i]) # ,mfc='none'
            else:
                plt.plot(time_lst[i],dat_lst[i])
    fig.autofmt_xdate()
    ax = plt.gca()
    plt.text(0.01, 0.99, capt,fontsize=size_label,ha='left', va='top', transform=ax.transAxes) #add fig caption
    # trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
    # plt.text(0.00, 1.00, capt,fontsize=size_label,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption

    ax.tick_params(axis="both", labelsize=size_tick)     
    if tlim is not None:
        ax.set_xlim(tlim)
#         plt.xlim(tlim)
    if axlab is not None:
        plt.xlabel(axlab[0],fontsize=size_label)
        plt.ylabel(axlab[1],fontsize=size_label)
    if leg is None:
        leg = [str(i) for i in range(ndat)]
    else:
        leg = leg[:ndat]
    plt.tight_layout()
    if legloc is None:
        plt.legend(leg,ncol=leg_col,fontsize=size_tick)
    else: # loc: 0best,1Ur,2Ul,3-Ll,4-Lr, 5-R,6-Cl,7-Cr,8-Lc,9Uc,10C
        plt.legend(leg,ncol=leg_col,fontsize=size_tick,loc=2,bbox_to_anchor=legloc)    
    plt.savefig(figname,dpi=150)
    plt.close(fig)
    plt.show()
    
    
def plot_errbar_list(xlst,dat_lst,err_lst,tlim=None,figname='Fig',axlab=None,leg=None,
                   leg_col=1, legloc=None,line_sty=None,style='default',capt=''):
    import matplotlib.transforms as mtransforms    
    
    size_tick = 14
    size_label = 16
    # size_title = 18
    fig = plt.figure()
    ndat = len(xlst)
    # line_sty=['k','b','r','m','g','c']
    with plt.style.context(style):
        for i in range(ndat): 
            if line_sty is not None and len(line_sty)>=ndat:
                plt.plot(xlst[i],dat_lst[i],line_sty[i]) # ,mfc='none'
                plt.errorbar(xlst[i],dat_lst[i],err_lst[i], linestyle='None', marker='^', capsize=3)
            else:
                plt.plot(xlst[i],dat_lst[i])
                plt.errorbar(xlst[i],dat_lst[i],err_lst[i], linestyle='None', marker='^', capsize=3)
    fig.autofmt_xdate()
    ax = plt.gca()
    plt.text(0.01, 0.99, capt,fontsize=size_label,ha='left', va='top', transform=ax.transAxes) #add fig caption
    # trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
    # plt.text(0.00, 1.00, capt,fontsize=size_label,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption

    ax.tick_params(axis="both", labelsize=size_tick)     
    if tlim is not None:
        ax.set_xlim(tlim)
#         plt.xlim(tlim)
    if axlab is not None:
        plt.xlabel(axlab[0],fontsize=size_label)
        plt.ylabel(axlab[1],fontsize=size_label)
    if leg is None:
        leg = [str(i) for i in range(ndat)]
    else:
        leg = leg[:ndat]
    plt.tight_layout()
    if legloc is None:
        plt.legend(leg,ncol=leg_col,fontsize=size_tick)
    else: # loc: 0best,1Ur,2Ul,3-Ll,4-Lr, 5-R,6-Cl,7-Cr,8-Lc,9Uc,10C
        plt.legend(leg,ncol=leg_col,fontsize=size_tick,loc=2,bbox_to_anchor=legloc)    
    plt.savefig(figname,dpi=150)
    plt.close(fig)
    plt.show()    
    

def plot_sites_cmp(time_TG,ssh_TG,time,ssh,tlim=None,figname=None,axlab=None,leg=None):
    fig = plt.figure()
    plt.plot(time_TG,ssh_TG,'k.')
    plt.plot(time,ssh,'b')
    fig.autofmt_xdate()
    if tlim is not None:
        ax = plt.gca()
        ax.set_xlim(tlim)
#         plt.xlim(tlim)
    if axlab is not None:
        plt.xlabel(axlab[0],fontsize=14)
        plt.ylabel(axlab[1],fontsize=14)
    if leg is None:
        leg = ['ref','mod']
    if figname is None:
        figname = 'Fig'
    plt.legend(leg)     
    plt.savefig(figname,dpi=100)
    plt.close(fig)
    plt.show()        
    
    
def plot_mod_vs_obs(mod,obs,figname,axlab=('Target','Mod',''),leg=None,alpha=0.5,
                    marker='o',figsize=(4,4),fontsize=16,capt=''):
    """
    Plot model data against observation data to visualize bias in the model.
    Parameters:
        mod (list of arrays): Model data to be plotted.
        obs (array-like): Observation data to be plotted.
        label (tuple, optional): Labels for x-axis, y-axis, and title.
    """
    import matplotlib.transforms as mtransforms    

    fig= plt.figure(figsize=figsize)
    # plt.style.use('seaborn-deep')
    # if leg is None:
    #     leg = [str(i) for i in range(len(mod))]
    if len(marker) < len(mod):
        marker = ['o' for i in range(len(mod))]
    # Plot the scatter plot
    for i in range(len(mod)):
        if leg is not None:
            plt.scatter(obs, mod[i], alpha=0.3, marker=marker[i],label=leg[i]) # marker=marker, color='blue',
        else:
            plt.scatter(obs, mod[i], alpha=0.3, marker=marker[i]) # marker=marker, color='blue',

    # Set the same limits for x and y axes
    max_val = max(np.nanmax(obs), np.nanmax(np.array(mod)))
    min_val = min(np.nanmin(obs), np.nanmin(np.array(mod)))
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)    
    # Set the same ticks for x and y axes
    ticks = plt.xticks()
    plt.xticks(ticks[0],fontsize=fontsize-2)
    plt.yticks(ticks[0],fontsize=fontsize-2)
    
    # Plot the perfect fit line (y = x)
    plt.plot(ticks[0], ticks[0], linestyle='dashed', color='black') 

    plt.xlabel(axlab[0], fontsize=fontsize)
    plt.ylabel(axlab[1], fontsize=fontsize)
    plt.title(axlab[2], fontsize=fontsize)
    # plt.legend(fontsize=fontsize)
    plt.grid(True)
    ax = plt.gca()
    plt.text(0.01, 0.99, capt,fontsize=fontsize,ha='left', va='top', transform=ax.transAxes) #add fig caption
    # trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
    # plt.text(0.00, 1.00, capt,fontsize=fontsize,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption

    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    plt.savefig(figname,dpi=100) #,dpi=100    
    plt.close(fig)
    
def plot_distri(data,figname,bins=10, axlab=('Val','P',''),leg=('1', ), 
                   figsize=(8, 6), fontsize=12,capt='',style='default'):
    """
    Compare the distribution of data using histograms.
    Parameters:
        data (list of arrays with same length): data, one array corresponds to 1 histogram.
        bins (int or sequence, optional): Number of bins or bin edges. Default is 10.
"""
    from matplotlib.ticker import PercentFormatter 
    import matplotlib.transforms as mtransforms    
    # from matplotlib import style 
    fig = plt.figure(figsize=figsize)
    # plt.style.use(style) #'seaborn-deep'
    plt.style.context(style)
    # Calculate the bin edges
    xmin = min([np.nanmin(np.array(data[i])) for i in range(len(data))])
    xmax = max([np.nanmax(np.array(data[i])) for i in range(len(data))])
    hist_range = (xmin, xmax)
    bins = np.linspace(hist_range[0], hist_range[1], bins+1)

    # Plot histogram for observation data
    # for i in range(length(data)):
    # plt.hist(data[i], bins=bins, color=color[i], alpha=0.5, label=label[i]) # , align='left'

    # to plot the histogam side by side
    weights=[np.ones(len(data[i])) / len(data[i]) for i in range(len(data))]
    plt.hist(data,bins=bins,weights=weights, alpha=0.5, label=leg) # , align='right'

    plt.xlabel(axlab[0], fontsize=fontsize)
    plt.ylabel(axlab[1], fontsize=fontsize)
    plt.title(axlab[2], fontsize=fontsize)
    plt.legend(fontsize=fontsize-2)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1)) # for array of the same length
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=len(data[0]))) # for array of the same length

    plt.grid(True)
    ax = plt.gca()
    plt.text(0.01, 0.99, capt,fontsize=fontsize,ha='left', va='top', transform=ax.transAxes) #add fig caption
    # trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
    # plt.text(0.00, 1.00, capt,fontsize=fontsize,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption

    plt.tight_layout()
    # plt.show()
    plt.savefig(figname,dpi=300) #,dpi=100    
    plt.close(fig)