#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:07:24 2023
functions to process nc files 
@author: g260218
"""
import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
# import netCDF4
from datetime import date, datetime, timedelta
import math

# read wave buoy data
# nc_f: name of nc files
def read_cmemes_MO(nc_f,cri_QC=None,varlim=None):
    nc_fid = Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file
                             # and create an instance of the ncCDF4 class
    nc_fid.variables.keys() # list(nc_fid.variables)
    # print(nc_fid)

    # Extract data from NetCDF file
    lon = float(nc_fid.geospatial_lon_min)
    lat = float(nc_fid.geospatial_lat_min)
    sid = nc_fid.id  # full name
    platform_code = nc_fid.platform_code  # suffix

    str_ts = nc_fid.time_coverage_start
    str_te = nc_fid.time_coverage_end
    
    ts = datetime.strptime(str_ts, '%Y-%m-%dT%H:%M:%SZ')#.date()
    te = datetime.strptime(str_te, '%Y-%m-%dT%H:%M:%SZ')#.date()

    time = nc_fid.variables['TIME'][:] # days refer to 1950.1.1
    dt = int((time[1]-time[0])*1440+0.5)
    tref = datetime(1950,1,1) # reference date in nc file 
    dtref = (datetime(1970,1,1)-datetime(1950,1,1)).days # reference in python 1970
    dcount = (ts - tref).days + 1
    
    sec = (time.data - dtref) * 86400.0 # seconds refer to 1970
    func = np.vectorize(datetime.utcfromtimestamp)
    timed= func(sec)
    # timed = timed.tolist()
    
    # time = np.ma.getdata(time)
    # sec = (time - dtref) * 86400.0 # seconds refer to 1970
    # timed = [datetime.fromtimestamp(sec[i]) for i in range(len(sec))]
        
    # Spectral significant wave height (Hm0) 
    VHM0 = nc_fid.variables['VHM0'][:]  # shape is (time,depth)
    VHM0_QC = nc_fid.variables['VHM0_QC'][:]  # shape is (time,depth)
    varmax = VHM0.max()
    # print(sid,'VHM0 max=',varmax)
    
    # sea_surface_wave_mean_period_from_variance_spectral_density_second_frequency_moment
    VTM02 = nc_fid.variables['VTM02'][:]  # shape is (time,depth)
    # VTM02_QC = nc_fid.variables['VHM02_QC'][:]  # shape is (time,depth)
    # varmax = VTM02.max()
    # sea_surface_wave_period_at_variance_spectral_density_maximum
    VTPK = nc_fid.variables['VTPK'][:]  # shape is (time,depth)
    # VTPK_QC = nc_fid.variables['VTPK_QC'][:]  # shape is (time,depth)
    # varmax = VTPK.max()
    # Mean wave direction from (Mdir)
    VMDR = nc_fid.variables['VMDR'][:]  # shape is (time,depth)
    # VMDR_QC = nc_fid.variables['VMDR_QC'][:]  # shape is (time,depth)
    # varmax = VMDR.max()
    
    # depth = nc_fid.variables['DEPH'][:]  # shape is time, 1
    # DEPH_QC = nc_fid.variables['DEPH_QC'][:]  # shape is time
    # temp = nc_fid.variables['TEMP'][:]  # shape is depth
    
    # remove possible bad values 
    # SLEV_QC: 0 "no_qc_performed; 1 good_data; 2 probably_good_data; 3 bad_data_that_are_potentially_correctable;  4 bad_data; 5 value_changed; 6 value_below_detection; 7 nominal_value; 8 interpolated_value; 9 missing_value" 
    if cri_QC is not None:
        VHM0[VHM0_QC>cri_QC]=np.nan
    if varlim is not None:
        VHM0[VHM0>varlim[0]]=np.nan
        VHM0[VHM0<varlim[1]]=np.nan
#     pref=infile.split('/')[-1].replace('.nc','')
#     fig = plt.figure()
#     plt.plot(timed,VHM0,'k.')
#     plt.xlabel('time',fontsize=16)
#     plt.ylabel('SWH (m)',fontsize=16)
#     figname = pref+'swh_.png'
#     plt.savefig(figname,dpi=100)
#     plt.close(fig)
#     plt.show()

    return sid,platform_code,timed,ts,te,lon,lat,VHM0,VHM0_QC,VTM02,VTPK,VMDR


# read tidal gauge data
# nc_f: name of nc files
def read_cmemes_TG(nc_f,cri_QC=None,sshlim=None):
    nc_fid = Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file
                             # and create an instance of the ncCDF4 class
    nc_fid.variables.keys() # list(nc_fid.variables)
    # print(nc_fid)

    # Extract data from NetCDF file
    lon = float(nc_fid.geospatial_lon_min)
    lat = float(nc_fid.geospatial_lat_min)
    sid = nc_fid.id
    str_ts = nc_fid.time_coverage_start
    str_te = nc_fid.time_coverage_end
    
    ts = datetime.strptime(str_ts, '%Y-%m-%dT%H:%M:%SZ')#.date()
    te = datetime.strptime(str_te, '%Y-%m-%dT%H:%M:%SZ')#.date()

    time = nc_fid.variables['TIME'][:] # days refer to 1950.1.1
    dt = int((time[1]-time[0])*1440+0.5)
    tref = datetime(1950,1,1) # reference date in nc file 
    dtref = (datetime(1970,1,1)-datetime(1950,1,1)).days # reference in python 1970
    dcount = (ts - tref).days + 1
    
    sec = (time.data - dtref) * 86400.0 # seconds refer to 1970
    func = np.vectorize(datetime.utcfromtimestamp)
    timed= func(sec)
    # timed = timed.tolist()
    
    # time = np.ma.getdata(time)
    # sec = (time - dtref) * 86400.0 # seconds refer to 1970
    # timed = [datetime.fromtimestamp(sec[i]) for i in range(len(sec))]
    
    ssh = nc_fid.variables['SLEV'][:]  # shape is time, 1
    SLEV_QC = nc_fid.variables['SLEV_QC'][:]  # shape is time
    depth = nc_fid.variables['DEPH'][:]  # shape is time, 1
    DEPH_QC = nc_fid.variables['DEPH_QC'][:]  # shape is time
    ssh_max = ssh.max()
    # print(sid,'sshmax=',ssh_max)
    
    # remove possible bad values 
    # SLEV_QC: 0 "no_qc_performed; 1 good_data; 2 probably_good_data; 3 bad_data_that_are_potentially_correctable;  4 bad_data; 5 value_changed; 6 value_below_detection; 7 nominal_value; 8 interpolated_value; 9 missing_value" 
    if cri_QC is not None:
        ssh[SLEV_QC>cri_QC]=np.nan
    if sshlim is not None:
        ssh[ssh>sshlim[0]]=np.nan
        ssh[ssh<sshlim[1]]=np.nan
#     pref=infile.split('/')[-1].replace('.nc','')
#     fig = plt.figure()
#     plt.plot(timed,ssh,'k.')
#     plt.xlabel('time',fontsize=16)
#     plt.ylabel('ssh (m)',fontsize=16)
#     figname = pref+'ssh_.png'
#     plt.savefig(figname,dpi=100)
#     plt.close(fig)
#     plt.show()

    return sid,timed,ts,te,lon,lat,ssh,SLEV_QC

def plot_sites_var(timed,ssh,tlim=None,figname='Fig'):
    fig = plt.figure()
    plt.plot(timed,ssh,'k.')
    fig.autofmt_xdate()
    if tlim is not None:
        ax = plt.gca()
        ax.set_xlim(tlim)
#         plt.xlim(tlim)
    plt.xlabel('time',fontsize=16)
    plt.ylabel('ssh (m)',fontsize=16)
    plt.savefig(figname,dpi=100)
    plt.close(fig)
    plt.show()

def plot_sites_location(lats,lons,figname):
    fig = plt.figure()
    plt.plot(lons,lats,'k.')
    plt.xlabel('lon',fontsize=16)
    plt.ylabel('lat',fontsize=16)
    plt.savefig(figname,dpi=100)
    plt.close(fig)
    plt.show()

    
def plt_pcolor_pnt(lat,lon,sample, figname,lat_sta=None,lon_sta=None,subsize = [5,4],
                   cmap='bwr',clim=None,unit=None,title=None,axoff=0,capt=None,txt=None,
                    loc_txt=None,xlim=None,ylim=None):  
    # sample is an array[nx,ny]
    # lat_sta & lon_sta are lat/lon of stations 
    import matplotlib.transforms as mtransforms    
    cm = 1/2.54  # centimeters in inches
    fig, ax = plt.subplots(1, 1, figsize=(subsize[0], subsize[1])) # default unit inch 2.54 cm
    size_tick = 16
    size_label = 18
    size_title = 18

    if clim:
        vmin, vmax = clim
        cf = ax.pcolormesh(lat, lon, sample, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        cf = ax.pcolormesh(lat, lon, sample, cmap=cmap)
    cbar = fig.colorbar(cf, ax=ax,location='top')
    
    # add mark for stations if exist
    if lat_sta is not None and lon_sta is not None:
        ax.scatter(lon_sta, lat_sta, s=40, marker="^",color='k')
    
    cbar.ax.tick_params(labelsize=size_tick)
    if title:
        ax.set_title(title,fontsize=size_title)
    if unit:
        cbar.set_label(unit,fontsize=size_tick-1)
    if not axoff: # keep axes or not 
        ax.set_xlabel('lon',fontsize=size_label)
        ax.set_ylabel('lat',fontsize=size_label)
        ax.tick_params(axis="both", labelsize=size_tick-1) 
    # plt.xticks(fontsize=size_tick)
    # plt.yticks(fontsize=size_tick)
    else:
        ax.axis('off')
    if txt is not None: 
        plt.text(loc_txt[0],loc_txt[1], txt,fontsize=size_tick,ha='left', va='top', transform=ax.transAxes) #add text
    if capt is not None: 
        trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
        plt.text(0.00, 1.00, capt,fontsize=size_label,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.tight_layout()

    plt.savefig(figname,bbox_inches='tight',dpi=300) #,dpi=100    
    plt.close(fig)
    

def write_sites_info(sids,lons,lats,ts_all,te_all,outfile):
    data_all = np.rec.fromarrays((lons,lats,ts_all,te_all,sids))
    np.savetxt(outfile+'.dat', data_all, fmt='%f,%f,%s,%s,%s')
    #b = np.loadtxt(outfile+'.dat')    


def check_dry(lon_sta,lat_sta,lon,lat,var):
    # lon_sta, lat_sta are longitude and latidude of 1 station
    # lon, lat are longitude and latidude of rectangular grids
    dx = lon[1]-lon[0]
    dy = lat[1]-lat[0]
    nx = len(lon)
    ny = len(lat)
    nnodes = 4
    ix0 = int((lon_sta-lon[0])/dx)
    iy0 = int((lat_sta-lat[0])/dy)
    ix1 = ix0+1
    iy1 = iy0+1
    index = [[iy0,ix0],[iy1,ix0],[iy1,ix1],[iy0,ix1]]
    # assert ix0 >= 0 and iy0 >= 0 and ix1<nx and iy1<ny, "out of domain"
    if ix0 >= 0 and iy0 >= 0 and ix1<nx and iy1<ny:
        outside=0
    else:
        return 1

    # check if the grid has meaningful value 
    ll_grd = []
    var_use = []
    for k in range(nnodes):
        if var[index[k][0],index[k][1]] != 0:
            ll_grd.append([lon[index[k][0]],lat[index[k][1]]])
            var_use.append(var[index[k][0],index[k][1]])

    nnode_ = len(ll_grd)
    if nnode_ > 0:
        dry = 0
    else:
        dry = 1
    return outside+dry


def interp_var(lon_sta,lat_sta,lon,lat,var,method='max'):
    # lon_sta, lat_sta are longitude and latidude of 1 station
    # lon, lat are longitude and latidude of rectangular grids
    dx = lon[1]-lon[0]
    dy = lat[1]-lat[0]
    nx = len(lon)
    ny = len(lat)
    nnodes = 4
    ix0 = int((lon_sta-lon[0])/dx)
    iy0 = int((lat_sta-lat[0])/dy)
    ix1 = ix0+1
    iy1 = iy0+1
    index = [[iy0,ix0],[iy1,ix0],[iy1,ix1],[iy0,ix1]]
    assert ix0 >= 0 and iy0 >= 0 and ix1<nx and iy1<ny, "out of domain"
    ll_sta = [lon_sta,lat_sta]
    # ll_grd0 = [[lon[ix0],lat[iy0]],[lon[ix0],lat[iy1]],
    #           [lon[ix1],lat[iy1]],[lon[ix1],lat[iy0]]]  # clockwise
    # check if the grid has meaningful value 
    ll_grd = []
    var_use = []
    for k in range(nnodes):
        if var[index[k][0],index[k][1]] != 0:
            ll_grd.append([lon[index[k][0]],lat[index[k][1]]])
            var_use.append(var[index[k][0],index[k][1]])

    nnode_ = len(ll_grd)
    # assert nnode_ > 0 , "locate on land!"
    
    if nnode_ > 0:
        weight = np.zeros(nnode_)
        dist = [] 
        for k in range(nnode_):
            dist.append(math.dist(ll_sta,ll_grd[k]))
            
        if method == 'max':
            ind_max = np.argmax(var_use)
            weight[ind_max] = 1.0
        elif method == 'nearest':
            ind_min = np.argmin(dist)
            weight[ind_min] = 1.0
        elif method == 'ave':
            weight = np.ones(nnodes)*1.0/nnode_
        elif method == 'idw':
            weight = 1/(dist+1e-10)/dist.sum()
        var_int = 0.0
        for k in range(nnode_):
            var_int += var_use[k]*weight[k]
        return var_int
    else:
        return np.nan

def index_stations(lon_sta,lat_sta,lon,lat):
    # lon_sta, lat_sta are longitude and latidude of stations
    # lon, lat are longitude and latidude of rectangular grids
    nsta = len(lon_sta)
    dx = lon[1]-lon[0]
    dy = lat[1]-lat[0]
    nx = len(lon)
    ny = len(lat)
    index = [None] * nsta
    for i in range(nsta):
        ix0 = int((lon_sta[i]-lon[0])/dx)
        iy0 = int((lat_sta[i]-lat[0])/dy)
        ix1 = ix0+1
        iy1 = iy0+1
        assert ix0 >= 0 and iy0 >= 0 and ix1<nx and iy1<ny, "out of domain"
        index[i] = np.array([[iy0,ix0],[iy1,ix0],[iy1,ix1],[iy0,ix1]])
    index = np.vstack(index)
    return index

def index_weight_stations(lon_sta,lat_sta,lon,lat,method='nearest'):
    # lon_sta, lat_sta are longitude and latidude of stations
    # lon, lat are longitude and latidude of rectangular grids
    nsta = len(lon_sta)
    dx = lon[1]-lon[0]
    dy = lat[1]-lat[0]
    nx = len(lon)
    ny = len(lat)
    dist = [None] * nsta
    index = [None] * nsta
    weight = [None] * nsta
    nnodes = 4
    for i in range(nsta):
        ix0 = int((lon_sta[i]-lon[0])/dx)
        iy0 = int((lat_sta[i]-lat[0])/dy)
        ix1 = ix0+1
        iy1 = iy0+1
        assert ix0 >= 0 and iy0 >= 0 and ix1<nx and iy1<ny, "out of domain"
        index[i] = [[iy0,ix0],[iy1,ix0],[iy1,ix1],[iy0,ix1]]
        ll_sta = [lon_sta[i],lat_sta[i]]
        ll_grd = [[lon[ix0],lat[iy0]],[lon[ix0],lat[iy1]],
                  [lon[ix1],lat[iy1]],[lon[ix1],lat[iy0]]]  # clockwise
        dist[i] = []
        weight[i] = np.zeros(nnodes)
        for j in range(nnodes):
            dist[i].append(math.dist(ll_sta,ll_grd[j]))
            
        if method == 'nearest':
            ind_min = np.argmin(dist)
            weight[i][ind_min] = 1.0
        elif method == 'ave':
            weight[i] = np.ones(nnodes)*1.0/nnodes
        elif method == 'idw':
            weight[i] = 1/(dist[i]+1e-10)/dist[i].sum()
    return index, weight
        

def lst_flatten(xss):
    return [x for xs in xss for x in xs]

# select points for post-analysis
def select_sta(var,ivar_hr,lon,lat,nskp = (40,40),kpshare=1):
    # var 4d array
    # estimate max min value for the selected period
    nchl = len(ivar_hr)
    sta_max = lst_flatten([['v%d_max'%ivar_hr[i], 'v%d_min'%ivar_hr[i]] for i in range(nchl)])
    varm_hr = np.ones((nchl,2))
    ind_varm = np.ones((nchl,2),dtype= np.int64)
    for i in range(nchl):
        varm_hr[i,0] = np.nanmax(var[:,i,:,:])
        ind_varm[i,0] = np.nanargmax(var[:,i,:,:])
        varm_hr[i,1] = np.nanmin(var[:,i,:,:])
        ind_varm[i,1] = np.nanargmin(var[:,i,:,:])
    temp = np.unravel_index(ind_varm.flatten(), (len(var),len(lat),len(lon)))
    ind_mm = np.array([temp[1],temp[2]]).transpose()
    
    # select several observation locations for comparison 
    # sta_user0 = ['WAVEB04', 'WAVEB05', 'WAVEB06']
    # sta_user1 = [sta_user0[i] + str(j) for i in range(len(sta_user0)) for j in range(4)]
    # ll_stas = np.array([[28.611600,43.539200],[27.906700,42.696400],[28.343800,43.371700]])
    # ll_shift = np.array([[0,0],[0,0],[0,0]]) # shift the station to the water region, lon,lat
    sta_user0 = ['WAVEB01', 'WAVEB02', 'WAVEB03','WAVEB04', 'WAVEB05', 'WAVEB06','SPOT0772','SPOT0773','SPOT0776']
    sta_user1 = [sta_user0[i] + str(j) for i in range(len(sta_user0)) for j in range(4)]
    ll_stas = np.array([[27.941500,43.194200],[27.556200,42.511700],[27.927700,42.114400],
                        [28.611600,43.539200],[27.906700,42.696400],[28.343800,43.371700],
                        [27.994700,43.182000],[27.899200,42.957600],[27.633200,42.504400]
                        ])
    ll_shift = np.array([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]) # shift the station to the water region, lon,lat
    ll_stas = ll_stas+ll_shift
    id_use = [0,1,4]
    ll_stas = ll_stas[id_use,:]
    sta_user1 = [sta_user0[i] + str(j) for i in id_use for j in range(4)]
    ind_sta = index_stations(ll_stas[:,0],ll_stas[:,1],lon,lat)
    # index = ind_sta
    
    # add points in the domain for testing
    nx_skp = nskp[0]
    ny_skp = nskp[1]
    if kpshare ==1: #  shared poins by lr & hr
        ix = np.arange(0, len(lat)-1, nx_skp) #  shared poins by lr & hr
        iy = np.arange(0, len(lon)-1, nx_skp)
    else:
        ix = np.arange(int(nx_skp/2), len(lat)-1, nx_skp) #  non-shared poins by lr & hr, initial index links to scale
        iy = np.arange(int(nx_skp/2), len(lon)-1, ny_skp)
    xv, yv = np.meshgrid(ix, iy)
    ind_add = np.vstack((np.int_(xv.flatten()), np.int_(yv.flatten()))).T
    sta_add = ['p'+ str(i).zfill(2) for i in range(len(ind_add))]
    
    # max + gauging stations + grid points
    # index = np.vstack(_mm, ind_add, ind_sta))
    # sta_user = sta_max + sta_add + sta_user1
    # no gauging station
    # index = np.vstack((ind_mm, ind_add))
    # sta_user = sta_max + sta_add 
    # max + gauging stations
    index = np.vstack((ind_mm, ind_sta))
    sta_user = sta_max + sta_user1 
    
    ll_sta = np.array([lat[index[:,0]],lon[index[:,1]]]).transpose() # should corresponds to (H,W), H[0]-lowest lat
    return index,sta_user,ll_sta,varm_hr,ind_varm