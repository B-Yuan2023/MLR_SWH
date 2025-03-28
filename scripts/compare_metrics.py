#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 09:11:07 2024
compare metrics between model runs
@author: g260218
"""

import os
import numpy as np

import pandas as pd
# import sys
# import importlib
path_par = "../"  # the path of parameter files, also used for output path

# Function to load a CSV file and convert it to a dictionary
def csv_to_dict_of_arrays(filename):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(filename)
    
    # Convert DataFrame to a dictionary where keys are columns (headers) and values are NumPy arrays
    result_dict = {col: df[col].to_numpy() for col in df.columns}
    
    return result_dict

# load metrics from multiple model with one run
def read_metric_mod(mod_name,up_factor,ivar_hr,kmask):
    metrics_md = {'mae': [], 'rmse': [], 'mae_99': [],'rmse_99': [],
                      'mae_01': [],'rmse_01': [],'mae_m': [],'rmse_m': [],
                      'mae_t': [],'rmse_t': [],} # 'mse': [], 
    for i,mod in enumerate(mod_name):
        suf = mod
        opath_st = path_par+'stat_' + suf +'/'
        ofname = "srf_%d_c%d_mask" % (up_factor[i],ivar_hr) + '_test_metrics.csv'
        # metrics = np.loadtxt(opath_st + ofname, delimiter=",",skiprows=1)
        metrics = csv_to_dict_of_arrays(opath_st + ofname)
        for key in metrics_md.keys():
            metrics_md[key].append(metrics[key])  # list of arrays
    for key in metrics_md.keys():
        metrics_md[key] = np.stack(metrics_md[key], axis=1) # combine lists to array
    return metrics_md


# Function to write dictionary of 2D arrays side by side into a CSV using pandas
def dict_2Darrays_to_csv_pd(data, filename):
    # Create a list to store the DataFrames
    dfs = []
    
    for key, array in data.items():
        # Convert the 2D array to a DataFrame
        df = pd.DataFrame(array)
        # Rename the columns to include the array name
        df.columns = [f'{key}_col{i+1}' for i in range(df.shape[1])]
        # Append the DataFrame to the list
        dfs.append(df)
    
    # Concatenate the DataFrames horizontally (side by side)
    result_df = pd.concat(dfs, axis=1)
    # Write the resulting DataFrame to a CSV file
    result_df.to_csv(filename, index=False)
    

# load metrics from sorted data from direct interpolation
def plot_bar(data,figname,figsize,axlab,leg,ticklab,size_label=12,capt='(a)',
             leg_col=2,style='default',ylim=None):
    # data(M,N) array: N bar at M xticks
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms    
    # from matplotlib import style 

    x = np.arange(data.shape[0])
    dx = (np.arange(data.shape[1])-data.shape[1]/2.)/(data.shape[1]+2.)
    d = 1./(data.shape[1]+2.)
    
    plt.style.context(style) # 'seaborn-deep', why not working
    plt.style.use(style) # 'seaborn-deep', why not working

    fig, ax=plt.subplots(figsize=figsize,layout="constrained")
    for i in range(data.shape[1]):
        ax.bar(x+dx[i],data[:,i], width=d, label=leg[i], zorder=3)
    ax.set_xticks(np.arange(len(ticklab))) # for missing 1st tick
    ax.set_xticklabels(ticklab,rotation = 30,fontsize=size_label)
    ax.set_xlabel(axlab[0],fontsize=size_label)
    ax.set_ylabel(axlab[1],fontsize=size_label)
    if len(axlab)>=3:
        ax.set_title(axlab[2], fontsize=size_label)
        
    ax.tick_params(axis="both", labelsize=size_label-1)
    ax.grid(zorder=0)
    if ylim:
        ax.set_ylim(ylim)
    plt.legend(ncol=leg_col,fontsize=size_label-2,borderpad=0.2,handlelength=1.0,
               handleheight=0.3,handletextpad=0.4,labelspacing=0.2,columnspacing=0.5)  # framealpha=1
    if capt is not None: 
        trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
        plt.text(0.03, 1.06, capt,fontsize=size_label,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption
    plt.savefig(figname,bbox_inches='tight',dpi=300) #,dpi=100    
    # plt.show()
    plt.close(fig) 


if __name__ == '__main__':
    
    kmask = 1 
    out_path = path_par+'cmp_metrics/'
    os.makedirs(out_path, exist_ok=True)

# =============================================================================
#   scale factor
    mod_name= ['par55e_md0','par55e_s40_md0','par55e_s80_md0'] # 'par55e_s30_md0',
    up_factor = [20,40,80] # 30,
    ivar_hr = 5
    metrics_md = read_metric_mod(mod_name,up_factor,ivar_hr,kmask)
    ofname = 'metrics_mk'+str(kmask)+ mod_name[0]+'_s%d'%len(mod_name)+'.csv' 
    # Write the dictionary to a CSV file
    dict_2Darrays_to_csv_pd(metrics_md, out_path + ofname)

    ticklab  = ['mae','rmse','mae_m','rmse_m','mae_99','rmse_99','mae_01','rmse_01']
    leg = ['s20','s40','s80']
    data =  []
    for key in ticklab:
        data.append(metrics_md[key][0])  
    data = np.stack(data,axis=0)
    figname = out_path + 'metrics_mk'+str(kmask)+ mod_name[0]+'_s%d'%len(mod_name)+'.png'
    axlab = ['','Error in SWH (m)','MLR']
    figsize = [3.3,2]
    ylim = [0,0.25]
    plot_bar(data,figname,figsize,axlab,leg,ticklab,size_label=12,capt='(a)',
             leg_col=3,ylim=ylim)
    
# =============================================================================
#   scale factor
    mod_name= ['par534e_md0','par534e_s20_md0','par534e_s40_md0'] # 'par55e_s30_md0',
    up_factor = [10,20,40] # 30,
    leg = ['s10','s20','s40']
    ivar_hr = 5
    metrics_md = read_metric_mod(mod_name,up_factor,ivar_hr,kmask)
    ofname = 'metrics_mk'+str(kmask)+ mod_name[0]+'_s%d'%len(mod_name)+'.csv' 
    # Write the dictionary to a CSV file
    dict_2Darrays_to_csv_pd(metrics_md, out_path + ofname)

    ticklab  = ['mae','rmse','mae_m','rmse_m','mae_99','rmse_99','mae_01','rmse_01']
    data =  []
    for key in ticklab:
        data.append(metrics_md[key][0])  
    data = np.stack(data,axis=0)
    figname = out_path + 'metrics_mk'+str(kmask)+ mod_name[0]+'_s%d'%len(mod_name)+'.png'
    axlab = ['','Error in SWH (m)','MLR']
    figsize = [3.3,2]
    ylim = [0,0.95]
    plot_bar(data,figname,figsize,axlab,leg,ticklab,size_label=12,capt='(a)',
             leg_col=3,ylim=ylim)    
# =============================================================================
    # no. of samples
    id_use = [0,2,3,4,5,6]
    mod_name= ['par55e_dt48_md0','par55e_dt24_md0','par55e_dt12_md0',
               'par55e_dt6_md0','par55e_md0',
               'par55e_dt2_md0','par55e_dt1_md0',]  # 'par55e_dt8_md0',
    mod_name = [mod_name[i] for i in id_use]
    up_factor = [20] * len(mod_name)
    ivar_hr = 5
    metrics_md = read_metric_mod(mod_name,up_factor,ivar_hr,kmask)
    ofname = 'metrics_mk'+str(kmask)+ mod_name[0]+'_dt%d'%len(mod_name)+'.csv' 
    # Write the dictionary to a CSV file
    dict_2Darrays_to_csv_pd(metrics_md, out_path + ofname)
    
    ticklab  = ['mae','rmse','mae_m','rmse_m','mae_99','rmse_99','mae_01','rmse_01']
    leg = ['dt48','dt24','dt12','dt6','dt3','dt2','dt1']
    leg = [leg[i] for i in id_use]
    data =  []
    for key in ticklab:
        data.append(metrics_md[key][0])  
    data = np.stack(data,axis=0)
    figname = out_path + 'metrics_mk'+str(kmask)+ mod_name[0]+'_dt%d'%len(mod_name)+'.png'
    axlab = ['','Error in SWH (m)','MLR']
    figsize = [3.3,2]
    ylim = [0,0.18]
    plot_bar(data,figname,figsize,axlab,leg,ticklab,size_label=12,capt='(c)',
             leg_col=3,ylim=ylim)

# =============================================================================
    # no. of samples
    id_use = [0,1,2,3,4,5]
    mod_name= ['par534e_dt48_md0','par534e_dt12_md0','par534e_dt6_md0',
               'par534e_md0','par534e_dt2_md0','par534e_dt1_md0']  # 
    leg = ['dt48','dt12','dt6','dt3','dt2','dt1']
    mod_name = [mod_name[i] for i in id_use]
    leg = [leg[i] for i in id_use]

    up_factor = [10] * len(mod_name)
    ivar_hr = 5
    metrics_md = read_metric_mod(mod_name,up_factor,ivar_hr,kmask)
    ofname = 'metrics_mk'+str(kmask)+ mod_name[0]+'_dt%d'%len(mod_name)+'.csv' 
    # Write the dictionary to a CSV file
    dict_2Darrays_to_csv_pd(metrics_md, out_path + ofname)
    
    ticklab  = ['mae','rmse','mae_m','rmse_m','mae_99','rmse_99','mae_01','rmse_01']
    data =  []
    for key in ticklab:
        data.append(metrics_md[key][0])  
    data = np.stack(data,axis=0)
    figname = out_path + 'metrics_mk'+str(kmask)+ mod_name[0]+'_dt%d'%len(mod_name)+'.png'
    axlab = ['','Error in SWH (m)','MLR']
    figsize = [3.3,2]
    ylim = [0,0.55]
    plot_bar(data,figname,figsize,axlab,leg,ticklab,size_label=12,capt='(c)',
             leg_col=3,ylim=ylim)

# =============================================================================
#  no of input channels 
    # mod_name= ['par534e_md0','par534e_ct3_1_md0','par534e_ct6_1_md0','par534e_ct12_md0','par534e_ct24_md0']  # 
    # up_factor = [10] * len(mod_name)
    # leg = ['0h','3h_dt1','6h_dt1','12h_dt6','24h_dt6']

    mod_name= ['par534e_md0','par534e_ct3_md0','par534e_ct6_md0','par534e_ct12_md0','par534e_ct24_md0']  # 
    up_factor = [10] * len(mod_name)
    leg = ['0h','3h','6h','12h','24h']

    ivar_hr = 5
    metrics_md = read_metric_mod(mod_name,up_factor,ivar_hr,kmask)
    ofname = 'metrics_mk'+str(kmask)+ mod_name[0]+'_dt%d'%len(mod_name)+'.csv' 
    # Write the dictionary to a CSV file
    dict_2Darrays_to_csv_pd(metrics_md, out_path + ofname)
    
    ticklab  = ['mae','rmse','mae_m','rmse_m','mae_99','rmse_99','mae_01','rmse_01']
    data =  []
    for key in ticklab:
        data.append(metrics_md[key][0])  
    data = np.stack(data,axis=0)
    figname = out_path + 'metrics_mk'+str(kmask)+ mod_name[0]+'_ct%d'%len(mod_name)+'.png'
    axlab = ['','Error in SWH (m)','MLR']
    figsize = [3.3,2]
    ylim = [0,0.8]
    plot_bar(data,figname,figsize,axlab,leg,ticklab,size_label=12,capt='(a)',
             leg_col=3,ylim=ylim)

    