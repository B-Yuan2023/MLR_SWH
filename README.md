# Multivariate Linear Regression: introduction

This repository is for downscaling physical fields using 
multivariate linear regression. 
Here the model is applied to downscale significant wave height (SWH) 
in the Black Sea using low-resolution data from ERA5 and high-resolution 
data from CMEMS. 
Both self-variable downscaling from low-resolutoin SWH and cross-variable 
downscaling from low-resolution wind fields are applied. 

Batch files are used to run the model on the DKRZ Levante cluster. 
The batch files contain the name of the parameter file (par*.py) to be read for runnning. 
Examples of the parameter files and the corresponding batch files are in *files_par_bash*. 
To run the model, these files should be in the parent directory of the directory 
*scripts*.

pytorch-msssim is from 
[https://github.com/VainF/pytorch-msssim]  

===========================================================
## How to use 

To run the model without batch file, change 'mod_name' from 'sys.argv[1]' 
to the name of the parameter file (e.g., 'par55e') in train.py. 
Next in a linux terminal: python train.py, or run with python IDE like spyder. 

To train the model:   
```	
	python train.py, or sbatch train_r55e_md0.sh

```
To test the model:   
```	python test.py or sbatch test_r55e_md0.sh   
```
To test the model with specified data period:   
```	python test_t.py or sbatch test_r55e_tuse1.sh   
```

Mainly used python scripts and steps to produce figures:  
===========================================================
Main scripts for training and testing:  
1. train.py:   
	
2. test.py:   
	generating hr data from lr testing dataset.  
	save mean, 99th 1st percentiles, and metrics rmse/mae for each epoch.  
	sort rmse and rmse_99 for all repeated runs.   
3. test_t.py:  
	as test.py, but use testing period defined in par*_tuse*.py (set rtra=0 such that period in tlim is all for testing).  

Scripts needed:  
datasets.py: class mydataset, loss function.  
funs_prepost.py: functions for pre- and post-processing.  
par*.py: parameter file (located in the parent path of \scripts), including model parameters and information of dataset (file dir, nc variable names etc.).  

===========================================================
## Scripts used for plotting

Comparison of metrics: (plot)  
4. compare_metrics.py: (plot)  
	compare metrics between different experiments (e.g. scale factor) using ensemble model.  

Scripts needed:  
funs_prepost.py: functions for post-processing.  
===========================================================
Comparison of 2D spatial pattern and time series at selected location in user defined period: (plot)  
5. test_tuse.py:  
	testing for the user defined (short) period instead of whole test set.  
	save hr, interpolation results, ensemble sr in the user defined period.  
	stations selected based on plot_study_domain.py (modify select_sta in funs_sites.py for custom use).   
6. compare_2D_pnt.py:  
	compare sr & hr at selected time (2d map plot) and selected stations (line plot).  

7. test_time.py:  
	check the referencing time after training for a selected period.  

===========================================================


