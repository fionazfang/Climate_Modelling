import numpy as np
import netCDF4 as nc
import xarray as xr

file_path_vssi = 'D:/OneDrive - University of Cambridge/学术蓄水/全新世模型/volcano_Sigl-etal_2021_allfiles/HolVol_volcanic_stratospheric_sulfur_injection_v1.0.nc'
file_path_sao = 'D:/OneDrive - University of Cambridge/学术蓄水/全新世模型/volcano_Sigl-etal_2021_allfiles/HolVol_SAOD_-9500_1900_v1.0.nc'

dataset_vssi = xr.open_dataset(file_path_vssi)
dataset_sao = xr.open_dataset(file_path_sao)

print(dataset_vssi)
print(dataset_sao)

import pandas as pd

years_vssi = dataset_vssi['year'].values
vssi_data = dataset_vssi['vssi'].values
df_vssi = pd.DataFrame({'Year': years_vssi, 'VSSI': vssi_data})

time_sao = dataset_sao['time'].values
aod550_data = dataset_sao['aod550'].values
df_sao = pd.DataFrame({'Time': time_sao, 'SAOD': aod550_data.mean(axis=1)})