#!/usr/bin/env python
# coding: utf-8

# # Holocene EBM
# by Fiona Fang zf276@cam.ac.uk
# ## Volcanic Forcing

# In[2]:


import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

file_path_saod = r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\Sigl_2021_SAOD.nc"
dataset_saod = xr.open_dataset(file_path_saod)

print(dataset_saod)

saod_ad = dataset_saod['time'].values  # AD time
saod_bp = 1950 - saod_ad  # Convert to BP
saod_data = dataset_saod['aod550'].values

df_saod = pd.DataFrame({'Time': saod_bp, 'SAOD': saod_data.mean(axis=1)})
df_saod['Radiative Forcing'] = -25 * df_saod['SAOD']

vol_rf_dict = df_saod.set_index('Time')['Radiative Forcing'].to_dict()


# In[3]:


# Plot 
plt.figure(figsize=(14, 7))

# Plot SAOD
plt.subplot(2, 1, 1)
plt.plot(df_saod['Time'], df_saod['SAOD'], label='SAOD', color='b')
plt.xlabel('Year (BP)')
plt.ylabel('Stratospheric Aerosol Optical Depth (SAOD)')
plt.title('Stratospheric Aerosol Optical Depth (SAOD) over Time')
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()  # Invert x-axis to show 12000 BP to 0 BP

# Plot Radiative Forcing
plt.subplot(2, 1, 2)
plt.plot(df_saod['Time'], df_saod['Radiative Forcing'], label='Radiative Forcing', color='r')
plt.xlabel('Year (BP)')
plt.ylabel('Radiative Forcing (W/m²)')
plt.title('Radiative Forcing of Volcanic Eruptions over Time')
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()  # Invert x-axis to show 12000 BP to 0 BP

plt.tight_layout()
plt.show()


# ## Ice Albedo Forcing

# In[4]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
from scipy.interpolate import interp1d

ice_albedo = 0.7
planetary_albedo = 0.3
solar_constant = 1361  # in W/m²

file_paths = [
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.0.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.0.5.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.1.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.1.5.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.2.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.2.5.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.3.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.3.5.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.4.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.4.5.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.5.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.5.5.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.6.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.6.5.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.7.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.7.5.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.8.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.8.5.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.9.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.9.5.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.10.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.10.5.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.11.nc",
    r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\I6_C.VM5a_1deg.11.5.nc"
]


# In[19]:


def calculate_ice_cover_and_albedo(file_path):
    data = xr.open_dataset(file_path)
    sftgif = data['sftgif']

    # Earth's radius in meters
    R = 6371000

    # Convert latitude and longitude to radians
    lat_rad = np.deg2rad(data['lat'])
    lon_rad = np.deg2rad(data['lon'])

    # Calculate the differences in latitude and longitude
    dlat = np.diff(lat_rad)
    dlon = np.diff(lon_rad)

    # Calculate the area for each grid cell
    lat_matrix, lon_matrix = np.meshgrid(lat_rad, lon_rad, indexing='ij')
    area_matrix = np.zeros((len(data['lat']), len(data['lon'])))

    for i in range(len(dlat)):
        for j in range(len(dlon)):
            lat1 = lat_matrix[i, j]
            lat2 = lat_matrix[i + 1, j]
            lon1 = lon_matrix[i, j]
            lon2 = lon_matrix[i, j + 1]

            area_matrix[i, j] = (R**2 * np.abs(lon2 - lon1) *
                                 (np.sin(lat2) - np.sin(lat1)))

    # For the last row and column, replicate the area of the second to last row and column
    area_matrix[-1, :] = area_matrix[-2, :]
    area_matrix[:, -1] = area_matrix[:, -2]

    # Calculate the ice-covered area 
    ice_cover_area = area_matrix * (sftgif.values / 100)  
    total_ice_cover_area_m2 = np.sum(ice_cover_area)
    total_ice_cover_area_km2 = total_ice_cover_area_m2 / 1e6
    radiative_forcing = -ice_albedo * solar_constant * ((total_ice_cover_area_m2 - 1.686223e+13) / (4 * np.pi * R**2))
    
    return total_ice_cover_area_km2, radiative_forcing
 
results = []
for path in file_paths:
    try:
        ice_cover_area, radiative_forcing = calculate_ice_cover_and_albedo(path)
        results.append((path, ice_cover_area, radiative_forcing))
    except Exception as e:
        results.append((path, None, None, str(e)))


# In[10]:


# Create a DataFrame with year, ice cover area, and radiative forcing data
years = np.arange(0, 12.5, 0.5)  # 0 kyr BP to 12 kyr BP with 0.5 kyr intervals
ice_cover_data = []
radiative_forcing_data = []

# Extract the ice cover area and radiative forcing for each year
for year in years:
    matching_files = [file for file, _, _ in results if f'{year}' in file]
    if matching_files:
        index = [file for file, _, _ in results].index(matching_files[0])
        ice_cover_data.append(results[index][1])
        radiative_forcing_data.append(results[index][2])
    else:
        ice_cover_data.append(None)
        radiative_forcing_data.append(None)

# Create the DataFrame
data = pd.DataFrame({
    'Year (kyr BP)': years,
    'Ice Cover Area (sq km)': ice_cover_data,
    'Radiative Forcing (W/m²)': radiative_forcing_data
})

print (data)

# Remove NaN values for interpolation
data = data.dropna()

# Spline interpolation
years_interp = np.linspace(data['Year (kyr BP)'].min(), data['Year (kyr BP)'].max(), 500)
ice_cover_spline = interp1d(data['Year (kyr BP)'], data['Ice Cover Area (sq km)'], kind='cubic')
radiative_forcing_spline = interp1d(data['Year (kyr BP)'], data['Radiative Forcing (W/m²)'], kind='cubic')

ice_rf_dict = dict(zip(years_interp, radiative_forcing_spline(years_interp)))


# In[20]:


fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Ice Cover Area
color = 'tab:blue'
ax1.set_xlabel('Time (kyr BP)')
ax1.set_ylabel('Ice Cover Area (sq km)', color=color)
ax1.plot(data['Year (kyr BP)'], data['Ice Cover Area (sq km)'], 'o', color=color, label='Ice Cover Data')
ax1.plot(years_interp, ice_cover_spline(years_interp), '-', color=color, label='Ice Cover Spline')
ax1.tick_params(axis='y', labelcolor=color)
ax1.invert_xaxis()  # Invert the x-axis to show years from 12 kyr to 0 kyr

# Plot Radiative Forcing
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('Radiative Forcing (W/m²)', color=color)
ax2.plot(data['Year (kyr BP)'], data['Radiative Forcing (W/m²)'], 'x', color=color, label='Radiative Forcing Data')
ax2.plot(years_interp, radiative_forcing_spline(years_interp), '--', color=color, label='Radiative Forcing Spline')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.suptitle('Ice Cover Area and Radiative Forcing from 12,000 BP to Present', y=1.03)
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
plt.show()


# ## GHG forcing

# In[21]:


import pandas as pd
import matplotlib.pyplot as plt

co2_file_path = r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\CO2_stack_156K_spline_V1.tab"
ch4_file_path = r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\CH4_stack_156K_spline_V1.tab"
n2o_file_path = r"D:\OneDrive - University of Cambridge\Tripos\Projects\Holocene EBM\N2O_stack_134K_spline_V1.tab"

def load_and_process_data(file_path, gas):
    start_line = 0
    if gas == 'CO2':
        start_line = 15
    elif gas == 'CH4':
        start_line = 15
    elif gas == 'N2O':
        start_line = 18
    
    data = pd.read_csv(file_path, sep='\t', skiprows=start_line, comment='/', names=['AGE [ka BP]', f'{gas} [nmol/mol]', f'dR_[{gas}]'])
    forcing = data[['AGE [ka BP]', f'dR_[{gas}]']].rename(columns={'AGE [ka BP]': 'Year', f'dR_[{gas}]': f'{gas}_RF'})
    forcing['Year'] = -forcing['Year'] * 1000 + 1950
    return forcing

co2_forcing = load_and_process_data(co2_file_path, 'CO2')
ch4_forcing = load_and_process_data(ch4_file_path, 'CH4')
n2o_forcing = load_and_process_data(n2o_file_path, 'N2O')

ghg_forcing = co2_forcing.merge(ch4_forcing, on='Year').merge(n2o_forcing, on='Year')


# In[22]:


ghg_forcing['Total_RF'] = ghg_forcing['CO2_RF'] + ghg_forcing['CH4_RF'] + ghg_forcing['N2O_RF']
ghg_forcing['Year_BP'] = 1950 - ghg_forcing['Year']

ghg_forcing_dict = ghg_forcing.set_index('Year_BP')['Total_RF'].to_dict()


# In[23]:


plt.figure(figsize=(14, 7))

plt.plot(ghg_forcing['Year_BP'], ghg_forcing['CO2_RF'], label='CO2 Forcing', color='red')
plt.plot(ghg_forcing['Year_BP'], ghg_forcing['CH4_RF'], label='CH4 Forcing', color='blue')
plt.plot(ghg_forcing['Year_BP'], ghg_forcing['N2O_RF'], label='N2O Forcing', color='purple')
plt.plot(ghg_forcing['Year_BP'], ghg_forcing['Total_RF'], label='Total GHG Forcing', color='black', linewidth=2)

plt.title('Greenhouse Gases Forcing Over the Past 12,000 Years')
plt.xlabel('Age (yr BP)')
plt.ylabel('Global radiative forcing (∆W m²)')
plt.legend()
plt.grid(True)

plt.xlim(12000, 0)
plt.ylim(-1.4, 0.4)

plt.show()


# ## Total Radiative Forcing

# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt 

# Filter GHG forcing data to the most recent 12,000 years
ghg_forcing_recent = {k: v for k, v in ghg_forcing_dict.items() if k <= 12000}

# Convert ice albedo years to years (BP) and divide by 10
ice_rf_dict_corrected = {int(k * 1000): v / 10 for k, v in ice_rf_dict.items()}

# Century-smoothing for volcanic forcing
def century_smoothing(rf_dict):
    df_rf = pd.DataFrame(list(rf_dict.items()), columns=['Year_BP', 'Radiative_Forcing'])
    df_rf.set_index('Year_BP', inplace=True)
    df_rf.sort_index(inplace=True)
    smoothed_rf = df_rf.rolling(window=100, center=True).mean().dropna()
    return smoothed_rf.to_dict()['Radiative_Forcing']
vol_rf_dict_smoothed = century_smoothing(vol_rf_dict)


# In[25]:


def combine_radiative_forcings(vol_rf_dict, ice_rf_dict, ghg_forcing_dict):
    all_years = sorted(set(vol_rf_dict.keys()).union(set(ice_rf_dict.keys())).union(set(ghg_forcing_dict.keys())))
    combined_rf = {}
    
    for year in all_years:
        vol_rf = vol_rf_dict.get(year, 0)
        ice_rf = ice_rf_dict.get(year, 0)
        ghg_rf = ghg_forcing_dict.get(year, 0)
        total_rf = vol_rf + ice_rf + ghg_rf
        combined_rf[year] = total_rf
    
    return combined_rf

combined_rf_dict = combine_radiative_forcings(vol_rf_dict_smoothed, ice_rf_dict_corrected, ghg_forcing_recent)

# Convert the dictionaries to DataFrames for plotting
df_vol_rf_smoothed = pd.DataFrame(list(vol_rf_dict_smoothed.items()), columns=['Year_BP', 'Volcanic_RF'])
df_ice_rf = pd.DataFrame(list(ice_rf_dict_corrected.items()), columns=['Year_BP', 'Ice_RF'])
df_ghg_rf = pd.DataFrame(list(ghg_forcing_recent.items()), columns=['Year_BP', 'GHG_RF'])
df_combined_rf = pd.DataFrame(list(combined_rf_dict.items()), columns=['Year_BP', 'Total_Radiative_Forcing'])

# Sort DataFrames by Year_BP
df_vol_rf_smoothed = df_vol_rf_smoothed.sort_values(by='Year_BP')
df_ice_rf = df_ice_rf.sort_values(by='Year_BP')
df_ghg_rf = df_ghg_rf.sort_values(by='Year_BP')
df_combined_rf = df_combined_rf.sort_values(by='Year_BP')


# Plot
plt.figure(figsize=(14, 7))
plt.plot(df_combined_rf['Year_BP'], df_combined_rf['Total_Radiative_Forcing'], label='$\Delta R_{Total, Smoothed}$', color='lightgrey', linewidth=2)
plt.plot(df_vol_rf_smoothed['Year_BP'], df_vol_rf_smoothed['Volcanic_RF'], label='$\Delta R_{Volc}$', color='forestgreen')
plt.plot(df_ice_rf['Year_BP'], df_ice_rf['Ice_RF'], label='$\Delta R_{Albedo}$ (divided by 10)', color='dodgerblue')
plt.plot(df_ghg_rf['Year_BP'], df_ghg_rf['GHG_RF'], label='$\Delta R_{GHG}$', color='firebrick')

plt.xlabel('Age (yr BP)')
plt.ylabel('Global radiative forcing (ΔW/m²)')
plt.title('Global Radiative Forcing over Time during the Holocene')
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()  # Invert x-axis to show 12000 BP to 0 BP
plt.ylim(-4, 0.5)
plt.xlim(12000, 0)
plt.show()

