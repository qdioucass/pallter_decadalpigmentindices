# =============================================================================
# Imports 
# =============================================================================
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pymannkendall as mk
from scipy.stats import kendalltau

import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature as cnef
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.mpl.ticker as ctk

import warnings
warnings.filterwarnings('ignore')

# Set global font properties
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.style'] = 'normal'
# =============================================================================


# =============================================================================
# Custom Colormap
# =============================================================================
import matplotlib.colors as mcolors
#define a slightly darker shade of lightblue
darker_lightblue = "#5C8DCE"

#define the custom color progression with adjusted endpoint
adjusted_colors = ["lightblue", darker_lightblue]
reversed_colors = adjusted_colors[::-1]
adjusted_blues = mcolors.LinearSegmentedColormap.from_list("AdjustedBlues", adjusted_colors)
reversed_blues = mcolors.LinearSegmentedColormap.from_list("ReversedBlues", reversed_colors)
# =============================================================================


# =============================================================================
# Functions
# =============================================================================
def kendall_corr_with_pvalues(df):
    cols = df.columns
    kendall_corr = pd.DataFrame(index=cols, columns=cols, dtype=float)
    p_values = pd.DataFrame(index=cols, columns=cols, dtype=float)
    
    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                kendall_corr.loc[col1, col2] = 1.0
                p_values.loc[col1, col2] = 0.0
            else:
                sub = df[[col1,col2]].dropna()
                # sub = df.dropna(subset=[col1,col2])
                corr, p_value = kendalltau(sub[col1], sub[col2])
                kendall_corr.loc[col1, col2] = corr
                p_values.loc[col1, col2] = p_value
    
    return kendall_corr, p_values

def remove_bad_years(df):
    parambadyears = {
        'MLD': [1995, 2011],
        'QI': [1995],
        'max_N2': [1995, 2011],
        
        'WWUpper': [None],
        'WWLower': [None],
        'WWThickness': [None],
        'WW%Obs': [None],
        'WWMinTemp': [None],
        'WWMinTempDepth': [None],
        'SIAdvance': [None],
        'SIRetreat': [None],
        'SIDuration': [None],
        'IceDays': [None],
        'SIRetrProx': [None],
        'SIExtent': [None],
        'SIArea': [None],
        'OWArea': [None],
        'TotalSIConc': [None],
        
        'WWMedTemp': [None],
        'WWMedSal': [None],
        'WWMedDens': [None],
        
        'BttmTemp': [1995],
        'BttmSal': [1995],
        'BttmDens': [1995, 2017, 2018],
        
        'Temperature': [None], 
        'Salinity': [None],
        'Density': [1995], 
        
        # 'Temp_ML': [1995], 
        # 'Sal_ML': [1995],
        # 'Dens_ML': [1995],
        
        'MinTemp': [1995],
        'MinTempDepth': [1995],
        
        'TCaro:Chla': [None], 
        'PSC:Chla': [None],
        'PPC:Chla': [None], 
        'PrimPPC:Chla': [None],
        'SecPPC:Chla': [None],
        
        'PPC:TCaro': [None],
        'PSC:TCaro': [None],
        
        # 'TCaro:TAcc': [None],
        # 'PSC:TAcc': [None],
        # 'PPC:TAcc': [None],
        
        # 'Allo:PPC': [None],
        # 'Diadino:PPC': [None],
        # 'Diato:PPC': [None],
        # 'DD+DT:PPC': [None],
        # 'Zea:PPC': [None],
        # 'BCar:PPC': [None],
        # 'PrimPPC:PPC': [None],
        # 'SecPPC:PPC': [None],
        
        # 'Fuco:PSC': [None],
        # 'Hex-Fuco:PSC': [None],
        # 'But-Fuco:PSC': [None],
        # 'Perid:PSC': [None],
        
        # 'Allo:TCaro': [None],
        # 'Diadino:TCaro': [None],
        # 'Diato:TCaro': [None],
        # 'DD+DT:TCaro': [None],
        # 'Zea:TCaro': [None],
        # 'BCar:TCaro': [None],
        # 'Fuco:TCaro': [None],
        # 'Hex-Fuco:TCaro': [None],
        # 'But-Fuco:TCaro': [None],
        # 'Perid:TCaro': [None],
        
        # 'Allo': [None],
        # 'Diadino': [None],
        # 'Diato': [None],
        # 'DD+DT': [None],
        # 'Zea': [None],
        # 'BCar': [None],
        # 'Fuco': [None],
        # 'Hex-Fuco': [None],
        # 'But-Fuco': [None],
        # 'Perid': [None],
        
        # 'mPF': [None],
        # 'nPF': [None],
        # 'pPF': [None],
        
        'Chlorophylla': [None],
        # 'POC': [1994, 2009, 2015, 2016],
        'PrimaryProduction': [2019],
        'SpecPrimProd': [None],
        
        'Diatoms': [None],
        'Cryptophytes': [None],
        'MixedFlagellates': [None],
        'Type4Haptophytes': [None],
        'Prasinophytes': [None],
        
        # 'DiatomBiomass': [1993, 1994, 1998, 2015],
        # 'CryptophyteBiomass': [1993, 1994],
        # 'MixedFlagellateBiomass': [2013, 2014, 2015, 2016],
        # 'Type4HaptophyteBiomass': [1993, 1994, 1998, 2009],
        # 'PrasinophyteBiomass': [1993, 1998, 2013, 2017],
        
        # 'TAcc2:POC': [2009, 2015, 2016],
        
        # 'Chla:POC': [1994, 2009, 2015, 2016],
        # 'TCaro:POC': [2009, 2015, 2016],
        # 'TAcc:POC': [2009, 2015, 2016],
        # 'TPig:POC': [2009, 2015, 2016],
        # 'TAcc:Chla': [None],
        
        'SiO4': [1998],
        'PO4': [1998],
        'NO2': [1998],
        'NO3': [1998],
        'NO3plusNO2': [None],
        
        # 'FIRERho': [2015], #these have not been adjusted/blank corrected/filtered
        # 'FIRESigma': [2015], #these have not been adjusted/blank corrected/filtered
        # 'FIRE_FvFm': [2015], #these have not been adjusted/blank corrected/filtered
        
        # 'TCaro:POC': [1994],
        
        'Evenness': [None]}
    
    for param, bad_years in parambadyears.items():
        if param in df.columns and bad_years is not None:
            df.loc[df['Year'].isin(bad_years), param] = np.nan
    
    return df

def fill_missing_years(df):    
    all_years = np.arange(1991, 2021)
    existing_years = df['Year'].unique()
    missing_years = np.setdiff1d(all_years, existing_years)
    missing_data = pd.DataFrame({'Year': missing_years})
    missing_data = missing_data.assign(**{column: np.nan for column in df.columns if column != 'Year'})
    df = pd.concat([df, missing_data], ignore_index=True)
    df = df.sort_values('Year')
    return df
# =============================================================================


# =============================================================================
# Load Core Dataframe
# =============================================================================
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_EventLevel_SurfaceAvgCoreDataframe.csv")
loadpath = str(current_directory / absolute_path / filename)
df_SM = pd.read_csv(loadpath)

data = df_SM[df_SM['Year'] == 2009].reset_index(drop=True)

s = 30
X = data['StandardLon']
Y = data['StandardLat']
Z = data['max_N2']

# =============================================================================
# Define Map Extent & Projections
# =============================================================================
# Load & Select Bathymetry
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_BottomTopo_ETOPO2v2c_f4.nc")
loadpath = str(current_directory / absolute_path / filename)
ds_etpo = xr.open_dataset(loadpath)

#broad region selection
grid_bath = ds_etpo.sel(x=slice(-101,-50), y=slice(-76,-58))

# Load & Filter Standard Grid
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_CruiseStandardGridPointCoordinates.csv")
loadpath = str(current_directory / absolute_path / filename)
grid = pd.read_csv(loadpath)

#manually drop overland points from overlay grid
grid = grid.drop(grid[(grid['GridLine'] == 600) & (grid['GridStation'] <= 0)].index)
grid = grid.drop(grid[(grid['GridLine'] == 500) & (grid['GridStation'] == 40)].index)
grid = grid.drop(grid[(grid['GridLine'] == 500) & (grid['GridStation'] <= -20)].index)
grid = grid.drop(grid[(grid['GridLine'] == 400) & (grid['GridStation'] <= -20)].index)
grid = grid.drop(grid[(grid['GridLine'] == 300) & (grid['GridStation'] == 0)].index)
grid = grid.drop(grid[(grid['GridLine'] == 300) & (grid['GridStation'] < -60)].index)
grid = grid.drop(grid[(grid['GridLine'] == 100) & (grid['GridStation'] <= -60)].index)
grid = grid.drop(grid[(grid['GridLine'] == 0) & (grid['GridStation'] <= -60)].index)
grid = grid.drop(grid[(grid['GridLine'] == -100) & (grid['GridStation'] <= -40)].index)
grid = grid.drop(grid[(grid['GridLine'] == -200) & (grid['GridStation'] <= -100)].index)

#   ^The standard grid coordinates file contains all points in the grid, whether they
#    are real points or not, and so the non-real points have to be removed

mapextent = [min(grid['StandardLon'])-1, max(grid['StandardLon'])+6, 
             min(grid['StandardLat'])-3, max(grid['StandardLat'])+1]

data_proj = ccrs.PlateCarree()

map_proj = ccrs.SouthPolarStereo(central_longitude=-69)
# map_proj = ccrs.SouthPolarStereo(central_longitude=-55)
# map_proj = ccrs.RotatedPole(pole_latitude=44, pole_longitude=-9, central_rotated_longitude=0)

width = 5
height = 4.5
fig, ax = plt.subplots(1, 1, figsize=(width, height), dpi=1200, subplot_kw={'projection':map_proj})

#define map corners
lon1, lon2, lat1, lat2 = [-78.5, #left
                          -60.0, #right
                          -62.0, #top
                          -70.0] #bottom

#define rectangular path in geographic coordinates based on corners
rect = mpath.Path([[lon1, lat1], 
                   [lon2, lat1], 
                   [lon2, lat2], 
                   [lon1, lat2], 
                   [lon1, lat1]]).interpolated(50) #interpolate the path with 50 additional points for smoother edges

#transformation from PlateCarree to ax coordinate system
proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData

#transform path from geographic coordinates to ax coordinates
rect_in_target = proj_to_data.transform_path(rect)

#set boundary of ax to be the transformed rectangular path
ax.set_boundary(rect_in_target)

#adjust the x and y axis limits to match transformed rectangle's coordinates range
ax.set_xlim(rect_in_target.vertices[:,0].min(), rect_in_target.vertices[:,0].max())
ax.set_ylim(rect_in_target.vertices[:,1].min(), rect_in_target.vertices[:,1].max())

#define map features
coastline = cnef('physical', 'coastline', '10m', 
                 edgecolor='face', linewidth=0.75, facecolor='grey', zorder=0)
iceshelves = cnef('physical', 'antarctic_ice_shelves_polys', '10m', 
                  edgecolor='face', linewidth=0.75, facecolor='#fafafa', zorder=0)
ocean = cnef('physical', 'ocean', '50m', 
             edgecolor='face', linewidth=0.75, facecolor='lightblue', zorder=0)

#define bathymetry and adjust levels to get desired colors/definition
lvls = list(range(0, -7001, -1000))[::-1]
bx = grid_bath.x
by = grid_bath.y
bz = grid_bath.z

#plot map features
ax.add_feature(coastline, zorder=1, edgecolor='black')
ax.add_feature(iceshelves, zorder=1, edgecolor='black')        
ax.add_feature(ocean, zorder=0, edgecolor='black')

#plot bathymetry contours
filled_contour = ax.contourf(bx, by, bz, levels=lvls, transform=data_proj, cmap=reversed_blues, zorder=0.1, alpha=0.75)
line_contour = ax.contour(bx, by, bz, levels=lvls, linestyles ='solid', linewidths=.20, colors='black', 
           transform=data_proj, zorder=0.2)


#plot points with event type distinction
data_scatter = ax.scatter(X, Y, c=Z, s=s, transform=data_proj, marker='D', cmap='viridis', zorder=10)
cbar = fig.colorbar(data_scatter, ax=ax, orientation='vertical', pad=0.025)
# cbar.set_label('Temperature (°C)', fontweight='normal', fontsize=12, labelpad=8)
# cbar.ax.tick_params(labelsize=10)
cbar.ax.minorticks_on()


