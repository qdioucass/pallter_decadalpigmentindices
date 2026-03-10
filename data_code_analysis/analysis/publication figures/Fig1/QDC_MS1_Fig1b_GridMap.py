# =============================================================================
# Imports 
# =============================================================================
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature as cnef
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.mpl.ticker as ctk

# Set global font properties
fs = 12
plt.rcParams['font.size'] = fs
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.style'] = 'normal'
# =============================================================================


# =============================================================================
# Functions
# =============================================================================
def assign_IORegion(row, criteria):
    for line, regions in criteria.items():
        if row['GridLine'] == line:
            if row['GridStation'] > regions['Slope']:
                return 'Sl'
            elif regions['Shelf'][0] <= row['GridStation'] <= regions['Shelf'][1]:
                return 'Sh'
            elif row['GridStation'] <= regions['Coast']:
                return 'C'
    # If no criteria match, return None or a default value
    return None
# =============================================================================


# =============================================================================
# Load & Select Bathymetry
# =============================================================================
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_BottomTopo_ETOPO2v2c_f4.nc")
loadpath = str(current_directory / absolute_path / filename)
ds_etpo = xr.open_dataset(loadpath)

#broad region selection
grid_bath = ds_etpo.sel(x=slice(-101,-50), y=slice(-76,-58))
# =============================================================================


# =============================================================================
# Load & Filter Standard Grid
# =============================================================================
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
# =============================================================================


# # =============================================================================
# # Assign Regional Identifiers
# # =============================================================================
# #define regional classification dictionary (based on EDI Cruise dataset)
# criteria = {
#     600: {'Slope': 160, 'Shelf': (60, 160), 'Coast': 40},
#     500: {'Slope': 180, 'Shelf': (80, 180), 'Coast': 60},
#     400: {'Slope': 160, 'Shelf': (60, 160), 'Coast': 40},
#     300: {'Slope': 160, 'Shelf': (60, 160), 'Coast': 40},
#     200: {'Slope': 140, 'Shelf': (60, 140), 'Coast': 40},
#     100: {'Slope': 120, 'Shelf': (20, 120), 'Coast': 0},
#     0:   {'Slope': 100, 'Shelf': (0, 100), 'Coast': -20},
#     -100:{'Slope': 100, 'Shelf': (40, 100), 'Coast': 20},
#     -200:{'Slope': 140, 'Shelf': (0, 140), 'Coast': -20},}

# #assign inshore-offshore regions
# grid['IORegion'] = grid.apply(assign_IORegion, axis=1, criteria=criteria)
# # =============================================================================


# =============================================================================
# Assign Event Type Identifiers
# =============================================================================
grid['EventType'] = np.nan
# Specific combinations for CTD events (determined based on modern sampling scheme and visual inspection of 2009-2020 cruises)
combinations = {(600, 40), (600, 100), (600, 200), 
                (500, 60), (500, 100), (500, 200), #500/40 does not exist due to island, so CTD is performed at 500/60 
                (400, 40), (400, 100), (400, 200), 
                (300, -40), (300, 40), (300, 100), (300, 200), 
                (200, -60), (200, 0), (200, 40), (200, 100), (200, 200)}
value1 = 'CTD'
value2 = 'Underway'
# Assign based on matching combinations
grid['EventType'] = grid.apply(
    lambda row: value1 if (row['GridLine'],row['GridStation']) in combinations else value2, axis=1)
# =============================================================================

        
# =============================================================================
# Separate by Region  
# =============================================================================
gridns = grid[(grid['GridLine'] >= 200)].reset_index(drop=True)
gridfs = grid[(grid['GridLine'] < 200)].reset_index(drop=True)

# gridnsc = gridns[(gridns['IORegion'] == 'C')].reset_index(drop=True)
# gridnssh = gridns[(gridns['IORegion'] == 'Sh')].reset_index(drop=True)
# gridnssl = gridns[(gridns['IORegion'] == 'Sl')].reset_index(drop=True)

# gridfsc = gridfs[(gridfs['IORegion'] == 'C')].reset_index(drop=True)
# gridfssh = gridfs[(gridfs['IORegion'] == 'Sh')].reset_index(drop=True)
# gridfssl = gridfs[(gridfs['IORegion'] == 'Sl')].reset_index(drop=True)
# =============================================================================


# =============================================================================
# Separate by Event Type  
# =============================================================================
gridns_ctd = gridns[(gridns['EventType'] == 'CTD')].reset_index(drop=True)
gridns_uw = gridns[(gridns['EventType'] == 'Underway')].reset_index(drop=True)

# gridfs_ctd = gridfs[(gridfs['EventType'] == 'CTD')].reset_index(drop=True)
# gridfs_uw = gridfs[(gridfs['EventType'] == 'Underway')].reset_index(drop=True)
# =============================================================================


# =============================================================================
# Define Map Extent & Projections
# =============================================================================
mapextent = [min(grid['StandardLon'])-1, max(grid['StandardLon'])+6, 
             min(grid['StandardLat'])-3, max(grid['StandardLat'])+1]

data_proj = ccrs.PlateCarree()

map_proj = ccrs.SouthPolarStereo(central_longitude=-69)
# map_proj = ccrs.SouthPolarStereo(central_longitude=-55)
# map_proj = ccrs.RotatedPole(pole_latitude=44, pole_longitude=-9, central_rotated_longitude=0)
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
# Plot Standard Grid Map - Use Inshore-Offshore Subregion Colors
# =============================================================================
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

#isolate point coordinates
nsx = gridns['StandardLon'] 
nsy = gridns['StandardLat']
fsx = gridfs['StandardLon'] 
fsy = gridfs['StandardLat']

ctd_nsx = gridns_ctd['StandardLon'] 
ctd_nsy = gridns_ctd['StandardLat']
uw_nsx = gridns_uw['StandardLon'] 
uw_nsy = gridns_uw['StandardLat']

# #isolate point coordinates
# nscx = gridnsc['StandardLon'] 
# nscy = gridnsc['StandardLat']
# nsshx = gridnssh['StandardLon'] 
# nsshy = gridnssh['StandardLat']
# nsslx = gridnssl['StandardLon'] 
# nssly = gridnssl['StandardLat']
# fscx = gridfsc['StandardLon'] 
# fscy = gridfsc['StandardLat']
# fsshx = gridfssh['StandardLon'] 
# fsshy = gridfssh['StandardLat']
# fsslx = gridfssl['StandardLon'] 
# fssly = gridfssl['StandardLat']

#plot map features
ax.add_feature(coastline, zorder=1, edgecolor='black')
ax.add_feature(iceshelves, zorder=1, edgecolor='black')        
ax.add_feature(ocean, zorder=0, edgecolor='black')

#plot bathymetry contours
filled_contour = ax.contourf(bx, by, bz, levels=lvls, transform=data_proj, cmap=reversed_blues, zorder=0.1, alpha=0.75)
line_contour = ax.contour(bx, by, bz, levels=lvls, linestyles ='solid', linewidths=.20, colors='black', 
           transform=data_proj, zorder=0.2)

# #add custom legend for bathymetry
# import matplotlib.patches as mpatches
# lighterbluespatch = mpatches.Patch(color=reversed_blues(0.9), label='< 1000 m', edgecolor='black')
# darkerbluespatch = mpatches.Patch(color=reversed_blues(0.1), label='> 1000 m', edgecolor='black')
# legend = ax.legend(handles=[lighterbluespatch, darkerbluespatch],
#                    loc='upper left',
#                    bbox_to_anchor=(0.04, 0.96), 
#                    title='Bathymetry',
#                    fontsize=8,
#                    title_fontsize=8,
#                    borderpad=0.5,  #space between the legend content and the border
#                    handlelength=1.5,  #length of the patches
#                    handleheight=1,  #height of the patches
#                    labelspacing=0.2)
# legend.get_frame().set_facecolor('grey')
# legend.get_frame().set_alpha(0.9)
# legend.get_title().set_fontweight('bold')

#define face colors for slope, shelf, coast
facecolor1 = '#95a5a6'
facecolor2 = '#34495e'
facecolor3 = '#ecf0f1'

# #alternate face colors
# facecolor1 = '#D55E00' #red       - Slope
# facecolor2 = '#FFA500' #orange    - Shelf
# facecolor3 = '#F0E442' #yellow    - Coast

# #plot coast, shelf, slope regions in north and south
# markersize = 25
# ax.scatter(nscx, nscy, s=markersize, transform=data_proj, marker='o',
#            edgecolor='black', linewidth=0.75, facecolor=facecolor3, zorder=10)
# ax.scatter(nsshx, nsshy, s=markersize, transform=data_proj, marker='o', 
#            edgecolor='black', linewidth=0.75, facecolor=facecolor2, zorder=10)
# ax.scatter(nsslx, nssly, s=markersize, transform=data_proj, marker='o', 
#            edgecolor='black', linewidth=0.75, facecolor=facecolor1, zorder=10)

# #plot coast, shelf, slope regions in far south 
# ax.scatter(fscx, fscy, s=s, transform=data_proj, marker='o', 
#             edgecolor='black', facecolor=facecolor3, zorder=10, alpha=0.5)
# ax.scatter(fsshx, fsshy, s=s, transform=data_proj, marker='o', 
#             edgecolor='black', facecolor=facecolor2, zorder=10, alpha=0.5)
# ax.scatter(fsslx, fssly, s=s, transform=data_proj, marker='o',
#             edgecolor='black', facecolor=facecolor1, zorder=10, alpha=0.5)



# #plot points without regional distinction
# markersize = 20
# ax.scatter(nsx, nsy, s=markersize, transform=data_proj, marker='o',
#            edgecolor='black', linewidth=0.75, facecolor='#FFA500', zorder=10)
# # ax.scatter(fsx, fsy, s=markersize, transform=data_proj, marker='o', 
# #             edgecolor='black', facecolor=facecolor3, zorder=10, alpha=0.25)



#plot points with event type distinction
markersize = 20
ax.scatter(ctd_nsx, ctd_nsy, s=markersize+5, transform=data_proj, marker='s',
            edgecolor='black', linewidth=0.75, facecolor='palegoldenrod', zorder=10) #Brown:825501 - DarkOrange: #B57602
ax.scatter(uw_nsx, uw_nsy, s=markersize, transform=data_proj, marker='o', 
            edgecolor='black', linewidth=0.75, facecolor='#FFA500', zorder=10)



#add grid lines and adjust kwargs
gl=ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, 
                linewidth=0.5, color='dimgray', alpha=0.5, linestyle='--')
gl.top_labels=False
gl.right_labels=True
gl.xlabel_style = {'size': 8, 'font': 'Times New Roman'}
gl.ylabel_style = {'size': 8, 'font': 'Times New Roman'}
gl.rotate_labels=False
gl.xlocator=ctk.LongitudeLocator(4)
gl.ylocator=ctk.LatitudeLocator(6)
gl.xformatter=ctk.LongitudeFormatter(zero_direction_label=False)
gl.yformatter=ctk.LatitudeFormatter()
# =============================================================================


# # =============================================================================
# # Save Figure as PDF for Compiling in Adobe Illustrator
# # =============================================================================
# current_directory = Path.cwd()
# absolute_path = Path("analysis/publication figures/Fig1/")
# filename = Path("Fig1b_WAPLTERGridMap.pdf")
# savepath = str(current_directory / absolute_path / filename)

# # fig.subplots_adjust(left=0.125, right=0.975, top=0.95, bottom=0.175)

# plt.savefig(savepath, transparent=True)
# # =============================================================================
