# =============================================================================
# Imports 
# =============================================================================
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import seaborn as sns
from scipy.stats import kendalltau
import warnings
warnings.filterwarnings('ignore')

# Set global font properties
fs = 12
plt.rcParams['font.size'] = fs
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.style'] = 'normal'
# =============================================================================


# =============================================================================
# Functions
# =============================================================================
def get_asterisks(p_value):
    if p_value < 0.0001:
        return '****'
    elif p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''

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
        'POC': [2009, 2015, 2016],
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
filename = Path("PALLTER_StationLevel_SurfaceAvgCoreDataframe.csv")
loadpath = str(current_directory / absolute_path / filename)
df_SM = pd.read_csv(loadpath)

#calculate yearly medians
temp = remove_bad_years(df_SM)
df_YM = temp.groupby(['Year']).median().reset_index()
df_YM = fill_missing_years(df_YM)

#determine climatological stats
climatology_stats = df_YM.describe()
# =============================================================================


# # =============================================================================
# # Miscellaneous Tests
# # =============================================================================
# x = 'Year'
# y = 'MLD'
# subset = df_YM[[x, y]]
# trenddata = df_YM.loc[:, [y]]
# results = mk.trend_free_pre_whitening_modification_test(trenddata)
# p_value = results.p
# slope = results.slope
# print(results)

# # Test for serial autocorrelation (significant p-value at specific lag = autocorrelation)
# from statsmodels.stats.diagnostic import acorr_ljungbox
# trenddata = trenddata.dropna()
# lb_test = acorr_ljungbox(trenddata, lags=10, return_df=True)
# print(lb_test)
# # =============================================================================


# =============================================================================
# Plot Grid-Level Regression
# =============================================================================
width = 2.25
height = 1.75
fig = plt.figure(figsize=(width,height), dpi=1200)
ax1 = fig.add_subplot(1, 1, 1)
markersize = 32
x='SIRetrProx'
y='Temperature'
c=None
cmap = 'viridis'
xscale = 'linear' #linear, log, symlog
yscale = 'linear' #linear, log, symlog
data = df_YM

# data = data[data['Year'] != 2009]

# # Move y-axis ticks to the right
# ax1.yaxis.tick_right()
# ax1.yaxis.set_label_position("right")

#plot points (use normalized colorbar if parameter for color given)
if c == None:
    points = ax1.scatter(x=x, y=y, c=None, cmap=cmap, data=data, marker='o',
                         s=markersize, color='grey', linewidths=0.75, edgecolor='k', zorder=3)
else:
    # points = ax1.scatter(x=x, y=y, c=c, cmap=cmap, data=data, zorder=0)
    # plt.colorbar(points, label=c)
    # ------------------------------------------
    norm = matplotlib.colors.Normalize(vmin=data[c].min(), vmax=data[c].max())
    # norm = matplotlib.colors.LogNorm(vmin=data[c].min(), vmax=data[c].max(), clip=True)
    # norm = matplotlib.colors.CenteredNorm(vcenter=data[c].median(), clip=True)
    # norm = matplotlib.colors.SymLogNorm(data[c].median(), linscale=1.0, vmin=data[c].min(), vmax=data[c].max())
    # norm = matplotlib.colors.TwoSlopeNorm(data[c].median(), vmin=data[c].min(), vmax=data[c].max())
    #       ^TwoSlopeNorm sets a midpoint (in this case, the parameter median) and then imposes two
    #       different linear scales on either side based on the distance to vmin/vmax and number
    #       of values in that half
    points = ax1.scatter(x=x, y=y, c=c, cmap=cmap, norm=norm, data=data, s=markersize, zorder=3)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), 
                      label=c, pad=0.035, shrink=0.95)
    cb.ax.tick_params(labelsize=10)
    cb.ax.minorticks_on()

#ajust axis parameters
ax1.set_xscale(xscale)
ax1.set_yscale(yscale)
ax1.set_xlabel('Retreat Proximity\n(relative days)', fontweight='normal', size=12, labelpad=2)
ax1.set_ylabel('$\mathregular{T_{Surf}}$ (°C)', fontweight='normal', size=12, labelpad=6)
ax1.tick_params('x', labelbottom=True)
ax1.tick_params('both', labelsize=8)
ax1.set_ylim(-1, 2.4)
# ax1.set_xlim(10, 75)
ax1.minorticks_on()

if y=='max_N2':
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1)) #adjust the range to use scientific notation
    ax1.yaxis.set_major_formatter(formatter)
    ax1.yaxis.offsetText.set_fontsize(8)

#plot theil-sen regression line
from scipy import stats
sub = data.loc[:, [x,y]]
sub = sub.dropna(subset=[x,y])
res = stats.theilslopes(sub[y], sub[x], 0.95) #Theil-Sen regression
x_range = np.linspace(ax1.get_xlim()[0], ax1.get_xlim()[1], 100) #plot to extent of axes
# x_range = np.linspace(min(sub[x]), max(sub[x]), 100) #plot to extent of points
y_range = res[0] * x_range + res[1]
ax1.plot(x_range, y_range, 'r--', lw=1, color='grey')

# #plot 1:1 regression line
# min_val = min(min(x), min(y))
# max_val = max(max(x), max(y))
# ax1.plot([0.01, 10], [0.01, 10], 'r--', label='1:1 Line')

#calculate kendall tau correlation
sub = data.dropna(subset=[x,y])
coef, pval = kendalltau(sub[x], sub[y])
print(f'---{x} vs {y}---')
print('Kendall correlation coefficient: %.3f' % coef)
print('p-value: %.3f' % pval)
print('----------------')

#add number of asterisks based on p-value
asterisks = get_asterisks(pval)

# ASTERISKS ON THE RIGHT
ax1.annotate(asterisks, xy=(0.95, 0.95), xycoords='axes fraction', 
            fontsize=10, fontweight='bold', color='black', ha='right', va='top')

# # ASTERISKS ON THE LEFT
# ax1.annotate(asterisks, xy=(0.05, 0.95), xycoords='axes fraction', 
#             fontsize=10, fontweight='bold', color='black', ha='left', va='top')


l = 0 + 0.34
r = 1 - 0.035
t = 1 - 0.125
b = 0 + 0.35
plt.subplots_adjust(left=l, right=r, top=t, bottom=b)
# =============================================================================

# =============================================================================
# Save Figure as PDF for Compiling in Adobe Illustrator
# =============================================================================
current_directory = Path.cwd()
absolute_path = Path("analysis/publication figures/Supporting Figures/FigS2/")
filename = Path("FigS2_TEMPvRETR.pdf")
savepath = str(current_directory / absolute_path / filename)

plt.savefig(savepath)
# =============================================================================
