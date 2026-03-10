# =============================================================================
# Imports 
# =============================================================================
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pymannkendall as mk
from scipy import stats
import warnings
from matplotlib.ticker import ScalarFormatter
warnings.filterwarnings('ignore')

# Set global font properties
fs = 12
plt.rcParams['font.size'] = fs
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.style'] = 'normal'
# =============================================================================

# =============================================================================
# Functions & Reference Dictionaries
# =============================================================================
def fill_missing_years(df):    
    all_years = np.arange(1991, 2021)
    existing_years = df['Year'].unique()
    missing_years = np.setdiff1d(all_years, existing_years)
    missing_data = pd.DataFrame({'Year': missing_years})
    missing_data = missing_data.assign(**{column: np.nan for column in df.columns if column != 'Year'})
    df = pd.concat([df, missing_data], ignore_index=True)
    df = df.sort_values('Year')
    return df

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

def filter_and_average_data(df, param):
    data = df
    
    # data_SM = fill_missing_years(data_SM)
    
    data_SM = remove_bad_years(data)
    
    return data_SM

def run_trend_analysis(df, param):
    x = 'Year'
    y = param
    data_YM = df.groupby('Year').median().reset_index()
    
    data_YM = fill_missing_years(data_YM)

    subset = data_YM[[x, y]]
    window_size = 4
    var = subset[y]
    RllAvg = var.rolling(window=window_size, center=True, min_periods=1).mean()
    data_YM['Rolling Average'] = RllAvg

    trenddata = data_YM.loc[:, [y]]
    results = mk.original_test(trenddata)
    p_value = results.p
    slope = results.slope
    
    return data_YM, slope, p_value

# =============================================================================

# =============================================================================
# Load Yearly Station Averaged Dataframe
# =============================================================================
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_StationLevel_SurfaceAvgCoreDataframe.csv")
loadpath = str(current_directory / absolute_path / filename)
df = pd.read_csv(loadpath)

# # -------------------------------------
# # Check Data Distributions
# # -------------------------------------
# # Plot Sample Distribution
# temp = df
# param = 'PPC:PSC'
# temp.sort_values('Year', inplace=True)
# temp = temp.reset_index(drop=True)
# # temp = temp.groupby(['Year','Cruise','RoundedGridLine','RoundedGridStation','Event']).median().reset_index()
# # temp = temp.groupby(['Year','RoundedGridLine','RoundedGridStation']).median().reset_index()
# temp = temp.dropna(subset=param)
# years = temp['Year'].unique()
# fig = plt.figure(figsize=(16,19), dpi=100.0, tight_layout=True)
# loc = 0
# for yr in years:
#     sub = temp[temp['Year'] == yr]
#     loc += 1
#     ax = fig.add_subplot(6,5, loc)
#     plt.scatter(x='RoundedGridStation', y='RoundedGridLine', data=sub)
#     plt.gca().invert_xaxis()
#     plt.xlim(260, -120)
#     plt.ylim(100, 700)
#     plt.title(yr)

# # Plot Yearly Station Counts Bar Plot
# param = 'RoundedGridStation'
# tempcount = temp.groupby('Year').count().reset_index()
# fig = plt.figure(figsize=(3,1.5), dpi=600, tight_layout=True)
# plt.bar(tempcount['Year'], tempcount[param], color='grey')
# plt.ylim(10,75)
# plt.xlim(1990,2021)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.ylabel('Count', fontsize=12)
# plt.title('Yearly Total Station Counts', fontsize=12, fontweight='bold')
# plt.minorticks_on()
# =============================================================================

# =============================================================================
# Plot Time Series ############################################################
# =============================================================================
plotrollingavg = False #True, False
plottheilsen = True #True, False

x = 'Year'
xticks = 'auto' #manual, auto

y = 'Prasinophytes'
yscale = 'linear' #linear, log, symlog
ylims = 'manual' #manual, auto, dynamic
ymin = 0
ymax = 101

identifier = None

units = ''
# units = ' (%)'
# units = ' (m)'
# units = ' (ratio)'
# units = ' (unitless)'
# units = '\n(mgChl $\mathregular{m^{-3}}$)' #Chlorophylla
# units = '\n(mgC $\mathregular{m^{-3}}$ $\mathregular{day^{-1}}$)' #PrimaryProduction
# units = '\n(mgC $\mathregular{(mgChl)^{-1}}$ $\mathregular{day^{-1}}$)' #SpecPrimProd

xlabel = None
ylabel = y + units
# ylabel = 'MLD (m)'
# ylabel = 'Max Buoyancy \nFreq. ($\mathregular{s^{-1}}$)'
# ylabel = 'Surface\nSalinity (PSU)'
# ylabel = 'Chla-Specific Production' + units
# ylabel = 'Quality Index\n(unitless)'
# ylabel = 'Temperature\nMin. Depth (m)'
# ylabel = 'Temperature\nMin. (C)'
# ylabel = 'WW Temperature\nMin. (C)'
# ylabel = 'WW Temperature\nMin. Depth (m)'
# ylabel = 'WW Upper\nBoundary Depth (m)'
# ylabel = 'WW Lower\nBoundary Depth (m)'
# ylabel = 'WW Thickness (m)'
# ylabel = 'WW % Observation'

font_size = 12
ident_size = 24
markersize = 10
linewidth = 1

width = 3.5
length = 1.5

fig = plt.figure(figsize=(width, length), dpi=1200)
ax1 = fig.add_subplot(1, 1, 1)

# ------------------------------------------
#set up defined variables
yscale = yscale
plotrollingavg = plotrollingavg
plottheilsen = plottheilsen
fs = font_size
lw = linewidth
ms = markersize
# ------------------------------------------

df[y] = df[y]*100

#run data filtering and trend test
data_SM = filter_and_average_data(df, y)
data_YM, slope, p_value = run_trend_analysis(data_SM, y)
data_YM2 = data_YM[data_YM[y].notna()]

#define Q1 and Q3
data_SMtemp = data_SM[data_SM[y].notna()]
df_stats = data_SMtemp.groupby('Year').describe()
Q1 = df_stats[(y, '25%')]
Q3 = df_stats[(y, '75%')]

#adjust axis params
ax1.tick_params('both', labelsize=8)
ax1.set_ylabel('Prasinophytes (%)', 
               color='Black', fontweight='normal', fontsize=12, labelpad=4)
ax1.set_yscale(yscale)
ax1.set_ylim(0, 100)
ax1.set_xlim(1990, 2021)
ax1.minorticks_on()
ax1.tick_params('x', labelbottom=True)
# ax1.set_title('Diatoms', fontweight='normal', fontsize=12)

if y == 'max_N2':
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1)) #adjust the range to use scientific notation
    ax1.yaxis.set_major_formatter(formatter)

#plot parameter points (markers, line, and gaps)
ax1.scatter(x, y, data=data_YM, s=ms, color='Black')
ax1.plot(x, y, linewidth=lw, data=data_YM, color='Black')
ax1.plot(x, y, linewidth=lw, data=data_YM2, color='Black', ls=':', zorder=0.1)

# #plot parameter bars w/ markers and trend
# ax1.bar(x, y, data=data_YM, color='green', zorder=0)
# ax1.plot(x, y, linewidth=lw, data=data_YM, marker='o', markersize=ms, color='k', zorder=10)
# ax1.plot(x, y, linewidth=lw, data=data_YM2, color='k', ls=':', zorder=10)

# #plot parameter as just bars
# ax1.bar(x, y, data=data_YM, color='black', zorder=0)
# # ax1.bar(x, y, data=data_YM, color='green', zorder=0)
# data_YM_dropped = data_YM[data_YM[y].isna()]
# data_YM_dropped[y] = 2.5
# ax1.scatter(data_YM_dropped[x], data_YM_dropped[y], s=12, marker='o', color='grey', linewidths=0, alpha=0.5, zorder=0)
# # ax1.scatter(data_YM2[x], Q1, s=6, marker='_', color='k', zorder=10)
# # ax1.scatter(data_YM2[x], Q3, s=6, marker='_', color='k', zorder=10)
# # ax1.scatter(data_YM2[x], data_YM2[y], s=2, marker='o', color='k', zorder=10)

#plot Q1 and Q3 shaded area
ax1.fill_between(data_YM2[x], Q1, Q3, color='#898989', zorder=0, alpha=0.250)

#plot Theil-Sen regression line
subsetx = data_YM2[x]
subsety = data_YM2[y]
res = stats.theilslopes(subsety, subsetx, 0.95) #Theil-Sen estimator (regression)
if plottheilsen==True:
    ax1.plot(subsetx, res[1] + res[0] * subsetx, '--', color='grey', zorder=0, alpha=0.85)

#add number of asterisks based on p-value
asterisks = get_asterisks(p_value)
ax1.annotate(asterisks, xy=(0.95, 0.95), xycoords='axes fraction', 
            fontsize=fs, fontweight='bold', color='black', ha='right', va='top')

# #add slope and p-value annotations
# if y == 'max_N2':
#     slope_str = f"Slope: {slope:.1e}"
# else:
#     slope_str = f"Slope: {slope:.3f}"
# p_val_str = f"P-Value: {p_value:.4f}"
# ax1.annotate(slope_str, xy=(0.04, 0.97), xycoords='axes fraction', 
#             fontsize=8, ha='left', va='top', alpha=0.75, zorder=10)
# ax1.annotate(p_val_str, xy=(0.04, 0.87), xycoords='axes fraction', 
#             fontsize=8, ha='left', va='top', alpha=0.75, zorder=10)

# #function-based plotting method
# plot_timeseries(ax1, df, x, y, xlabel, ylabel, ymin, ymax, yscale, 
#                 font_size, linewidth, markersize, ident_size, 
#                 identifier=identifier, identifier_location='upper left', 
#                 ylims=ylims, xticks=xticks, plotrollingavg=plotrollingavg,
#                 plottheilsen=plottheilsen)

l = 0 + 0.175
r = 1 - 0.03
t = 1 - 0.15
b = 0 + 0.125
plt.subplots_adjust(left=l, right=r, top=t, bottom=b)
# plt.tight_layout()
# =============================================================================


# =============================================================================
# Save Figure as PDF for Compiling in Adobe Illustrator
# =============================================================================
current_directory = Path.cwd()
absolute_path = Path("analysis/publication figures/Supporting Figures/FigS4/")
filename = Path("FigS4_Prasinophytes.pdf")
savepath = str(current_directory / absolute_path / filename)

plt.savefig(savepath)
# =============================================================================
