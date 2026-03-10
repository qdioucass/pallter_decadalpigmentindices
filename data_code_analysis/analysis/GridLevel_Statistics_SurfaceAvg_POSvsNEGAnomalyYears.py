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
# =============================================================================


# =============================================================================
# Plot KDE and Stats Results ##################################################
# =============================================================================
def PosNegAnomDiffTest(df, anomaly, param, level='station'):
    # PLOT AND TEST SIG DIFFERENCE BETWEEN POS AND NEG YEARS
    plt.figure(figsize=(8, 8), dpi=600)
    anomaly = anomaly
    param = param
    
    #run data filtering and trend test
    data_SM = filter_and_average_data(df, anomaly)
    data_YM, slope, p_value = run_trend_analysis(data_SM, anomaly)
    if level == 'grid':
        data = data_YM
    if level == 'station':
        data = data_SM
    #calculate anomaly per year
    anomparam = anomaly+'_Anomaly'
    anom_median = data[anomaly].median()
    data[anomparam] = data[anomaly] - anom_median
    data = fill_missing_years(data)
    positive_data = data[data[anomparam] > 0]
    negative_data = data[data[anomparam] < 0]
    data_pos = positive_data[param].dropna()
    data_neg = negative_data[param].dropna()
    data_all = data[param].dropna()
    
    # Compute quartiles and median for anomaly years
    q1_pos, median_pos, q3_pos = np.percentile(data_pos, [25, 50, 75])
    q1_neg, median_neg, q3_neg = np.percentile(data_neg, [25, 50, 75])
    q1_all, median_all, q3_all = np.percentile(data_all, [25, 50, 75])
    
    import seaborn as sns
    # KDE plot for both datasets
    sns.kdeplot(data_pos, label="Positive Anomaly Years", fill=True, color="red", alpha=0.5, clip=(0, None))
    sns.kdeplot(data_neg, label="Negative Anomaly Years", fill=True, color="blue", alpha=0.5, clip=(0, None))
    sns.kdeplot(data_all, label="All Years", fill=True, color="grey", alpha=0.5, clip=(0, None))
    
    # # Plot vertical lines for positive anomaly years
    # plt.axvline(x=median_pos, color='red', linestyle='-', linewidth=2)
    # plt.axvline(x=q1_pos, color='red', linestyle='--', linewidth=2)
    # plt.axvline(x=q3_pos, color='red', linestyle='--', linewidth=2)
    
    # # Plot vertical lines for negative anomaly years
    # plt.axvline(x=median_neg, color='blue', linestyle='-', linewidth=2)
    # plt.axvline(x=q1_neg, color='blue', linestyle='--', linewidth=2)
    # plt.axvline(x=q3_neg, color='blue', linestyle='--', linewidth=2)
    
    # # Plot vertical lines for all years
    # plt.axvline(x=median_all, color='grey', linestyle='-', linewidth=2)
    # plt.axvline(x=q1_all, color='grey', linestyle='--', linewidth=2)
    # plt.axvline(x=q3_all, color='grey', linestyle='--', linewidth=2)
    
    # # Add title and labels
    # plt.title(anomparam+": KDE with Median, Q1, and Q3")
    # plt.xlabel(param)
    # plt.ylabel("Density")
    
    # Ensure legend shows only one label per quartile
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # Mann-Whitney U Test (for median comparison)
    from scipy.stats import mannwhitneyu
    mw_stat, mw_p_value = mannwhitneyu(data_pos, data_neg)
    print("POS Years vs NEG Years")
    print(f"Mann-Whitney U test statistic: {mw_stat}, p-value: {mw_p_value}")
    if mw_p_value < 0.05:
        print("The medians are significantly different.")
    else:
        print("The medians are not significantly different.")
    
    # mw_stat, mw_p_value = mannwhitneyu(data_neg, data_all)
    # print("NEG Years vs ALL Years")
    # print(f"Mann-Whitney U test statistic: {mw_stat}, p-value: {mw_p_value}")
    # if mw_p_value < 0.05:
    #     print("The medians are significantly different.")
    # else:
    #     print("The medians are not significantly different.")
        
    # mw_stat, mw_p_value = mannwhitneyu(data_pos, data_all)
    # print("POS Years vs ALL Years")
    # print(f"Mann-Whitney U test statistic: {mw_stat}, p-value: {mw_p_value}")
    # if mw_p_value < 0.05:
    #     print("The medians are significantly different.")
    # else:
    #     print("The medians are not significantly different.")
    
    print("------DATA MEDIANS +/-IQR------")
    print(f"POS Median: {median_pos} +/-{ q3_pos-q1_pos}")
    print(f"NEG Median: {median_neg} +/-{ q3_neg-q1_neg}")
    print(f"ALL Median: {median_all} +/-{ q3_all-q1_all}")
# =============================================================================

anomalyparam = 'MLD'
testparam = 'Cryptophytes'
PosNegAnomDiffTest(df, anomalyparam, testparam, level='grid')
# =============================================================================
