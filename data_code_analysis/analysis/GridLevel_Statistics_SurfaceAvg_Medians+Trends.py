# =============================================================================
# Imports 
# =============================================================================
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pymannkendall as mk
from scipy.stats import kendalltau
import warnings
warnings.filterwarnings('ignore')

# Set global font properties
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.style'] = 'normal'
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
filename = Path("PALLTER_EventLevel_DepthAvgCoreDataframe.csv")
loadpath = str(current_directory / absolute_path / filename)
df_SM = pd.read_csv(loadpath)

df_SM = df_SM[df_SM['Year']==2009]

#calculate yearly medians
temp = remove_bad_years(df_SM)
df_YM = temp.groupby(['Year']).median().reset_index()
df_YM = fill_missing_years(df_YM)

#determine climatological stats
climatology_stats = temp.describe()
ccc = df_SM[['RoundedGridLine', 'RoundedGridStation', 'Event', 'Depth', 
             'Diatoms', 'Cryptophytes', 'MixedFlagellates', 'MLD', 'QI']].dropna()
ccc = ccc[ccc['Depth']<=7]
# =============================================================================


# =============================================================================
# Define list of parameters and regions to run calculations on
# =============================================================================
parameters = [
              # 'MLD',
              # 'QI',
              # 'max_N2',
              
              # 'WWUpper',
              # 'WWLower',
              # 'WWThickness',
              # 'WW%Obs',
              
              # 'WWMinTemp',
              # 'WWMinTempDepth',
              # # 'MinTemp',
              # # 'MinTempDepth',
              
              # 'Temperature', 
              # 'Salinity',
              # 'Density',
              
              # 'WWMedTemp',
              # 'WWMedSal',
              # 'WWMedDens',
              
              # # 'BttmTemp',
              # # 'BttmSal',
              # # 'BttmDens',
              
              #  'SIDuration',
              #  'SIRetrProx',
              
              # 'Chlorophylla',
              # 'PrimaryProduction',
              # 'SpecPrimProd',
              
               'Diatoms',
               'Cryptophytes',
               'MixedFlagellates',
               'Type4Haptophytes',
               'Prasinophytes',
               'Evenness',
              
              # 'TCaro:Chla',
              # 'PSC:Chla',
              # 'PPC:Chla',
              # 'PrimPPC:Chla',
              # 'SecPPC:Chla',
              # 'PPC:TCaro',
              # 'PSC:TCaro',
              # 'PPC:PSC',
              # 'PrimPPC:PPC',
              # 'SecPPC:PPC'
              ]

regions = ['Full Grid', 
           'North', 
           'South', 
           'Coast', 
           'Shelf', 
           'Slope']
# =============================================================================


# =============================================================================
# Initiate results table
# =============================================================================
results_table = pd.MultiIndex.from_product([parameters, regions], 
                                           names=['Measurement', 'Region']).to_frame(index=False)
results_table = results_table.assign(n=np.nan,
                                     Minimum=np.nan,
                                     Q1=np.nan, 
                                     Median=np.nan, 
                                     Q3=np.nan,
                                     Maximum=np.nan,
                                     Trend=np.nan, 
                                     Slope=np.nan, 
                                     Tau=np.nan, 
                                     P_value=np.nan)
# =============================================================================


# =============================================================================
# Run calculations per parameter and region
# =============================================================================
for param in parameters:
    # filter down to necessary columns
    df_filt = df_SM[['Year', 'RoundedGridLine', 'RoundedGridStation', 
                  'Region', 'NSRegion', 'IORegion', param]]
    
    for region_name in regions:
        # classify region identifiers
        region_mapping = {
            1: ["Full Grid", "North", "Slope"],
            2: ["Full Grid", "North", "Shelf"],
            3: ["Full Grid", "North", "Coast"],
            4: ["Full Grid", "South", "Slope"],
            5: ["Full Grid", "South", "Shelf"],
            6: ["Full Grid", "South", "Coast"]}
        
        # locate region data
        region_identifiers = [key for key, value in region_mapping.items() if region_name in value]
        region_data = df_filt[df_filt['Region'].isin(region_identifiers)]
        
        # calculate yearly medians
        yearly_data = region_data.groupby('Year').median().reset_index()
        n = yearly_data[param].dropna().count()
        yearly_data = fill_missing_years(yearly_data)
        
        # # calculate climatelogical summary (using yearly grid-level medians)
        # q1 = yearly_data[param].quantile(0.25)
        # median = yearly_data[param].median()
        # q3 = yearly_data[param].quantile(0.75)
        
        # calculate climatelogical summary (using station-level medians)
        minimum = region_data[param].min()
        q1 = region_data[param].quantile(0.25)
        median = region_data[param].median()
        q3 = region_data[param].quantile(0.75)
        maximum = region_data[param].max()
        
        # run mann-kendall trend test (using yearly grid-level medians)
        trenddata = yearly_data.loc[:, [param]]
        results = mk.original_test(trenddata)
        trend = results.trend
        slope = results.slope
        tau = results.Tau
        p_value = results.p
        
        # # calculate theil-sen slope estimator seperately
        # yearly_data2 = yearly_data[yearly_data[param].notna()]
        # subsetx = yearly_data2['Year']
        # subsety = yearly_data2[param]
        # res = stats.theilslopes(subsety, subsetx, 0.95) #Theil-Sen estimator (regression)
        
        # insert results into results table
        results_table.loc[(results_table['Measurement'] == param) & (results_table['Region'] == region_name), 
                 ['n', 'Minimum', 'Q1', 'Median', 'Q3', 'Maximum','Trend', 'Slope', 'Tau', 'P_value']] = [n, minimum, q1, median, q3, maximum, trend, slope, tau, p_value]
# =============================================================================

# =============================================================================
# Change measurement and region labels in results table
# =============================================================================
# #rename parameters for publication
# measurement_rename = {
#     'MLD': 'Mixed Layer Depth',
#     'Chlorophylla': 'Chlorophylla',
#     'PPC:Chla': 'PPC:Chla'}
# results_table['Measurement'] = results_table['Measurement'].replace(measurement_rename)

#rename grid regions for simplicity
region_rename = {'Full Grid': 'FG',
                  'North': 'N',
                  'South': 'S',
                  'Slope': 'Sl',
                  'Shelf': 'Sh',
                  'Coast': 'C'}
results_table['Region'] = results_table['Region'].replace(region_rename)
# =============================================================================

# =============================================================================
# Reduce dataframe to region and/or parameter of interest
# =============================================================================
filt_results_table = results_table[results_table['Region'] == 'FG']
# filt_results_table = filt_results_table[filt_results_table['Measurement'] == 'PPC:PSC']
# =============================================================================

# # =============================================================================
# # Save results in Excel file
# # =============================================================================
# current_directory = Path.cwd()
# absolute_path = Path("local data/")

# filename1 = Path("PALLTER_Regional_Medians+TrendsTable.xlsx")
# savepath1 = str(current_directory / absolute_path / filename1)
# results_table.to_excel(str(savepath1), index=False)

# filename2 = Path("PALLTER_FullGrid_Medians+TrendsTable.xlsx")
# savepath2 = str(current_directory / absolute_path / filename2)
# filt_results_table.to_excel(str(savepath2), index=True)
# # =============================================================================