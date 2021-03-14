'''
Download of Eurostat demographic data using the Eurostat Python Package
https://pypi.org/project/eurostat/

Downloads all the population, fertility & mortality data needed for 
OG-MOD (to be processed in demographics.py).
'''
import eurostat
import xlsxwriter
import pandas as pd
import sys
import numpy as np

###### USER ENTRY: country and year ########
Country = 'UK'
Year = 2018
############################################

############## Download Basic Data - START ##################
StartPeriod = Year
EndPeriod = Year

filter_pars = {'GEO': [Country]}
df_pop = eurostat.get_sdmx_data_df('demo_pjan', StartPeriod, EndPeriod, filter_pars, flags = True, verbose=True)
df_mort = eurostat.get_sdmx_data_df('demo_magec', StartPeriod, EndPeriod, filter_pars, flags = True, verbose=True)
df_fert = eurostat.get_sdmx_data_df('demo_fasec', StartPeriod, EndPeriod, filter_pars, flags = True, verbose=True)

# print(df_pop)
# print(df_mort)
# print(df_fert)
############## Download Basic Data - END ##################

############## Isolate Required Mortality Data - START ##################
# STEP 1: Remove totals and other unused rows 
indexNames = df_mort[(df_mort['AGE'] == 'TOTAL') |
                     (df_mort['AGE'] == 'UNK') |
                     (df_mort['AGE'] == 'Y_LT1') |
                     (df_mort['AGE'] == 'Y_OPEN')].index 
df_mort.drop(indexNames , inplace=True)

# STEP 2: Keep only totals
df_mort = df_mort[(df_mort['SEX'] == 'T')]

# STEP 3: Select columns = Age, Frequency; drop others
df_mort = df_mort.drop(columns=['UNIT', 'SEX', 'GEO', 'FREQ', '2018_OBS_STATUS'])

print(df_mort)
############## Isolate Required Mortality Data - END ##################

############## Isolate Required Population Data - START ##################
# STEP 1: Remove totals and other unused rows 
indexNames = df_pop[(df_pop['AGE'] == 'TOTAL') |
                    (df_pop['AGE'] == 'UNK') |
                    (df_pop['AGE'] == 'Y_LT1') |
                    (df_pop['AGE'] == 'Y_OPEN')].index 
df_pop.drop(indexNames , inplace=True)

# STEP 2: Keep gender specific population, to calculate fertility per person
df_pop_m = df_pop[(df_pop['SEX'] == 'M')]
df_pop_f = df_pop[(df_pop['SEX'] == 'F')]
df_pop = df_pop[(df_pop['SEX'] == 'T')]

# STEP 3: Select columns = Age, Frequency; drop others
df_pop_m = df_pop_m.drop(columns=['UNIT', 'SEX', 'GEO', 'FREQ', '2018_OBS_STATUS'])
df_pop_f = df_pop_f.drop(columns=['UNIT', 'SEX', 'GEO', 'FREQ', '2018_OBS_STATUS'])
df_pop = df_pop.drop(columns=['UNIT', 'SEX', 'GEO', 'FREQ', '2018_OBS_STATUS'])

print(df_pop_m, df_pop_f, df_pop)
############## Isolate Required Population Data - END ##################

############## Isolate Required Fertility Data - START ##################
# STEP 1: Select Sex = T (meaning "Total" of boys and girls born); drop others
df_fert = df_fert[(df_fert['SEX'] == 'T')]

# STEP 2: Select columns = Age, Frequency; drop others
df_fert = df_fert.drop(columns=['UNIT', 'SEX', 'GEO', 'FREQ', '2018_OBS_STATUS'])

# STEP 3: Change Y10-14 to 14 (i.e. assume all under 15 as age 14)
        #    & Y_GE50  split between 50 (half) & 51 (half)
df_fert.AGE[df_fert.AGE=='Y10-14'] = 'Y14'
df_fert.AGE[df_fert.AGE=='Y_GE50'] = 'Y50'
df1row = pd.DataFrame([['Y51', 0]], columns=list(('AGE',2018)))
df_fert = df_fert.append(df1row, ignore_index=True)

# STEP 4: spread value for age 50+ evenly across age 50 and 51
df_fert2 = df_fert.copy()
mask = df_fert2['AGE'].str.startswith('Y50') | df_fert2['AGE'].str.startswith('Y51')
df_fert_age50_float = df_fert[2018].loc[df_fert['AGE'] == 'Y50'].values.astype(np.float)
df_fert.loc[mask, 2018] = df_fert_age50_float*0.5

# STEP 5: Remove remaining total and subtotals 
indexNames = df_fert[(df_fert['AGE'] == 'TOTAL') |
                     (df_fert['AGE'] == 'UNK') |
                     (df_fert['AGE'] == 'Y15-19') |
                     (df_fert['AGE'] == 'Y20-24') |
                     (df_fert['AGE'] == 'Y25-29') |
                     (df_fert['AGE'] == 'Y30-34') |
                     (df_fert['AGE'] == 'Y35-39') | 
                     (df_fert['AGE'] == 'Y40-44') | 
                     (df_fert['AGE'] == 'Y45-49')].index 
df_fert.drop(indexNames , inplace=True)

print('df_fert: ', df_fert)
############## Isolate Required Fertility Data - END ##################
