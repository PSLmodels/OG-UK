"""
Download of Eurostat demographic data using the Eurostat Python Package
https://pypi.org/project/eurostat/

Downloads all the population, fertility & mortality data needed for 
OG-MOD (to be processed in demographics.py).
"""
import eurostat
import xlsxwriter
import pandas as pd
import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


###### USER ENTRY: country and year ########
Country = "UK"
Year = 2018
############################################

############## Download Basic Data - START ##################
StartPeriod = Year
EndPeriod = Year

filter_pars = {"GEO": [Country]}
df_pop = eurostat.get_sdmx_data_df(
    "demo_pjan", StartPeriod, EndPeriod, filter_pars, flags=True, verbose=True
)
# df_mort = eurostat.get_sdmx_data_df('demo_magec', StartPeriod, EndPeriod, filter_pars, flags = True, verbose=True)
df_fert = eurostat.get_sdmx_data_df(
    "demo_fasec", StartPeriod, EndPeriod, filter_pars, flags=True, verbose=True
)
############## Download Basic Data - END ##################

############## Isolate Required Population Data - START ##################
# STEP 1: Remove totals and other unused rows
indexNames = df_pop[
    (df_pop["AGE"] == "TOTAL")
    | (df_pop["AGE"] == "UNK")
    | (df_pop["AGE"] == "Y_OPEN")
].index
df_pop.drop(indexNames, inplace=True)

# STEP: Rename Y_LT1 to 0 (means 'less than one year')
df_pop.AGE[df_pop.AGE == "Y_LT1"] = "Y0"

# STEP 2: Remove leading 'Y' from 'AGE' (e.g. 'Y23' --> '23')
df_pop["AGE"] = df_pop["AGE"].str[1:]

# STEP 3: Keep gender specific population, to calculate fertility per person
df_pop_m = df_pop[(df_pop["SEX"] == "M")]
df_pop_f = df_pop[(df_pop["SEX"] == "F")]
df_pop = df_pop[(df_pop["SEX"] == "T")]

# Name of 1 column includes the year - create column name before dropping
Obs_status_col = str(Year) + "_OBS_STATUS"
# Drop columns except: Age, Frequency
df_pop = df_pop.drop(columns=["UNIT", "SEX", "GEO", "FREQ", Obs_status_col])

# convert strings to float
df_pop = df_pop.astype(float)

# sort values by AGE
df_pop = df_pop.sort_values(by=["AGE"])
print("df_pop[-20:]: ", df_pop[-20:])

np_pop = df_pop[Year].to_numpy().astype(np.float)
############## Isolate Required Population Data - END ##################

# ############## Isolate Required Mortality Data - START ##################
# # STEP: Remove totals and other unused rows
# indexNames = df_mort[(df_mort['AGE'] == 'TOTAL') |
#                      (df_mort['AGE'] == 'UNK') |
#                      (df_mort['AGE'] == 'Y_OPEN')].index
# df_mort.drop(indexNames , inplace=True)

# # STEP: Rename Y_LT1 to 0 (means 'less than one year')
# df_mort.AGE[df_mort.AGE=='Y_LT1'] = 'Y0'

# # STEP: Remove leading 'Y' from 'AGE' (e.g. 'Y23' --> '23')
# df_mort['AGE'] = df_mort['AGE'].str[1:]

# # STEP: Keep only totals
# df_mort = df_mort[(df_mort['SEX'] == 'T')]

# # STEP: Select columns = Age, Frequency; drop others
# df_mort = df_mort.drop(columns=['UNIT', 'SEX', 'GEO', 'FREQ', Obs_status_col])

# # convert strings to float
# df_mort = df_mort.astype(float)

# # sort values by AGE
# df_mort = df_mort.sort_values(by=['AGE'])

# print('df_mort[-20:]: ', df_mort[-20:])
# ############## Isolate Required Mortality Data - END ##################

# ############## Calculate Mortality Rates & make csv - START ##################
# # adding total population column from df_pop
# df_mort['POP'] = df_pop[Year]

# # divide mortality number by total population = mortality rate
# df_mort['mort_rate'] = df_mort[Year] / df_mort['POP']

# print('df_mort[: 20]: ', df_mort[: 20])
# print('df_mort[-20:]: ', df_mort[-20:])

# TO DO: Assign to parameter: mort_rates = df_mort[].as_numpy ?
# ############## Calculate Mortality Rates & make csv - END ##################

############## Isolate Required Fertility Data - START ##################
# Select Sex = T (meaning "Total" of boys and girls born); drop others
df_fert = df_fert[(df_fert["SEX"] == "T")]

# Drop columns except: Age, Frequency
df_fert = df_fert.drop(columns=["UNIT", "SEX", "GEO", "FREQ", Obs_status_col])


# see: https://stackoverflow.com/questions/21420792/exponential-curve-fitting-in-scipy

# record total values for 10-14 year old and over 50 year old for tails
under15total = (
    df_fert[Year].loc[df_fert["AGE"] == "Y10-14"].values.astype(np.float)
)
over50total = (
    df_fert[Year].loc[df_fert["AGE"] == "Y_GE50"].values.astype(np.float)
)
print("under15total: ", under15total)
print("over50total: ", over50total)


# STEP 5: Remove remaining total and subtotals
indexNames = df_fert[
    (df_fert["AGE"] == "TOTAL")
    | (df_fert["AGE"] == "UNK")
    | (df_fert["AGE"] == "Y10-14")
    | (df_fert["AGE"] == "Y15-19")
    | (df_fert["AGE"] == "Y20-24")
    | (df_fert["AGE"] == "Y25-29")
    | (df_fert["AGE"] == "Y30-34")
    | (df_fert["AGE"] == "Y35-39")
    | (df_fert["AGE"] == "Y40-44")
    | (df_fert["AGE"] == "Y45-49")
    | (df_fert["AGE"] == "Y_GE50")
].index
df_fert.drop(indexNames, inplace=True)

# # STEP 6: Remove leading 'Y' from 'AGE' (e.g. 'Y23' --> '23')
# df_fert['AGE'] = df_fert['AGE'].str[1:]


np_fert = df_fert[Year].to_numpy().astype(np.float)

# STEP: spread values for age 50+ between 50 and 55
# using an exponential distribution.

# select final 6 values (ages 44-49) from which to estimate the top tail
Y_44_49 = np_fert[-7:-1]


def expon(x, a, b):
    return a * np.exp(-b * x)


x_44_49 = np.linspace(1, len(Y_44_49), len(Y_44_49))
print("Y: ", Y_44_49, " x_44_49: ", x_44_49)

popt_top, pcov = curve_fit(expon, x_44_49, Y_44_49)
print("popt_top: ", popt_top)

# plt.plot(x_44_49, Y_44_49, 'b-', label='fert data')
# plt.plot(x_44_49, expon(x_44_49, *popt_top), 'r-',
#          label='fit: a=%5.3f, b=%5.3f' % tuple(popt_top))
# plt.show()

# num_over50 is the number of years beyond age 49, e.g. 6 --> 50 to 55
num_over50 = 11
x_over50 = np.linspace(len(Y_44_49) + 1, len(Y_44_49) + num_over50, num_over50)
over50pred = expon(x_over50, *popt_top)

x_44_over50 = np.linspace(
    1, len(Y_44_49) + num_over50, len(Y_44_49) + num_over50
)

# plot
# plt.title('Fertility data ages 44-49 and predictions ages 44-60')
# plt.plot(x_44_49, Y_44_49, 'b-', label='fert data')
# plt.plot(x_44_over50, expon(x_44_over50, *popt_top), 'r-',
#          label='a.exp(-b x) fit: a=%5.3f, b=%5.3f' % tuple(popt_top))
# plt.legend()
# plt.show()

print("over50pred unscaled: ", over50pred)
over50pred = over50pred * over50total / over50pred.sum()
print("over50pred scaled: ", over50pred)

# STEP: spread single value for ages 10 to 14 between 10 and 14
# using an exponential distribution.

# select initial 3 values (ages 15-17) from which to estimate the bottom tail
# Note: taking more values misses how steep a decline there is in these ages
Y_15_17 = np_fert[:3]
print("Y_15_17 pre-flip: ", Y_15_17)
Y_15_17 = np.flip(Y_15_17)
print("Y_15_17 post-flip: ", Y_15_17)

x_15_17 = np.linspace(1, len(Y_15_17), len(Y_15_17))
print("Y_15_17: ", Y_15_17, " x_15_17: ", x_15_17)

popt_low, pcov = curve_fit(expon, x_15_17, Y_15_17)
print("popt_low: ", popt_low)

# num_under15 is the number of years below age 15: ages 10-14
num_under15 = 5
x_under15 = np.linspace(
    len(Y_15_17) + 1, len(Y_15_17) + num_under15, num_under15
)
under15pred = expon(x_under15, *popt_low)

x_under15_17 = np.linspace(
    1, len(Y_15_17) + num_under15, len(Y_15_17) + num_under15
)

# plot
# plt.title('Fertility data ages 17-15 and predictions ages 14-10')
# plt.plot(x_15_17, Y_15_17, 'b-', label='fert data')
# plt.plot(x_under15_17, expon(x_under15_17, *popt_low), 'r-',
#          label='a.exp(-b x) fit: a=%5.3f, b=%5.3f' % tuple(popt_low))
# plt.legend()
# plt.show()

print("under15pred unscaled: ", under15pred)
under15pred = under15pred * under15total / under15pred.sum()
under15pred = np.flip(under15pred)
print("under15pred scaled: ", under15pred)

# extend fert to 100 ages with zeros; ages 0-14 & 50-99
fert100 = np.hstack((under15pred, np_fert))
print("fert100: ", fert100)
fert100 = np.hstack((np.zeros(15 - num_under15), fert100))
print("fert100: ", fert100)
fert100 = np.hstack((fert100, over50pred))
print("fert100: ", fert100)
fert100 = np.hstack((fert100, np.zeros(50 - num_over50)))
print("fert100: ", fert100)
print("fert100.shape: ", fert100.shape)

# plt.plot(fert100)
# plt.show()

############## Isolate Required Fertility Data - END ##################

############## Calculate Fertility per person - START ##################
print("np_pop: ", np_pop)
fert_rates = fert100 / np_pop
print("fert_rates: ", fert_rates)

plt.plot(fert_rates)
plt.show()
############## Calculate Fertility per person - END ##################
