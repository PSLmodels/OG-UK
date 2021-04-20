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

a = np.array(['1.1', '2.2', '3.3'])
print(a)
# ['"1.1"' '"2.2"' '"3.3"']

# b = np.array([x.strip('"') for x in a])
# print(b)
# ['1.1' '2.2' '3.3']

c = a.astype(np.float)
print(c)
# [ 1.1  2.2  3.3]




dfmi = pd.DataFrame([list('abcd'),
                    list('efgh'),
                    list('ijkl'),
                    list('mnop')],
                    columns=pd.MultiIndex.from_product([['one', 'two'],
                                                ['first', 'second']]))

print('dfmi: ', dfmi)
print('dfmi - one - second: ', dfmi['one']['second'])
print('dfmi - one - second wi loc: ', dfmi.loc[:, ('one', 'second')])

dfmi.loc[:, ('one', 'second')] = 100
print('dfmi: ', dfmi)

import numpy as np
dfc = pd.DataFrame({'a': ['one', 'one', 'two',
                    'three', 'two', 'one', 'six'],
                    'c': np.arange(7)})
print('dfc: ', dfc)

dfd = dfc.copy()
print('dfd: ', dfd)
mask = dfd['a'].str.startswith('o')
dfd.loc[mask, 'c'] = 42
print('dfd: ', dfd)

# sys.exit('stop after indexing practicing')


# df = pd.DataFrame({'angles': [0, 3, 4],
#                    'degrees': [360, 180, 360]},
#                   index=['circle', 'triangle', 'rectangle'])
# print('df: ', df)

# df = df + 1
# print('df: ', df)

# df = df.add(1)
# print('df: ', df)

# df = df - [1, 2]
# print('df: ', df)

# other = pd.DataFrame({'angles': [0, 3, 4]},
#                      index=['circle', 'triangle', 'rectangle'])
# print('other: ', other)

# # df["angles"] = df["angles"] * other["angles"]
# # print('df: ', df)
# # print('count & describe df: ', df.count(), df.describe())

# df = df.mul(other, fill_value=0)
# print('df mul: ', df)

# df = df.replace(0, 7777)
# print('df replace: ', df)

# df = df.replace({7777: 5555, 12: 4444})
# print('df replace 2: ', df)

# # fails: df["angles", "circle"] = df.values["angles", "triangle"]

# # import numpy as np
# # df['First Season'] = np.where(df['First Season'] > 1990, 1, df['First Season'])

# # works: df['angles'] = (df['angles'] > 1000).astype(int)
# # print('df boolean: ', df)

# print('df angles triangle: ', df["angles"]["triangle"])

# # df_values = df.values["angles"]["triangle"]
# # print('df_values: ', df_values)


# df['angles'].loc[(df['angles'] == 4444)] = 2222
# df['degrees'].loc[(df['degrees'] == 5555)] = 0.5 * df["angles"]["triangle"]

# # import numpy as np

# # df2 = pd.DataFrame({'A': 'foo bar foo bar foo bar foo foo'.split(),
# #                'B': 'one one two three two two one three'.split(),
# #                'C': np.arange(8), 'D': np.arange(8) * 2})

# # print('df2: ', df2)

# # print(df2.loc[df['D'] == 14]['A'].index.values)

# # print(df2.loc[df['D'] == 14]['A'].values)
# # works: df['angles'].loc[(df['angles'] == 4444)] = 2222
# # df['angles'].loc[(df['angles'] == 4444)] = df.values[('angles', 'circle')]

#     #     df_trnc_gph = df[(df['Total Labor Income'] > 5) &
#     #         (df['Total Labor Income'] < 800000) &
#     #         (df['Total Capital Income'] > 5) &
#     #         (df['Total Capital Income'] < 800000)]

#     #     df_etr = df[['MTR Labor', 'MTR capital income',
#     #         'Total Labor Income', 'Total Capital Income',
#     #         'Effective Tax Rate', 'Weights']]
#     #     df_etr = df_etr[
#     #         (np.isfinite(df_etr['Effective Tax Rate'])) &
#     #         (np.isfinite(df_etr['Total Labor Income'])) &
#     #         (np.isfinite(df_etr['Total Capital Income'])) &
#     #         (np.isfinite(df_etr['Weights']))]

#     # data_trnc = \
#     #     data_trnc.drop(data_trnc[data_trnc['Effective Tax Rate'] < -0.15].index)



# # fails: df["angles", "triangle"] = df["angles", "triangle"] * other["angles", "triangle"]

# # fails: df[("angles", "triangle")] = df[("angles", "triangle")] * other[("angles", "triangle")]

# # fails: df(["angles", "triangle"]) = df(["angles", "triangle"]) * other(["angles", "triangle"])


# print('df: ', df)

# import pandas as pd
# df = pd.DataFrame({"Salesman" : ["Patrick", "Sara", "Randy"],
#                   "order date" : pd.date_range(start='2020-08-01', periods=3),
#                   "Item Desc " : ["White Wine", "Whisky", "Red Wine"],
#                   "Price Per-Unit": [10, 20, 30], 
#                   "Order Quantity" : [50, 10, 40],
#                   99: ["remak1", "remark2", "remark3"]})

# print('df: ',  df)
# df.columns = df.columns.astype("str")
# print('df: ',  df)
# df.columns = df.columns.map(lambda x : x.replace("-", "_").replace(" ", "_"))
# print('df: ',  df)
import numpy as np
df = pd.DataFrame(np.ones((5,6)),columns=['one','two','three',
                                            'four','five','six'])
print('df: ', df)
df.one = df.one*5

df.three = df.three.multiply(5)
df['four'] = df['four']*5
df.loc[:, 'five'] = df.loc[:, 'five'] * 7
df.iloc[:, 1] = df.iloc[:, 1]*4

df.loc[3, 'five'] = df.loc[3, 'five'] * 100

print('df: ', df)

# sys.exit('temp try out commands')


###### USER ENTRY: country and year ########
Country = 'UK'
Year = 2018
############################################

############## Download Basic Data - START ##################
StartPeriod = Year
EndPeriod = Year

filter_pars = {'GEO': [Country]}
# df_pop = eurostat.get_sdmx_data_df('demo_pjan', StartPeriod, EndPeriod, filter_pars, flags = True, verbose=True)
# df_mort = eurostat.get_sdmx_data_df('demo_magec', StartPeriod, EndPeriod, filter_pars, flags = True, verbose=True)
df_fert = eurostat.get_sdmx_data_df('demo_fasec', StartPeriod, EndPeriod, filter_pars, flags = True, verbose=True)

# print(df_pop)
# print(df_mort)
print(df_fert)
############## Download Basic Data - END ##################

############## Create Fertility csv file - START ##################
# Create a Pandas dataframe from some data.
# Create a Pandas Excel writer using XlsxWriter as the engine.
# writer = pd.ExcelWriter('df_fert.xlsx', engine='xlsxwriter')
# # Convert the dataframe to an XlsxWriter Excel object.
# df_fert.to_excel(writer, sheet_name='df_fert')
# # Close the Pandas Excel writer and output the Excel file.
# writer.save()

# STEP 2: Select Sex = T (meaning "Total" of boys and girls born)
df_fert = df_fert[(df_fert['SEX'] == 'T')]

# STEP 1: Select columns = Age, Frequency
df_fert = df_fert.drop(columns=['UNIT', 'SEX', 'GEO', 'FREQ', '2018_OBS_STATUS'])


    #     txrates = df['MTR Labor']
    # elif rate_type == 'mtry':
    #     txrates = df['MTR capital income']
    # x_10pctl = df['Total Labor Income'].quantile(0.1)
    # y_10pctl = df['Total Capital Income'].quantile(0.1)
    # ###############before it was quantile(.2), but giving us Nan's#############
    # x_20pctl = df['Total Labor Income'].quantile(.3)
    # y_20pctl = df['Total Capital Income'].quantile(.3)
    # ###########################################################################
    # min_x = txrates[(df['Total Capital Income'] < y_10pctl)].min()
    # min_y = txrates[(df['Total Labor Income'] < x_10pctl)].min()
 
print('df_fert: ', df_fert)

# # Create a Pandas dataframe from some data.
# # Create a Pandas Excel writer using XlsxWriter as the engine.
# writer = pd.ExcelWriter('df_fert_slim.xlsx', engine='xlsxwriter')
# # Convert the dataframe to an XlsxWriter Excel object.
# df_fert.to_excel(writer, sheet_name='df_fert')
# # Close the Pandas Excel writer and output the Excel file.
# writer.save()

# STEP 3: Change Y10-14 to 14 (i.e. assume all under 15 as age 14)
        #    & Y_GE50  split between 50 (half) & 51 (half)
df_fert.AGE[df_fert.AGE=='Y10-14'] = 'Y14'
df_fert.AGE[df_fert.AGE=='Y_GE50'] = 'Y50'

df1row = pd.DataFrame([['Y51', 0]], columns=list(('AGE',2018)))
print('df1row: ', df1row)
df_fert = df_fert.append(df1row, ignore_index=True)

# df_fert.columns = df_fert.columns.astype("str")
# df_fert.columns = df_fert.columns.map(lambda x : x.replace("2018", "Y2018"))

# df["col"] = 2 * df["col"]

# print('df_fert.Y2018[df_fert.AGE==Y50]: ', df_fert.Y2018[df_fert.AGE=='Y50'])
# print('type(df_fert.Y2018[df_fert.AGE==Y50]): ', type(df_fert.Y2018[df_fert.AGE=='Y50']))
# print('df_fert[Y2018].loc[(df[AGE] == Y50)]: ', df_fert['Y2018'].loc[(df_fert['AGE'] == 'Y50')])
# print('df_fert[Y2018].values[(df[AGE] == Y50)]: ', df_fert['Y2018'].values[(df_fert['AGE'] == 'Y50')])
# # data_trnc = \
#     data_trnc.drop(data_trnc[data_trnc['Effective Tax Rate'] < -0.15].index)

print('df_fert: ', df_fert)

print('df_fert.loc[df_fert[AGE] == Y50]: ', df_fert.loc[df_fert['AGE'] == 'Y50'])
print('df_fert[Y2018].loc[df_fert[AGE] == Y50]: ', df_fert[2018].loc[df_fert['AGE'] == 'Y50'])


df_fert2 = df_fert.copy()
mask = df_fert2['AGE'].str.startswith('Y50') | df_fert2['AGE'].str.startswith('Y51')
df_fert2.loc[mask, 2018] = 88
print('df_fert2: ', df_fert2)
# df_fert_to_numpy = df_fert[2018].loc[df_fert['AGE'] == 'Y50'].to_numpy()
df_fert_to_numpy = df_fert[2018].loc[df_fert['AGE'] == 'Y50'].values
print('df_fert_to_numpy: ', df_fert_to_numpy)
print('type df_fert_to_numpy: ', type(df_fert_to_numpy))
df_fert_to_numpy3 = df_fert_to_numpy
print('df_fert_to_numpy3: ', df_fert_to_numpy3)
df_fert_to_numpy4 = df_fert_to_numpy3.astype(np.float)
print(df_fert_to_numpy4)        # answer=263 as a number!
print(df_fert_to_numpy4*3)

df_fert_to_numpy5 = df_fert_to_numpy.astype(np.float)
print(df_fert_to_numpy5)       
print(df_fert_to_numpy5*0.5)

df_fert_to_numpy6 = df_fert[2018].loc[df_fert['AGE'] == 'Y50'].values.astype(np.float)
print(df_fert_to_numpy6)       
print(df_fert_to_numpy6*0.5)


df_fert.loc[mask, 2018] = df_fert_to_numpy6*0.5

# half_age50plus = 0.5 *  df_fert2[2018].loc[df_fert2['AGE'] == 'Y50'].astype(np.float)
# print('half_age50plus: ', half_age50plus)   # has the row label in it: half_age50plus:  45    44.0
# # df_fert.loc[mask, 2018] = 0.5 *  df_fert2[2018].loc[df_fert2['AGE'] == 'Y50'].astype(np.float)
# df_fert.loc[mask, 2018] = half_age50plus
print('df_fert: ', df_fert)

sys.exit('stop after loc')

# df_fert.loc['Y2018'] = df.loc['Y2018'] * 0.5
# write value three times: df_fert[2018].loc[df_fert['AGE'] == 'Y50'] = df_fert[2018].loc[df_fert['AGE'] == 'Y50'] * 3
# (df_fert['AGE'] == 'Y50')


# df = pd.DataFrame(np.ones((5,6)),columns=['one','two','three',
#                                        'four','five','six'])
# df.one *=5
# df.two = df.two*5
# df.three = df.three.multiply(5)
# df['four'] = df['four']*5
# df.loc[:, 'five'] *=5
# df.iloc[:, 5] = df.iloc[:, 5]*5


# df_fert['Y2018'].loc[(df_fert['AGE'] == 'Y51')] = df_fert["Y2018"].values[(df_fert['AGE'] == 'Y50')]

# df_fert["Y2018"] = 1000 * df_fert["Y2018"] 

# sys.exit()


# df_fert.Y2018[df_fert.AGE=='Y51'] = 0.5 * df_fert.Y2018[df_fert.AGE=='Y50']
# df_fert.Y2018[df_fert.AGE=='Y50'] = 0.5 * df_fert.Y2018[df_fert.AGE=='Y50']


# STEP 4: Select rows = All but remaining aggregates: Y15-19, etc, TOTAL, UNK 
print('df_fert: ', df_fert)


############## Create Fertility csv file - END ##################
