"""
-------------------------------------------------------------------------------
Functions for generating demographic objects necessary for the OG-UK model
-------------------------------------------------------------------------------
"""
# Import packages
import os
import numpy as np
import pandas as pd
import eurostat
import parameter_plots as pp
from scipy.optimize import curve_fit

# Create current directory path object
CUR_PATH = os.path.split(os.path.abspath(__file__))[0]

# Create demographic data directory path object
DATA_DIR = os.path.join(CUR_PATH, "data", "demographic")
if os.access(DATA_DIR, os.F_OK) is False:
    os.makedirs(DATA_DIR)

# Create demographic figures directory path object
FIG_DIR = os.path.join(CUR_PATH, "figures", "demographic")
if os.access(FIG_DIR, os.F_OK) is False:
    os.makedirs(FIG_DIR)

"""
------------------------------------------------------------------------
Define functions
------------------------------------------------------------------------
"""


def get_fert(totpers, base_yr, graph=False):
    """
    This function generates a vector of fertility rates by model period
    age that corresponds to the fertility rate data by age in years
    using data from Eurostat.

    Args:
        totpers (int): total number of agent life periods (E+S), >= 3
        base_yr: base year
        graph (bool): =True if want graphical output

    Returns:
        fert_rates (Numpy array): fertility rates for each model period
            of life
    """
    Country = "UK"
    Year = base_yr

    ############## Download Eurostat Data - START ##################
    StartPeriod = Year
    EndPeriod = Year

    filter_pars = {"GEO": [Country]}
    df_pop = eurostat.get_sdmx_data_df(
        "demo_pjan",
        StartPeriod,
        EndPeriod,
        filter_pars,
        flags=True,
        verbose=True,
    )
    df_fert = eurostat.get_sdmx_data_df(
        "demo_fasec",
        StartPeriod,
        EndPeriod,
        filter_pars,
        flags=True,
        verbose=True,
    )
    ############## Download Eurostat Data - END ##################

    # TO DO: Decide how to load population
    #        probably don't want to load separately in get_fert

    ############## Process Population Data - START ##################
    # Remove totals and other unused rows
    indexNames = df_pop[
        (df_pop["AGE"] == "TOTAL")
        | (df_pop["AGE"] == "UNK")
        | (df_pop["AGE"] == "Y_OPEN")
    ].index
    df_pop.drop(indexNames, inplace=True)

    # Rename Y_LT1 to 0 (means 'less than one year')
    df_pop.AGE[df_pop.AGE == "Y_LT1"] = "Y0"

    #  Remove leading 'Y' from 'AGE' (e.g. 'Y23' --> '23')
    df_pop["AGE"] = df_pop["AGE"].str[1:]

    # Drop gender specific population, keep only total
    df_pop = df_pop[(df_pop["SEX"] == "T")]

    # Name of 1 column includes the year - create column name before dropping
    Obs_status_col = str(Year) + "_OBS_STATUS"
    # Drop columns except: Age, Frequency
    df_pop = df_pop.drop(
        columns=["UNIT", "SEX", "GEO", "FREQ", Obs_status_col]
    )

    # convert strings to float to allow for sort_values
    df_pop = df_pop.astype(float)

    # sort values by AGE
    df_pop = df_pop.sort_values(by=["AGE"])

    np_pop = df_pop[Year].to_numpy().astype(np.float)
    ############## Process Population Data - END ##################

    ############## Select Fertility Data - START ##################
    # Select Sex = T (meaning "Total" of boys and girls born); drop others
    df_fert = df_fert[(df_fert["SEX"] == "T")]

    # Drop columns except: Age, Frequency
    df_fert = df_fert.drop(
        columns=["UNIT", "SEX", "GEO", "FREQ", Obs_status_col]
    )

    # Record values for 10-14 year old and over 50 year old for tail estimation
    under15total = (
        df_fert[Year].loc[df_fert["AGE"] == "Y10-14"].values.astype(np.float)
    )
    over50total = (
        df_fert[Year].loc[df_fert["AGE"] == "Y_GE50"].values.astype(np.float)
    )

    # Remove remaining total and subtotals
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

    # convert to numpy array, keeping only fertility values
    np_fert = df_fert[Year].to_numpy().astype(np.float)
    ############## Select Fertility Data - START ##################

    ############## Add tails for under 15 and over 50 - START ######
    # data contains single values for ages 10-14 & over 50
    # spread data from ages 10-14 and 50-60
    # using expontial function, based on shape of adjacent data

    # Top tail estimation:
    # select final 6 single-age values (ages 44-49)
    Y_44_49 = np_fert[-7:-1]
    x_44_49 = np.linspace(1, len(Y_44_49), len(Y_44_49))

    # define negative exponential curve
    def expon(x, a, b):
        return a * np.exp(-b * x)

    # estimate the best fit
    popt_top, pcov = curve_fit(expon, x_44_49, Y_44_49)

    # num_over50 is the number of years beyond age 49, e.g. 11 --> 50 to 60
    num_over50 = 11
    x_over50 = np.linspace(
        len(Y_44_49) + 1, len(Y_44_49) + num_over50, num_over50
    )

    # predict over 50 values based on estimated curve
    over50pred_unscaled = expon(x_over50, *popt_top)

    # scale predicted values to match the total over 50 births
    over50pred = over50pred_unscaled * over50total / over50pred_unscaled.sum()

    if graph:
        x_44_over50 = np.linspace(
            1, len(Y_44_49) + num_over50, len(Y_44_49) + num_over50
        )
        plt.title("Fertility data ages 44-49 and predictions ages 44-60")
        plt.plot(x_44_49, Y_44_49, "b-", label="fert data")
        plt.plot(
            x_44_over50,
            expon(x_44_over50, *popt_top),
            "r-",
            label="a.exp(-b x) fit: a=%5.3f, b=%5.3f" % tuple(popt_top),
        )
        plt.legend()
        plt.show()

    # Bottom tail estimation:
    # select initial 3 values (ages 15-17)
    # Note: taking more than 3 values misses the steep decline in the data
    Y_15_17 = np_fert[:3]
    Y_15_17 = np.flip(Y_15_17)
    x_15_17 = np.linspace(1, len(Y_15_17), len(Y_15_17))

    # estimate the best fit
    popt_low, pcov = curve_fit(expon, x_15_17, Y_15_17)

    # num_under15 is the number of years below age 15: ages 10-14
    num_under15 = 5
    x_under15 = np.linspace(
        len(Y_15_17) + 1, len(Y_15_17) + num_under15, num_under15
    )

    # predict under 15 values based on estimated curve
    under15pred_unscaled = expon(x_under15, *popt_low)

    # scale predicted values to match the total under 15 births
    under15pred = (
        under15pred_unscaled * under15total / under15pred_unscaled.sum()
    )
    under15pred = np.flip(under15pred)

    if graph:
        x_under15_17 = np.linspace(
            1, len(Y_15_17) + num_under15, len(Y_15_17) + num_under15
        )
        plt.title("Fertility data ages 17-15 and predictions ages 14-10")
        plt.plot(x_15_17, Y_15_17, "b-", label="fert data")
        plt.plot(
            x_under15_17,
            expon(x_under15_17, *popt_low),
            "r-",
            label="a.exp(-b x) fit: a=%5.3f, b=%5.3f" % tuple(popt_low),
        )
        plt.legend()
        plt.show()
    ############## Add tails for under 15 and over 50 - END ########

    ############## Calculate rate for all ages - START #############
    # extend fert to 100 ages with values for tails and zero elsewhere
    # under 15 year olds
    fert100 = np.hstack((under15pred, np_fert))
    fert100 = np.hstack((np.zeros(15 - num_under15), fert100))
    # over 50 year olds
    fert100 = np.hstack((fert100, over50pred))
    fert100 = np.hstack((fert100, np.zeros(50 - num_over50)))

    # convert to fertility rates per person
    fert_rates = fert100 / np_pop

    if graph:
        plt.title("Fertility rate by age per person")
        plt.plot(fert_rates)
        plt.show()
    ############## Calculate rate for all ages - END #############

    return fert_rates

    def get_mort(
        totpers,
        min_age_yr,
        max_age_yr,
        beg_yr=2018,
        end_yr=2018,
        download=False,
        save_data=False,
        graph=False,
    ):
    """
    This function generates a vector of mortality rates by model period age.
    Source: Eurostat demographic data using the Eurostat Python package
            https://pypi.org/project/eurostat/

    Args:
        totpers (int): total number of agent life periods (E+S), >= 3
        min_age_yr (int): age in years at which agents are born, >= 0
        max_age_yr (int): age in years at which agents die with certainty,
            > min_age_yr. For example, max_age_yr = 100 means that a model
            agent dies at the beginning of the year in which they turn 100 (at
            the end of their 99th year)
        download (bool): =True to download data from Eurostat. Otherwise, load
            data from mort_rate_data.csv in DATA_DIR
        save_data (bool): =True and download=True then save df_mort DataFrame
            as mort_rate_data.csv file in DATA_DIR
        graph (bool): =True if want graphical output

    Returns:
        mort_rates (Numpy array): mortality rates that correspond to each model
            period of life
        infmort_rate (scalar): infant mortality rate
    """
    # The infant mortality rate in the U.K. is reported to be 3.507 deaths per
    # 1,000 live births in 2021 (see https://www.macrotrends.net/countries/...
    # GBR/united-kingdom/infant-mortality-rate).
    infmort_rate = 3.507 / 1000

    if download:
        # Get U.K. mortality rate and total population data from Eurostat and
        # clean it
        country = "UK"
        mort_data_beg_yr = beg_yr
        mort_data_end_yr = end_yr
        pop_data_beg_yr = beg_yr
        pop_data_end_yr = end_yr
        filter_pars = {"GEO": [country]}
        df_mort = eurostat.get_sdmx_data_df(
            "demo_magec",
            mort_data_beg_yr,
            mort_data_end_yr,
            filter_pars,
            flags=True,
            verbose=True,
        )
        df_pop = eurostat.get_sdmx_data_df(
            "demo_pjan",
            pop_data_beg_yr,
            pop_data_end_yr,
            filter_pars,
            flags=True,
            verbose=True,
        )
        # Delete columns that we don't use (keep only columns that do use)
        df_mort = df_mort[["SEX", "AGE", beg_yr]]
        df_pop = df_pop[["SEX", "AGE", beg_yr]]
        # Rename the total deaths column, and the other columns
        df_mort.rename(
            columns={"SEX": "sex", "AGE": "age_str", beg_yr: "tot_deaths"},
            inplace=True,
        )
        df_pop.rename(
            columns={"SEX": "sex", "AGE": "age_str", beg_yr: "tot_pop"},
            inplace=True,
        )
        # Keep only all gender ('T') deaths and population by age
        df_mort = df_mort[df_mort["sex"] == "T"]
        df_pop = df_pop[df_pop["sex"] == "T"]
        # Now drop 'sex' variable
        df_mort = df_mort[["age_str", "tot_deaths"]]
        df_pop = df_pop[["age_str", "tot_pop"]]
        # Drop the age categories that we don't use ('TOTAL', 'UNK', 'Y_OPEN')
        indexNames_mort = df_mort[
            (df_mort["age_str"] == "TOTAL")
            | (df_mort["age_str"] == "UNK")
            | (df_mort["age_str"] == "Y_OPEN")
        ].index
        df_mort.drop(indexNames_mort, inplace=True)
        indexNames_pop = df_pop[
            (df_pop["age_str"] == "TOTAL")
            | (df_pop["age_str"] == "UNK")
            | (df_pop["age_str"] == "Y_OPEN")
        ].index
        df_pop.drop(indexNames_pop, inplace=True)
        # Rename age='Y_LT1' to 'Y0'
        df_mort.age_str[df_mort.age_str == "Y_LT1"] = "Y0"
        df_pop.age_str[df_pop.age_str == "Y_LT1"] = "Y0"
        # Generate new age variable that is numeric (remove 'Y' prefix)
        df_mort["age"] = df_mort["age_str"].str[1:].astype(int)
        df_pop["age"] = df_pop["age_str"].str[1:].astype(int)
        # Remove age_str variable and sort DataFrame by age
        df_mort = df_mort[["age", "tot_deaths"]]
        df_mort = df_mort.sort_values(by=["age"])
        df_mort.reset_index(drop=True, inplace=True)
        df_pop = df_pop[["age", "tot_pop"]]
        df_pop = df_pop.sort_values(by=["age"])
        df_pop.reset_index(drop=True, inplace=True)
        # Merge total population data into total deaths data
        df_mort = pd.merge(df_mort, df_pop, on="age", validate="1:1")
        # Change 'tot_deaths' and 'tot_pop' to numeric float
        df_mort["tot_deaths"] = df_mort["tot_deaths"].astype(np.float64)
        df_mort["tot_pop"] = df_mort["tot_pop"].astype(np.float64)
        # Create mortality rates variable
        df_mort["mort_rate_yr_data"] = np.divide(
            df_mort["tot_deaths"], df_mort["tot_pop"]
        )
        # Set the mortality rate in max_age_yr = 1.0
        df_mort["mort_rate_yr_mod"] = df_mort["mort_rate_yr_data"]
        df_mort["mort_rate_yr_mod"][df_mort["age"] == max_age_yr - 1] = 1.0
        if save_data:
            mort_data_csv_path = os.path.join(DATA_DIR, "mort_rate_data.csv")
            df_mort.to_csv(mort_data_csv_path, index=False)

    else:
        # Make sure the mort_rate_data.csv file is accessible in DATA_DIR
        mort_data_csv_path = os.path.join(DATA_DIR, "mort_rate_data.csv")
        assert os.access(mort_data_csv_path, os.F_OK)
        df_mort = pd.read_csv(mort_data_csv_path)

    # Create the model-ages mort_rates variable
    if totpers == 100 and min_age_yr == 0 and max_age_yr == 100:
        # This is case in which model age periods correspond to years
        mort_rates = df_mort["mort_rate_yr_mod"].to_numpy()

    else:
        # This is case in which model age periods do not correspond to years or
        # in which the initial age does not start at 0 and the ending age does
        # not end at 100
        yr_cut_pts = np.linspace(min_age_yr, max_age_yr, totpers + 1)

        tot_deaths = np.zeros(totpers)
        tot_pop = np.zeros(totpers)
        for per in range(totpers):
            # Get relevant vector of total deaths yearly data
            deaths_yr_data = df_mort["tot_deaths"][
                (
                    (df_mort["age"] >= np.floor(yr_cut_pts[per]))
                    & (df_mort["age"] <= np.ceil(yr_cut_pts[per + 1]))
                )
            ].to_numpy()
            # Get relevant vector of total population yearly data
            totpop_yr_data = df_mort["tot_pop"][
                (
                    (df_mort["age"] >= np.floor(yr_cut_pts[per]))
                    & (df_mort["age"] <= np.ceil(yr_cut_pts[per + 1]))
                )
            ].to_numpy()
            # Calculate the percent of the first and last bins to be included
            # in totals
            pct_first_yr_bin = 1 - (
                yr_cut_pts[per] - np.floor(yr_cut_pts[per])
            )
            pct_last_yr_bin = 1 - (
                np.ceil(yr_cut_pts[per + 1]) - yr_cut_pts[per + 1]
            )
            # Calculate total deaths in model age period
            deaths_yr_data[0] = pct_first_yr_bin * deaths_yr_data[0]
            deaths_yr_data[-1] = pct_last_yr_bin * deaths_yr_data[-1]
            tot_deaths[per] = deaths_yr_data.sum()
            # Calculate total population in model age period
            totpop_yr_data[0] = pct_first_yr_bin * totpop_yr_data[0]
            totpop_yr_data[-1] = pct_last_yr_bin * totpop_yr_data[-1]
            tot_pop[per] = totpop_yr_data.sum()

        mort_rates = tot_deaths / tot_pop
        mort_rates[-1] = 1.0

    if graph:
        mort_rates_yr = df_mort["mort_rate_yr_data"].to_numpy()
        ages_yr = np.arange(0, 100)
        pp.plot_mort_rates_data(
            totpers,
            min_age_yr,
            max_age_yr,
            mort_rates_yr,
            ages_yr,
            mort_rates,
            infmort_rate,
            output_dir=FIG_DIR,
        )

    return mort_rates, infmort_rate
