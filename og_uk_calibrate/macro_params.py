"""
This script produces the data required for macro_parameters for 
the OG-UK model. The sources are documented in the corresponding code. 
"""
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import datetime
import statsmodels.api as sm
from pandas_datareader import data
import eurostat
import os, sys
import pickle as pkl
import matplotlib.pyplot as plt


def get_macro_params():
    ##############################################################################
    ########### initial_debt_ratio - START ###################################
    # Use: https://www.ons.gov.uk/economy/governmentpublicsectorandtaxes/publicsectorfinance/datasets/governmentdeficitanddebtreturn
    # read from xlsx: worksheet "M1"; cell "G64"
    # "Gross consolidated debt as a percentage of GDP" "Calendar Year" "2018"
    initial_debt_ratio = 0.858
    ########### initial_debt_ratio - END #####################################
    ##############################################################################

    ##############################################################################
    ########### ALPHA_T & ALPHA_G - START ####################################
    # Transfer payments & Government expenditure, both as share of nominal GDP
    # Nominal GDP: https://www.ons.gov.uk/economy/grossdomesticproductgdp/datasets/uksecondestimateofgdpdatatables
    # worksheet 'A2 AGGREGATES', cell C77 (in GDP millions):
    nominal_GDP = 2141792
    # Government expenditure: https://www.ons.gov.uk/economy/governmentpublicsectorandtaxes/publicspending/datasets/esatable2mainaggregatesofgeneralgovernment
    # Expenditure category description: worksheet 'COFOG Description'
    # Data: worksheet '2018'
    # Total government expenditure (_T): cell AG25
    Total_gov_expre = 879383
    # Public Debt Transactions (GF0107): cell AG33
    Public_debt_payment = 52715
    # Social Protection (GF10), i.e. Public Transfers: cell AG95
    Public_transfers = 321008
    # Public Transfers-to-GDP
    alpha_T = Public_transfers / nominal_GDP
    # Government expenditure-to-GDP (excl transfers & debt payments)
    alpha_G = (
        Total_gov_expre - Public_transfers - Public_debt_payment
    ) / nominal_GDP
    ########### ALPHA_T & ALPHA_G - END ######################################
    ##############################################################################

    ##############################################################################
    ########### gamma = capital_share = 1 - labour_share - START #############
    # previously used EU-KLEMS as in this report:
    # https://ec.europa.eu/economy_finance/publications/pages/publication15147_en.pdf
    # https://euklems.eu/
    # This OECD report gives the labour share as 0.7 (Figure 4)
    # https://www.oecd.org/g20/topics/employment-and-social-policy/The-Labour-Share-in-G20-Economies.pdf
    # This Office of National Statistics chart give labour share in 2016 as 54.44
    # https://www.ons.gov.uk/economy/nationalaccounts/uksectoraccounts/adhocs/006610labourshareofgdpandprivatenonfinancialcorporationgrossoperatingsurpluscorporateprofitabilitycurrentpricesquarter1jantomar1997toquarter3julytosept2016
    # According to this report (Figure 2):
    # https://bankunderground.co.uk/2019/08/07/is-there-really-a-global-decline-in-the-non-housing-labour-share/
    # Values depends on whether you are dealing with
    # - the "unadjusted corporate sector" (around 60% labour share) versus
    # - the "adjusted business sector (using KLEMS)" (around 70% labour share)
    labour_share = 0.70
    gamma = 1 - labour_share
    ########### gamma = capital_share = 1 - labour_share - END ###############
    ##############################################################################

    ##############################################################################
    ### g_y (Exogenous growth rate of labor augmenting tech change) - START ##
    # use real_GDP_growth rate as proxy. Needs to be averaged over time periods.
    # http://appsso.eurostat.ec.europa.eu/nui/show.do?dataset=tec00115&lang=en
    # UK data for 2015 - 2018 is:
    # 2.4 1.7 1.7 1.3 = average of 1.775 percent
    g_y = 1.775
    # Alternative source: Office of National Statistics
    # https://www.ons.gov.uk/economy/economicoutputandproductivity/productivitymeasures/datasets/internationalcomparisonsofproductivityfirstestimates
    # International Comparisons of Productivity - Final Estimates, 2016
    # Table 3: Constant price GDP per hour worked
    # Index values for 2012-2016:
    # 100 100.4 100.3 101.1 101.6 = average growth of 0.4 percent per year
    ### g_y (Exogenous growth rate of labor augmenting tech change) - END ####
    ##############################################################################

    ##############################################################################
    ########### r_gov_shift & r_gov_scale - START ############################
    # collect data on UK government bond yields
    gbond = data.DataReader(
        "irt_lt_mcby_d", start="1986-1-1", end="2020-12-31", data_source="eurostat"
    )
    # note column format:
    gbond_UK = gbond.loc[
        :, ("EMU convergence criterion bond yields", "United Kingdom", "Daily")
    ]
    # collect US corporate bond yields from FRED
    # Note: Global value may be preferred; neither UK nor global values readily
    # available.
    start_fred = datetime.datetime(1986, 1, 1)
    end_fred = datetime.datetime(2020, 12, 31)  # go through today
    fred_cbond = web.DataReader("DBAA", "fred", start_fred, end_fred)
    # ######### to pkl once, then use pkl file #########
    # pkl.dump(gbond_UK, open( "gbond_UK.pkl", "wb" ) )
    # pkl.dump(fred_cbond, open( "fred_cbond.pkl", "wb" ) )
    # ###### use pkl files instead:
    # gbond_UK = pkl.load(open("gbond_UK.pkl", "rb"))
    # fred_cbond = pkl.load(open("fred_cbond.pkl", "rb"))

    # convert panda Series to panda Dataframe
    gbond_UK_df = gbond_UK.to_frame()
    # Transform three-level column for UKbonds in to single-level column
    gbond_UK_df.columns = ["".join(col).strip() for col in gbond_UK_df.columns.values]
    # Rename UKbonds column
    gbond_UK_df = gbond_UK_df.rename(
        columns={
            "EMU convergence criterion bond yieldsUnited KingdomDaily": "UKbonds"
        }
    )
    # merge UK govt bond and US corp bond in one Dataframe
    rate_data = fred_cbond.merge(
        gbond_UK_df, left_index=True, right_index=True
    ).dropna()
    # add constant column
    rate_data["constant"] = np.ones(len(rate_data.index))
    # run OLS regression: UKbonds = a*constant + b*DBAA
    mod = sm.OLS(rate_data["UKbonds"], rate_data[["constant", "DBAA"]])
    res = mod.fit()
    # use OLS solution for parameter values
    r_gov_shift = res.params["DBAA"]
    r_gov_scale = res.params["constant"]

    # display scatterplot:
    # UKbonds_np = rate_data['UKbonds'].to_numpy().astype(float)
    # DBAA_np = rate_data['DBAA'].to_numpy().astype(float)
    # plt.scatter(UKbonds_np, DBAA_np, s=1)
    # plt.show()
    ########### r_gov_shift & r_gov_scale - END ##############################
    ##############################################################################

    ##############################################################################
    ########### collect key values as macro_parameters - START ###################
    # # initialize a dictionary of parameters
    macro_parameters = {}
    macro_parameters["initial_debt_ratio"] = initial_debt_ratio
    macro_parameters['alpha_T'] = alpha_T
    macro_parameters['alpha_G'] = alpha_G
    macro_parameters["gamma"] = gamma
    macro_parameters["g_y"] = g_y
    macro_parameters["r_gov_shift"] = r_gov_shift
    macro_parameters["r_gov_scale"] = r_gov_scale

    return macro_parameters
    ########### collect key values as macro_parameters - END #####################
    ##############################################################################