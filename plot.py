from ogusa.utils import safe_read_pickle
from ogusa.parameter_plots import plot_2D_taxfunc
from ogusa import txfunc
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# read in tax function parameters from pickle
tax_funcs_base = safe_read_pickle('./examples/OG-UK-Example/OUTPUT_BASELINE/TxFuncEst_baseline.pkl')
tax_funcs_reform = safe_read_pickle('./examples/OG-UK-Example/OUTPUT_REFORM/TxFuncEst_policy.pkl')

# read in micro data from pickle
micro_data = safe_read_pickle('./examples/OG-UK-Example/OUTPUT_BASELINE/micro_data_baseline.pkl')

# create plot

'''for rate_type in ("etr", "mtrx", "mtry"):
    for labinc_val in (True, False):
        fig = plot_2D_taxfunc(
            )

        plt.show()'''


def get_tax_fn(year, start_year, tax_param_list, age=None,
                    tax_func_type=['DEP'], rate_type='etr',
                    over_labinc=True, other_inc_val=1000,
                    max_inc_amt=1000000, data_list=None,
                    labels=['1st Functions'], title=None, path=None):
    # Check that inputs are valid
    assert isinstance(start_year, int)
    assert isinstance(year, int)
    assert (year >= start_year)
    # if list of tax function types less than list of params, assume
    # all the same functional form
    if len(tax_func_type) < len(tax_param_list):
        tax_func_type = [tax_func_type[0]] * len(tax_param_list)
    for i, v in enumerate(tax_func_type):
        assert (v in ['DEP', 'DEP_totalinc', 'GS', 'linear'])
    assert (rate_type in ['etr', 'mtrx', 'mtry'])
    assert (len(tax_param_list) == len(labels))

    # Set age and year to look at
    if age is not None:
        assert isinstance(age, int)
        s = age - 21
    else:
        s = 0  # if not age-specific, all ages have the same values
    t = year - start_year

    # create rate_key to correspond to keys in tax func dicts
    rate_key = 'tfunc_' + rate_type + '_params_S'

    # Set income range to plot over (min income value hard coded to 5)
    inc_sup = np.exp(np.linspace(np.log(5), np.log(max_inc_amt), 100))
    # Set income value for other income
    inc_fix = other_inc_val

    if over_labinc:
        key1 = 'total_labinc'
        X = inc_sup
        Y = inc_fix
    else:
        key1 = 'total_capinc'
        X = inc_fix
        Y = inc_sup

    # get tax rates for each point in the income support and plot
    for i, tax_params in enumerate(tax_param_list):
        rates = txfunc.get_tax_rates(
            tax_params[rate_key][s, t, :], X, Y, None, tax_func_type[i],
            rate_type, for_estimation=False)
    
    return lambda x, y: txfunc.get_tax_rates(
            tax_params[rate_key][s, t, :], x, y, None, tax_func_type[i],
            rate_type, for_estimation=False)


tax = get_tax_fn(2018, 2018, [tax_funcs_base, tax_funcs_reform], age=None,
            tax_func_type=['DEP'], rate_type="etr", over_labinc=True, other_inc_val=1000,
            max_inc_amt=100000, data_list=[micro_data], labels=['Baseline', 'Reform'],
            title=f'Rate type: DEP, over labour income',
            path=None)

results = tax(micro_data["2018"]["total_labinc"], micro_data["2018"]["total_capinc"])

import plotly.express as px
import plotly.graph_objects as go

df = pd.DataFrame({
    "Labour income": micro_data["2018"]["total_labinc"],
    "Capital income": micro_data["2018"]["total_capinc"],
    "ETR":  micro_data["2018"]["etr"],
    "Type": "Actual"
})
second_df = df.copy()

second_df["Fitted ETR"] = tax(df["Labour income"], df["Capital income"])
second_df["Type"] = "Fitted"

df = pd.concat([df, second_df])

px.scatter_3d(df, x="Labour income", y="Capital income", z="ETR", opacity=0.1, color="Type").show()