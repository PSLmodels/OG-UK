from ogusa.utils import safe_read_pickle
from ogusa.parameter_plots import plot_2D_taxfunc
from matplotlib import pyplot as plt

# read in tax function parameters from pickle
tax_funcs_base = safe_read_pickle('./examples/OG-UK-Example/OUTPUT_BASELINE/TxFuncEst_baseline.pkl')
tax_funcs_reform = safe_read_pickle('./examples/OG-UK-Example/OUTPUT_REFORM/TxFuncEst_policy.pkl')

# read in micro data from pickle
micro_data = safe_read_pickle('./examples/OG-UK-Example/OUTPUT_BASELINE/micro_data_baseline.pkl')

# create plot

for rate_type in ("etr", "mtrx", "mtry"):
    for labinc_val in (True, False):
        fig = plot_2D_taxfunc(
            2018, 2018, [tax_funcs_base, tax_funcs_reform], age=None,
            tax_func_type=['DEP'], rate_type=rate_type, over_labinc=labinc_val, other_inc_val=0,
            max_inc_amt=100000, data_list=[micro_data], labels=['Baseline', 'Reform'],
            title=f'Rate type: {rate_type}, over labour income: {labinc_val}',
            path=None)

        plt.show()