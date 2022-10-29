[![Python 3.7+](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/release/python-377/)
[![Build Status](https://travis-ci.com/PSLmodels/OG-UK.svg?branch=master)](https://travis-ci.com/PSLmodels/OG-UK)
[![Codecov](https://codecov.io/gh/PSLmodels/OG-UK/branch/main/graph/badge.svg)](https://codecov.io/gh/PSLmodels/OG-UK)

# OG-UK

OG-UK is an overlapping-generations (OG) model of the economy of the United Kingom (UK) that allows for dynamic general equilibrium analysis of fiscal policy. OG-UK builds on the OG-Core platform, using a calibration specific to the UK.  The model output focuses changes in macroeconomic aggregates (GDP, investment, consumption), wages, interest rates, and the stream of tax revenues over time. Regularly updated documentation of the model and the Python API are available [here](https://pslmodels.github.io/OG-UK).


## Disclaimer

The model is currently under development.  PLEASE DO NOT USE OR CITE THE MODEL'S OUTPUT FOR ANY PURPOSE.


## Using/contributing to OG-UK

* Install the [Anaconda distribution](https://www.anaconda.com/distribution/) of Python
* Clone this repository to a directory on your computer
* From the terminal (or Conda command prompt), navigate to the directory to which you cloned this repository and run `conda env create -f environment.yml`
* Then, `conda activate oguk-calibrate-dev`
* Then install by `pip install -e .`
* Navigate to `./examples`
* Run the model with an example reform from terminal/command prompt by typing `python run_og_uk.py examples.small_ubi_reform.ubi_reform`
* You can adjust the `./run_examples/run_og_uk.py` by adjusting the individual income tax reform (using an OpenFisca UK `Reform` object) or other model parameters specified in a dictionary and passed to the `Specifications.update_specification()` method.
* Model outputs will be saved in the following files:
  * `./examples/OG-UK_example_plots`
    * This folder will contain a number of plots generated from OG-UK to help you visualize the output from your run
  * `./examples/og_uk_example_output.csv`
    * This is a summary of the percentage changes in macro variables over the first ten years and in the steady-state.
  * `./examples/OG-UK-Example/OUTPUT_BASELINE/model_params.pkl`
    * Model parameters used in the baseline run
    * See `execute.py` in the OG-Core model for items in the dictionary object in this pickle file
  * `./examples/OG-UK-Example/OUTPUT_BASELINE/TxFuncEst_baseline.pkl`
    * Tax function parameters used for the baseline model run
    * See `txfunc.py` in the OG-Core model for what is in the dictionary object in this pickle file
  * `./examples/OG-UK-Example/OUTPUT_BASELINE/SS/SS_vars.pkl`
    * Outputs from the model steady state solution under the baseline policy
    * See `SS.py` in the OG-Core model for what is in the dictionary object in this pickle file
  * `./examples/OG-UK-Example/OUTPUT_BASELINE/TPI/TPI_vars.pkl`
    * Outputs from the model timepath solution under the baseline policy
    * See `TPI.py` in the OG-Core model for what is in the dictionary object in this pickle file
  * An analogous set of files in the `./examples/OG-UK-Example/OUTPUT_REFORM` directory, which represent objects from the simulation of the reform policy

Note that, depending on your machine, a full model run (solving for the full time path equilibrium for the baseline and reform policies) can take from a few to several hours of compute time.

If you run into errors running the example script, please open a new issue in the OG-UK repo with a description of the issue and any relevant tracebacks you receive.


## Citing OG-UK

PLEASE DO NOT CITE
