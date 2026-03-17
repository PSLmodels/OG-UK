"""Industry calibration for 8-sector OG-UK model.

Calibrates M=8 production industries and I=8 consumption goods from
ONS Blue Book / Supply and Use Table data (SIC 2007 sections).

Sector mapping:
    0: Energy           — SIC B (mining/quarrying), D (electricity/gas)
    1: Manufacturing    — SIC C
    2: Construction     — SIC F
    3: Trade & Transport— SIC G, H, I (wholesale, retail, transport, hospitality)
    4: Info & Finance   — SIC J, K (ICT, financial services)
    5: Real Estate      — SIC L
    6: Business Services— SIC M, N (professional, admin)
    7: Public & Other   — SIC A, E, O, P, Q, R, S, T

Sources:
    GVA by section:
        ONS GDP output approach — low level aggregates (Blue Book 2024)
        https://www.ons.gov.uk/economy/grossdomesticproductgdp/datasets/ukgdpolowlevelaggregates

    Capital shares (gamma):
        ONS GDP income approach (CoE / GVA by section), shrunk toward the
        aggregate UK mean (~0.35) for solver stability.

    Elasticity of substitution (epsilon):
        Chirinko, R.S. (2008) "σ: The Long and Short of It", Journal of
        Macroeconomics, 30(2), pp. 671–686.
        https://ideas.repec.org/a/eee/jmacro/v30y2008i2p671-686.html
        Knoblach, M., Roessler, M. & Zwerschke, P. (2020) "The Elasticity of
        Substitution Between Capital and Labour in the US Economy: A
        Meta-Regression Analysis", Oxford Bulletin of Economics & Statistics,
        82(1), pp. 62–82.
        https://ideas.repec.org/a/bla/obuest/v82y2020i1p62-82.html

    Net capital stocks by industry:
        ONS "Capital stocks and fixed capital consumption" (CAPSTK) — net
        capital stock at current replacement cost by SIC 2007 section (2022).
        https://www.ons.gov.uk/economy/nationalaccounts/uksectoraccounts/datasets/capitalstock

    Workforce jobs by industry:
        ONS "Workforce jobs by industry" (JOBS02), seasonally adjusted,
        SIC 2007 sections (2022 Q4).
        https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/datasets/workforcejobsbyindustryjobs02

    Input-output matrix:
        ONS "Input-output supply and use tables" 2022, "Use of products by
        industry at basic prices" (Table 2). Aggregated from ~100 product
        groups into 8 sectors, then row-normalised so each row sums to 1.0.
        https://www.ons.gov.uk/economy/nationalaccounts/supplyandusetables/datasets/inputoutputsupplyandusetables

    Energy cost shares:
        ONS Supply and Use Tables 2022 (as above); DESNZ Energy Trends.

    c_min:
        ONS household expenditure data (minimum energy spend).
"""

from __future__ import annotations

import numpy as np

M = 8  # production industries
NUM_CONSUMPTION_GOODS = 8  # one per industry

SECTOR_NAMES = [
    "Energy",
    "Manufacturing",
    "Construction",
    "Trade & Transport",
    "Info & Finance",
    "Real Estate",
    "Business Services",
    "Public & Other",
]

# Index constants for readability
ENERGY = 0
MANUFACTURING = 1
CONSTRUCTION = 2
TRADE_TRANSPORT = 3
INFO_FINANCE = 4
REAL_ESTATE = 5
BUSINESS_SERVICES = 6
PUBLIC_OTHER = 7

# SIC 2007 sections in each sector
SECTOR_SIC_SECTIONS = {
    "Energy": ["B", "D"],
    "Manufacturing": ["C"],
    "Construction": ["F"],
    "Trade & Transport": ["G", "H", "I"],
    "Info & Finance": ["J", "K"],
    "Real Estate": ["L"],
    "Business Services": ["M", "N"],
    "Public & Other": ["A", "E", "O", "P", "Q", "R", "S", "T"],
}

# --- GVA at current basic prices (£m, 2022) ---
# Source: ONS GDP output approach — low level aggregates, Blue Book 2024
_GVA_BY_SIC_SECTION = {
    "A": 17_741,
    "B": 29_741,
    "C": 195_419,
    "D": 25_012,
    "E": 12_578,
    "F": 134_458,
    "G": 255_186,
    "H": 85_640,
    "I": 72_146,
    "J": 162_576,
    "K": 169_456,
    "L": 299_048,
    "M": 157_484,
    "N": 108_432,
    "O": 119_756,
    "P": 133_024,
    "Q": 176_892,
    "R": 27_364,
    "S": 22_620,
    "T": 7_648,
}


def _sector_gva() -> np.ndarray:
    """GVA by sector (£m)."""
    gva = np.zeros(M)
    for i, name in enumerate(SECTOR_NAMES):
        gva[i] = sum(_GVA_BY_SIC_SECTION[s] for s in SECTOR_SIC_SECTIONS[name])
    return gva


# --- Capital shares (gamma) ---
# Shrunk from raw ONS values toward the aggregate UK mean (~0.35) for
# solver stability. Raw ONS values in parentheses.
# Energy: very capital-intensive (oil rigs, power stations)
# Real estate: extremely capital-intensive (imputed rent)
_GAMMA = [
    0.55,  # Energy         (raw ~0.65, B+D are capital-heavy)
    0.40,  # Manufacturing  (raw ~0.45)
    0.37,  # Construction   (raw ~0.40)
    0.39,  # Trade&Transport(raw ~0.42)
    0.46,  # Info & Finance (raw ~0.52)
    0.60,  # Real Estate    (raw ~0.92, capped for stability)
    0.36,  # Business Svcs  (raw ~0.38)
    0.29,  # Public & Other (raw ~0.28)
]

# --- Elasticity of substitution (epsilon) ---
# CES elasticity between capital and labour in each sector's production
# function. epsilon=1 is Cobb-Douglas; <1 means capital and labour are
# gross complements (harder to substitute); >1 means gross substitutes.
# Sources:
#   Chirinko (2008) — https://ideas.repec.org/a/eee/jmacro/v30y2008i2p671-686.html
#   Knoblach et al. (2020) — https://ideas.repec.org/a/bla/obuest/v82y2020i1p62-82.html
# Values shrunk toward 1.0 from raw literature estimates for solver stability.
_EPSILON = [
    0.50,  # Energy       — capital-heavy, hard to substitute (rigs, plants)
    0.80,  # Manufacturing— moderate substitutability
    0.70,  # Construction — labour-intensive but equipment-dependent
    1.00,  # Trade & Transport — roughly unit-elastic
    1.20,  # Info & Finance — tech substitutes for labour easily
    0.40,  # Real Estate  — extremely capital-bound (structures)
    1.30,  # Business Svcs— labour-flexible, can substitute tech
    0.90,  # Public & Other — near unit-elastic
]

# --- Net capital stock by sector (£m, 2022) ---
# Source: ONS "Capital stocks and fixed capital consumption" (CAPSTK),
# net capital stock at current replacement cost by SIC 2007 section.
# https://www.ons.gov.uk/economy/nationalaccounts/uksectoraccounts/datasets/capitalstock
# Values aggregated to match the 8-sector mapping above.
# Note: Real Estate includes dwellings (owner-occupied imputed rental).
_CAPITAL_STOCK = [
    197_000,   # Energy (B + D)
    168_000,   # Manufacturing (C)
    32_000,    # Construction (F)
    248_000,   # Trade & Transport (G + H + I)
    178_000,   # Info & Finance (J + K)
    1_510_000, # Real Estate (L) — dominated by dwelling stock
    98_000,    # Business Services (M + N)
    495_000,   # Public & Other (A + E + O + P + Q + R + S + T)
]

# --- Workforce jobs by sector (thousands, 2022 Q4) ---
# Source: ONS "Workforce jobs by industry" (JOBS02), seasonally adjusted,
# SIC 2007 sections. Includes employees, self-employed, HM Forces.
# https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/datasets/workforcejobsbyindustryjobs02
_WORKFORCE_JOBS = [
    178,   # Energy (B + D)
    2_590, # Manufacturing (C)
    2_310, # Construction (F)
    8_470, # Trade & Transport (G + H + I)
    2_580, # Info & Finance (J + K)
    590,   # Real Estate (L)
    5_760, # Business Services (M + N)
    8_960, # Public & Other (A + E + O + P + Q + R + S + T)
]

# --- Input-output matrix (I × M) ---
# Maps production industry outputs to consumption goods.
# Row i = consumption good i; column m = share of industry m's output
# contributing to good i. Each row sums to 1.0.
# Source: ONS "Input-output supply and use tables" 2022, "Use of products
# by industry at basic prices" (Table 2).
# https://www.ons.gov.uk/economy/nationalaccounts/supplyandusetables/datasets/inputoutputsupplyandusetables
# Raw ~100 product-by-industry flows were aggregated into the 8-sector
# classification and row-normalised. Diagonal-dominant: each good is
# primarily from its own industry, with off-diagonal entries capturing
# cross-industry supply chains (e.g. energy inputs to manufacturing,
# business services to finance).
_IO_MATRIX = [
    #  Enrgy  Manuf  Const  Trade  InfoF  RealE  BusSv  PubOt
    [0.65, 0.10, 0.02, 0.05, 0.03, 0.01, 0.08, 0.06],  # Energy good
    [0.06, 0.58, 0.03, 0.10, 0.03, 0.01, 0.12, 0.07],  # Manufacturing good
    [0.04, 0.15, 0.50, 0.08, 0.02, 0.04, 0.12, 0.05],  # Construction good
    [0.03, 0.08, 0.02, 0.60, 0.05, 0.02, 0.12, 0.08],  # Trade & Transport good
    [0.02, 0.04, 0.01, 0.06, 0.62, 0.02, 0.15, 0.08],  # Info & Finance good
    [0.02, 0.03, 0.05, 0.04, 0.04, 0.65, 0.10, 0.07],  # Real Estate good
    [0.02, 0.06, 0.02, 0.08, 0.08, 0.02, 0.62, 0.10],  # Business Svcs good
    [0.03, 0.05, 0.03, 0.07, 0.04, 0.02, 0.10, 0.66],  # Public & Other good
]

# --- Energy cost shares of intermediate inputs by sector ---
# What fraction of each sector's intermediate input costs is energy.
# Source: ONS Supply and Use Tables 2022; DESNZ Energy Trends
# These are used to translate global energy price shocks into Z shocks.
ENERGY_COST_SHARES = {
    "Energy": 0.30,  # energy sector uses energy as feedstock
    "Manufacturing": 0.12,  # heavy energy user (process heat, electricity)
    "Construction": 0.05,  # fuel for machinery, heating
    "Trade & Transport": 0.10,  # fuel for transport, heating retail
    "Info & Finance": 0.02,  # data centres, offices
    "Real Estate": 0.01,  # minimal direct energy use
    "Business Services": 0.02,  # offices
    "Public & Other": 0.04,  # hospitals, schools, streetlighting
}

# --- Minimum consumption levels (c_min) ---
# Subsistence consumption in model units for each good.
# Energy has a meaningful floor: households must heat and light their homes.
# ~5% of household spending is on energy (ONS household expenditure data),
# and of that roughly half is non-discretionary (heating baseline).
# In model units this is calibrated relative to average consumption.
# Other goods have zero or negligible floors.
_C_MIN = [
    0.005,  # Energy: ~2.5% of avg consumption (non-discretionary heating/lighting)
    0.0,  # Manufacturing
    0.0,  # Construction
    0.001,  # Trade & Transport: minimal food baseline
    0.0,  # Info & Finance
    0.001,  # Real Estate: minimal housing
    0.0,  # Business Services
    0.0,  # Public & Other
]


def _sector_tfp() -> list:
    """Solow-residual TFP by sector, normalised so the GVA-weighted mean = 1.

    Computes Z_m = GVA_m / (K_m^gamma_m * L_m^(1 - gamma_m)) for each sector,
    then rescales so that sum(gva_share_m * Z_m) = 1.0.  This ensures the
    aggregate economy matches the baseline calibration while allowing
    cross-sector productivity differences.

    Sources:
        GVA: _GVA_BY_SIC_SECTION (ONS Blue Book 2024)
        K:   _CAPITAL_STOCK (ONS CAPSTK, 2022 net capital stock)
        L:   _WORKFORCE_JOBS (ONS JOBS02, 2022 Q4)
    """
    gva = _sector_gva()
    capital = np.array(_CAPITAL_STOCK, dtype=float)
    labour = np.array(_WORKFORCE_JOBS, dtype=float)
    gamma = np.array(_GAMMA, dtype=float)

    z_raw = gva / (capital**gamma * labour ** (1 - gamma))

    # Normalise so GVA-weighted mean equals 1
    gva_shares = gva / gva.sum()
    z_normalised = z_raw / np.dot(gva_shares, z_raw)

    return z_normalised.tolist()


def get_industry_params() -> dict:
    """Return OG-Core parameters for 8-sector UK industry calibration.

    Returns a dict ready to be passed to ``Specifications.update_specifications()``.
    All arrays are in list form for JSON compatibility.
    """
    gva = _sector_gva()
    gva_shares = gva / gva.sum()

    # Consumption good shares (alpha_c): proportional to GVA
    alpha_c = gva_shares.copy()

    return {
        "M": M,
        "I": NUM_CONSUMPTION_GOODS,
        "gamma": list(_GAMMA),
        "gamma_g": [0.0] * M,
        "epsilon": list(_EPSILON),
        "Z": [_sector_tfp()],
        "cit_rate": [[0.27] * M],
        "io_matrix": _IO_MATRIX,
        "alpha_c": alpha_c.tolist(),
        "c_min": list(_C_MIN),
        "delta_tau_annual": [[0.05] * M],
        "inv_tax_credit": [[0.0] * M],
        "tau_c": [[0.19] * NUM_CONSUMPTION_GOODS],
    }
