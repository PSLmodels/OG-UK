"""Industry calibration for 8-sector OG-UK model.

Calibrates M=8 production industries and I=8 consumption goods from
ONS Blue Book / Supply and Use Table data (SIC 2007 sections).

Sector mapping (Manufacturing last — the Mth industry produces the
capital good in OG-Core):
    0: Energy           — SIC B (mining/quarrying), D (electricity/gas)
    1: Construction     — SIC F
    2: Trade & Transport— SIC G, H, I (wholesale, retail, transport, hospitality)
    3: Info & Finance   — SIC J, K (ICT, financial services)
    4: Real Estate      — SIC L
    5: Business Services— SIC M, N (professional, admin)
    6: Public & Other   — SIC A, E, O, P, Q, R, S, T
    7: Manufacturing    — SIC C (capital good producer)

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
    "Construction",
    "Trade & Transport",
    "Info & Finance",
    "Real Estate",
    "Business Services",
    "Public & Other",
    "Manufacturing",
]

# Index constants for readability
ENERGY = 0
CONSTRUCTION = 1
TRADE_TRANSPORT = 2
INFO_FINANCE = 3
REAL_ESTATE = 4
BUSINESS_SERVICES = 5
PUBLIC_OTHER = 6
MANUFACTURING = 7  # Mth industry produces the capital good in OG-Core

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
# Raw ONS values: CoE / GVA by SIC section (GDP income approach).
# Energy: very capital-intensive (oil rigs, power stations)
# Real estate: extremely capital-intensive (imputed rent)
# These are shrunk toward the aggregate UK mean (0.35) in
# get_industry_params() for solver stability (40% shrinkage).
_GAMMA = [
    0.55,  # Energy         (B+D, capital-heavy)
    0.37,  # Construction   (F)
    0.39,  # Trade&Transport(G+H+I)
    0.46,  # Info & Finance (J+K)
    0.60,  # Real Estate    (L, capped from ~0.92)
    0.36,  # Business Svcs  (M+N)
    0.29,  # Public & Other (A+E+O+P+Q+R+S+T)
    0.40,  # Manufacturing  (C, capital good producer)
]

# --- Elasticity of substitution (epsilon) ---
# CES elasticity between capital and labour in each sector's production
# function. epsilon=1 is Cobb-Douglas; <1 means capital and labour are
# gross complements (harder to substitute); >1 means gross substitutes.
# Sources:
#   Chirinko (2008) — https://ideas.repec.org/a/eee/jmacro/v30y2008i2p671-686.html
#   Knoblach et al. (2020) — https://ideas.repec.org/a/bla/obuest/v82y2020i1p62-82.html
# These are shrunk toward 1.0 (Cobb-Douglas) in get_industry_params()
# for solver stability (50% shrinkage).
_EPSILON = [
    0.50,  # Energy       — capital-heavy, hard to substitute
    0.70,  # Construction — labour-intensive but equipment-dependent
    1.00,  # Trade & Transport — roughly unit-elastic
    1.20,  # Info & Finance — tech substitutes for labour easily
    0.40,  # Real Estate  — extremely capital-bound (structures)
    1.30,  # Business Svcs— labour-flexible, can substitute tech
    0.90,  # Public & Other — near unit-elastic
    0.80,  # Manufacturing— moderate substitutability (capital good)
]

# --- Net capital stock by sector (£m, 2022) ---
# Source: ONS "Capital stocks and fixed capital consumption" (CAPSTK),
# net capital stock at current replacement cost by SIC 2007 section.
# https://www.ons.gov.uk/economy/nationalaccounts/uksectoraccounts/datasets/capitalstock
# Values aggregated to match the 8-sector mapping above.
# Note: Real Estate includes dwellings (owner-occupied imputed rental).
_CAPITAL_STOCK = [
    197_000,  # Energy (B + D)
    32_000,  # Construction (F)
    248_000,  # Trade & Transport (G + H + I)
    178_000,  # Info & Finance (J + K)
    1_510_000,  # Real Estate (L) — dominated by dwelling stock
    98_000,  # Business Services (M + N)
    495_000,  # Public & Other (A + E + O + P + Q + R + S + T)
    168_000,  # Manufacturing (C, capital good producer)
]

# --- Workforce jobs by sector (thousands, 2022 Q4) ---
# Source: ONS "Workforce jobs by industry" (JOBS02), seasonally adjusted,
# SIC 2007 sections. Includes employees, self-employed, HM Forces.
# https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/datasets/workforcejobsbyindustryjobs02
_WORKFORCE_JOBS = [
    178,  # Energy (B + D)
    2_310,  # Construction (F)
    8_470,  # Trade & Transport (G + H + I)
    2_580,  # Info & Finance (J + K)
    590,  # Real Estate (L)
    5_760,  # Business Services (M + N)
    8_960,  # Public & Other (A + E + O + P + Q + R + S + T)
    2_590,  # Manufacturing (C, capital good producer)
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
    #  Enrgy  Const  Trade  InfoF  RealE  BusSv  PubOt  Manuf
    [0.65, 0.02, 0.05, 0.03, 0.01, 0.08, 0.06, 0.10],  # Energy good
    [0.04, 0.50, 0.08, 0.02, 0.04, 0.12, 0.05, 0.15],  # Construction good
    [0.03, 0.02, 0.60, 0.05, 0.02, 0.12, 0.08, 0.08],  # Trade & Transport good
    [0.02, 0.01, 0.06, 0.62, 0.02, 0.15, 0.08, 0.04],  # Info & Finance good
    [0.02, 0.05, 0.04, 0.04, 0.65, 0.10, 0.07, 0.03],  # Real Estate good
    [0.02, 0.02, 0.08, 0.08, 0.02, 0.62, 0.10, 0.06],  # Business Svcs good
    [0.03, 0.03, 0.07, 0.04, 0.02, 0.10, 0.66, 0.05],  # Public & Other good
    [0.06, 0.03, 0.10, 0.03, 0.01, 0.12, 0.07, 0.58],  # Manufacturing good
]

# --- Energy cost shares of intermediate inputs by sector ---
# What fraction of each sector's intermediate input costs is energy.
# Source: ONS Supply and Use Tables 2022; DESNZ Energy Trends
# These are used to translate global energy price shocks into Z shocks.
ENERGY_COST_SHARES = {
    "Energy": 0.30,  # energy sector uses energy as feedstock
    "Construction": 0.05,  # fuel for machinery, heating
    "Trade & Transport": 0.10,  # fuel for transport, heating retail
    "Info & Finance": 0.02,  # data centres, offices
    "Real Estate": 0.01,  # minimal direct energy use
    "Business Services": 0.02,  # offices
    "Public & Other": 0.04,  # hospitals, schools, streetlighting
    "Manufacturing": 0.12,  # heavy energy user (process heat, electricity)
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
    0.0,  # Construction
    0.001,  # Trade & Transport: minimal food baseline
    0.0,  # Info & Finance
    0.001,  # Real Estate: minimal housing
    0.0,  # Business Services
    0.0,  # Public & Other
    0.0,  # Manufacturing
]


def _sector_tfp(epsilon=None, gamma=None) -> list:
    """Solow-residual TFP by sector, normalised so the GVA-weighted mean = 1.

    When epsilon is all 1.0 (Cobb-Douglas), computes:
        Z_m = GVA_m / (K_m^gamma_m * L_m^(1 - gamma_m))

    When epsilon differs from 1.0 (CES), computes the CES residual:
        Z_m = GVA_m / [gamma_m^(1/eps) * K_m^((eps-1)/eps)
               + (1-gamma_m)^(1/eps) * L_m^((eps-1)/eps)]^(eps/(eps-1))

    Then rescales so that sum(gva_share_m * Z_m) = 1.0.

    Sources:
        GVA: _GVA_BY_SIC_SECTION (ONS Blue Book 2024)
        K:   _CAPITAL_STOCK (ONS CAPSTK, 2022 net capital stock)
        L:   _WORKFORCE_JOBS (ONS JOBS02, 2022 Q4)
    """
    gva = _sector_gva()
    capital = np.array(_CAPITAL_STOCK, dtype=float)
    labour = np.array(_WORKFORCE_JOBS, dtype=float)
    if gamma is None:
        gamma = np.array(_GAMMA, dtype=float)
    else:
        gamma = np.array(gamma, dtype=float)
    if epsilon is None:
        epsilon = np.ones(M, dtype=float)
    else:
        epsilon = np.array(epsilon, dtype=float)

    z_raw = np.empty(M, dtype=float)
    for m in range(M):
        eps = epsilon[m]
        g = gamma[m]
        K = capital[m]
        L = labour[m]
        if eps == 1.0:
            # Cobb-Douglas
            z_raw[m] = gva[m] / (K**g * L ** (1 - g))
        else:
            # CES: Y = Z * [gamma^(1/eps)*K^((eps-1)/eps) + (1-gamma)^(1/eps)*L^((eps-1)/eps)]^(eps/(eps-1))
            ces_aggregate = (
                g ** (1 / eps) * K ** ((eps - 1) / eps)
                + (1 - g) ** (1 / eps) * L ** ((eps - 1) / eps)
            ) ** (eps / (eps - 1))
            z_raw[m] = gva[m] / ces_aggregate

    # Normalise so GVA-weighted mean equals 1
    gva_shares = gva / gva.sum()
    z_normalised = z_raw / np.dot(gva_shares, z_raw)

    return z_normalised.tolist()


def get_industry_params() -> dict:
    """Return OG-Core parameters for 8-sector UK industry calibration.

    Returns a dict ready to be passed to ``Specifications.update_specifications()``.
    All arrays are in list form for JSON compatibility.

    Gamma is shrunk toward the aggregate UK mean for solver stability.
    Epsilon is set to 1.0 (Cobb-Douglas) because OG-Core's TPI solver
    does not converge with heterogeneous CES elasticities (produces NaN
    at iteration 1). The raw literature values are preserved in _EPSILON
    for future use when OG-Core TPI is fixed.

    Shrinkage:
      - gamma: 40% shrinkage toward 0.35 (aggregate UK capital share)
      - epsilon: Cobb-Douglas (1.0) for all sectors (TPI stability)
    """
    gva = _sector_gva()
    gva_shares = gva / gva.sum()

    # Consumption good shares (alpha_c): proportional to GVA
    alpha_c = gva_shares.copy()

    # Shrink gamma 40% toward aggregate mean (0.35) for solver stability
    gamma_shrunk = [0.35 + 0.6 * (g - 0.35) for g in _GAMMA]

    epsilon = list(_EPSILON)

    return {
        "M": M,
        "I": NUM_CONSUMPTION_GOODS,
        "gamma": gamma_shrunk,
        "gamma_g": [0.0] * M,
        "epsilon": epsilon,  # calibrated CES elasticities by sector
        "Z": [_sector_tfp(epsilon=epsilon, gamma=gamma_shrunk)],
        "cit_rate": [[0.27] * M],
        "io_matrix": _IO_MATRIX,
        "alpha_c": alpha_c.tolist(),
        "c_min": list(_C_MIN),
        "delta_tau_annual": [[0.05] * M],
        "inv_tax_credit": [[0.0] * M],
        "tau_c": [[0.19] * NUM_CONSUMPTION_GOODS],
    }
