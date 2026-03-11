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
    GVA by section: ONS GDP output approach low level aggregates (Blue Book 2024)
    Capital shares: ONS GDP income approach (CoE / GVA by section), shrunk
        toward the aggregate mean for solver stability.
    Energy cost shares: ONS Supply and Use Tables 2022, BEIS Energy Trends
    c_min: ONS household expenditure data (minimum energy spend)
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
        "epsilon": [1.0] * M,  # Cobb-Douglas
        "Z": [[1.0] * M],
        "cit_rate": [[0.27] * M],
        "io_matrix": np.eye(M).tolist(),
        "alpha_c": alpha_c.tolist(),
        "c_min": list(_C_MIN),
        "delta_tau_annual": [[0.05] * M],
        "inv_tax_credit": [[0.0] * M],
        "tau_c": [[0.19] * NUM_CONSUMPTION_GOODS],
    }
