"""Data fetching functions for UK macroeconomic parameters.

Sources: ONS, Bank of England, GOV.UK.
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 30


def fetch_ons_timeseries(
    cdid: str,
    dataset: str,
    topic_path: str,
    frequency: str = "years",
    fallback: float | None = None,
) -> float:
    """Fetch the latest value from an ONS time series.

    Args:
        cdid: ONS series identifier (e.g. "HF6X").
        dataset: ONS dataset identifier (e.g. "pusf").
        topic_path: Topic URL path (e.g.
            "economy/governmentpublicsectorandtaxes/publicsectorfinance").
        frequency: One of "years", "quarters", "months".
        fallback: Value to return if the API is unreachable.

    Returns:
        The latest numeric value from the series.
    """
    url = (
        f"https://www.ons.gov.uk/generator?format=csv"
        f"&uri=/{topic_path}/timeseries/{cdid.lower()}/{dataset.lower()}"
    )
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        return _parse_ons_csv(resp.text, frequency)
    except Exception as exc:
        if fallback is not None:
            logger.warning("Could not fetch ONS %s/%s: %s — using fallback", cdid, dataset, exc)
            return fallback
        raise


def _parse_ons_csv(text: str, frequency: str) -> float:
    """Parse an ONS generator CSV and return the latest numeric value.

    ONS CSVs have a metadata header section followed by data rows.
    Data rows have a date-like first column and a numeric second column.
    """
    df = pd.read_csv(io.StringIO(text), header=None, names=["period", "value"])

    # Detect where the data starts — rows with numeric second column
    df["value_num"] = pd.to_numeric(df["value"], errors="coerce")
    data = df.dropna(subset=["value_num"]).copy()

    if data.empty:
        raise ValueError("No numeric data found in ONS response")

    # Filter by frequency hint: yearly rows are 4-digit, quarterly have Q,
    # monthly have e.g. "2024 JAN"
    if frequency == "years":
        mask = data["period"].str.strip().str.match(r"^\d{4}$")
        if mask.any():
            data = data[mask]
    elif frequency == "quarters":
        mask = data["period"].str.contains("Q", na=False)
        if mask.any():
            data = data[mask]
    elif frequency == "months":
        mask = data["period"].str.strip().str.match(r"^\d{4}\s+\w{3}$")
        if mask.any():
            data = data[mask]

    return float(data["value_num"].iloc[-1])


def fetch_boe_series(
    series_code: str = "IUDBEDR",
    fallback: float | None = None,
) -> float:
    """Fetch the latest value from the Bank of England statistical database.

    Args:
        series_code: BoE series code. Default IUDBEDR = official bank rate.
        fallback: Value to return if unreachable.

    Returns:
        Latest value as a float.
    """
    url = (
        "https://www.bankofengland.co.uk/boeapps/database/"
        "_iadb-FromShowColumns.asp?csv.x=yes"
        "&Datefrom=01/Jan/2020&Dateto=01/Jan/2030"
        f"&SeriesCodes={series_code}&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N"
    )
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        # BoE CSV has columns like "DATE" and the series code
        val_col = [c for c in df.columns if c.strip() != "DATE"][0]
        values = pd.to_numeric(df[val_col], errors="coerce").dropna()
        if values.empty:
            raise ValueError("No numeric data in BoE response")
        return float(values.iloc[-1])
    except Exception as exc:
        if fallback is not None:
            logger.warning("Could not fetch BoE %s: %s — using fallback", series_code, exc)
            return fallback
        raise


def get_uk_tax_rates() -> dict:
    """Return current UK tax rates from GOV.UK.

    All values sourced from GOV.UK and HMRC publications.
    """
    return {
        # Corporation tax main rate: 25% from April 2023
        # Source: https://www.gov.uk/corporation-tax-rates
        "cit_rate": [[0.25]],
        # Employer NICs: 15% from April 2025
        # Source: https://www.gov.uk/guidance/rates-and-thresholds-for-employers-2025-to-2026
        "tau_payroll": [0.15],
        # VAT effective rate on total consumption ~10% (standard rate 20%
        # but many goods are zero-rated or exempt)
        # Source: HMRC VAT receipts / ONS household final consumption
        "tau_c": [[0.10]],
        # Inheritance tax: 40% above £325k nil-rate band
        # Effective rate ~8% accounting for band, exemptions, and reliefs
        # Source: https://www.gov.uk/inheritance-tax
        "tau_bq": [0.08],
        # Capital allowances: 18% writing-down allowance (main pool)
        # Source: https://www.gov.uk/capital-allowances
        "delta_tau_annual": [[0.18]],
        # State pension age: 66, rising to 67 from 2026-28
        # In model periods: 66 - starting_age(20) = 46
        # Source: https://www.gov.uk/state-pension-age
        "retirement_age": [46],
    }
