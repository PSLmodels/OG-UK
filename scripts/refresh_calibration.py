"""Refresh OG-UK default parameters from UK official data sources.

Scrapes macroeconomic parameters from ONS, Bank of England, and GOV.UK,
then writes a clean oguk/oguk_default_parameters.json.

Usage:
    uv run python scripts/refresh_calibration.py
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from oguk.sources import fetch_boe_series, fetch_ons_timeseries, get_uk_tax_rates

console = Console()

JSON_PATH = Path(__file__).parent.parent / "oguk" / "oguk_default_parameters.json"

# US-specific parameters to remove
US_ONLY_KEYS = [
    "AIME_num_years",
    "AIME_bkt_1",
    "AIME_bkt_2",
    "PIA_rate_bkt_1",
    "PIA_rate_bkt_2",
    "PIA_rate_bkt_3",
    "PIA_maxpayment",
    "PIA_minpayment",
    "surv_rate",
]


def main() -> None:
    console.print("[bold]Refreshing OG-UK default parameters[/bold]\n")

    # ------------------------------------------------------------------
    # 1. Fetch from ONS
    # ------------------------------------------------------------------
    console.print("[cyan]Fetching ONS data...[/cyan]")

    # Public sector net debt as % GDP
    # ONS series HF6X in dataset pusf
    debt_pct = fetch_ons_timeseries(
        "HF6X",
        "pusf",
        "economy/governmentpublicsectorandtaxes/publicsectorfinance",
        frequency="months",
        fallback=92.9,
    )
    initial_debt_ratio = round(debt_pct / 100, 4)
    console.print(f"  initial_debt_ratio = {initial_debt_ratio} (ONS HF6X: {debt_pct}%)")

    # Nominal GDP at market prices (£m, SA)
    # ONS series YBHA in dataset ukea
    gdp_nominal = fetch_ons_timeseries(
        "YBHA",
        "ukea",
        "economy/grossdomesticproductgdp",
        frequency="years",
        fallback=2_890_664,
    )
    console.print(f"  GDP nominal = £{gdp_nominal:,.0f}m (ONS YBHA)")

    # General government final consumption expenditure (£m, CP SA)
    # ONS series NMRP in dataset ukea
    gov_consumption = fetch_ons_timeseries(
        "NMRP",
        "ukea",
        "economy/grossdomesticproductgdp",
        frequency="years",
        fallback=583_561,
    )
    alpha_G = round(gov_consumption / gdp_nominal, 4)
    console.print(f"  alpha_G = {alpha_G} (ONS NMRP/YBHA)")

    # Net international investment position (£m, NSA)
    # ONS series HBQC in dataset pnbp
    niip = fetch_ons_timeseries(
        "HBQC",
        "pnbp",
        "economy/nationalaccounts/balanceofpayments",
        frequency="quarters",
        fallback=-261_400,
    )
    initial_foreign_debt_ratio = round(abs(niip) / gdp_nominal, 4)
    console.print(f"  initial_foreign_debt_ratio = {initial_foreign_debt_ratio} (ONS HBQC/YBHA)")

    # ------------------------------------------------------------------
    # 2. Fetch from Bank of England
    # ------------------------------------------------------------------
    console.print("\n[cyan]Fetching Bank of England data...[/cyan]")

    bank_rate = fetch_boe_series("IUDBEDR", fallback=3.75)
    # Real interest rate = bank rate - BoE 2% inflation target
    world_int_rate = round((bank_rate / 100) - 0.02, 4)
    console.print(f"  Bank rate = {bank_rate}%")
    console.print(f"  world_int_rate_annual = {world_int_rate} (bank rate - 2% target)")

    # ------------------------------------------------------------------
    # 3. Tax rates (hardcoded with GOV.UK citations)
    # ------------------------------------------------------------------
    console.print("\n[cyan]Loading UK tax rates...[/cyan]")
    tax_rates = get_uk_tax_rates()
    for k, v in tax_rates.items():
        console.print(f"  {k} = {v}")

    # ------------------------------------------------------------------
    # 4. Load existing JSON and update
    # ------------------------------------------------------------------
    console.print(f"\n[cyan]Loading existing {JSON_PATH.name}...[/cyan]")
    with open(JSON_PATH) as f:
        params = json.load(f)

    # Remove US-specific parameters
    removed = []
    for key in US_ONLY_KEYS:
        if key in params:
            del params[key]
            removed.append(key)

    # Fiscal position
    params["initial_debt_ratio"] = initial_debt_ratio
    params["debt_ratio_ss"] = 0.80  # OBR long-run anchor
    params["initial_foreign_debt_ratio"] = initial_foreign_debt_ratio
    params["alpha_G"] = [alpha_G]
    # Social protection spending ~15% of GDP (ONS PESA)
    params["alpha_T"] = [0.15]
    params["budget_balance"] = False
    # Fiscal rule timing: tG1 is when closure begins, tG2 when debt
    # ratio must hit target. Push out so reform effects have time to play out.
    params["tG1"] = 80
    params["tG2"] = 256

    # Tax rates
    params.update(tax_rates)

    # Economic parameters
    # OBR medium-term productivity growth forecast: ~1.0% pa
    # Source: OBR Economic and Fiscal Outlook, March 2025
    params["g_y_annual"] = 0.01
    params["world_int_rate_annual"] = [world_int_rate]
    params["delta_annual"] = 0.05  # Standard depreciation rate
    params["c_corp_share_of_assets"] = 0.55  # ONS national accounts

    # ------------------------------------------------------------------
    # 5. Fix parameter formats for OG-Core compatibility
    # ------------------------------------------------------------------

    # r_gov_scale: scalar -> list
    val = params.get("r_gov_scale", 1.0)
    if not isinstance(val, list):
        params["r_gov_scale"] = [val]

    # r_gov_shift: scalar -> list
    val = params.get("r_gov_shift", 0.02)
    if not isinstance(val, list):
        params["r_gov_shift"] = [val]

    # replacement_rate_adjust: 1D -> 2D
    val = params.get("replacement_rate_adjust", [1.0])
    if isinstance(val, list) and val and not isinstance(val[0], list):
        params["replacement_rate_adjust"] = [val]

    # rho: 1D -> 2D
    rho = params.get("rho", [])
    if isinstance(rho, list) and rho and not isinstance(rho[0], list):
        params["rho"] = [rho]

    # Update start_year to 2026
    params["start_year"] = 2026

    # ------------------------------------------------------------------
    # 6. Write output
    # ------------------------------------------------------------------
    with open(JSON_PATH, "w") as f:
        json.dump(params, f, indent=2)
    console.print(f"\n[green]Wrote {JSON_PATH}[/green]")

    # ------------------------------------------------------------------
    # 7. Display summary
    # ------------------------------------------------------------------
    table = Table(title="Updated parameters")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")
    table.add_column("Source", style="dim")

    rows = [
        ("initial_debt_ratio", f"{initial_debt_ratio}", "ONS HF6X"),
        ("debt_ratio_ss", "0.80", "OBR projection"),
        ("initial_foreign_debt_ratio", f"{initial_foreign_debt_ratio}", "ONS HBQC/YBHA"),
        ("alpha_G", f"{alpha_G}", "ONS NMRP/YBHA"),
        ("alpha_T", "0.15", "ONS PESA"),
        ("cit_rate", "0.25", "GOV.UK"),
        ("tau_payroll", "0.15", "GOV.UK employer NICs"),
        ("tau_c", "0.10", "HMRC/OBR effective VAT"),
        ("tau_bq", "0.08", "GOV.UK effective IHT"),
        ("delta_tau_annual", "0.18", "GOV.UK capital allowances"),
        ("retirement_age", "46 (age 66)", "GOV.UK"),
        ("g_y_annual", "0.01", "OBR forecast"),
        ("world_int_rate_annual", f"{world_int_rate}", "BoE rate - 2%"),
    ]
    for name, val, src in rows:
        table.add_row(name, val, src)
    console.print(table)

    if removed:
        console.print(f"\n[yellow]Removed {len(removed)} US-specific parameters: {', '.join(removed)}[/yellow]")
    console.print("[yellow]Fixed r_gov_scale, r_gov_shift (scalar -> list)[/yellow]")
    console.print("[yellow]Fixed rho, replacement_rate_adjust (1D -> 2D)[/yellow]")
    console.print("[yellow]Updated start_year to 2026[/yellow]")


if __name__ == "__main__":
    main()
