"""Estimate the fiscal headroom impact of a +0.1pp productivity growth shock by 2029-30.

We model this as a TFP (Z) level shock rather than a g_y_annual change, because
g_y_annual in OG-Core is a normalisation parameter (it defines the balanced growth
path that the model detrends by), not a production function input. Shocking g_y
changes the *units* the model works in, not the real economy.

A permanent +0.1pp annual productivity growth rate over 4 years (2026-2030)
compounds to ~0.4% higher TFP by 2029-30. We model this as a permanent Z
increase to capture the level effect, then read off the fiscal impact at 2029-30.

Usage:
    .venv/bin/python3 scripts/productivity_headroom.py
"""

from __future__ import annotations

import os
import time

from oguk import (
    map_transition_to_real_world,
    run_transition_path,
)

START_YEAR = 2026
TARGET_FY = "2029-30"
GROWTH_PP = 0.001  # +0.1pp per year
YEARS_TO_TARGET = 4  # 2026-27 to 2029-30
Z_SHOCK = (1 + GROWTH_PP) ** YEARS_TO_TARGET  # ~1.004, cumulative TFP gain


def main():
    from dask.distributed import Client

    n_workers = os.cpu_count()
    print(f"Starting Dask client ({n_workers} workers)...")
    client = Client(n_workers=n_workers, threads_per_worker=1)

    try:
        print(f"\nRunning baseline + reform (Z={Z_SHOCK:.4f}) transition paths...")
        t0 = time.time()
        base_tp, reform_tp = run_transition_path(
            start_year=START_YEAR,
            client=client,
            param_overrides={"Z": [[Z_SHOCK]]},
        )
        print(f"  Done in {time.time() - t0:.1f}s")
    finally:
        client.close()

    print("\nMapping to real-world £bn...")
    impact = map_transition_to_real_world(base_tp, reform_tp)

    years = list(impact.years)
    idx = years.index(TARGET_FY) if TARGET_FY in years else 3
    fy = impact.years[idx]

    print("\n" + "=" * 65)
    print(f"  +0.1pp productivity growth (Z shock): fiscal impact in {fy}")
    print("=" * 65)
    print(f"  {'Metric':<28} {'Baseline':>10} {'Reform':>10} {'Change':>10}")
    print("  " + "-" * 60)

    rows = [
        ("GDP (£bn)", impact.gdp, impact.gdp_change),
        ("Tax revenue (£bn)", impact.tax_revenue, impact.tax_revenue_change),
        ("Debt (£bn)", impact.debt, impact.debt_change),
        ("Consumption (£bn)", impact.consumption, impact.consumption_change),
        ("Investment (£bn)", impact.investment, impact.investment_change),
    ]

    for label, level, change in rows:
        base_val = level[idx] - change[idx]
        print(
            f"  {label:<28} {base_val:>10.1f} {level[idx]:>10.1f} {change[idx]:>+10.1f}"
        )

    print("=" * 65)
    print(
        f"\n  Fiscal headroom (tax revenue change in {fy}): £{impact.tax_revenue_change[idx]:+.1f}bn"
    )

    print("\n  Full transition path (£bn change from baseline):")
    print(f"  {'FY':<10} {'GDP':>10} {'Tax rev':>10} {'Investment':>10} {'Debt':>10}")
    print("  " + "-" * 52)
    T = len(impact.years)
    show = list(range(min(10, T))) + list(range(10, T, 5))
    if T - 1 not in show:
        show.append(T - 1)
    for i in sorted(set(show)):
        print(
            f"  {str(impact.years[i]):<10}"
            f" {impact.gdp_change[i]:>+10.1f}"
            f" {impact.tax_revenue_change[i]:>+10.1f}"
            f" {impact.investment_change[i]:>+10.1f}"
            f" {impact.debt_change[i]:>+10.1f}"
        )


if __name__ == "__main__":
    main()
