"""Example: Run OG-UK baseline and reform simulations."""

from __future__ import annotations

import sys
import time
from datetime import datetime

from policyengine.core import ParameterValue, Policy
from policyengine.tax_benefit_models.uk import uk_latest

from oguk import (
    map_to_real_world,
    map_transition_to_real_world,
    run_transition_path,
    solve_steady_state,
)

# Reform: increase basic rate from 20% to 21%
basic_rate_param = uk_latest.get_parameter("gov.hmrc.income_tax.rates.uk[0].rate")
REFORM = Policy(
    name="Basic rate 21%",
    parameter_values=[
        ParameterValue(
            parameter=basic_rate_param,
            value=0.21,
            start_date=datetime(2026, 1, 1),
        )
    ],
)


def run_steady_state(age_specific: str = "pooled", multi_sector: bool = False):
    """Run baseline and reform steady state, print results."""
    sector_label = "8-sector" if multi_sector else "1-sector"
    print(f"Solving baseline steady state (age_specific='{age_specific}', {sector_label})...")
    t0 = time.time()
    baseline = solve_steady_state(start_year=2026, age_specific=age_specific, multi_sector=multi_sector)
    print(f"  Done in {time.time() - t0:.1f}s")

    print(f"Solving reform steady state (age_specific='{age_specific}', {sector_label})...")
    t0 = time.time()
    reform = solve_steady_state(
        start_year=2026, policy=REFORM, age_specific=age_specific, multi_sector=multi_sector,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    impact = map_to_real_world(baseline, reform)

    print("\n" + "=" * 60)
    print("Steady state impact (£bn, current prices)")
    print("=" * 60)
    print(f"{'Variable':<15} {'Baseline':>12} {'Reform':>12} {'Change':>10} {'%':>8}")
    print("-" * 57)
    for label, level, change, pct in [
        ("GDP", impact.gdp, impact.gdp_change, impact.gdp_pct),
        (
            "Consumption",
            impact.consumption,
            impact.consumption_change,
            impact.consumption_pct,
        ),
        (
            "Investment",
            impact.investment,
            impact.investment_change,
            impact.investment_pct,
        ),
        (
            "Government",
            impact.government,
            impact.government_change,
            impact.government_pct,
        ),
        (
            "Tax revenue",
            impact.tax_revenue,
            impact.tax_revenue_change,
            impact.tax_revenue_pct,
        ),
        ("Debt", impact.debt, impact.debt_change, impact.debt_pct),
    ]:
        base_bn = level - change
        print(
            f"{label:<15} {base_bn:>10.1f} {level:>12.1f} {change:>+9.1f} {pct:>+7.3f}%"
        )
    print(f"\nInterest rate:  {impact.r_baseline:.2%} -> {impact.r_reform:.2%}")
    print("=" * 60)


def run_tpi(multi_sector: bool = False):
    """Run baseline and reform transition paths, print results."""
    from dask.distributed import Client

    sector_label = "8-sector" if multi_sector else "1-sector"
    print(f"Running baseline + reform transition paths ({sector_label})...")
    print("(This solves SS + TPI for both scenarios — may take a while.)")
    client = Client(n_workers=2, threads_per_worker=1, memory_limit="2GB")
    t0 = time.time()
    try:
        base_tp, reform_tp = run_transition_path(
            start_year=2026, policy=REFORM, client=client, multi_sector=multi_sector,
        )
    finally:
        client.close()
    print(f"Done in {time.time() - t0:.1f}s")

    impact = map_transition_to_real_world(base_tp, reform_tp)

    print("\n" + "=" * 70)
    print("Transition path: reform impact (£bn change from baseline)")
    print("=" * 70)
    print(
        f"{'Year':<8} {'GDP':>8} {'Cons':>8} {'Inv':>8} {'Gov':>8} {'Tax rev':>8} {'Debt':>8}"
    )
    print("-" * 70)

    # Show first 10 years, then every 10th, then final
    T = len(impact.years)
    indices = list(range(min(10, T)))
    indices += list(range(10, T, 10))
    if T - 1 not in indices:
        indices.append(T - 1)
    indices = sorted(set(indices))

    for i in indices:
        print(
            f"{str(impact.years[i]):<8}"
            f" {impact.gdp_change[i]:>+7.1f}"
            f" {impact.consumption_change[i]:>+7.1f}"
            f" {impact.investment_change[i]:>+7.1f}"
            f" {impact.government_change[i]:>+7.1f}"
            f" {impact.tax_revenue_change[i]:>+7.1f}"
            f" {impact.debt_change[i]:>+7.1f}"
        )
    print("=" * 70)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "ss"
    age_specific = sys.argv[2] if len(sys.argv) > 2 else "pooled"
    multi_sector = "multi-sector" in sys.argv
    if mode == "ss":
        run_steady_state(age_specific=age_specific, multi_sector=multi_sector)
    elif mode == "tpi":
        run_tpi(multi_sector=multi_sector)
    else:
        print(f"Usage: {sys.argv[0]} [ss|tpi] [pooled|brackets|each] [multi-sector]")
        sys.exit(1)


if __name__ == "__main__":
    main()
