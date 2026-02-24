"""Quick SS solve to check revenue/GDP breakdown."""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oguk.api import _build_specs, _suppress_output


def main():
    from ogcore.SS import run_SS

    with tempfile.TemporaryDirectory() as tmpdir, _suppress_output():
        p = _build_specs(2026, None, tmpdir, tmpdir)
        ss = run_SS(p, client=None)

    Y = float(ss["Y"])
    keys = [
        "total_tax_revenue",
        "iit_revenue",
        "payroll_tax_revenue",
        "cons_tax_revenue",
        "business_tax_revenue",
        "wealth_tax_revenue",
        "bequest_tax_revenue",
        "agg_pension_outlays",
        "G",
        "TR",
        "D",
        "I_g",
        "debt_service",
        "new_borrowing",
    ]
    print(f"{'Variable':<28} {'Level':>10} {'% of GDP':>10}")
    print("-" * 50)
    for k in keys:
        if k in ss:
            v = float(ss[k])
            print(f"{k:<28} {v:>10.4f} {v / Y * 100:>9.1f}%")
    print("-" * 50)
    print(f"{'GDP (Y)':<28} {Y:>10.4f}")

    # Spending total
    G = float(ss.get("G", 0))
    TR = float(ss.get("TR", 0))
    pensions = float(ss.get("agg_pension_outlays", 0))
    I_g = float(ss.get("I_g", 0))
    debt_svc = float(ss.get("debt_service", 0))
    total_spending = G + TR + pensions + I_g + debt_svc
    total_rev = float(ss["total_tax_revenue"])
    print(
        f"\n{'Total spending':<28} {total_spending:>10.4f} {total_spending / Y * 100:>9.1f}%"
    )
    print(f"{'Total revenue':<28} {total_rev:>10.4f} {total_rev / Y * 100:>9.1f}%")
    print(
        f"{'Deficit (spend - rev)':<28} {total_spending - total_rev:>10.4f} {(total_spending - total_rev) / Y * 100:>9.1f}%"
    )


if __name__ == "__main__":
    main()
