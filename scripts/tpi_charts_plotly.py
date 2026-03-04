"""Interactive Plotly charts: OBR outturn + forecast baseline vs reform.

One page, six panels (2×3).  Shows history through the end of the OBR
forecast horizon.  Reform line = OBR baseline × (1 + TPI % change).
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots

# ── Constants ─────────────────────────────────────────────────────────────────

REFORM_YEAR  = 2027
OBR_FORECAST_END = 2030   # last year in OBR Nov 2025 EFO
OUTTURN_LAST = 2024       # last year with ONS / HMRC outturn

OBR_DATA = Path("/Users/nikhil.woodruff/10ds/obr-macroeconomic-model/data")

BLUE   = "#1a3a6e"
RED    = "#c0392b"
ORANGE = "#e67e22"
GREEN  = "#27ae60"
PURPLE = "#8e44ad"
GREY   = "#95a5a6"
GOLD   = "#b8860b"

# ── OBR / ONS data ────────────────────────────────────────────────────────────

def _load_economy_12() -> pd.DataFrame:
    df = pd.read_excel(
        OBR_DATA / "obr_efo_november_2025_economy.xlsx",
        sheet_name="1.2", header=None,
    )
    rows = []
    for _, row in df.iterrows():
        period = str(row[1]).strip()
        if len(period) == 4 and period.isdigit():
            rows.append({
                "year":        int(period),
                "gdp":         row[14],
                "consumption": row[2],
                "investment":  row[4],
                "government":  row[3],
            })
    return pd.DataFrame(rows).set_index("year")


def _tax_revenue_series() -> pd.Series:
    """HMRC outturn 2008-2024; OBR forecast 2025-2030."""
    return pd.Series({
        2008: 516, 2009: 469, 2010: 496, 2011: 530, 2012: 552,
        2013: 568, 2014: 600, 2015: 627, 2016: 656, 2017: 692,
        2018: 729, 2019: 742, 2020: 669, 2021: 718, 2022: 786,
        2023: 827, 2024: 859,
        # OBR Nov 2025 EFO total HMRC receipts (fiscal yr → calendar yr)
        2025: 893, 2026: 951, 2027: 1001,
        2028: 1037, 2029: 1082, 2030: 1124,
    })


def _debt_series(gdp: pd.Series) -> pd.Series:
    """ONS HF6X outturn; OBR forecast % × OBR nominal GDP for 2025+."""
    topic = "economy/governmentpublicsectorandtaxes/publicsectorfinance"
    url = (f"https://www.ons.gov.uk/generator?format=csv"
           f"&uri=/{topic}/timeseries/hf6x/pusf")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), header=None, names=["period","value"])
        df["v"] = pd.to_numeric(df["value"], errors="coerce")
        ann = df.dropna(subset=["v"])
        ann = ann[ann["period"].str.strip().str.match(r"^\d{4}$")].copy()
        ann["year"] = ann["period"].str.strip().astype(int)
        pct = ann.set_index("year")["v"].sort_index()
    except Exception as exc:
        print(f"  HF6X fetch failed ({exc}), using fallback")
        pct = pd.Series({
            2008:52,2009:65,2010:75,2011:82,2012:84,2013:86,
            2014:88,2015:87,2016:87,2017:87,2018:86,2019:85,
            2020:103,2021:97,2022:97,2023:97,2024:96,
        })

    # OBR forecast pct for years beyond outturn
    obr_pct = {2025:95.5, 2026:96.1, 2027:96.5, 2028:96.3, 2029:95.8, 2030:95.1}
    for yr, p in obr_pct.items():
        if yr not in pct.index and yr in gdp.index:
            pct[yr] = p

    common = pct.index.intersection(gdp.index)
    return (pct.loc[common] / 100 * gdp.loc[common]).sort_index()


def fetch_obr_baseline() -> pd.DataFrame:
    print("  OBR EFO table 1.2 (nominal GDP components)...")
    bl = _load_economy_12()
    bl["tax_revenue"] = _tax_revenue_series()
    print("  ONS HF6X debt % GDP...")
    bl["debt"] = _debt_series(bl["gdp"])
    return bl.loc[2008:OBR_FORECAST_END]


# ── TPI % changes ─────────────────────────────────────────────────────────────

def load_tpi_pct_changes() -> pd.DataFrame:
    xl   = pd.ExcelFile("scripts/tpi_results.xlsx")
    base = xl.parse("Baseline levels", header=2)
    ref  = xl.parse("Reform levels",   header=2)
    cols = {
        "GDP (£bn)":         "gdp",
        "Consumption (£bn)": "consumption",
        "Investment (£bn)":  "investment",
        "Government (£bn)":  "government",
        "Tax revenue (£bn)": "tax_revenue",
        "Debt (£bn)":        "debt",
    }
    pct = pd.DataFrame({"year": base["Year"].astype(int)})
    for tc, key in cols.items():
        pct[key] = (ref[tc] - base[tc]) / base[tc] * 100
    return pct.set_index("year")


# ── Single-page chart ─────────────────────────────────────────────────────────

def fig_main(baseline: pd.DataFrame, pct: pd.DataFrame) -> go.Figure:
    # GDP is excluded from the % of GDP panels (it would be a flat 100% line)
    variables = [
        ("Consumption (% GDP)", "consumption", ORANGE),
        ("Investment (% GDP)",  "investment",  GREEN),
        ("Government (% GDP)",  "government",  PURPLE),
        ("Tax revenue (% GDP)", "tax_revenue", RED),
        ("Debt (% GDP)",        "debt",        GREY),
        ("GDP (£bn)",           "gdp",         BLUE),
    ]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[v[0] for v in variables],
        vertical_spacing=0.16, horizontal_spacing=0.08,
    )
    positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]

    gdp = baseline["gdp"].dropna()
    shown_base = shown_ref = False

    for (label, key, color), (row, col) in zip(variables, positions):
        s = baseline[key].dropna()

        # Express as % of GDP (GDP panel stays in £bn)
        as_pct_gdp = key != "gdp"
        if as_pct_gdp:
            common = s.index.intersection(gdp.index)
            s = (s.loc[common] / gdp.loc[common] * 100)

        # Single continuous baseline line (outturn + forecast, same style)
        fig.add_trace(go.Scatter(
            x=s.index.tolist(), y=s.tolist(),
            mode="lines", line=dict(color=color, width=2.5),
            name="OBR outturn / forecast" if not shown_base else None,
            showlegend=not shown_base, legendgroup="base",
            hovertemplate=(
                f"%{{x}}: %{{y:,.1f}}%<extra>Baseline</extra>"
                if as_pct_gdp else
                f"%{{x}}: £%{{y:,.0f}}bn<extra>Baseline</extra>"
            ),
        ), row=row, col=col)
        shown_base = True

        # Reform: only over OBR forecast window, same colour, dashed
        if key in pct.columns:
            reform_years = s.index.intersection(pct.index)
            reform_years = reform_years[
                (reform_years >= REFORM_YEAR) & (reform_years <= OBR_FORECAST_END)
            ]
            if len(reform_years):
                # Apply % change to the (possibly pct-of-GDP) baseline values
                ref_s = s.loc[reform_years] * (1 + pct.loc[reform_years, key] / 100)
                anchor_yr = REFORM_YEAR - 1
                if anchor_yr in s.index:
                    ref_s = pd.concat([s[[anchor_yr]], ref_s]).sort_index()

                fig.add_trace(go.Scatter(
                    x=ref_s.index.tolist(), y=ref_s.tolist(),
                    mode="lines", line=dict(color=color, width=2.5, dash="dash"),
                    name="Reform +1pp basic rate" if not shown_ref else None,
                    showlegend=not shown_ref, legendgroup="ref",
                    hovertemplate=(
                        f"%{{x}}: %{{y:,.1f}}%<extra>Reform</extra>"
                        if as_pct_gdp else
                        f"%{{x}}: £%{{y:,.0f}}bn<extra>Reform</extra>"
                    ),
                ), row=row, col=col)
                shown_ref = True

        # Outturn / forecast boundary
        fig.add_vline(x=OUTTURN_LAST + 0.5, line_width=1, line_dash="dot",
                      line_color=GREY, row=row, col=col)
        # Reform start
        fig.add_vline(x=REFORM_YEAR, line_width=1.2, line_dash="dash",
                      line_color=GOLD, row=row, col=col)

    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=12),
        paper_bgcolor="white", plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#ddd",
            borderwidth=1, font=dict(size=11),
            orientation="h", x=0.5, xanchor="center", y=-0.06,
        ),
        title=dict(
            text=(
                "<b>OG-UK: Basic Rate +1pp from 2027 — Impact on OBR Forecast</b><br>"
                "<sup>Solid = OBR outturn / forecast · Dashed = reform · "
                "Grey dotted = outturn/forecast boundary</sup>"
            ),
            x=0.5, font=dict(size=15),
        ),
        height=780,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#ececec", zeroline=False,
                     range=[2008, OBR_FORECAST_END + 0.5])
    # % of GDP axes for first 5 panels, £bn for GDP panel
    for i, (_, key, _) in enumerate(variables):
        r, c = positions[i]
        if key != "gdp":
            fig.update_yaxes(showgrid=True, gridcolor="#ececec",
                             ticksuffix="%", tickformat=".1f", row=r, col=c)
        else:
            fig.update_yaxes(showgrid=True, gridcolor="#ececec",
                             tickprefix="£", ticksuffix="bn", tickformat=",.0f",
                             row=r, col=c)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading TPI % changes...")
    pct = load_tpi_pct_changes()

    print("Fetching OBR baseline...")
    baseline = fetch_obr_baseline()

    print("Building chart...")
    fig = fig_main(baseline, pct)
    fig.write_html("scripts/tpi_charts.html")
    print("Saved: scripts/tpi_charts.html")


if __name__ == "__main__":
    main()
