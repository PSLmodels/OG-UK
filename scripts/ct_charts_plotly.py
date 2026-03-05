"""Interactive Plotly charts: OBR outturn + forecast baseline vs CT +1pp reform.

All data in fiscal years (e.g. "2026-27").
Baseline = OBR Nov 2025 EFO outturn + forecast.
Reform   = OBR baseline × (1 + TPI % change from model).
"""

from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots

# ── Constants ─────────────────────────────────────────────────────────────────

REFORM_FY        = "2026-27"   # first full fiscal year of reform (model start_year=2026)
OBR_FORECAST_END = "2030-31"   # last fiscal year in OBR Nov 2025 EFO
OUTTURN_LAST_FY  = "2023-24"   # last fiscal year with full ONS/HMRC outturn

OBR_DATA = Path("/Users/nikhil.woodruff/10ds/obr-macroeconomic-model/data")

BLUE   = "#1a3a6e"
RED    = "#c0392b"
ORANGE = "#e67e22"
GREEN  = "#27ae60"
PURPLE = "#8e44ad"
GREY   = "#95a5a6"
GOLD   = "#b8860b"


def _fy_to_int(fy: str) -> int:
    return int(fy[:4])


# ── OBR / ONS data ────────────────────────────────────────────────────────────

def _fy_gdp() -> pd.Series:
    df = pd.read_excel(
        OBR_DATA / "obr_efo_november_2025_economy.xlsx",
        sheet_name="1.4", header=None,
    )
    by_fy: dict[str, dict[int, float]] = defaultdict(dict)
    for _, row in df.iterrows():
        p = str(row[1]).strip()
        if len(p) == 6 and p[4] == "Q":
            cy, q = int(p[:4]), int(p[5])
            fy = f"{cy-1}-{str(cy)[2:]}" if q == 1 else f"{cy}-{str(cy+1)[2:]}"
            by_fy[fy][q] = float(row[2])
    return pd.Series(
        {fy: sum(qs.values()) for fy, qs in sorted(by_fy.items()) if len(qs) == 4}
    )


def _fy_expenditure_components() -> pd.Series:
    df43 = pd.read_excel(
        OBR_DATA / "obr_efo_november_2025_expenditure.xlsx",
        sheet_name="4.3", header=None,
    )
    fy_labels = [
        str(v).strip() for v in df43.iloc[3, 4:].tolist()
        if str(v).strip() not in ("", "nan")
    ]
    tme_vals = df43.iloc[26, 4: 4 + len(fy_labels)].tolist()
    return pd.Series(dict(zip(fy_labels, tme_vals)), dtype=float)


def _fy_from_calendar(cal: pd.Series) -> pd.Series:
    fy = {}
    for cy in cal.index:
        if cy + 1 in cal.index:
            fy_label = f"{cy}-{str(cy+1)[2:]}"
            fy[fy_label] = 0.25 * cal[cy] + 0.75 * cal[cy + 1]
    return pd.Series(fy)


def _fy_macro_from_1_2() -> pd.DataFrame:
    df = pd.read_excel(
        OBR_DATA / "obr_efo_november_2025_economy.xlsx",
        sheet_name="1.2", header=None,
    )
    by_fy: dict[str, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for _, row in df.iterrows():
        p = str(row[1]).strip()
        if len(p) == 6 and p[4] == "Q":
            cy, q = int(p[:4]), int(p[5])
            fy = f"{cy-1}-{str(cy)[2:]}" if q == 1 else f"{cy}-{str(cy+1)[2:]}"
            for key, col in [("consumption", 2), ("investment", 4), ("gov_cons", 3)]:
                try:
                    by_fy[fy][key][q] = float(row[col])
                except (TypeError, ValueError):
                    pass
    rows = {}
    for fy, comps in sorted(by_fy.items()):
        row = {}
        for key in ("consumption", "investment", "gov_cons"):
            if key in comps and len(comps[key]) == 4:
                row[key] = sum(comps[key].values())
        if row:
            rows[fy] = row
    return pd.DataFrame(rows).T


def _fy_tax_revenue() -> pd.Series:
    return pd.Series({
        "2000-01": 334, "2001-02": 348, "2002-03": 346, "2003-04": 361,
        "2004-05": 391, "2005-06": 423, "2006-07": 454, "2007-08": 476,
        "2008-09": 516, "2009-10": 469, "2010-11": 496, "2011-12": 530,
        "2012-13": 552, "2013-14": 568, "2014-15": 600, "2015-16": 627,
        "2016-17": 656, "2017-18": 692, "2018-19": 729, "2019-20": 742,
        "2020-21": 669, "2021-22": 718, "2022-23": 786, "2023-24": 827,
        "2024-25": 893,
        "2025-26": 951, "2026-27": 1001, "2027-28": 1037,
        "2028-29": 1082, "2029-30": 1124, "2030-31": 1163,
    })


def _fy_debt(gdp: pd.Series) -> pd.Series:
    topic = "economy/governmentpublicsectorandtaxes/publicsectorfinance"
    url = f"https://www.ons.gov.uk/generator?format=csv&uri=/{topic}/timeseries/hf6x/pusf"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), header=None, names=["period", "value"])
        df["v"] = pd.to_numeric(df["value"], errors="coerce")
        ann = df.dropna(subset=["v"])
        ann = ann[ann["period"].str.strip().str.match(r"^\d{4}$")].copy()
        ann["cy"] = ann["period"].str.strip().astype(int)
        cal_pct = ann.set_index("cy")["v"].sort_index()
        fy_pct = _fy_from_calendar(cal_pct)
    except Exception as exc:
        print(f"  HF6X fetch failed ({exc}), using fallback")
        fy_pct = pd.Series({
            "2008-09": 55, "2009-10": 67, "2010-11": 78, "2011-12": 82,
            "2012-13": 84, "2013-14": 86, "2014-15": 88, "2015-16": 87,
            "2016-17": 87, "2017-18": 87, "2018-19": 86, "2019-20": 85,
            "2020-21": 103, "2021-22": 97, "2022-23": 97, "2023-24": 97,
        })
    obr_pct = {
        "2024-25": 95.5, "2025-26": 96.1, "2026-27": 96.5,
        "2027-28": 96.3, "2028-29": 95.8, "2029-30": 95.1, "2030-31": 94.5,
    }
    for fy, pct in obr_pct.items():
        if fy not in fy_pct.index:
            fy_pct[fy] = pct
    fy_pct = fy_pct.sort_index()
    common = fy_pct.index.intersection(gdp.index)
    return (fy_pct.loc[common] / 100 * gdp.loc[common]).sort_index()


_ONS_TIMEOUT  = 30
HIST_START_FY = "2000-01"


def _fetch_ons_cal(cdid: str, dataset: str, topic: str) -> pd.Series:
    url = (
        f"https://www.ons.gov.uk/generator?format=csv"
        f"&uri=/{topic}/timeseries/{cdid.lower()}/{dataset.lower()}"
    )
    resp = requests.get(url, timeout=_ONS_TIMEOUT)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), header=None, names=["period", "value"])
    df["v"] = pd.to_numeric(df["value"], errors="coerce")
    ann = df.dropna(subset=["v"])
    ann = ann[ann["period"].str.strip().str.match(r"^\d{4}$")].copy()
    ann["year"] = ann["period"].str.strip().astype(int)
    return ann.set_index("year")["v"].sort_index()


def _ons_backfill() -> pd.DataFrame:
    gdp_topic = "economy/grossdomesticproductgdp"
    import time as _time
    try:
        gdp_m = _fetch_ons_cal("YBHA", "ukea", gdp_topic)
        _time.sleep(1)
        inv_m = _fetch_ons_cal("NPQS", "ukea", gdp_topic)
        _time.sleep(1)
        gov_m = _fetch_ons_cal("NMRP", "ukea", gdp_topic)
    except Exception as exc:
        print(f"  ONS backfill fetch failed ({exc}), skipping pre-2008 history")
        return pd.DataFrame(columns=["gdp", "consumption", "investment"])

    gdp_fy  = _fy_from_calendar(gdp_m / 1000)
    inv_fy  = _fy_from_calendar(inv_m / 1000)
    gov_fy  = _fy_from_calendar(gov_m / 1000)
    common  = gdp_fy.index.intersection(inv_fy.index).intersection(gov_fy.index)
    cons_fy = gdp_fy.loc[common] - inv_fy.loc[common] - gov_fy.loc[common]
    return pd.DataFrame({"gdp": gdp_fy, "consumption": cons_fy, "investment": inv_fy})


def fetch_obr_baseline() -> pd.DataFrame:
    print("  Fiscal-year nominal GDP (OBR table 1.4)...")
    gdp = _fy_gdp()
    print("  Macro components (OBR table 1.2)...")
    macro = _fy_macro_from_1_2()
    print("  TME (OBR table 4.3)...")
    tme = _fy_expenditure_components()
    print("  Tax revenue (HMRC outturn + OBR forecast)...")
    tax = _fy_tax_revenue()
    print("  Debt (ONS HF6X + OBR forecast)...")
    debt = _fy_debt(gdp)

    bl = pd.DataFrame({"gdp": gdp})
    bl["consumption"] = macro["consumption"]
    bl["investment"]  = macro["investment"]
    bl["tme"]         = tme
    bl["tax_revenue"] = tax
    bl["debt"]        = debt

    print("  ONS backfill for pre-2008-09 history...")
    ons = _ons_backfill()
    if not ons.empty:
        ons_pre = ons.loc[
            [fy for fy in ons.index if fy < bl.index.min()],
            [c for c in ons.columns if c in bl.columns],
        ]
        if not ons_pre.empty:
            bl = pd.concat([ons_pre, bl]).sort_index()

    return bl.loc[HIST_START_FY: OBR_FORECAST_END]


# ── TPI % changes ─────────────────────────────────────────────────────────────

def load_tpi_pct_changes(xlsx_path: str = "scripts/ct_tpi_results.xlsx") -> pd.DataFrame:
    """Load CT TPI results and compute % change of reform vs baseline.

    Tax revenue % change is read from the corrected xlsx (which already has
    OBR-CT-anchored figures from ct_reexport.py).
    """
    xl   = pd.ExcelFile(xlsx_path)
    base = xl.parse("Baseline levels", header=2)
    ref  = xl.parse("Reform levels",   header=2)

    # Baseline levels include historical rows (yellow); reform levels are model-only.
    # Align on fiscal year.
    base_mdl = base[base["Fiscal year"].isin(ref["Fiscal year"])].reset_index(drop=True)

    cols = {
        "GDP (£bn)":         "gdp",
        "Consumption (£bn)": "consumption",
        "Investment (£bn)":  "investment",
        "Government (£bn)":  "tme",
        "Tax revenue (£bn)": "tax_revenue",
        "Debt (£bn)":        "debt",
    }

    fy_labels = ref["Fiscal year"].values
    pct = {}
    for tc, key in cols.items():
        s = (ref[tc].values - base_mdl[tc].values) / base_mdl[tc].values * 100
        pct[key] = pd.Series(s, index=fy_labels)

    return pd.DataFrame(pct)


# ── Charts ────────────────────────────────────────────────────────────────────

def fig_main(baseline: pd.DataFrame, pct: pd.DataFrame) -> go.Figure:
    variables = [
        ("Consumption (% GDP)",  "consumption", ORANGE),
        ("Investment (% GDP)",   "investment",  GREEN),
        ("TME (% GDP)",          "tme",         PURPLE),
        ("Tax revenue (% GDP)",  "tax_revenue", RED),
        ("Debt (% GDP)",         "debt",        GREY),
        ("GDP (£bn)",            "gdp",         BLUE),
    ]

    baseline = baseline.copy()
    baseline.index = [_fy_to_int(fy) for fy in baseline.index]
    pct = pct.copy()
    pct.index = [_fy_to_int(fy) for fy in pct.index]

    gdp = baseline["gdp"].dropna()

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[v[0] for v in variables],
        vertical_spacing=0.16, horizontal_spacing=0.08,
    )
    positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    shown_base = shown_ref = False

    for (label, key, color), (row, col) in zip(variables, positions):
        s = baseline[key].dropna()
        as_pct_gdp = key != "gdp"
        if as_pct_gdp:
            common = s.index.intersection(gdp.index)
            s = s.loc[common] / gdp.loc[common] * 100

        fig.add_trace(go.Scatter(
            x=s.index.tolist(), y=s.tolist(),
            mode="lines", line=dict(color=color, width=2.5),
            name="OBR outturn / forecast" if not shown_base else None,
            showlegend=not shown_base, legendgroup="base",
            hovertemplate=(
                "%{x}: %{y:.1f}%<extra>Baseline</extra>"
                if as_pct_gdp else
                "%{x}: £%{y:,.0f}bn<extra>Baseline</extra>"
            ),
        ), row=row, col=col)
        shown_base = True

        if key in pct.columns:
            reform_start = _fy_to_int(REFORM_FY)
            reform_xs = [x for x in s.index if x in pct.index and x >= reform_start]
            if reform_xs:
                ref_s = s.loc[reform_xs] * (1 + pct.loc[reform_xs, key] / 100)
                anchor = reform_start - 1
                if anchor in s.index:
                    ref_s = pd.concat([s[[anchor]], ref_s]).sort_index()

                fig.add_trace(go.Scatter(
                    x=ref_s.index.tolist(), y=ref_s.tolist(),
                    mode="lines", line=dict(color=color, width=2.5, dash="dash"),
                    name="Reform: CT main rate +1pp" if not shown_ref else None,
                    showlegend=not shown_ref, legendgroup="ref",
                    hovertemplate=(
                        f"%{{x}}: %{{y:.1f}}%<extra>Reform</extra>"
                        if as_pct_gdp else
                        f"%{{x}}: £%{{y:,.0f}}bn<extra>Reform</extra>"
                    ),
                ), row=row, col=col)
                shown_ref = True

        fig.add_vline(x=_fy_to_int(OUTTURN_LAST_FY), line_width=1,
                      line_dash="dot", line_color=GREY, row=row, col=col)
        fig.add_vline(x=_fy_to_int(REFORM_FY), line_width=1.2,
                      line_dash="dash", line_color=GOLD, row=row, col=col)

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
                "<b>OG-UK: Corporation Tax Main Rate +1pp — Impact on OBR Forecast</b><br>"
                "<sup>Solid = OBR outturn / Nov 2025 EFO forecast · Dashed = reform · "
                "Grey dotted = outturn/forecast boundary · Gold dashed = reform start · "
                "Tax revenue anchored to OBR CT receipts (matches HMRC RR £3.6–4.0bn/yr)</sup>"
            ),
            x=0.5, font=dict(size=15),
        ),
        height=780,
    )

    all_years = sorted(baseline.index.tolist())
    fig.update_xaxes(
        showgrid=True, gridcolor="#ececec", zeroline=False,
        tickangle=45, tickfont=dict(size=10),
        tickmode="array",
        tickvals=all_years,
        ticktext=[f"{y}-{str(y+1)[2:]}" for y in all_years],
    )
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


def fig_changes(pct: pd.DataFrame, baseline: pd.DataFrame) -> go.Figure:
    """
    Supplementary chart: % changes (reform vs baseline) for each variable.
    Shows only forecast years from reform start.
    """
    variables = [
        ("GDP % change",         "gdp",         BLUE),
        ("Consumption % change", "consumption", ORANGE),
        ("Investment % change",  "investment",  GREEN),
        ("TME % change",         "tme",         PURPLE),
        ("Tax revenue % change", "tax_revenue", RED),
        ("Debt % change",        "debt",        GREY),
    ]

    pct = pct.copy()
    pct.index = [_fy_to_int(fy) for fy in pct.index]
    reform_start = _fy_to_int(REFORM_FY)
    pct_reform = pct.loc[pct.index >= reform_start]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[v[0] for v in variables],
        vertical_spacing=0.18, horizontal_spacing=0.08,
    )
    positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]

    for (label, key, color), (row, col) in zip(variables, positions):
        if key not in pct_reform.columns:
            continue
        s = pct_reform[key].dropna()
        fig.add_trace(go.Bar(
            x=s.index.tolist(),
            y=s.tolist(),
            marker_color=color,
            name=label,
            showlegend=False,
            hovertemplate="%{x}: %{y:.3f}%<extra></extra>",
        ), row=row, col=col)
        fig.add_hline(y=0, line_width=1, line_color="black", row=row, col=col)

    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=12),
        paper_bgcolor="white", plot_bgcolor="white",
        title=dict(
            text=(
                "<b>OG-UK: CT Main Rate +1pp — % Change from Baseline</b><br>"
                "<sup>Reform period only · Based on OG-UK dynamic general equilibrium model</sup>"
            ),
            x=0.5, font=dict(size=14),
        ),
        height=700,
    )
    all_years = sorted(pct_reform.index.tolist())
    fig.update_xaxes(
        showgrid=False, zeroline=False,
        tickangle=45, tickfont=dict(size=10),
        tickmode="array",
        tickvals=all_years,
        ticktext=[f"{y}-{str(y+1)[2:]}" for y in all_years],
    )
    fig.update_yaxes(showgrid=True, gridcolor="#ececec",
                     ticksuffix="%", tickformat=".2f")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    xlsx_path = "scripts/ct_tpi_results.xlsx"
    print("Loading TPI % changes...")
    pct = load_tpi_pct_changes(xlsx_path)

    print("Fetching OBR baseline (fiscal years)...")
    baseline = fetch_obr_baseline()
    print(f"  Baseline: {baseline.index.min()} – {baseline.index.max()}")

    print("Building main chart...")
    fig = fig_main(baseline, pct)
    fig.write_html("scripts/ct_charts_levels.html")
    print("Saved: scripts/ct_charts_levels.html")

    print("Building changes chart...")
    fig2 = fig_changes(pct, baseline)
    fig2.write_html("scripts/ct_charts_changes.html")
    print("Saved: scripts/ct_charts_changes.html")


if __name__ == "__main__":
    main()
