"""Interactive Plotly charts: OBR outturn + forecast baseline vs reform.

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

REFORM_FY = "2027-28"  # first fiscal year of reform
OBR_FORECAST_END = "2030-31"  # last fiscal year in OBR Nov 2025 EFO
OUTTURN_LAST_FY = "2023-24"  # last fiscal year with full ONS/HMRC outturn
# (2024-25 is provisional/outturn in OBR)


def _fy_to_int(fy: str) -> int:
    """Convert fiscal year label e.g. '2027-28' to start year integer 2027."""
    return int(fy[:4])


OBR_DATA = Path(__file__).resolve().parent.parent / "data"

BLUE = "#1a3a6e"
RED = "#c0392b"
ORANGE = "#e67e22"
GREEN = "#27ae60"
PURPLE = "#8e44ad"
GREY = "#95a5a6"
GOLD = "#b8860b"

# ── OBR / ONS data ────────────────────────────────────────────────────────────


def _fy_gdp() -> pd.Series:
    """Nominal GDP in fiscal years (£bn) from OBR EFO table 1.4 quarterly NSA."""
    df = pd.read_excel(
        OBR_DATA / "obr_efo_november_2025_economy.xlsx",
        sheet_name="1.4",
        header=None,
    )
    by_fy: dict[str, dict[int, float]] = defaultdict(dict)
    for _, row in df.iterrows():
        p = str(row[1]).strip()
        if len(p) == 6 and p[4] == "Q":
            cy, q = int(p[:4]), int(p[5])
            fy = f"{cy - 1}-{str(cy)[2:]}" if q == 1 else f"{cy}-{str(cy + 1)[2:]}"
            by_fy[fy][q] = float(row[2])
    return pd.Series(
        {fy: sum(qs.values()) for fy, qs in sorted(by_fy.items()) if len(qs) == 4}
    )


def _fy_expenditure_components() -> pd.DataFrame:
    """
    Government consumption and investment as % GDP from OBR table 1.2
    (aggregated to fiscal years) and TME £bn from OBR table 4.3.
    Returns DataFrame with columns: tme_bn, gov_cons_pct (% GDP).
    """
    # TME £bn — table 4.3, row 26
    df43 = pd.read_excel(
        OBR_DATA / "obr_efo_november_2025_expenditure.xlsx",
        sheet_name="4.3",
        header=None,
    )
    fy_labels = [
        str(v).strip()
        for v in df43.iloc[3, 4:].tolist()
        if str(v).strip() not in ("", "nan")
    ]
    tme_vals = df43.iloc[26, 4 : 4 + len(fy_labels)].tolist()
    tme = pd.Series(dict(zip(fy_labels, tme_vals)), dtype=float)
    return tme


def _fy_from_calendar(cal: pd.Series) -> pd.Series:
    """
    Convert a calendar-year series to fiscal years by averaging adjacent years.
    Fiscal year YYYY-YY ≈ 0.25*YYYY + 0.75*(YYYY+1)  [Apr-Mar convention].
    """
    fy = {}
    for cy in cal.index:
        if cy + 1 in cal.index:
            fy_label = f"{cy}-{str(cy + 1)[2:]}"
            fy[fy_label] = 0.25 * cal[cy] + 0.75 * cal[cy + 1]
    return pd.Series(fy)


def _fy_macro_from_1_2() -> pd.DataFrame:
    """
    Aggregate OBR table 1.2 quarterly nominal GDP components to fiscal years.
    Returns columns: consumption, investment, government (all £bn).
    """
    df = pd.read_excel(
        OBR_DATA / "obr_efo_november_2025_economy.xlsx",
        sheet_name="1.2",
        header=None,
    )
    # col 2=private cons, col 3=gov cons, col 4=fixed inv
    by_fy: dict[str, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for _, row in df.iterrows():
        p = str(row[1]).strip()
        if len(p) == 6 and p[4] == "Q":
            cy, q = int(p[:4]), int(p[5])
            fy = f"{cy - 1}-{str(cy)[2:]}" if q == 1 else f"{cy}-{str(cy + 1)[2:]}"
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
    """Total HMRC receipts £bn by fiscal year. HMRC outturn back to 2000-01, OBR forecast beyond."""
    return pd.Series(
        {
            # HMRC outturn (National Statistics: HMRC Tax & NIC Receipts bulletin)
            "2000-01": 334,
            "2001-02": 348,
            "2002-03": 346,
            "2003-04": 361,
            "2004-05": 391,
            "2005-06": 423,
            "2006-07": 454,
            "2007-08": 476,
            "2008-09": 516,
            "2009-10": 469,
            "2010-11": 496,
            "2011-12": 530,
            "2012-13": 552,
            "2013-14": 568,
            "2014-15": 600,
            "2015-16": 627,
            "2016-17": 656,
            "2017-18": 692,
            "2018-19": 729,
            "2019-20": 742,
            "2020-21": 669,
            "2021-22": 718,
            "2022-23": 786,
            "2023-24": 827,
            "2024-25": 893,
            # OBR Nov 2025 EFO forecast
            "2025-26": 951,
            "2026-27": 1001,
            "2027-28": 1037,
            "2028-29": 1082,
            "2029-30": 1124,
            "2030-31": 1163,
        }
    )


def _fy_debt(gdp: pd.Series) -> pd.Series:
    """PSND ex BoE as % GDP → £bn. ONS HF6X mapped to fiscal years, OBR forecast beyond."""
    topic = "economy/governmentpublicsectorandtaxes/publicsectorfinance"
    url = (
        f"https://www.ons.gov.uk/generator?format=csv&uri=/{topic}/timeseries/hf6x/pusf"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), header=None, names=["period", "value"])
        df["v"] = pd.to_numeric(df["value"], errors="coerce")
        ann = df.dropna(subset=["v"])
        ann = ann[ann["period"].str.strip().str.match(r"^\d{4}$")].copy()
        ann["cy"] = ann["period"].str.strip().astype(int)
        cal_pct = ann.set_index("cy")["v"].sort_index()
        # Convert calendar year pct to fiscal years (approx Apr-Mar average)
        fy_pct = _fy_from_calendar(cal_pct)
    except Exception as exc:
        print(f"  HF6X fetch failed ({exc}), using fallback")
        fy_pct = pd.Series(
            {
                "2008-09": 55,
                "2009-10": 67,
                "2010-11": 78,
                "2011-12": 82,
                "2012-13": 84,
                "2013-14": 86,
                "2014-15": 88,
                "2015-16": 87,
                "2016-17": 87,
                "2017-18": 87,
                "2018-19": 86,
                "2019-20": 85,
                "2020-21": 103,
                "2021-22": 97,
                "2022-23": 97,
                "2023-24": 97,
            }
        )
    # OBR Nov 2025 EFO forecast % GDP (PSND ex BoE)
    obr_pct = {
        "2024-25": 95.5,
        "2025-26": 96.1,
        "2026-27": 96.5,
        "2027-28": 96.3,
        "2028-29": 95.8,
        "2029-30": 95.1,
        "2030-31": 94.5,
    }
    for fy, pct in obr_pct.items():
        if fy not in fy_pct.index:
            fy_pct[fy] = pct
    fy_pct = fy_pct.sort_index()
    common = fy_pct.index.intersection(gdp.index)
    return (fy_pct.loc[common] / 100 * gdp.loc[common]).sort_index()


_ONS_TIMEOUT = 30
HIST_START_FY = "2000-01"  # how far back to show in charts


def _fetch_ons_cal(cdid: str, dataset: str, topic: str) -> pd.Series:
    """Fetch ONS annual calendar-year series (£m) and return as Series indexed by year."""
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
    """
    Fetch ONS calendar-year national accounts and convert to fiscal years.
    Used to backfill pre-2008-09 history where OBR tables don't reach.
    Returns DataFrame with columns: gdp, consumption, investment (all £bn).
    Consumption = GDP - investment - gov_consumption (residual approximation).
    """
    gdp_topic = "economy/grossdomesticproductgdp"
    import time as _time

    try:
        gdp_m = _fetch_ons_cal("YBHA", "ukea", gdp_topic)  # nominal GDP £m
        _time.sleep(1)
        inv_m = _fetch_ons_cal(
            "NPQS", "ukea", gdp_topic
        )  # gross fixed capital formation £m
        _time.sleep(1)
        gov_m = _fetch_ons_cal("NMRP", "ukea", gdp_topic)  # gov final consumption £m
    except Exception as exc:
        print(f"  ONS backfill fetch failed ({exc}), skipping pre-2008 history")
        return pd.DataFrame(columns=["gdp", "consumption", "investment"])

    gdp_fy = _fy_from_calendar(gdp_m / 1000)
    inv_fy = _fy_from_calendar(inv_m / 1000)
    gov_fy = _fy_from_calendar(gov_m / 1000)
    # Consumption as residual (excludes net exports — close enough for chart context)
    common = gdp_fy.index.intersection(inv_fy.index).intersection(gov_fy.index)
    cons_fy = gdp_fy.loc[common] - inv_fy.loc[common] - gov_fy.loc[common]

    bl = pd.DataFrame({"gdp": gdp_fy, "consumption": cons_fy, "investment": inv_fy})
    return bl


def fetch_obr_baseline() -> pd.DataFrame:
    print("  Fiscal-year nominal GDP (OBR table 1.4)...")
    gdp = _fy_gdp()

    print("  Macro components (OBR table 1.2 aggregated to fiscal years)...")
    macro = _fy_macro_from_1_2()

    print("  TME (OBR table 4.3)...")
    tme = _fy_expenditure_components()

    print("  Tax revenue (HMRC outturn + OBR forecast)...")
    tax = _fy_tax_revenue()

    print("  Debt (ONS HF6X + OBR forecast)...")
    debt = _fy_debt(gdp)

    bl = pd.DataFrame({"gdp": gdp})
    bl["consumption"] = macro["consumption"]
    bl["investment"] = macro["investment"]
    bl["tme"] = tme
    bl["tax_revenue"] = tax
    bl["debt"] = debt

    # Backfill pre-2008-09 from ONS calendar-year series
    print("  ONS backfill for pre-2008-09 history...")
    ons = _ons_backfill()
    if not ons.empty:
        # Only use ONS rows that are not already covered by OBR
        ons_pre = ons.loc[
            [fy for fy in ons.index if fy < bl.index.min()],
            [c for c in ons.columns if c in bl.columns],
        ]
        if not ons_pre.empty:
            bl = pd.concat([ons_pre, bl]).sort_index()

    return bl.loc[HIST_START_FY:OBR_FORECAST_END]


# ── TPI % changes ─────────────────────────────────────────────────────────────


def load_tpi_pct_changes() -> pd.DataFrame:
    """
    Load TPI results (model years = fiscal years, e.g. 2026 = 2026-27)
    and compute % change of reform vs baseline.
    """
    xl = pd.ExcelFile("scripts/tpi_results.xlsx")
    base = xl.parse("Baseline levels", header=2)
    ref = xl.parse("Reform levels", header=2)

    cols = {
        "GDP (£bn)": "gdp",
        "Consumption (£bn)": "consumption",
        "Investment (£bn)": "investment",
        "Government (£bn)": "tme",  # model G maps structurally to TME % change
        "Tax revenue (£bn)": "tax_revenue",
        "Debt (£bn)": "debt",
    }

    # Align baseline and reform by fiscal year (baseline may have
    # historical rows prepended that reform doesn't have)
    fy_col = "Fiscal year" if "Fiscal year" in base.columns else "Year"
    base = base.set_index(fy_col)
    ref = ref.set_index(fy_col)
    common = base.index.intersection(ref.index)
    base = base.loc[common]
    ref = ref.loc[common]

    pct = {}
    for tc, key in cols.items():
        s = (ref[tc] - base[tc]) / base[tc] * 100
        pct[key] = s

    return pd.DataFrame(pct)


# ── Chart ─────────────────────────────────────────────────────────────────────


def fig_main(baseline: pd.DataFrame, pct: pd.DataFrame) -> go.Figure:
    variables = [
        ("Consumption (% GDP)", "consumption", ORANGE),
        ("Investment (% GDP)", "investment", GREEN),
        ("Gov. Consumption (% GDP)", "tme", PURPLE),
        ("Tax revenue (% GDP)", "tax_revenue", RED),
        ("Debt (% GDP)", "debt", GREY),
        ("GDP (£bn)", "gdp", BLUE),
    ]

    # Use integer start-year as x (numeric axis — no categorical axis bugs)
    baseline = baseline.copy()
    baseline.index = [_fy_to_int(fy) for fy in baseline.index]
    pct = pct.copy()
    pct.index = [_fy_to_int(fy) for fy in pct.index]

    gdp = baseline["gdp"].dropna()

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[v[0] for v in variables],
        vertical_spacing=0.16,
        horizontal_spacing=0.08,
    )
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    shown_base = shown_ref = False

    for (label, key, color), (row, col) in zip(variables, positions):
        s = baseline[key].dropna()
        as_pct_gdp = key != "gdp"
        if as_pct_gdp:
            common = s.index.intersection(gdp.index)
            s = s.loc[common] / gdp.loc[common] * 100

        fig.add_trace(
            go.Scatter(
                x=s.index.tolist(),
                y=s.tolist(),
                mode="lines",
                line=dict(color=color, width=2.5),
                name="OBR outturn / forecast" if not shown_base else None,
                showlegend=not shown_base,
                legendgroup="base",
                hovertemplate=(
                    "%{x}: %{y:.1f}%<extra>Baseline</extra>"
                    if as_pct_gdp
                    else "%{x}: £%{y:,.0f}bn<extra>Baseline</extra>"
                ),
            ),
            row=row,
            col=col,
        )
        shown_base = True

        # Reform line — dashed, from reform FY onwards
        if key in pct.columns:
            reform_start = _fy_to_int(REFORM_FY)
            reform_xs = [x for x in s.index if x in pct.index and x >= reform_start]
            if reform_xs:
                ref_s = s.loc[reform_xs] * (1 + pct.loc[reform_xs, key] / 100)
                # Anchor at year before reform so line connects
                anchor = reform_start - 1
                if anchor in s.index:
                    ref_s = pd.concat([s[[anchor]], ref_s]).sort_index()

                fig.add_trace(
                    go.Scatter(
                        x=ref_s.index.tolist(),
                        y=ref_s.tolist(),
                        mode="lines",
                        line=dict(color=color, width=2.5, dash="dash"),
                        name="Reform +1pp basic rate" if not shown_ref else None,
                        showlegend=not shown_ref,
                        legendgroup="ref",
                        hovertemplate=(
                            "%{x}: %{y:.1f}%<extra>Reform</extra>"
                            if as_pct_gdp
                            else "%{x}: £%{y:,.0f}bn<extra>Reform</extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
                shown_ref = True

        # Vertical reference lines (numeric x — works cleanly)
        fig.add_vline(
            x=_fy_to_int(OUTTURN_LAST_FY),
            line_width=1,
            line_dash="dot",
            line_color=GREY,
            row=row,
            col=col,
        )
        fig.add_vline(
            x=_fy_to_int(REFORM_FY),
            line_width=1.2,
            line_dash="dash",
            line_color=GOLD,
            row=row,
            col=col,
        )

    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ddd",
            borderwidth=1,
            font=dict(size=11),
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.06,
        ),
        title=dict(
            text=(
                "<b>OG-UK 8-Sector Model: Basic Rate +1pp from 2027-28</b><br>"
                "<sup>Solid = OBR outturn / Nov 2025 EFO forecast · Dashed = reform (OG-UK TPI) · "
                "Grey dotted = outturn/forecast boundary</sup>"
            ),
            x=0.5,
            font=dict(size=15),
        ),
        height=780,
    )
    # Format x ticks — show every 5th year to avoid overlap
    all_years = sorted(baseline.index.tolist())
    tick_years = [y for y in all_years if y % 5 == 0]
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#ececec",
        zeroline=False,
        tickangle=0,
        tickfont=dict(size=10),
        tickmode="array",
        tickvals=tick_years,
        ticktext=[f"{y}-{str(y + 1)[2:]}" for y in tick_years],
    )
    for i, (_, key, _) in enumerate(variables):
        r, c = positions[i]
        if key != "gdp":
            fig.update_yaxes(
                showgrid=True,
                gridcolor="#ececec",
                ticksuffix="%",
                tickformat=".1f",
                row=r,
                col=c,
            )
        else:
            fig.update_yaxes(
                showgrid=True,
                gridcolor="#ececec",
                tickprefix="£",
                ticksuffix="bn",
                tickformat=",.0f",
                row=r,
                col=c,
            )
    return fig


# ── Sector charts ─────────────────────────────────────────────────────────────

SECTOR_COLORS = [
    "#e6194b",  # Energy — red
    "#3cb44b",  # Manufacturing — green
    "#ffe119",  # Construction — yellow
    "#4363d8",  # Trade & Transport — blue
    "#f58231",  # Info & Finance — orange
    "#911eb4",  # Real Estate — purple
    "#42d4f4",  # Business Services — cyan
    "#f032e6",  # Public & Other — magenta
]

# ── Historical GVA by 8-sector (£bn, current basic prices) ──────────────────
# Source: ONS GDP output approach — low level aggregates (Blue Book 2024)
# Aggregated from SIC 2007 sections to our 8-sector mapping.
# https://www.ons.gov.uk/economy/grossdomesticproductgdp/datasets/ukgdpolowlevelaggregates
_HIST_GVA_BY_SECTOR = {
    # fmt: off
    # Source: ONS GDP output approach — low level aggregates (Blue Book 2024)
    # Aggregated from SIC 2007 sections. £bn, current basic prices.
    #         Energy  Const  Trade  InfoF  RealE  BusSv  PubOt  Manuf
    "2000-01": [21.5, 55.6, 194.1, 119.2, 126.7, 108.0, 272.7, 128.7],
    "2001-02": [21.2, 58.4, 201.9, 127.5, 138.3, 114.0, 286.0, 126.5],
    "2002-03": [20.6, 63.0, 210.2, 131.5, 148.9, 120.3, 303.2, 124.0],
    "2003-04": [20.9, 67.3, 219.8, 137.1, 156.0, 128.7, 321.8, 124.1],
    "2004-05": [22.3, 70.8, 228.5, 143.4, 161.8, 136.7, 337.5, 126.0],
    "2005-06": [25.6, 75.4, 237.9, 150.6, 169.0, 145.3, 353.1, 127.6],
    "2006-07": [27.0, 81.0, 249.6, 162.0, 177.6, 157.0, 364.6, 130.4],
    "2007-08": [29.8, 87.0, 258.7, 176.9, 182.7, 167.5, 376.4, 133.0],
    "2008-09": [34.5, 81.2, 251.3, 179.8, 179.5, 163.3, 391.3, 127.1],
    "2009-10": [28.0, 72.7, 240.2, 170.3, 179.0, 153.6, 397.5, 118.5],
    "2010-11": [31.2, 77.0, 254.1, 178.3, 187.4, 161.9, 404.9, 127.1],
    "2011-12": [35.0, 80.5, 262.5, 185.6, 192.5, 167.7, 411.8, 130.1],
    "2012-13": [33.5, 82.1, 270.5, 192.1, 200.8, 173.4, 419.1, 131.4],
    "2013-14": [33.0, 87.8, 281.5, 200.0, 210.3, 182.8, 428.3, 136.2],
    "2014-15": [28.2, 95.5, 296.2, 214.0, 223.0, 193.6, 441.6, 140.7],
    "2015-16": [22.1, 99.5, 310.0, 228.2, 236.5, 197.8, 451.4, 144.3],
    "2016-17": [41.8, 111.4, 349.5, 259.4, 257.4, 214.2, 474.2, 174.8],
    "2017-18": [44.0, 118.3, 363.4, 270.5, 262.7, 224.1, 489.2, 178.6],
    "2018-19": [48.3, 125.5, 377.3, 282.0, 268.6, 233.0, 501.6, 181.5],
    "2019-20": [40.7, 126.4, 370.1, 290.3, 271.2, 237.3, 510.5, 181.3],
    "2020-21": [30.1, 113.1, 303.5, 289.3, 273.7, 216.5, 514.5, 168.1],
    "2021-22": [46.9, 127.8, 376.2, 308.8, 284.3, 243.4, 524.1, 183.6],
    "2022-23": [54.8, 134.5, 413.0, 332.0, 299.0, 265.9, 537.6, 195.4],
    "2023-24": [48.2, 140.1, 421.3, 344.4, 310.2, 274.6, 555.1, 195.0],
    # fmt: on
}

# Historical capital stock by 8-sector (£bn, net current replacement cost)
# Source: ONS CAPSTK
_HIST_CAPITAL_BY_SECTOR = {
    # Source: ONS CAPSTK — net capital stock, current replacement cost, £bn
    #         Energy  Const  Trade  InfoF  RealE   BusSv  PubOt  Manuf
    "2000-01": [105, 14, 128, 72, 610, 38, 270, 105],
    "2002-03": [100, 15, 133, 80, 660, 40, 280, 100],
    "2004-05": [100, 16, 140, 85, 730, 43, 295, 97],
    "2006-07": [108, 18, 153, 95, 840, 50, 315, 100],
    "2008-09": [130, 20, 170, 108, 940, 58, 340, 110],
    "2010-11": [135, 20, 168, 112, 970, 59, 350, 112],
    "2012-13": [145, 21, 178, 120, 1040, 64, 365, 118],
    "2014-15": [150, 22, 185, 128, 1100, 69, 380, 122],
    "2016-17": [155, 24, 195, 135, 1170, 74, 400, 128],
    "2018-19": [170, 26, 208, 146, 1250, 82, 425, 140],
    "2019-20": [175, 27, 210, 148, 1280, 83, 430, 143],
    "2020-21": [172, 27, 207, 150, 1310, 82, 440, 142],
    "2021-22": [181, 29, 223, 161, 1400, 89, 465, 152],
    "2022-23": [197, 32, 248, 178, 1510, 98, 495, 168],
}

# Historical workforce jobs by 8-sector (thousands)
# Source: ONS JOBS02, seasonally adjusted, Q4
_HIST_LABOUR_BY_SECTOR = {
    # Source: ONS JOBS02 — workforce jobs by industry, thousands, seasonally adjusted
    #         Energy  Const  Trade  InfoF  RealE  BusSv  PubOt  Manuf
    "2000-01": [125, 1820, 7400, 1650, 380, 3900, 7800, 3800],
    "2002-03": [115, 1870, 7500, 1700, 400, 4100, 8000, 3510],
    "2004-05": [110, 1950, 7600, 1750, 420, 4350, 8150, 3250],
    "2006-07": [115, 2100, 7750, 1850, 450, 4650, 8300, 3050],
    "2008-09": [120, 2150, 7700, 1900, 470, 4800, 8450, 2910],
    "2010-11": [115, 2050, 7500, 1950, 470, 4700, 8500, 2650],
    "2012-13": [125, 2050, 7700, 2050, 490, 4950, 8550, 2600],
    "2014-15": [135, 2100, 7900, 2150, 510, 5150, 8600, 2600],
    "2016-17": [155, 2190, 8150, 2300, 540, 5380, 8640, 2620],
    "2017-18": [157, 2250, 8260, 2350, 550, 5480, 8700, 2630],
    "2018-19": [160, 2280, 8340, 2410, 560, 5560, 8760, 2620],
    "2019-20": [161, 2300, 8320, 2450, 570, 5600, 8800, 2610],
    "2020-21": [152, 2170, 7750, 2420, 530, 5240, 8700, 2490],
    "2021-22": [162, 2250, 8180, 2510, 570, 5580, 8850, 2530],
    "2022-23": [178, 2310, 8470, 2580, 590, 5760, 8960, 2590],
    "2023-24": [175, 2330, 8500, 2620, 595, 5820, 9020, 2570],
}

from oguk.industry_params import SECTOR_NAMES  # noqa: E402


def _load_sector_model_levels() -> dict[str, pd.DataFrame]:
    """Load per-sector baseline and reform levels from xlsx (model units)."""
    xl = pd.ExcelFile("scripts/tpi_results.xlsx")
    result = {}
    for label in ["Output", "Capital", "Labour"]:
        base_sheet = f"Sector {label.lower()} (base)"
        ref_sheet = f"Sector {label.lower()} (reform)"
        if base_sheet not in xl.sheet_names or ref_sheet not in xl.sheet_names:
            continue
        base = xl.parse(base_sheet, header=2)
        ref = xl.parse(ref_sheet, header=2)
        fy_col = "Fiscal year" if "Fiscal year" in base.columns else base.columns[0]
        base = base.set_index(fy_col)
        ref = ref.set_index(fy_col)
        result[label] = {"base": base, "reform": ref}
    return result


def _extrapolate_anchor(hist_dict: dict, model_start_fy: str) -> list[float]:
    """Extrapolate historical outturn to the model start year.

    Uses the average annual growth rate from the last 3 years of outturn
    to project forward to the model start fiscal year.
    """
    sorted_fys = sorted(hist_dict.keys())
    last_fy = sorted_fys[-1]
    last_vals = hist_dict[last_fy]
    n_sectors = len(last_vals)

    # Compute avg growth rate from last 3 years
    if len(sorted_fys) >= 4:
        prev_fy = sorted_fys[-4]
        prev_vals = hist_dict[prev_fy]
        growth_rates = []
        for m in range(n_sectors):
            if prev_vals[m] > 0:
                growth_rates.append((last_vals[m] / prev_vals[m]) ** (1 / 3) - 1)
            else:
                growth_rates.append(0.03)  # fallback 3% nominal growth
    else:
        growth_rates = [0.03] * n_sectors

    gap_years = _fy_to_int(model_start_fy) - _fy_to_int(last_fy)
    return [last_vals[m] * (1 + growth_rates[m]) ** gap_years for m in range(n_sectors)]


def _bridge_gap(
    hist_dict: dict, model_start_fy: str, anchor_values: list[float]
) -> dict:
    """Fill the gap between last outturn and model start with linear interpolation."""
    sorted_fys = sorted(hist_dict.keys())
    last_fy = sorted_fys[-1]
    last_year = _fy_to_int(last_fy)
    start_year = _fy_to_int(model_start_fy)
    last_vals = hist_dict[last_fy]
    n_sectors = len(last_vals)

    bridge = {}
    gap = start_year - last_year
    for step in range(1, gap):
        yr = last_year + step
        fy = f"{yr}-{str(yr + 1)[2:]}"
        frac = step / gap
        bridge[fy] = [
            last_vals[m] + frac * (anchor_values[m] - last_vals[m])
            for m in range(n_sectors)
        ]
    return bridge


def _scale_to_real(
    model_df: pd.DataFrame,
    anchor_values: list[float],
    sector_names: list[str],
) -> pd.DataFrame:
    """Scale model-unit sector levels to real-world units.

    Uses anchor_values (real-world value at model t=0) to compute
    a per-sector scale factor: scale_m = anchor_m / model_m[0].
    """
    scaled = pd.DataFrame(index=model_df.index)
    for m, name in enumerate(sector_names):
        if name in model_df.columns:
            model_t0 = model_df[name].iloc[0]
            if abs(model_t0) > 1e-12:
                scale = anchor_values[m] / model_t0
                scaled[name] = model_df[name] * scale
            else:
                scaled[name] = 0.0
    return scaled


SECTOR_CHART_START = 2000
SECTOR_CHART_END = 2030


def fig_sector_real(sector_pct: dict, gdp: pd.Series | None = None) -> go.Figure | None:
    """Build sector charts: ONS outturn projected forward, reform = baseline × (1 + %chg).

    Same approach as macro charts: one smooth baseline, reform branches off using
    model % changes. No model absolute levels — avoids stitching artefacts.
    """
    hist_data = {
        "Output": (_HIST_GVA_BY_SECTOR, "£bn"),
        "Capital": (_HIST_CAPITAL_BY_SECTOR, "£bn"),
        "Labour": (_HIST_LABOUR_BY_SECTOR, "thousands"),
    }

    labels = [lb for lb in ["Output", "Capital", "Labour"] if lb in sector_pct]
    if not labels:
        return None

    fig = make_subplots(
        rows=len(SECTOR_NAMES),
        cols=len(labels),
        subplot_titles=[
            f"{name} — {label}" for name in SECTOR_NAMES for label in labels
        ],
        vertical_spacing=0.03,
        horizontal_spacing=0.06,
    )

    for col_idx, label in enumerate(labels, 1):
        hist_dict, unit = hist_data[label]
        pct_df = sector_pct[label]
        # Trim % changes to chart range
        pct_df = pct_df[[_fy_to_int(fy) <= SECTOR_CHART_END for fy in pct_df.index]]

        # Build baseline: outturn + GDP-growth projection for each sector
        sorted_fys = sorted(hist_dict.keys())
        last_outturn_fy = sorted_fys[-1]
        last_outturn_vals = hist_dict[last_outturn_fy]
        last_yr = _fy_to_int(last_outturn_fy)

        # Project forward using OBR GDP growth (year by year)
        proj_fys = [fy for fy in pct_df.index if _fy_to_int(fy) > last_yr]
        projected = {}  # fy -> [vals per sector]
        for fy in proj_fys:
            yr = _fy_to_int(fy)
            if gdp is not None and last_yr in gdp.index and yr in gdp.index:
                ratio = gdp.loc[yr] / gdp.loc[last_yr]
            else:
                ratio = 1.035 ** (yr - last_yr)
            projected[fy] = [v * ratio for v in last_outturn_vals]

        for row_idx, name in enumerate(SECTOR_NAMES, 1):
            color = SECTOR_COLORS[row_idx - 1]
            m = SECTOR_NAMES.index(name)
            col_name = f"{name} (%)"

            # Baseline: outturn + projection (one continuous solid line)
            base_x, base_y = [], []
            for fy, vals in sorted(hist_dict.items()):
                base_x.append(_fy_to_int(fy))
                base_y.append(vals[m])
            for fy in sorted(projected.keys(), key=_fy_to_int):
                base_x.append(_fy_to_int(fy))
                base_y.append(projected[fy][m])

            fig.add_trace(
                go.Scatter(
                    x=base_x,
                    y=base_y,
                    mode="lines",
                    line=dict(color=color, width=2.5),
                    name="ONS outturn / forecast"
                    if row_idx == 1 and col_idx == 1
                    else None,
                    showlegend=(row_idx == 1 and col_idx == 1),
                    legendgroup="baseline",
                    hovertemplate=f"%{{x}}: %{{y:,.1f}} {unit}<extra>Baseline</extra>",
                ),
                row=row_idx,
                col=col_idx,
            )

            # Reform: baseline × (1 + % change), from reform start
            reform_start = _fy_to_int(REFORM_FY)
            ref_x, ref_y = [], []

            # Anchor at year before reform
            anchor_yr = reform_start - 1
            if anchor_yr in base_x:
                idx = base_x.index(anchor_yr)
                ref_x.append(anchor_yr)
                ref_y.append(base_y[idx])

            for fy in sorted(pct_df.index, key=_fy_to_int):
                yr = _fy_to_int(fy)
                if yr < reform_start:
                    continue
                if col_name not in pct_df.columns:
                    continue
                pct_val = pct_df.loc[fy, col_name]
                # Get baseline value for this year
                if yr in base_x:
                    idx = base_x.index(yr)
                    ref_x.append(yr)
                    ref_y.append(base_y[idx] * (1 + pct_val / 100))

            if ref_x:
                fig.add_trace(
                    go.Scatter(
                        x=ref_x,
                        y=ref_y,
                        mode="lines",
                        line=dict(color=color, width=2.5, dash="dash"),
                        name="Reform +1pp basic rate"
                        if row_idx == 1 and col_idx == 1
                        else None,
                        showlegend=(row_idx == 1 and col_idx == 1),
                        legendgroup="reform",
                        hovertemplate=f"%{{x}}: %{{y:,.1f}} {unit}<extra>Reform</extra>",
                    ),
                    row=row_idx,
                    col=col_idx,
                )

            # Vertical lines
            fig.add_vline(
                x=_fy_to_int(OUTTURN_LAST_FY),
                line_width=1,
                line_dash="dot",
                line_color=GREY,
                row=row_idx,
                col=col_idx,
            )
            fig.add_vline(
                x=reform_start,
                line_width=1.2,
                line_dash="dash",
                line_color=GOLD,
                row=row_idx,
                col=col_idx,
            )

    # Collect all years for tick marks
    all_years = set()
    for hist_dict, _ in hist_data.values():
        all_years.update(_fy_to_int(fy) for fy in hist_dict)
    for label in labels:
        all_years.update(
            y
            for y in (_fy_to_int(fy) for fy in sector_pct[label].index)
            if y <= SECTOR_CHART_END
        )
    tick_years = sorted(y for y in all_years if y % 5 == 0)

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#ececec",
        tickangle=0,
        tickfont=dict(size=9),
        tickmode="array",
        tickvals=tick_years,
        ticktext=[f"{y}-{str(y + 1)[2:]}" for y in tick_years],
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#ececec",
        tickfont=dict(size=9),
    )
    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=11),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ddd",
            borderwidth=1,
            font=dict(size=11),
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.02,
        ),
        title=dict(
            text=(
                "<b>OG-UK 8-Sector Model: Per-Industry Levels with Historical Outturn</b><br>"
                "<sup>Solid = ONS outturn · Dotted = OG-UK baseline · "
                "Dashed = reform (+1pp basic rate) · Gold dashed = reform start</sup>"
            ),
            x=0.5,
            font=dict(size=15),
        ),
        height=300 * len(SECTOR_NAMES),
    )
    return fig


def load_sector_pct_changes() -> dict[str, pd.DataFrame]:
    """Load per-sector % change sheets from tpi_results.xlsx.

    Returns dict mapping variable label (e.g. "Output") to a DataFrame
    indexed by fiscal year with one column per sector.
    """
    xl = pd.ExcelFile("scripts/tpi_results.xlsx")
    result = {}
    for label in ["Output", "Capital", "Labour", "Prices"]:
        sheet = f"Sector {label.lower()} (%chg)"
        if sheet not in xl.sheet_names:
            continue
        df = xl.parse(sheet, header=2)
        fy_col = "Fiscal year" if "Fiscal year" in df.columns else df.columns[0]
        df = df.set_index(fy_col)
        result[label] = df
    return result


def fig_sector_pct(sector_pct: dict[str, pd.DataFrame]) -> go.Figure | None:
    """Build a 1×3 subplot of per-sector % changes (Output, Capital, Labour)."""
    labels = [lb for lb in ["Output", "Capital", "Labour"] if lb in sector_pct]
    if not labels:
        return None

    fig = make_subplots(
        rows=1,
        cols=len(labels),
        subplot_titles=[f"Sector {lb} (% change)" for lb in labels],
        horizontal_spacing=0.08,
    )

    for col_idx, label in enumerate(labels, 1):
        df = sector_pct[label]
        # Trim to chart range
        df = df[[_fy_to_int(fy) <= SECTOR_CHART_END for fy in df.index]]
        x = [_fy_to_int(fy) for fy in df.index]
        for i, col_name in enumerate(df.columns):
            sector_name = col_name.replace(" (%)", "")
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df[col_name].tolist(),
                    mode="lines",
                    line=dict(color=SECTOR_COLORS[i % len(SECTOR_COLORS)], width=2),
                    name=sector_name if col_idx == 1 else None,
                    showlegend=(col_idx == 1),
                    legendgroup=sector_name,
                    hovertemplate=f"%{{x}}: %{{y:.3f}}%<extra>{sector_name}</extra>",
                ),
                row=1,
                col=col_idx,
            )
        fig.add_vline(
            x=_fy_to_int(REFORM_FY),
            line_width=1.2,
            line_dash="dash",
            line_color=GOLD,
            row=1,
            col=col_idx,
        )

    all_years = sorted(
        set(
            y
            for df in sector_pct.values()
            for y in (_fy_to_int(fy) for fy in df.index)
            if y <= SECTOR_CHART_END
        )
    )
    tick_years = [y for y in all_years if y % 5 == 0]

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#ececec",
        tickangle=0,
        tickfont=dict(size=10),
        tickmode="array",
        tickvals=tick_years,
        ticktext=[f"{y}-{str(y + 1)[2:]}" for y in tick_years],
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#ececec",
        ticksuffix="%",
        tickformat=".3f",
    )
    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ddd",
            borderwidth=1,
            font=dict(size=10),
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.12,
        ),
        title=dict(
            text=(
                "<b>OG-UK 8-Sector Model: Per-Industry Impacts of Basic Rate +1pp</b><br>"
                "<sup>% change from baseline by sector · Gold dashed = reform start</sup>"
            ),
            x=0.5,
            font=dict(size=15),
        ),
        height=500,
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    print("Loading TPI % changes...")
    pct = load_tpi_pct_changes()

    print("Fetching OBR baseline (fiscal years)...")
    baseline = fetch_obr_baseline()
    print(f"  Baseline: {baseline.index.min()} – {baseline.index.max()}")

    print("Building macro chart...")
    fig = fig_main(baseline, pct)

    # Sector charts
    print("Loading sector data...")
    sector_pct_fig = None
    sector_real_fig = None
    try:
        sector_pct = load_sector_pct_changes()
        sector_pct_fig = fig_sector_pct(sector_pct) if sector_pct else None
    except Exception as exc:
        print(f"  Sector % change charts skipped: {exc}")

    try:
        # Pass OBR GDP (indexed by FY as integers) for growth-rate projection
        gdp_fy = baseline["gdp"].copy()
        gdp_fy.index = [_fy_to_int(fy) for fy in gdp_fy.index]
        sector_real_fig = (
            fig_sector_real(sector_pct, gdp=gdp_fy) if sector_pct else None
        )
    except Exception as exc:
        print(f"  Sector real-world charts skipped: {exc}")
        import traceback  # noqa: E402

        traceback.print_exc()

    # Combine into single HTML
    parts = [fig.to_html(full_html=False, include_plotlyjs="cdn")]
    if sector_pct_fig is not None:
        parts.append(sector_pct_fig.to_html(full_html=False, include_plotlyjs=False))
    if sector_real_fig is not None:
        parts.append(sector_real_fig.to_html(full_html=False, include_plotlyjs=False))

    print("Building combined HTML...")
    combined = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>OG-UK TPI Charts</title></head>
<body style="font-family: Inter, sans-serif; max-width: 1400px; margin: auto;">
{"<hr style='margin: 30px 0;'>".join(parts)}
</body></html>"""
    with open("scripts/tpi_charts.html", "w") as f:
        f.write(combined)

    print("Saved: scripts/tpi_charts.html")


if __name__ == "__main__":
    main()
