"""
dashboard/theme.py - colours, number formatting, CSS and small HTML helpers.

Agent colours are categorical slots 1,2,3,7 of a CVD-validated palette
(blue, orange, aqua, violet); assets and regimes have their own fixed colours
so a given entity is always drawn in the same colour on every page.
"""

import numpy as np
import streamlit as st

import config as cfg

# ---- colours -------------------------------------------------------------- #
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8985"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"
ACCENT = "#2a78d6"

AGENT_COLORS = {"Growth": "#2a78d6", "Conservative": "#eb6834", "Balanced": "#1baf7a", "RiskLiquidity": "#6250d6"}
TOTAL_COLOR = "#0b0b0b"
ASSET_COLORS = {"Equity": "#6250d6", "Bonds": "#2a78d6", "Gold": "#eda100", "Cash": "#9a9994"}
REGIME_COLORS = {"Bear": "#e34948", "Neutral": "#9a9994", "Bull": "#1baf7a", "HighVol": "#eda100"}
EXPERIMENT_COLORS = {"A": "#9a9994", "B": "#eda100", "C": "#2a78d6", "D": "#6250d6"}
GOOD, BAD = "#1baf7a", "#e34948"


def agent_color(role: str) -> str:
    return AGENT_COLORS.get(role, ACCENT)


# ---- formatting ------------------------------------------------------------ #
def inr(x: float, decimals: int = 0) -> str:
    """Indian digit grouping: 1234567.8 -> 'Rs 12,34,568'."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    sign = "-" if x < 0 else ""
    x = abs(float(x))
    whole, frac = f"{x:.{decimals}f}".split(".") if decimals else (f"{x:.0f}", "")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return f"{sign}₹{whole}" + (f".{frac}" if frac else "")


def pct(x: float, decimals: int = 2, signed: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x * 100:+.{decimals}f}%" if signed else f"{x * 100:.{decimals}f}%"


# ---- CSS ------------------------------------------------------------------- #
_CSS = """
<style>
:root { --ink:#0b0b0b; --ink2:#52514e; --ink3:#8a8985; --line:#e6e5e1; --card:#ffffff; --accent:#2a78d6; }
.block-container { padding-top: 3.4rem; padding-bottom: 3rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.syn-banner { background:#fff6e0; border:1px solid #f0d58a; border-left:5px solid #eda100; color:#4a3500;
  padding:.55rem .9rem; border-radius:8px; font-size:.88rem; margin-bottom:.9rem; }
.syn-banner b { color:#2e2100; }
.kpi { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:.8rem 1rem; height:100%; }
.kpi .lbl { color:var(--ink2); font-size:.76rem; text-transform:uppercase; letter-spacing:.06em; font-weight:600; }
.kpi .val { font-size:1.55rem; font-weight:700; color:var(--ink); line-height:1.25; font-variant-numeric: tabular-nums; }
.kpi .sub { color:var(--ink2); font-size:.8rem; }
.kpi .pos { color:#0a7a52; } .kpi .neg { color:#c0302f; }
.pill { display:inline-block; padding:.12rem .65rem; border-radius:999px; font-weight:600; font-size:.85rem; color:#fff; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:.9rem 1.05rem; }
.card h4 { margin:.1rem 0 .4rem 0; font-size:1rem; }
.small { color:var(--ink2); font-size:.84rem; }
.note { background:#eef4fd; border:1px solid #cfe0f6; border-radius:8px; padding:.6rem .85rem; font-size:.88rem; color:#1b3a63; }
.warn { background:#fdf0ef; border:1px solid #f1c4c3; border-radius:8px; padding:.6rem .85rem; font-size:.88rem; color:#6b1f1e; }
.agent-dot { display:inline-block; width:.7rem; height:.7rem; border-radius:50%; margin-right:.4rem; vertical-align:middle; }
[data-testid="stSidebar"] { background:#f6f5f2; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
div.stButton > button { border-radius:8px; font-weight:600; }
@media (max-width: 760px) { .kpi .val { font-size:1.2rem; } .block-container { padding-left:.8rem; padding-right:.8rem; } }
</style>
"""

SYNTHETIC_TEXT = (
    "<b>Synthetic data.</b> Every price, return and market regime in this app comes from a "
    "<b>simulated</b> Markov-regime market model (see the Learn page) - it is <b>not real market "
    "data</b>, and results are <b>not</b> real investment performance, advice or predictions. "
    "No real-money trading or broker connection exists in this project."
)


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def synthetic_banner():
    st.markdown(f'<div class="syn-banner">{SYNTHETIC_TEXT}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", tone: str = "") -> str:
    cls = {"pos": "pos", "neg": "neg"}.get(tone, "")
    return (f'<div class="kpi"><div class="lbl">{label}</div><div class="val {cls}">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def regime_pill(regime: int) -> str:
    name = cfg.REGIME_NAMES[regime]
    return f'<span class="pill" style="background:{REGIME_COLORS[name]}">{name}</span>'


def agent_label(role: str, aid: int = None) -> str:
    return f'<span class="agent-dot" style="background:{agent_color(role)}"></span><b>{role}</b>'
