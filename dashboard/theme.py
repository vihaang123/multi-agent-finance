"""
dashboard/theme.py - colours and number formatting (no UI-framework imports).

Agent colours are categorical slots 1,2,3,7 of a CVD-validated palette
(blue, orange, aqua, violet); assets and regimes have their own fixed colours
so a given entity is always drawn in the same colour on every page.
"""

import numpy as np

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


SYNTHETIC_TEXT = (
    "Synthetic data. Every price, return and market regime in this app comes from a simulated Markov-regime "
    "market model (see the Learn page) - it is not real market data, and results are not real investment "
    "performance, advice or predictions. No real-money trading or broker connection exists in this project."
)

REGIME_PILL = {name: REGIME_COLORS[name] for name in REGIME_COLORS}
