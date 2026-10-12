"""
server/blocks.py - the "view blocks" the API returns and the Next.js front end renders.

The Python side decides WHAT to show (tables, Plotly figures, text) from real simulation output; the browser
only knows how to draw a handful of generic block types. Every block is a plain JSON-able dict.

  heading   {text, level}                      markdown   {text}   (**bold**, `code`, *italic*, "- " bullets)
  caption   {text}                             callout    {tone: note|warn|ok|error, text}
  kpis      {items: [{label, value, sub, tone, pill}]}
  plot      {figure: <Plotly figure JSON>}     table      {columns, rows, caption?}
  columns   {cols: [{weight, blocks}]}         tabs       {tabs: [{label, blocks}]}
  switch    {label, options: [{label, blocks}]}   (client-side selector, no round trip)
  cards     {items: [{title, tag, color, body, blocks}]}
  download  {label, filename, mime, content}   math       {text}
"""

import json
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.io as pio

# Keep Plotly figure JSON small: no embedded copy of the default template (charts set their own styling).
pio.templates.default = "none"


def heading(text: str, level: int = 3) -> dict:
    return {"type": "heading", "text": text, "level": level}


def md(text: str) -> dict:
    return {"type": "markdown", "text": text}


def caption(text: str) -> dict:
    return {"type": "caption", "text": text}


def callout(tone: str, text: str) -> dict:
    assert tone in ("note", "warn", "ok", "error")
    return {"type": "callout", "tone": tone, "text": text}


def kpi(label: str, value: str, sub: str = "", tone: str = "", pill: Optional[str] = None) -> dict:
    d = {"label": label, "value": value, "sub": sub, "tone": tone}
    if pill:
        d["pill"] = pill          # hex colour: render the value as a coloured pill
    return d


def kpis(*items: dict) -> dict:
    return {"type": "kpis", "items": list(items)}


def plot(fig) -> dict:
    return {"type": "plot", "figure": json.loads(pio.to_json(fig, validate=False))}


def table(data, columns: Optional[Sequence[str]] = None, index_name: Optional[str] = None,
          caption_text: Optional[str] = None, max_height: Optional[int] = None) -> dict:
    """data: a DataFrame (its index becomes the first column when index_name is given) or a list of rows."""
    if isinstance(data, pd.DataFrame):
        cols = ([index_name] if index_name is not None else []) + [str(c) for c in data.columns]
        rows = []
        for idx, r in data.iterrows():
            rows.append(([idx] if index_name is not None else []) + list(r.values))
    else:
        cols, rows = list(columns or []), [list(r) for r in data]
    out = {"type": "table", "columns": cols, "rows": rows}
    if caption_text:
        out["caption"] = caption_text
    if max_height:
        out["max_height"] = max_height
    return out


def columns(*cols: List[dict], weights: Optional[Sequence[float]] = None) -> dict:
    ws = list(weights) if weights else [1] * len(cols)
    return {"type": "columns", "cols": [{"weight": w, "blocks": list(b)} for w, b in zip(ws, cols)]}


def tabs(*items) -> dict:
    """tabs(("Label", [blocks]), ...)"""
    return {"type": "tabs", "tabs": [{"label": lbl, "blocks": list(b)} for lbl, b in items]}


def switch(label: str, options: Sequence[tuple]) -> dict:
    return {"type": "switch", "label": label, "options": [{"label": lbl, "blocks": list(b)} for lbl, b in options]}


def cards(items: List[dict]) -> dict:
    return {"type": "cards", "items": items}


def download(label: str, filename: str, mime: str, content: str) -> dict:
    return {"type": "download", "label": label, "filename": filename, "mime": mime, "content": content}


def math_text(text: str) -> dict:
    return {"type": "math", "text": text}


# --------------------------------------------------------------------------- #
def sanitize(obj: Any) -> Any:
    """Make a structure strictly JSON-safe: numpy -> python, NaN/inf -> None, tuples -> lists, keys -> str."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int,)) and not isinstance(obj, bool):
        return obj
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)
