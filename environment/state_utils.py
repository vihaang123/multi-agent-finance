"""
environment/state_utils.py

Helpers to encode / decode / describe the 108-state discretised state.

    state = regime * 27 + equity_bucket * 9 + bond_bucket * 3 + gold_bucket

Pure functions of the config; used by the dashboard so a state index like
"58" can be shown as "Bull regime, Equity High, Bonds Low, Gold Medium".
"""

from typing import Dict

import config as cfg

_B = cfg.N_ALLOC_BUCKETS

BUCKET_LABELS = [
    f"Low (<{cfg.ALLOC_BUCKET_EDGES[0]:.0%})",
    f"Medium ({cfg.ALLOC_BUCKET_EDGES[0]:.0%}-{cfg.ALLOC_BUCKET_EDGES[1]:.0%})",
    f"High (>={cfg.ALLOC_BUCKET_EDGES[1]:.0%})",
]
BUCKET_SHORT = ["Low", "Med", "High"]


def encode_state(regime: int, eq_bucket: int, bond_bucket: int, gold_bucket: int) -> int:
    for name, v, hi in (("regime", regime, len(cfg.REGIME_NAMES)), ("eq_bucket", eq_bucket, _B),
                        ("bond_bucket", bond_bucket, _B), ("gold_bucket", gold_bucket, _B)):
        if not 0 <= int(v) < hi:
            raise ValueError(f"{name}={v} out of range [0, {hi})")
    return int(regime * _B ** 3 + eq_bucket * _B ** 2 + bond_bucket * _B + gold_bucket)


def decode_state(state: int) -> Dict[str, int]:
    state = int(state)
    if not 0 <= state < cfg.N_STATES:
        raise ValueError(f"state={state} out of range [0, {cfg.N_STATES})")
    regime, rem = divmod(state, _B ** 3)
    eq_b, rem = divmod(rem, _B ** 2)
    bond_b, gold_b = divmod(rem, _B)
    return {"regime": regime, "equity_bucket": eq_b, "bond_bucket": bond_b, "gold_bucket": gold_b}


def describe_state(state: int) -> str:
    d = decode_state(state)
    return (f"{cfg.REGIME_NAMES[d['regime']]} regime | Equity {BUCKET_SHORT[d['equity_bucket']]}, "
            f"Bonds {BUCKET_SHORT[d['bond_bucket']]}, Gold {BUCKET_SHORT[d['gold_bucket']]}")
