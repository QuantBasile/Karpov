from __future__ import annotations

from typing import Callable, Optional
import os, sys
import pandas as pd

from controller.state import AppState
from controller.warmup import warmup_after_load


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fake_mm_logs import generate_mm_logs
from cleanData import clean_mm_logs

def load_data(state: AppState, *, progress: Optional[Callable[[str], None]] = None) -> None:
    if progress: progress("Generating fake logs...")
    df_raw = generate_mm_logs(
        state.filters.start_date,
        state.filters.end_date,
        trades_per_day=30_000,
        seed=42,
    )

    if progress: progress("Cleaning...")
    df_clean = clean_mm_logs(df_raw)

    # Notional for later usage
    if "Quantity" in df_clean.columns and "Price" in df_clean.columns:
        qty = df_clean["Quantity"].astype("Float64")
        px = df_clean["Price"].astype("Float64")
        df_clean["Notional"] = (qty * px).astype("Float64")

    state.df_raw = df_raw
    state.df_clean = df_clean

    # 🔥 NEW: warm-up heavy tabs
    if progress: progress("Warming up views...")
    warmup_after_load(state.app)   # 👈 explicado abajo

    if progress: progress("Ready.")


def summarize(state: AppState) -> dict:
    n = 0 if state.df_clean is None else int(len(state.df_clean))
    return {"n_trades": n}
