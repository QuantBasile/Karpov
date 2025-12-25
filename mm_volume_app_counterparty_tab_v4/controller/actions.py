from __future__ import annotations

from typing import Callable, Optional
import os, sys
import pandas as pd
import numpy as np

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
    
    # Precompute filter values once (fast tab switching) - unique()
    # Precompute filter values once (fast tab switching)
    d = df_clean
    state.filter_values = {
        "days":   ["ALL"] + sorted(pd.Series(d["_Day"]).dropna().astype(str).unique().tolist()) if "_Day" in d else ["ALL"],
        "months": ["ALL"] + sorted(pd.Series(d["_Month"]).dropna().astype(str).unique().tolist()) if "_Month" in d else ["ALL"],
        "calls":  ["ALL"] + sorted(pd.Series(d["CALL_OPTION"]).dropna().astype(str).unique().tolist()) if "CALL_OPTION" in d else ["ALL"],
        "cps":    ["ALL"] + sorted(pd.Series(d["Counterparty"]).dropna().astype(str).unique().tolist()) if "Counterparty" in d else ["ALL"],
        "unds":   ["ALL"] + sorted(pd.Series(d["UND_NAME"]).dropna().astype(str).unique().tolist()) if "UND_NAME" in d else ["ALL"],
    }

    # Precompute filter values once (fast tab switching)
    d = df_clean
    state.df_raw = df_raw
    state.df_clean = df_clean

    # 🔥 NEW: warm-up heavy tabs
    if progress: progress("Warming up views...")
    warmup_after_load(state.app)   # 👈 explicado abajo
    
    if progress: progress("Precomputing CP time aggs (Notional + Flow)...")
    state.cp_time_aggs = _precompute_cp_time_aggs(df_clean)
    state.cp_list = sorted(df_clean["Counterparty"].astype(str).unique().tolist())
    if progress: progress("Precompute CP done.")
    
    if progress: progress("Precomputing Flow Puro...")
    state.flow_puro = _precompute_flow_puro(df_clean)
    if progress: progress("Flow Puro ready.")

    if progress: progress("Precomputing UND_NAME...")
    state.und_name = _precompute_und_name(df_clean)
    if progress: progress("UND_NAME ready.")
    
    if progress: progress("Precomputing UND Evolution...")
    state.und_evolution = _precompute_und_evolution(df_clean)
    if progress: progress("UND Evolution ready.")
    
    if progress: progress("Precomputing UND Flow...")
    state.und_flow = _precompute_und_flow(df_clean)
    if progress: progress("UND Flow ready.")
    
    if progress: progress("Precomputing Size/Percentiles...")
    state.size_percentiles = _precompute_size_percentiles(df_clean)
    if progress: progress("Size/Percentiles ready.")

    if progress: progress("Ready.")


def summarize(state: AppState) -> dict:
    n = 0 if state.df_clean is None else int(len(state.df_clean))
    return {"n_trades": n}

def _precompute_cp_time_aggs(df_clean: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Returns aggs for Daily/Weekly/Monthly with columns:
      CP, Period, Notional, Flow, Trades
    """
    base = df_clean[["Counterparty", "_Day", "_Week", "_Month", "Notional", "Flow"]].copy()

    base["Counterparty"] = base["Counterparty"].astype(str)
    base["Notional"] = pd.to_numeric(base["Notional"], errors="coerce").fillna(0.0).astype(float)
    base["Flow"] = pd.to_numeric(base["Flow"], errors="coerce").fillna(0.0).astype(float)
    base["_Trades"] = 1.0

    def agg_for(period_col: str) -> pd.DataFrame:
        out = (
            base.groupby(["Counterparty", period_col], as_index=False, observed=False)[["Notional", "Flow", "_Trades"]]
                .sum()
                .rename(columns={"Counterparty": "CP", period_col: "Period", "_Trades": "Trades"})
                .sort_values(["CP", "Period"])
                .reset_index(drop=True)
        )
        out["Notional"] = out["Notional"].astype(float)
        out["Flow"] = out["Flow"].astype(float)
        out["Trades"] = out["Trades"].astype(float)
        return out

    return {
        "Daily": agg_for("_Day"),
        "Weekly": agg_for("_Week"),
        "Monthly": agg_for("_Month"),
    }

def _precompute_flow_puro(df_clean: pd.DataFrame) -> dict:
    """
    Precompute for Flow Puro tab:
      - base slim df with only needed columns (fast filtering)
      - aggs for Daily/Weekly/Monthly when no extra filters are applied
      - unique values per filter column for dropdowns
    Expects columns (if exist): Counterparty, UND_NAME, ISIN, b/s, C/P, Flow, Notional, _Day, _Week, _Month
    """
    # --- choose available filter columns ---
    filter_cols = []
    for c in ["Counterparty", "UND_NAME", "ISIN", "b/s", "CALL_OPTION"]:
        if c in df_clean.columns:
            filter_cols.append(c)

    needed = ["Flow", "Notional", "_Day", "_Week", "_Month"] + filter_cols
    needed = [c for c in needed if c in df_clean.columns]

    base = df_clean[needed].copy()

    # normalize types
    if "Flow" in base.columns:
        base["Flow"] = pd.to_numeric(base["Flow"], errors="coerce").fillna(0.0).astype(float)
    else:
        base["Flow"] = 0.0

    if "Notional" in base.columns:
        base["Notional"] = pd.to_numeric(base["Notional"], errors="coerce").fillna(0.0).astype(float)
    else:
        base["Notional"] = base["Flow"].abs()

    base["_Trades"] = 1.0

    # normalize categoricals to string (for stable dropdowns / filtering)
    for c in filter_cols:
        base[c] = base[c].astype(str)

    for p in ["_Day", "_Week", "_Month"]:
        if p in base.columns:
            base[p] = base[p].astype(str)

    # unique dropdown values
    values = {}
    for c in filter_cols:
        vals = sorted(v for v in base[c].dropna().unique().tolist() if v != "nan")
        values[c] = vals

    def agg_for(period_col: str) -> pd.DataFrame:
        if period_col not in base.columns:
            return pd.DataFrame(columns=["Period", "Total Flow", "Total Notional", "Total Trades"])
        out = (
            base.groupby(period_col, as_index=False, observed=False)[["Flow", "Notional", "_Trades"]]
                .sum()
                .rename(columns={
                    period_col: "Period",
                    "Flow": "Total Flow",
                    "Notional": "Total Notional",
                    "_Trades": "Total Trades",
                })
                .sort_values("Period")
                .reset_index(drop=True)
        )
        return out

    aggs = {
        "Daily": agg_for("_Day"),
        "Weekly": agg_for("_Week"),
        "Monthly": agg_for("_Month"),
    }

    return {
        "base": base,
        "aggs": aggs,
        "values": values,
        "filter_cols": filter_cols,
    }

def _precompute_und_name(df_clean: pd.DataFrame) -> dict:
    # columnas mínimas
    cols = []
    for c in ["_Day", "_Month", "CALL_OPTION", "Counterparty", "UND_NAME", "b/s", "Notional"]:
        if c in df_clean.columns:
            cols.append(c)

    base = df_clean[cols].copy()

    # types
    base["UND_NAME"] = base["UND_NAME"].astype(str)
    if "Counterparty" in base.columns:
        base["Counterparty"] = base["Counterparty"].astype(str)
    if "_Day" in base.columns:
        base["_Day"] = base["_Day"].astype(str)
    if "_Month" in base.columns:
        base["_Month"] = base["_Month"].astype(str)
    if "CALL_OPTION" in base.columns:
        base["CALL_OPTION"] = base["CALL_OPTION"].astype(str)
    if "b/s" in base.columns:
        base["b/s"] = base["b/s"].astype(str)

    notional = pd.to_numeric(base.get("Notional"), errors="coerce").fillna(0.0).astype(float)

    call = base["CALL_OPTION"].eq("C") if "CALL_OPTION" in base.columns else pd.Series(False, index=base.index)
    put  = base["CALL_OPTION"].eq("P") if "CALL_OPTION" in base.columns else pd.Series(False, index=base.index)
    buy  = base["b/s"].eq("buy")       if "b/s" in base.columns else pd.Series(False, index=base.index)
    sell = base["b/s"].eq("sell")      if "b/s" in base.columns else pd.Series(False, index=base.index)

    # columnas numéricas vectorizadas (CRÍTICO: evita lambdas por grupo)
    base["Trades"]  = 1.0
    base["N"]       = notional
    base["N_C"]     = notional.where(call, 0.0)
    base["N_P"]     = notional.where(put, 0.0)
    base["N_buy"]   = notional.where(buy, 0.0)
    base["N_sell"]  = notional.where(sell, 0.0)

    base["T_C"]     = call.astype(float)
    base["T_P"]     = put.astype(float)
    base["T_buy"]   = buy.astype(float)
    base["T_sell"]  = sell.astype(float)

    # Agregado ALL (lo que pintas al entrar por primera vez)
    grp = base.groupby("UND_NAME", observed=False)[
        ["N", "Trades", "N_C", "N_P", "N_buy", "N_sell", "T_C", "T_P", "T_buy", "T_sell"]
    ].sum()

    # derivadas (en agregado, ya es baratísimo)
    und_all = grp.copy()
    und_all["CP$R"]   = und_all["N_C"] / und_all["N_P"].replace(0, pd.NA)
    und_all["Call%#"] = 100.0 * und_all["T_C"] / (und_all["T_C"] + und_all["T_P"]).replace(0, pd.NA)
    und_all["BS$R"]   = und_all["N_buy"] / und_all["N_sell"].replace(0, pd.NA)
    und_all["Buy%#"]  = 100.0 * und_all["T_buy"] / (und_all["T_buy"] + und_all["T_sell"]).replace(0, pd.NA)

    und_all = und_all.sort_values("N", ascending=False)

    und_top20_all = und_all["N"].head(20)

    return {
        "base": base,
        "und_all": und_all,         # tabla completa ALL
        "und_top20_all": und_top20_all,  # chart ALL
        "cache": {},                # cache por filtros en runtime (dict)
    }

def _precompute_und_evolution(df_clean: pd.DataFrame) -> dict:
    """
    Precompute aggs for UND Evolution:
      aggs["Daily"/"Weekly"/"Monthly"] => DataFrame columns: UND, Period, Notional, Trades
      und_list => sorted list of all UND_NAME (as str)
    """
    needed = ["UND_NAME", "_Day", "_Week", "_Month"]
    for c in needed:
        if c not in df_clean.columns:
            return {"aggs": {"Daily": pd.DataFrame(), "Weekly": pd.DataFrame(), "Monthly": pd.DataFrame()},
                    "und_list": []}

    # Notional
    if "Notional" in df_clean.columns:
        notional = pd.to_numeric(df_clean["Notional"], errors="coerce").fillna(0.0).astype(float)
    else:
        q = pd.to_numeric(df_clean.get("Quantity", 0.0), errors="coerce").fillna(0.0).abs().astype(float)
        p = pd.to_numeric(df_clean.get("Price", 0.0), errors="coerce").fillna(0.0).astype(float)
        notional = (q * p).astype(float)

    base = pd.DataFrame({
        "UND": df_clean["UND_NAME"].astype(str),
        "_Day": df_clean["_Day"].astype(str),
        "_Week": df_clean["_Week"].astype(str),
        "_Month": df_clean["_Month"].astype(str),
        "Notional": notional,
        "Trades": 1.0,
    })

    def agg_for(col: str) -> pd.DataFrame:
        out = (
            base.groupby(["UND", col], as_index=False, observed=False)[["Notional", "Trades"]]
                .sum()
                .rename(columns={col: "Period"})
                .sort_values(["UND", "Period"])
                .reset_index(drop=True)
        )
        # force numeric
        out["Notional"] = pd.to_numeric(out["Notional"], errors="coerce").fillna(0.0).astype(float)
        out["Trades"] = pd.to_numeric(out["Trades"], errors="coerce").fillna(0.0).astype(float)
        return out

    aggs = {
        "Daily": agg_for("_Day"),
        "Weekly": agg_for("_Week"),
        "Monthly": agg_for("_Month"),
    }
    und_list = sorted(base["UND"].unique().tolist())

    return {"aggs": aggs, "und_list": und_list}

def _precompute_und_flow(df_clean: pd.DataFrame) -> dict:
    """
    Precompute aggs for UND Flow tab:
      aggs["Daily"/"Weekly"/"Monthly"] columns: UND, Period, Notional, Flow, Trades
      und_list: sorted list of all UND_NAME
    Requires df_clean to already have 'Flow' and 'Notional' (recommended).
    """
    needed = ["UND_NAME", "_Day", "_Week", "_Month", "Flow"]
    for c in needed:
        if c not in df_clean.columns:
            return {"aggs": {"Daily": pd.DataFrame(), "Weekly": pd.DataFrame(), "Monthly": pd.DataFrame()},
                    "und_list": []}

    if "Notional" in df_clean.columns:
        notional = pd.to_numeric(df_clean["Notional"], errors="coerce").fillna(0.0).astype(float)
    else:
        q = pd.to_numeric(df_clean.get("Quantity", 0.0), errors="coerce").fillna(0.0).abs().astype(float)
        p = pd.to_numeric(df_clean.get("Price", 0.0), errors="coerce").fillna(0.0).astype(float)
        notional = (q * p).astype(float)

    flow = pd.to_numeric(df_clean["Flow"], errors="coerce").fillna(0.0).astype(float)

    base = pd.DataFrame({
        "UND": df_clean["UND_NAME"].astype(str),
        "_Day": df_clean["_Day"].astype(str),
        "_Week": df_clean["_Week"].astype(str),
        "_Month": df_clean["_Month"].astype(str),
        "Notional": notional,
        "Flow": flow,
        "Trades": 1.0,
    })

    def agg_for(col: str) -> pd.DataFrame:
        out = (
            base.groupby(["UND", col], as_index=False, observed=False)[["Notional", "Flow", "Trades"]]
                .sum()
                .rename(columns={col: "Period"})
                .sort_values(["UND", "Period"])
                .reset_index(drop=True)
        )
        out["Notional"] = pd.to_numeric(out["Notional"], errors="coerce").fillna(0.0).astype(float)
        out["Flow"] = pd.to_numeric(out["Flow"], errors="coerce").fillna(0.0).astype(float)
        out["Trades"] = pd.to_numeric(out["Trades"], errors="coerce").fillna(0.0).astype(float)
        out["UND"] = out["UND"].astype(str)
        out["Period"] = out["Period"].astype(str)
        return out

    aggs = {"Daily": agg_for("_Day"), "Weekly": agg_for("_Week"), "Monthly": agg_for("_Month")}
    und_list = sorted(base["UND"].unique().tolist())

    return {"aggs": aggs, "und_list": und_list}

def _precompute_size_percentiles(df_clean: pd.DataFrame) -> dict:
    """
    Precompute pack for Size/Percentiles analysis.
    Keeps a slim base df + dropdown values + (optional) fast ALL-filter quantile edges.
    """

    # Required-ish cols
    # Notional/Flow should already exist in your pipeline. If Notional missing, we derive.
    cols = []
    for c in ["_Day", "_Week", "_Month", "CALL_OPTION", "Counterparty", "UND_NAME", "b/s", "Quantity", "Price", "Notional", "Flow", "TradeNo"]:
        if c in df_clean.columns:
            cols.append(c)

    base = df_clean[cols].copy()

    # Types
    for c in ["_Day", "_Week", "_Month", "CALL_OPTION", "Counterparty", "UND_NAME", "b/s"]:
        if c in base.columns:
            base[c] = base[c].astype(str)

    # Quantity
    if "Quantity" in base.columns:
        qty = pd.to_numeric(base["Quantity"], errors="coerce").fillna(0.0).astype(float)
    else:
        qty = pd.Series(0.0, index=base.index, dtype=float)

    # Notional
    if "Notional" in base.columns:
        notional = pd.to_numeric(base["Notional"], errors="coerce").fillna(0.0).astype(float)
    else:
        price = pd.to_numeric(base.get("Price", 0.0), errors="coerce").fillna(0.0).astype(float)
        notional = (qty.abs() * price).astype(float)
        base["Notional"] = notional

    # Flow (signed)
    flow = pd.to_numeric(base.get("Flow", 0.0), errors="coerce").fillna(0.0).astype(float)
    base["Flow"] = flow

    base["_Trades"] = 1.0
    base["_AbsFlow"] = flow.abs()
    base["_AbsNotional"] = notional.abs()
    base["_AbsQty"] = qty.abs()

    # Dropdown values (fast)
    values = {}
    for c in ["CALL_OPTION", "Counterparty", "UND_NAME", "b/s"]:
        if c in base.columns:
            values[c] = sorted(pd.Series(base[c]).dropna().astype(str).unique().tolist())

    # Fast global quantile edges (ALL universe) for standard schemes
    def q_edges(series: pd.Series, probs: list[float]) -> np.ndarray:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) == 0:
            return np.array([0.0, 1.0], dtype=float)
        return np.quantile(s.to_numpy(dtype=float), probs)

    schemes = {
        "Quartiles": [0.0, 0.25, 0.5, 0.75, 1.0],
        "Deciles":   [i/10 for i in range(0, 11)],
        # “Whales”: muy útil para tus preguntas
        "Whales":    [0.0, 0.50, 0.80, 0.90, 0.95, 0.99, 1.0],
    }

    edges = {
        "Notional": {name: q_edges(base["_AbsNotional"], probs) for name, probs in schemes.items()},
        "Quantity": {name: q_edges(base["_AbsQty"], probs) for name, probs in schemes.items()},
        "schemes": schemes,
    }

    return {"base": base, "values": values, "edges": edges}