from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd

DEFAULT_SYNONYMS: Dict[str, List[str]] = {
    "TradeTime": ["tradetime", "trade_time", "time", "timestamp", "datetime", "trade_datetime"],
    "TradeNo": ["tradeno", "trade_no", "id", "tradeid", "trade_id", "ticket", "ticketno"],
    "ISIN": ["isin", "isin_code", "instrument", "instrument_id", "security", "securityid"],
    "b/s": ["b/s", "side", "bs", "buy_sell", "buy/sell", "direction"],
    "Quantity": ["quantity", "qty", "size", "amount", "volume", "contracts", "units"],
    "Price": ["price", "tradeprice", "px", "execution_price", "fill_price"],
    "Counterparty": ["counterparty", "cp", "cpty", "broker", "client", "participant"],
    "Ref": ["ref", "reference", "ref_price", "spot", "underlying_ref", "underlying_px"],
    "CALL_OPTION": ["call_option", "cp_flag", "callput", "call_put", "putcall", "option_type", "c/p"],
    "Expiry": ["expiry", "expiration", "exp", "maturity", "mat_date"],
    "Knock_Date": ["knock_date", "ko_date", "knockout_date", "barrier_date", "knock"],
    "Ratio": ["ratio", "multiplier", "mult", "parity", "conversion_ratio"],
    "Strike": ["strike", "k", "strike_price"],
    "UND_NAME": ["und_name", "underlying", "underlying_name", "underlier", "ul_name"],
    "UND_TYPE": ["und_type", "underlying_type", "ul_type", "asset_class", "class"],
}

EXPECTED_COLS = [
    "TradeTime", "TradeNo", "ISIN", "b/s", "Quantity", "Price", "Counterparty",
    "Ref", "CALL_OPTION", "Expiry", "Knock_Date", "Ratio", "Strike", "UND_NAME", "UND_TYPE"
]

DEFAULT_CATEGORICALS = ["ISIN", "b/s", "Counterparty", "CALL_OPTION", "UND_NAME", "UND_TYPE"]

SIDE_MAP = {
    "b": "buy", "buy": "buy", "bid": "buy", "long": "buy", "+": "buy",
    "s": "sell", "sell": "sell", "ask": "sell", "short": "sell", "-": "sell",
}
CP_MAP = {"c": "C", "call": "C", "p": "P", "put": "P"}


@dataclass
class CleanReport:
    renamed: Dict[str, str]
    added_missing: List[str]
    dropped_unknown: List[str]
    nan_summary: Dict[str, int]
    dtype_summary: Dict[str, str]


def _norm_colname(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[\s\-_/]+", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def _build_rename_map(columns: Iterable[str],
                      synonyms: Dict[str, List[str]]) -> Dict[str, str]:
    canon_by_norm = {}
    for canon, alts in synonyms.items():
        for a in [canon] + list(alts):
            canon_by_norm[_norm_colname(a)] = canon

    rename = {}
    for c in columns:
        key = _norm_colname(c)
        if key in canon_by_norm:
            rename[c] = canon_by_norm[key]
    return rename


def _to_string_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_string_dtype(s):
        return s
    return s.astype("string")


def _coerce_numeric(series: pd.Series,
                    *,
                    allow_percent: bool = True,
                    allow_parentheses_negative: bool = True) -> pd.Series:
    s = _to_string_series(series).str.strip()

    if allow_parentheses_negative:
        s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)

    s = s.str.replace(r"[^\d\-\+\.\,\%]", "", regex=True)

    is_percent = allow_percent & s.str.endswith("%", na=False)
    if allow_percent:
        s = s.str.replace("%", "", regex=False)

    has_dot = s.str.contains(r"\.", na=False)
    has_comma = s.str.contains(r",", na=False)
    both = has_dot & has_comma

    s2 = s.copy()

    if both.any():
        idx = both[both].index
        tmp = s2.loc[idx]
        last_dot = tmp.str.rfind(".")
        last_com = tmp.str.rfind(",")
        dec_is_com = last_com > last_dot

        idx_com = idx[dec_is_com.values]
        if len(idx_com) > 0:
            t = s2.loc[idx_com]
            t = t.str.replace(".", "", regex=False)
            t = t.str.replace(",", ".", regex=False)
            s2.loc[idx_com] = t

        idx_dot = idx[~dec_is_com.values]
        if len(idx_dot) > 0:
            t = s2.loc[idx_dot]
            t = t.str.replace(",", "", regex=False)
            s2.loc[idx_dot] = t

    only_comma = has_comma & ~has_dot
    if only_comma.any():
        s2.loc[only_comma] = s2.loc[only_comma].str.replace(",", ".", regex=False)

    out = pd.to_numeric(s2, errors="coerce").astype("Float64")

    if allow_percent and is_percent.any():
        out.loc[is_percent] = out.loc[is_percent] / 100.0

    return out


def _coerce_int(series: pd.Series) -> pd.Series:
    x = _coerce_numeric(series, allow_percent=False)
    return x.round().astype("Int64")


def _coerce_datetime(series: pd.Series,
                     *,
                     dayfirst: bool = True,
                     utc: bool = False) -> pd.Series:
    s = series

    if pd.api.types.is_datetime64_any_dtype(s):
        dt = pd.to_datetime(s, errors="coerce", utc=utc)
        return dt.dt.tz_localize(None) if utc else dt

    sn = pd.to_numeric(_to_string_series(s), errors="coerce")
    med = sn.median(skipna=True)
    dt = None
    if pd.notna(med) and med > 1e10:
        unit = "ms" if med < 1e15 else "ns"
        dt = pd.to_datetime(sn, unit=unit, errors="coerce", utc=utc)
    elif pd.notna(med) and med > 1e9:
        dt = pd.to_datetime(sn, unit="s", errors="coerce", utc=utc)

    if dt is None:
        dt = pd.to_datetime(_to_string_series(s), errors="coerce", dayfirst=dayfirst, utc=utc)

    return dt.dt.tz_localize(None) if utc else dt


def _coerce_date_to_string(series: pd.Series, *, dayfirst: bool = True) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        dt = pd.to_datetime(series, errors="coerce")
    else:
        dt = pd.to_datetime(_to_string_series(series), errors="coerce", format="%Y-%m-%d")


    out = dt.dt.strftime("%Y-%m-%d").astype("string")
    out = out.where(dt.notna(), pd.NA)
    return out


def _standardize_side(series: pd.Series) -> pd.Series:
    s = _to_string_series(series).str.strip().str.lower()
    s = s.replace(SIDE_MAP)

    m = ~s.isin(["buy", "sell"]) & s.notna()
    if m.any():
        s.loc[m] = s.loc[m].str[0].replace(SIDE_MAP)

    s = s.where(s.isin(["buy", "sell"]), pd.NA)
    return s.astype("string")


def _standardize_call_put(series: pd.Series) -> pd.Series:
    s = _to_string_series(series).str.strip().str.lower()
    s = s.replace(CP_MAP)

    m = ~s.isin(["C", "P"]) & s.notna()
    if m.any():
        s.loc[m] = s.loc[m].str[0].str.upper()

    s = s.where(s.isin(["C", "P"]), pd.NA)
    return s.astype("string")


def _to_category(series: pd.Series) -> pd.Series:
    return series.astype("category")


def clean_mm_logs(
    df: pd.DataFrame,
    *,
    synonyms: Dict[str, List[str]] = DEFAULT_SYNONYMS,
    keep_unknown_columns: bool = True,
    ensure_expected_columns: bool = True,
    dayfirst: bool = True,
    trade_time_utc: bool = False,
    categorical_cols: Iterable[str] = tuple(DEFAULT_CATEGORICALS),
    return_report: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, CleanReport]]:

    if df is None or len(df) == 0:
        out = pd.DataFrame(columns=EXPECTED_COLS)
        if return_report:
            rep = CleanReport(renamed={}, added_missing=EXPECTED_COLS.copy(),
                              dropped_unknown=[], nan_summary={}, dtype_summary={})
            return out, rep
        return out

    df = df.copy()

    rename_map = _build_rename_map(df.columns, synonyms)
    df = df.rename(columns=rename_map)

    added_missing = []
    if ensure_expected_columns:
        for c in EXPECTED_COLS:
            if c not in df.columns:
                df[c] = pd.NA
                added_missing.append(c)

    dropped_unknown = []
    if not keep_unknown_columns:
        unknown = [c for c in df.columns if c not in EXPECTED_COLS]
        if unknown:
            df = df.drop(columns=unknown)
            dropped_unknown = unknown

    if "TradeTime" in df.columns:
        df["TradeTime"] = _coerce_datetime(df["TradeTime"], dayfirst=dayfirst, utc=trade_time_utc)

    if "TradeNo" in df.columns:
        df["TradeNo"] = _to_string_series(df["TradeNo"]).str.strip()

    if "ISIN" in df.columns:
        df["ISIN"] = _to_string_series(df["ISIN"]).str.strip().str.upper()

    if "b/s" in df.columns:
        df["b/s"] = _standardize_side(df["b/s"])

    if "Counterparty" in df.columns:
        df["Counterparty"] = _to_string_series(df["Counterparty"]).str.strip().str.upper()

    if "CALL_OPTION" in df.columns:
        df["CALL_OPTION"] = _standardize_call_put(df["CALL_OPTION"])

    if "UND_NAME" in df.columns:
        df["UND_NAME"] = _to_string_series(df["UND_NAME"]).str.strip().str.upper()

    if "UND_TYPE" in df.columns:
        s = _to_string_series(df["UND_TYPE"]).str.strip().str.upper()
        s = s.where(s.isin(["FUT", "STO", "COM"]), pd.NA)
        df["UND_TYPE"] = s

    if "Quantity" in df.columns:
        df["Quantity"] = _coerce_int(df["Quantity"])

    for col in ["Price", "Ref", "Strike"]:
        if col in df.columns:
            df[col] = _coerce_numeric(df[col], allow_percent=False)

    if "Ratio" in df.columns:
        df["Ratio"] = _coerce_numeric(df["Ratio"], allow_percent=True)

    for col in ["Expiry", "Knock_Date"]:
        if col in df.columns:
            df[col] = _coerce_date_to_string(df[col], dayfirst=dayfirst)
            
    # --- NOTIONAL ---
    q = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0).abs()
    p = pd.to_numeric(df["Price"], errors="coerce").fillna(0.0)
    df["Notional"] = q * p

    # --- Flow (signed) using "b/s": buy -> negative, sell -> positive ---
    # normalize b/s to lowercase string
    bs = df.get("b/s")
    if bs is None:
        # if column missing, create 0 flow (safe)
        df["Flow"] = 0.0
    else:
        bs_norm = bs.astype(str).str.strip().str.lower()
        sign = bs_norm.map({"sell": 1.0, "buy": -1.0}).fillna(0.0)
        df["Flow"] = sign * (df["Quantity"].abs() * df["Price"])

    # ---------------------------------------------------------
    # Derived date keys for all tabs: _Day / _Month / _Week (Mon)
    # ---------------------------------------------------------
    if "TradeTime" in df.columns and pd.api.types.is_datetime64_any_dtype(df["TradeTime"]):
        tt = pd.to_datetime(df["TradeTime"], errors="coerce")

        # Day / Month string keys
        df["_Day"] = tt.dt.strftime("%Y-%m-%d").astype("string")
        df["_Month"] = tt.dt.strftime("%Y-%m").astype("string")

        # Week starting Monday => store monday date YYYY-MM-DD
        # week_start = date - weekday(days since monday)
        week_start = tt.dt.normalize() - pd.to_timedelta(tt.dt.weekday, unit="D")
        df["_Week"] = week_start.dt.strftime("%Y-%m-%d").astype("string")

        # keep NAs consistent
        df["_Day"] = df["_Day"].where(tt.notna(), pd.NA)
        df["_Month"] = df["_Month"].where(tt.notna(), pd.NA)
        df["_Week"] = df["_Week"].where(tt.notna(), pd.NA)
    else:
        df["_Day"] = pd.NA
        df["_Month"] = pd.NA
        df["_Week"] = pd.NA

    for c in categorical_cols:
        if c in df.columns:
            df[c] = _to_category(_to_string_series(df[c]).str.strip())

    if "TradeTime" in df.columns:
        df = df.sort_values(["TradeTime"], kind="mergesort", na_position="last").reset_index(drop=True)

    if return_report:
        nan_summary = {c: int(df[c].isna().sum()) for c in df.columns}
        dtype_summary = {c: str(df[c].dtype) for c in df.columns}
        rep = CleanReport(
            renamed=rename_map,
            added_missing=added_missing,
            dropped_unknown=dropped_unknown,
            nan_summary=nan_summary,
            dtype_summary=dtype_summary,
        )
        return df, rep

    return df
