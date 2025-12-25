# ui/views/und_name.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import tkinter as tk
from tkinter import ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm

from ui.views.counterparty import SortableTable  # reuse the same pro table


def _fmt_km(x, _pos=None) -> str:
    try:
        x = float(x)
    except Exception:
        return ""
    ax = abs(x)
    if ax >= 1_000_000:
        return f"{x/1_000_000:.0f}M" if ax >= 10_000_000 else f"{x/1_000_000:.1f}M"
    if ax >= 1_000:
        return f"{x/1_000:.0f}k" if ax >= 10_000 else f"{x/1_000:.1f}k"
    return f"{x:.0f}"


def _safe_ratio(num: float, den: float):
    if den is None or den == 0 or (isinstance(den, float) and math.isnan(den)):
        if num == 0 or num is None or (isinstance(num, float) and math.isnan(num)):
            return None
        return float("inf")
    return num / den


def _ratio_to_str(r):
    if r is None:
        return "—"
    if isinstance(r, float) and math.isinf(r):
        return "∞"
    try:
        return f"{float(r):.2f}"
    except Exception:
        return "—"


def _pct_to_str(p):
    if p is None:
        return "—"
    if isinstance(p, float) and math.isinf(p):
        return "∞"
    try:
        return f"{float(p):.1f}%"
    except Exception:
        return "—"


def _comma(x: float, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if digits == 0:
        return f"{x:,.0f}"
    return f"{x:,.{digits}f}"


@dataclass(frozen=True)
class _FilterKey:
    day: str
    month: str
    call: str
    cp: str
    und: str


class UndNameView(ttk.Frame):
    """
    2.1 UND_NAME
    - Chart: TOP 20 underlyings
    - Table: ALL underlyings
    - Instant first render using state.und_name precompute
    """

    def __init__(self, master):
        super().__init__(master, style="TFrame")

        self._df: Optional[pd.DataFrame] = None
        self._prepared: bool = False
        self._filter_values: Optional[dict] = None

        # NEW: precompute pack from actions.py
        self._pack: Optional[dict] = None  # {"base","und_all","und_top20_all","cache"}

        # Filters row
        self.filters_row = ttk.Frame(self, style="TFrame")
        self.filters_row.pack(side="top", fill="x", padx=14, pady=(12, 6))

        self.var_day = tk.StringVar(value="ALL")
        self.var_month = tk.StringVar(value="ALL")
        self.var_call = tk.StringVar(value="ALL")
        self.var_cp = tk.StringVar(value="ALL")
        self.var_und = tk.StringVar(value="ALL")

        self.cb_day = self._make_combo(self.filters_row, "Day", self.var_day, width=12)
        self.cb_month = self._make_combo(self.filters_row, "Month", self.var_month, width=10)
        self.cb_call = self._make_combo(self.filters_row, "C/P", self.var_call, width=6)
        self.cb_cp = self._make_combo(self.filters_row, "CP", self.var_cp, width=10)
        self.cb_und = self._make_combo(self.filters_row, "UND", self.var_und, width=14)

        self.btn_apply = ttk.Button(self.filters_row, text="Apply", command=self.apply_filters)
        self.btn_reset = ttk.Button(self.filters_row, text="Reset", command=self.reset_filters)
        self.btn_apply.pack(side="left", padx=(14, 6))
        self.btn_reset.pack(side="left", padx=(0, 6))

        # KPIs row
        self.kpi_row = ttk.Frame(self, style="TFrame")
        self.kpi_row.pack(side="top", fill="x", padx=14, pady=(4, 10))
        for c in range(6):
            self.kpi_row.columnconfigure(c, weight=1)

        self.kpi_total_notional = self._kpi_inline_card("Total Notional", col=0)
        self.kpi_total_trades = self._kpi_inline_card("Total Trades", col=1)
        self.kpi_cp_notional = self._kpi_inline_card("C/P $ Ratio", col=2)
        self.kpi_cp_trades = self._kpi_inline_card("Call % Trades", col=3)
        self.kpi_bs_notional = self._kpi_inline_card("B/S $ Ratio", col=4)
        self.kpi_bs_trades = self._kpi_inline_card("Buy % Trades", col=5)

        # Bottom: chart + table
        self.bottom = ttk.Frame(self, style="TFrame")
        self.bottom.pack(side="top", fill="both", expand=True, padx=14, pady=(0, 14))
        self.bottom.rowconfigure(0, weight=1)
        self.bottom.columnconfigure(0, weight=1, uniform="half")
        self.bottom.columnconfigure(1, weight=1, uniform="half")

        # Chart
        self.chart_frame = ttk.Frame(self.bottom, style="TFrame")
        self.chart_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.chart_frame.rowconfigure(1, weight=1)
        self.chart_frame.columnconfigure(0, weight=1)

        ttk.Label(self.chart_frame, text="Notional by Underlying (Top 20)", style="Title.TLabel") \
            .grid(row=0, column=0, sticky="w", pady=(0, 6))

        self._fig = Figure(figsize=(6, 4), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._ax.yaxis.set_major_formatter(FuncFormatter(_fmt_km))
        self._fig.tight_layout()

        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=self.chart_frame)
        self._mpl_widget = self._mpl_canvas.get_tk_widget()
        self._mpl_widget.grid(row=1, column=0, sticky="nsew")
        self.chart_frame.bind("<Configure>", self._on_chart_resize)

        # Table
        self.table_frame = ttk.Frame(self.bottom, style="TFrame")
        self.table_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.table_frame.rowconfigure(1, weight=1)
        self.table_frame.columnconfigure(0, weight=1)

        ttk.Label(self.table_frame, text="Underlying summary (click headers to sort)", style="Title.TLabel") \
            .grid(row=0, column=0, sticky="w", pady=(0, 6))

        self.table_cols = ["CP", "Notional", "Trades", "CP$R", "Call%#", "BS$R", "Buy%#"]
        self.table = SortableTable(self.table_frame, columns=self.table_cols, height=18)
        self.table.grid(row=1, column=0, sticky="nsew")

        self._last_chart_series: Optional[pd.Series] = None

    def _make_combo(self, parent, label, var, width=10):
        ttk.Label(parent, text=f"{label}:", style="Muted.TLabel").pack(side="left", padx=(0, 6))
        cb = ttk.Combobox(parent, textvariable=var, width=width, state="readonly")
        cb["values"] = ("ALL",)
        cb.pack(side="left", padx=(0, 10))
        return cb

    def _kpi_inline_card(self, title: str, col: int) -> ttk.Label:
        fr = ttk.Frame(self.kpi_row, style="Card.TFrame")
        fr.grid(row=0, column=col, sticky="ew", padx=(0, 10), pady=0)
        fr.columnconfigure(0, weight=0)
        fr.columnconfigure(1, weight=1)

        ttk.Label(fr, text=f"{title}:", style="Muted.TLabel", font=("TkDefaultFont", 9)).grid(
            row=0, column=0, sticky="w", padx=(6, 4), pady=5
        )
        val = ttk.Label(fr, text="—", style="KPI.TLabel", font=("TkDefaultFont", 10, "bold"))
        val.grid(row=0, column=1, sticky="w", padx=(0, 6), pady=5)
        return val

    # -------- public --------

    def render(self, df: Optional[pd.DataFrame], filter_values: Optional[dict] = None, und_pack: Optional[dict] = None):
        """
        und_pack should be state.und_name from actions.py precompute.
        """
        if df is self._df and self._prepared and filter_values is self._filter_values and und_pack is self._pack:
            return

        self._df = df
        self._filter_values = filter_values
        self._pack = und_pack
        self._prepared = False

        self._populate_filter_values()
        self._prepared = True

        # IMPORTANT: instant first render (no heavy compute)
        self.var_day.set("ALL")
        self.var_month.set("ALL")
        self.var_call.set("ALL")
        self.var_cp.set("ALL")
        self.var_und.set("ALL")
        self._render_all_from_precompute()

    def _populate_filter_values(self):
        if self._filter_values is not None:
            fv = self._filter_values
            self.cb_day["values"] = tuple(fv.get("days", ["ALL"]))
            self.cb_month["values"] = tuple(fv.get("months", ["ALL"]))
            self.cb_call["values"] = tuple(fv.get("calls", ["ALL"]))
            self.cb_cp["values"] = tuple(fv.get("cps", ["ALL"]))
            self.cb_und["values"] = tuple(fv.get("unds", ["ALL"]))
            return

        if self._df is None or self._df.empty:
            for cb in (self.cb_day, self.cb_month, self.cb_call, self.cb_cp, self.cb_und):
                cb["values"] = ("ALL",)
            return

        d = self._df
        days = ["ALL"] + sorted(pd.Series(d["_Day"]).dropna().astype(str).unique().tolist()) if "_Day" in d else ["ALL"]
        months = ["ALL"] + sorted(pd.Series(d["_Month"]).dropna().astype(str).unique().tolist()) if "_Month" in d else ["ALL"]
        calls = ["ALL"] + sorted(pd.Series(d["CALL_OPTION"]).dropna().astype(str).unique().tolist()) if "CALL_OPTION" in d else ["ALL"]
        cps = ["ALL"] + sorted(pd.Series(d["Counterparty"]).dropna().astype(str).unique().tolist()) if "Counterparty" in d else ["ALL"]
        unds = ["ALL"] + sorted(pd.Series(d["UND_NAME"]).dropna().astype(str).unique().tolist()) if "UND_NAME" in d else ["ALL"]

        self.cb_day["values"] = tuple(days)
        self.cb_month["values"] = tuple(months)
        self.cb_call["values"] = tuple(calls)
        self.cb_cp["values"] = tuple(cps)
        self.cb_und["values"] = tuple(unds)

    def _filter_key(self) -> _FilterKey:
        return _FilterKey(
            day=self.var_day.get(),
            month=self.var_month.get(),
            call=self.var_call.get(),
            cp=self.var_cp.get(),
            und=self.var_und.get(),
        )

    # -------- instant ALL render --------

    def _render_all_from_precompute(self):
        if not self._pack:
            # fallback: if precompute not provided, compute via apply_filters
            self.apply_filters()
            return

        und_all = self._pack.get("und_all")
        top20 = self._pack.get("und_top20_all")

        if und_all is None or top20 is None:
            self.apply_filters()
            return

        self._render_kpis_from_und_all(und_all)
        self._update_chart(top20)
        self.table.set_records(self._rows_from_und_all(und_all))

    # -------- filtering + compute (fast) --------

    def _filtered_base(self) -> pd.DataFrame:
        """
        Uses precomputed slim base if available; fallback to df.
        """
        if self._pack and "base" in self._pack:
            base = self._pack["base"]
        else:
            if self._df is None or self._df.empty:
                return pd.DataFrame()
            base = self._df

        if base is None or len(base) == 0:
            return pd.DataFrame()

        df = base
        if self.var_day.get() != "ALL" and "_Day" in df.columns:
            df = df[df["_Day"] == self.var_day.get()]
        if self.var_month.get() != "ALL" and "_Month" in df.columns:
            df = df[df["_Month"] == self.var_month.get()]
        if self.var_call.get() != "ALL" and "CALL_OPTION" in df.columns:
            df = df[df["CALL_OPTION"] == self.var_call.get()]
        if self.var_cp.get() != "ALL" and "Counterparty" in df.columns:
            df = df[df["Counterparty"] == self.var_cp.get()]
        if self.var_und.get() != "ALL" and "UND_NAME" in df.columns:
            df = df[df["UND_NAME"] == self.var_und.get()]

        return df

    def apply_filters(self):
        k = self._filter_key()

        if self._pack is not None:
            cache = self._pack.setdefault("cache", {})
            if k in cache:
                und_all, top20 = cache[k]
                self._render_kpis_from_und_all(und_all)
                self._update_chart(top20)
                self.table.set_records(self._rows_from_und_all(und_all))
                return

        df_f = self._filtered_base()
        und_all = self._agg_und(df_f)  # fast groupby sum (no lambdas)
        top20 = und_all["N"].head(20) if len(und_all) else pd.Series(dtype="float64")

        self._render_kpis_from_und_all(und_all)
        self._update_chart(top20)
        self.table.set_records(self._rows_from_und_all(und_all))

        if self._pack is not None:
            self._pack["cache"][k] = (und_all, top20)

    def reset_filters(self):
        self.var_day.set("ALL")
        self.var_month.set("ALL")
        self.var_call.set("ALL")
        self.var_cp.set("ALL")
        self.var_und.set("ALL")
        # after reset, use instant ALL if available
        self._render_all_from_precompute()

    def _agg_und(self, df_f: pd.DataFrame) -> pd.DataFrame:
        """
        Fast aggregation on vectorized columns if present.
        Expected cols in precomputed base:
          N, Trades, N_C, N_P, N_buy, N_sell, T_C, T_P, T_buy, T_sell
        """
        if df_f is None or df_f.empty:
            return pd.DataFrame(
                columns=["N", "Trades", "N_C", "N_P", "N_buy", "N_sell", "T_C", "T_P", "T_buy", "T_sell",
                         "CP$R", "Call%#", "BS$R", "Buy%#"]
            )

        # If not vectorized (fallback), build minimal vectorization here
        if "N" not in df_f.columns:
            notional = pd.to_numeric(df_f.get("Notional"), errors="coerce").fillna(0.0).astype(float)
            call = df_f["CALL_OPTION"].eq("C") if "CALL_OPTION" in df_f.columns else pd.Series(False, index=df_f.index)
            put  = df_f["CALL_OPTION"].eq("P") if "CALL_OPTION" in df_f.columns else pd.Series(False, index=df_f.index)
            buy  = df_f["b/s"].eq("buy")       if "b/s" in df_f.columns else pd.Series(False, index=df_f.index)
            sell = df_f["b/s"].eq("sell")      if "b/s" in df_f.columns else pd.Series(False, index=df_f.index)

            tmp = pd.DataFrame({
                "UND_NAME": df_f["UND_NAME"].astype(str),
                "N": notional,
                "Trades": 1.0,
                "N_C": notional.where(call, 0.0),
                "N_P": notional.where(put, 0.0),
                "N_buy": notional.where(buy, 0.0),
                "N_sell": notional.where(sell, 0.0),
                "T_C": call.astype(float),
                "T_P": put.astype(float),
                "T_buy": buy.astype(float),
                "T_sell": sell.astype(float),
            })
            grp = tmp.groupby("UND_NAME", observed=False).sum(numeric_only=True)
        else:
            grp = df_f.groupby("UND_NAME", observed=False)[
                ["N", "Trades", "N_C", "N_P", "N_buy", "N_sell", "T_C", "T_P", "T_buy", "T_sell"]
            ].sum(numeric_only=True)

        out = grp.copy()
        out["CP$R"] = out["N_C"] / out["N_P"].replace(0, pd.NA)
        out["Call%#"] = 100.0 * out["T_C"] / (out["T_C"] + out["T_P"]).replace(0, pd.NA)
        out["BS$R"] = out["N_buy"] / out["N_sell"].replace(0, pd.NA)
        out["Buy%#"] = 100.0 * out["T_buy"] / (out["T_buy"] + out["T_sell"]).replace(0, pd.NA)

        out = out.sort_values("N", ascending=False)
        return out

    # -------- KPIs & table rows from aggregated UND --------

    def _render_kpis_from_und_all(self, und_all: pd.DataFrame):
        if und_all is None or len(und_all) == 0:
            self.kpi_total_notional.config(text="—")
            self.kpi_total_trades.config(text="—")
            self.kpi_cp_notional.config(text="—")
            self.kpi_cp_trades.config(text="—")
            self.kpi_bs_notional.config(text="—")
            self.kpi_bs_trades.config(text="—")
            return

        total_notional = float(und_all["N"].sum())
        total_trades = int(round(float(und_all["Trades"].sum())))

        # overall ratios/pcts from totals
        tot_NC = float(und_all["N_C"].sum())
        tot_NP = float(und_all["N_P"].sum())
        tot_Nbuy = float(und_all["N_buy"].sum())
        tot_Nsell = float(und_all["N_sell"].sum())

        tot_TC = float(und_all["T_C"].sum())
        tot_TP = float(und_all["T_P"].sum())
        tot_Tbuy = float(und_all["T_buy"].sum())
        tot_Tsell = float(und_all["T_sell"].sum())

        cp_ratio = _safe_ratio(tot_NC, tot_NP)
        bs_ratio = _safe_ratio(tot_Nbuy, tot_Nsell)

        call_pct = None if (tot_TC + tot_TP) == 0 else 100.0 * tot_TC / (tot_TC + tot_TP)
        buy_pct = None if (tot_Tbuy + tot_Tsell) == 0 else 100.0 * tot_Tbuy / (tot_Tbuy + tot_Tsell)

        self.kpi_total_notional.config(text=_comma(total_notional))
        self.kpi_total_trades.config(text=f"{total_trades:,}")
        self.kpi_cp_notional.config(text=_ratio_to_str(cp_ratio))
        self.kpi_cp_trades.config(text=_pct_to_str(call_pct))
        self.kpi_bs_notional.config(text=_ratio_to_str(bs_ratio))
        self.kpi_bs_trades.config(text=_pct_to_str(buy_pct))

    def _rows_from_und_all(self, und_all: pd.DataFrame) -> List[dict]:
        rows: List[dict] = []
        if und_all is None or len(und_all) == 0:
            return rows

        for und, r in und_all.iterrows():
            und_s = str(und)
            cp_r = r.get("CP$R", None)
            bs_r = r.get("BS$R", None)
            call_p = r.get("Call%#", None)
            buy_p = r.get("Buy%#", None)

            rows.append({
                "CP": und_s,
                "Notional": _comma(float(r["N"])),
                "Trades": f"{int(round(float(r['Trades']))):,}",
                "CP$R": _ratio_to_str(None if pd.isna(cp_r) else float(cp_r)),
                "Call%#": _pct_to_str(None if pd.isna(call_p) else float(call_p)),
                "BS$R": _ratio_to_str(None if pd.isna(bs_r) else float(bs_r)),
                "Buy%#": _pct_to_str(None if pd.isna(buy_p) else float(buy_p)),
                "__raw__CP": und_s,
                "__raw__Notional": float(r["N"]),
                "__raw__Trades": int(round(float(r["Trades"]))),
                "__raw__CP$R": None if pd.isna(cp_r) else float(cp_r),
                "__raw__Call%#": None if pd.isna(call_p) else float(call_p),
                "__raw__BS$R": None if pd.isna(bs_r) else float(bs_r),
                "__raw__Buy%#": None if pd.isna(buy_p) else float(buy_p),
            })
        return rows

    # -------- chart --------

    def _on_chart_resize(self, event):
        if self._last_chart_series is None:
            return
        w_px = max(200, int(event.width))
        h_px = max(200, int(event.height - 30))
        dpi = 100
        self._fig.set_size_inches(w_px / dpi, h_px / dpi, forward=True)
        self._fig.tight_layout()
        self._mpl_canvas.draw_idle()

    def _update_chart(self, und_series: pd.Series):
        self._last_chart_series = und_series

        ax = self._ax
        ax.clear()
        ax.set_xlabel("Underlying (Top 20)")
        ax.set_ylabel("Notional")
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_km))

        if und_series is None or len(und_series) == 0:
            self._mpl_canvas.draw_idle()
            return

        x = und_series.index.astype(str).tolist()
        y = und_series.values.tolist()

        n = len(x)
        cmap = cm.get_cmap("tab20", max(n, 1))
        colors = [cmap(i) for i in range(n)]

        ax.bar(x, y, color=colors)
        ax.tick_params(axis="x", rotation=60, labelsize=9)
        ax.grid(True, axis="y", alpha=0.25)

        self.chart_frame.update_idletasks()
        w_px = max(200, self._mpl_widget.winfo_width())
        h_px = max(200, self._mpl_widget.winfo_height())
        dpi = 100
        self._fig.set_size_inches(w_px / dpi, h_px / dpi, forward=True)
        self._fig.tight_layout()
        self._mpl_canvas.draw_idle()
