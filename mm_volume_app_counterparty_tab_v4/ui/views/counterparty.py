# ui/views/counterparty.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import tkinter as tk
from tkinter import ttk

from ui.widgets.table import DataTable

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm


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


def _safe_ratio(num: float, den: float) -> Optional[float]:
    if den is None or den == 0 or (isinstance(den, float) and math.isnan(den)):
        if num == 0 or num is None or (isinstance(num, float) and math.isnan(num)):
            return None
        return float("inf")
    return num / den


def _ratio_to_str(r: Optional[float]) -> str:
    if r is None:
        return "—"
    if math.isinf(r):
        return "∞"
    return f"{r:.2f}"


def _safe_pct(num: float, den: float) -> Optional[float]:
    if den is None or den == 0 or (isinstance(den, float) and math.isnan(den)):
        return None
    return 100.0 * (num / den)


def _pct_to_str(p: Optional[float]) -> str:
    if p is None:
        return "—"
    if math.isinf(p):
        return "∞"
    return f"{p:.1f}%"


def _comma(x: float, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if digits == 0:
        return f"{x:,.0f}"
    return f"{x:,.{digits}f}"


class SortableTable(DataTable):
    """
    - Click headers to sort.
    - CP left-aligned, rest right-aligned.
    - NO horizontal scrollbar; columns always fit to available width.
    """
    def __init__(self, master, columns: List[str], height: int = 18):
        super().__init__(master, columns=columns, height=height)
        self._rows: List[dict] = []
        self._cols = list(columns)
        self._sort_dir = {c: False for c in self._cols}

        # remove horizontal scroll
        try:
            self.hsb.grid_remove()
        except Exception:
            pass
        try:
            self.tree.configure(xscrollcommand="")
        except Exception:
            pass

        self._setup_columns(self._cols)
        self._apply_alignment()
        self._apply_sortable_headings()
        self.autofit_columns()

        self.tree.bind("<Configure>", lambda e: self.autofit_columns())

    def _apply_alignment(self):
        for c in self._cols:
            self.tree.column(c, anchor="w" if c == "CP" else "e")

    def _apply_sortable_headings(self):
        for c in self._cols:
            self.tree.heading(c, text=c, command=lambda col=c: self.sort_by(col))

    def autofit_columns(self):
        w = self.tree.winfo_width()
        if w <= 80:
            return
        avail = max(280, w - 18)

        weights = {
            "CP": 1.45,
            "Notional": 1.35,
            "Trades": 0.90,
            "CP$R": 0.75,
            "Call%#": 0.85,
            "BS$R": 0.75,
            "Buy%#": 0.85,
        }
        sw = sum(weights.get(c, 1.0) for c in self._cols)

        for c in self._cols:
            cw = int(avail * (weights.get(c, 1.0) / sw))
            if c == "CP":
                cw = max(105, cw)
            elif c == "Notional":
                cw = max(120, cw)
            else:
                cw = max(78, cw)
            self.tree.column(c, width=cw, stretch=True)

    def set_records(self, rows: List[dict]):
        self._rows = rows
        for iid in self.tree.get_children():
            self.tree.delete(iid)

        self._setup_columns(self._cols)
        self._apply_alignment()
        self._apply_sortable_headings()

        for r in rows:
            self.tree.insert("", "end", values=[r.get(c, "") for c in self._cols])

        self.autofit_columns()

    def sort_by(self, col: str):
        asc = self._sort_dir.get(col, False)
        self._sort_dir[col] = not asc

        raw_key = f"__raw__{col}"

        def key_fn(r):
            if raw_key in r:
                v = r.get(raw_key)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    return float("-inf") if not asc else float("inf")
                return v
            return str(r.get(col, ""))

        rows = sorted(self._rows, key=key_fn, reverse=not asc)
        self.set_records(rows)


@dataclass(frozen=True)
class _FilterKey:
    day: str
    month: str
    call: str
    cp: str
    und: str


class CounterpartyView(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, style="TFrame")

        self._df: Optional[pd.DataFrame] = None
        self._prepared: bool = False
        self._cache: Dict[_FilterKey, Tuple[dict, pd.Series, List[dict]]] = {}

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
        self.cb_und = self._make_combo(self.filters_row, "UND", self.var_und, width=12)

        self.btn_apply = ttk.Button(self.filters_row, text="Apply", command=self.apply_filters)
        self.btn_reset = ttk.Button(self.filters_row, text="Reset", command=self.reset_filters)
        self.btn_apply.pack(side="left", padx=(14, 6))
        self.btn_reset.pack(side="left", padx=(0, 6))

        # KPIs row (forced single row)
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

        # Bottom area: chart + table (50/50)
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

        ttk.Label(self.chart_frame, text="Notional by Counterparty", style="Title.TLabel") \
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

        ttk.Label(self.table_frame, text="Counterparty summary (click headers to sort)", style="Title.TLabel") \
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

        # smaller title + smaller value so it fits
        lbl = ttk.Label(fr, text=f"{title}:", style="Muted.TLabel", font=("TkDefaultFont", 9))
        lbl.grid(row=0, column=0, sticky="w", padx=(6, 4), pady=5)

        val = ttk.Label(fr, text="—", style="KPI.TLabel", font=("TkDefaultFont", 10, "bold"))
        val.grid(row=0, column=1, sticky="w", padx=(0, 6), pady=5)

        return val

    # ---- render: no recompute on navigation
    def render(self, df: Optional[pd.DataFrame]):
        if df is self._df and self._prepared:
            return

        self._df = df
        self._cache.clear()
        self._prepared = False

        # NOTE: _Day/_Month/_Week now come from cleanData.py, so nothing heavy here.
        self._populate_filter_values()

        self._prepared = True
        self.reset_filters()

    def _populate_filter_values(self):
        if self._df is None or self._df.empty:
            for cb in (self.cb_day, self.cb_month, self.cb_call, self.cb_cp, self.cb_und):
                cb["values"] = ("ALL",)
            return

        d = self._df

        # Prefer precomputed columns from cleanData.py; fallback if missing
        if "_Day" not in d.columns or "_Month" not in d.columns:
            tt = pd.to_datetime(d["TradeTime"], errors="coerce")
            d["_Day"] = tt.dt.strftime("%Y-%m-%d").astype("string")
            d["_Month"] = tt.dt.strftime("%Y-%m").astype("string")

        days = ["ALL"] + sorted(pd.Series(d["_Day"]).dropna().unique().tolist())
        months = ["ALL"] + sorted(pd.Series(d["_Month"]).dropna().unique().tolist())
        calls = ["ALL"] + sorted(pd.Series(d["CALL_OPTION"]).dropna().astype(str).unique().tolist())
        cps = ["ALL"] + sorted(pd.Series(d["Counterparty"]).dropna().astype(str).unique().tolist())
        unds = ["ALL"] + sorted(pd.Series(d["UND_NAME"]).dropna().astype(str).unique().tolist())

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

    def _filtered_df(self) -> pd.DataFrame:
        if self._df is None or self._df.empty:
            return pd.DataFrame()

        d = self._df
        mask = pd.Series(True, index=d.index)

        if self.var_day.get() != "ALL":
            mask &= (d["_Day"] == self.var_day.get())
        if self.var_month.get() != "ALL":
            mask &= (d["_Month"] == self.var_month.get())
        if self.var_call.get() != "ALL":
            mask &= (d["CALL_OPTION"].astype(str) == self.var_call.get())
        if self.var_cp.get() != "ALL":
            mask &= (d["Counterparty"].astype(str) == self.var_cp.get())
        if self.var_und.get() != "ALL":
            mask &= (d["UND_NAME"].astype(str) == self.var_und.get())

        return d.loc[mask]

    def apply_filters(self):
        k = self._filter_key()
        if k in self._cache:
            kpis, cp_series, rows = self._cache[k]
        else:
            df_f = self._filtered_df()
            kpis = self._compute_kpis(df_f)
            cp_series = self._compute_cp_series(df_f)
            rows = self._compute_table_rows(df_f)
            self._cache[k] = (kpis, cp_series, rows)

        self._render_kpis(kpis)
        self._update_chart(cp_series)
        self.table.set_records(rows)

    def reset_filters(self):
        self.var_day.set("ALL")
        self.var_month.set("ALL")
        self.var_call.set("ALL")
        self.var_cp.set("ALL")
        self.var_und.set("ALL")
        self.apply_filters()

    def _compute_kpis(self, df_f: pd.DataFrame) -> dict:
        if df_f is None or df_f.empty:
            return {
                "total_notional": None,
                "total_trades": None,
                "cp_ratio_notional": None,
                "call_pct_trades": None,
                "bs_ratio_notional": None,
                "buy_pct_trades": None,
            }

        notional = pd.to_numeric(df_f.get("Notional"), errors="coerce").fillna(0.0)

        total_notional = float(notional.sum())
        n_trades = int(len(df_f))

        call_mask = df_f["CALL_OPTION"].astype(str) == "C"
        put_mask = df_f["CALL_OPTION"].astype(str) == "P"
        buy_mask = df_f["b/s"].astype(str) == "buy"
        sell_mask = df_f["b/s"].astype(str) == "sell"

        call_ntl = float(notional[call_mask].sum())
        put_ntl = float(notional[put_mask].sum())
        buy_ntl = float(notional[buy_mask].sum())
        sell_ntl = float(notional[sell_mask].sum())

        call_n = int(call_mask.sum())
        put_n = int(put_mask.sum())
        buy_n = int(buy_mask.sum())
        sell_n = int(sell_mask.sum())

        return {
            "total_notional": total_notional,
            "total_trades": n_trades,
            "cp_ratio_notional": _safe_ratio(call_ntl, put_ntl),
            "call_pct_trades": _safe_pct(call_n, call_n + put_n),
            "bs_ratio_notional": _safe_ratio(buy_ntl, sell_ntl),
            "buy_pct_trades": _safe_pct(buy_n, buy_n + sell_n),
        }

    def _render_kpis(self, k: dict):
        if not k or k["total_notional"] is None:
            self.kpi_total_notional.config(text="—")
            self.kpi_total_trades.config(text="—")
            self.kpi_cp_notional.config(text="—")
            self.kpi_cp_trades.config(text="—")
            self.kpi_bs_notional.config(text="—")
            self.kpi_bs_trades.config(text="—")
            return

        self.kpi_total_notional.config(text=_comma(k["total_notional"]))
        self.kpi_total_trades.config(text=f"{int(k['total_trades']):,}")
        self.kpi_cp_notional.config(text=_ratio_to_str(k["cp_ratio_notional"]))
        self.kpi_cp_trades.config(text=_pct_to_str(k["call_pct_trades"]))
        self.kpi_bs_notional.config(text=_ratio_to_str(k["bs_ratio_notional"]))
        self.kpi_bs_trades.config(text=_pct_to_str(k["buy_pct_trades"]))

    def _compute_cp_series(self, df_f: pd.DataFrame) -> pd.Series:
        if df_f is None or df_f.empty:
            return pd.Series(dtype="float64")

        notional = pd.to_numeric(df_f.get("Notional"), errors="coerce").fillna(0.0)
        s = df_f.assign(_Notional=notional).groupby("Counterparty")["_Notional"].sum()
        return s.sort_values(ascending=False)

    def _on_chart_resize(self, event):
        if self._last_chart_series is None:
            return
        w_px = max(200, int(event.width))
        h_px = max(200, int(event.height - 30))
        dpi = 100
        self._fig.set_size_inches(w_px / dpi, h_px / dpi, forward=True)
        self._fig.tight_layout()
        self._mpl_canvas.draw_idle()

    def _update_chart(self, cp_series: pd.Series):
        self._last_chart_series = cp_series

        ax = self._ax
        ax.clear()
        ax.set_xlabel("Counterparty")
        ax.set_ylabel("Notional")
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_km))

        if cp_series is None or cp_series.empty:
            self._mpl_canvas.draw_idle()
            return

        x = cp_series.index.astype(str).tolist()
        y = cp_series.values.tolist()

        n = len(x)
        if n <= 20:
            cmap = cm.get_cmap("tab20", n)
            colors = [cmap(i) for i in range(n)]
        else:
            cmap = cm.get_cmap("hsv", n)
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

    def _compute_table_rows(self, df_f: pd.DataFrame) -> List[dict]:
        if df_f is None or df_f.empty:
            return []

        notional = pd.to_numeric(df_f.get("Notional"), errors="coerce").fillna(0.0)
        dd = df_f.assign(_Notional=notional)

        grp = dd.groupby("Counterparty").agg(
            Notional=("_Notional", "sum"),
            Trades=("TradeNo", "count"),
            CallNotional=("_Notional", lambda s: float(s[dd.loc[s.index, "CALL_OPTION"].astype(str) == "C"].sum())),
            PutNotional=("_Notional", lambda s: float(s[dd.loc[s.index, "CALL_OPTION"].astype(str) == "P"].sum())),
            BuyNotional=("_Notional", lambda s: float(s[dd.loc[s.index, "b/s"].astype(str) == "buy"].sum())),
            SellNotional=("_Notional", lambda s: float(s[dd.loc[s.index, "b/s"].astype(str) == "sell"].sum())),
            CallTrades=("CALL_OPTION", lambda s: int((s.astype(str) == "C").sum())),
            PutTrades=("CALL_OPTION", lambda s: int((s.astype(str) == "P").sum())),
            BuyTrades=("b/s", lambda s: int((s.astype(str) == "buy").sum())),
            SellTrades=("b/s", lambda s: int((s.astype(str) == "sell").sum())),
        )

        grp["CP$R"] = grp.apply(lambda r: _safe_ratio(r["CallNotional"], r["PutNotional"]), axis=1)
        grp["Call%#"] = grp.apply(lambda r: _safe_pct(r["CallTrades"], r["CallTrades"] + r["PutTrades"]), axis=1)
        grp["BS$R"] = grp.apply(lambda r: _safe_ratio(r["BuyNotional"], r["SellNotional"]), axis=1)
        grp["Buy%#"] = grp.apply(lambda r: _safe_pct(r["BuyTrades"], r["BuyTrades"] + r["SellTrades"]), axis=1)

        grp = grp.sort_values("Notional", ascending=False)

        rows: List[dict] = []
        for cp, r in grp.iterrows():
            rows.append({
                "CP": str(cp),
                "Notional": _comma(float(r["Notional"])),
                "Trades": f"{int(r['Trades']):,}",
                "CP$R": _ratio_to_str(r["CP$R"]),
                "Call%#": _pct_to_str(r["Call%#"]),
                "BS$R": _ratio_to_str(r["BS$R"]),
                "Buy%#": _pct_to_str(r["Buy%#"]),
                "__raw__CP": str(cp),
                "__raw__Notional": float(r["Notional"]),
                "__raw__Trades": int(r["Trades"]),
                "__raw__CP$R": (float(r["CP$R"]) if (r["CP$R"] is not None and not math.isinf(r["CP$R"])) else None),
                "__raw__Call%#": (float(r["Call%#"]) if r["Call%#"] is not None else None),
                "__raw__BS$R": (float(r["BS$R"]) if (r["BS$R"] is not None and not math.isinf(r["BS$R"])) else None),
                "__raw__Buy%#": (float(r["Buy%#"]) if r["Buy%#"] is not None else None),
            })

        return rows
