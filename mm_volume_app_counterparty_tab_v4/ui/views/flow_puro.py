# ui/views/flow_puro.py
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from ui.widgets.table import DataTable


class FitTable(DataTable):
    """No horizontal scroll; columns always fit available width."""
    def __init__(self, master, columns, height=18):
        super().__init__(master, columns=columns, height=height)

        # remove horizontal scroll
        try:
            self.hsb.grid_remove()
        except Exception:
            pass
        try:
            self.tree.configure(xscrollcommand="")
        except Exception:
            pass

        self._cols = list(columns)
        self.autofit_columns()
        self.tree.bind("<Configure>", lambda e: self.autofit_columns())

        # alignment: text left, numbers right
        self.tree.column("Period", anchor="w")
        self.tree.column("Total Flow", anchor="e")
        self.tree.column("Total Notional", anchor="e")
        self.tree.column("Total Trades", anchor="e")

    def autofit_columns(self):
        w = self.tree.winfo_width()
        if w <= 80:
            return

        avail = max(280, w - 18)  # 18px ~ scrollbar vertical
        weights = {c: 1.0 for c in self._cols}
        weights["Period"] = 1.05
        weights["Total Flow"] = 1.15
        weights["Total Notional"] = 1.20
        weights["Total Trades"] = 0.80

        sw = sum(weights.get(c, 1.0) for c in self._cols)
        for c in self._cols:
            cw = int(avail * (weights.get(c, 1.0) / sw))
            cw = max(80, cw)
            if c == "Period":
                cw = max(95, cw)
            self.tree.column(c, width=cw, stretch=True)


class FlowPuroView(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, style="TFrame")

        self._flow_puro_pack = None
        self._cache: dict[tuple, pd.DataFrame] = {}

        self.period_var = tk.StringVar(value="Daily")  # Daily/Weekly/Monthly

        self.filter_vars: dict[str, tk.StringVar] = {}
        self.filter_boxes: dict[str, ttk.Combobox] = {}

        # KPI labels
        self.lbl_notional = None
        self.lbl_flow = None
        self.lbl_trades = None
        self.lbl_callpct = None
        self.lbl_buypct = None

        self._build()

    # ---------- formatting ----------

    def _fmt_kmb(self, x, _pos=None):
        try:
            x = float(x)
        except Exception:
            return "0"
        ax = abs(x)
        if ax >= 1e9:
            return f"{x/1e9:.2f}B"
        if ax >= 1e6:
            return f"{x/1e6:.2f}M"
        if ax >= 1e3:
            return f"{x/1e3:.2f}k"
        return f"{x:.0f}"

    def _build_kpi(self, parent, title: str):
        # NOTE: compact padding so 5 KPIs always fit on one line
        box = ttk.Frame(parent, style="Card.TFrame")
        box.grid_propagate(True)

        lbl = ttk.Label(box, text=f"{title}:", style="Muted.TLabel")
        lbl.grid(row=0, column=0, sticky="w", padx=(8, 4), pady=4)

        val = ttk.Label(box, text="—", style="KPI.TLabel")
        val.grid(row=0, column=1, sticky="e", padx=(0, 8), pady=4)

        box.columnconfigure(0, weight=1)
        box.columnconfigure(1, weight=0)

        return box, val

    # ---------- UI ----------

    def _build(self):
        header = ttk.Label(self, text="Flow Puro", style="Title.TLabel")
        header.pack(anchor="w", padx=14, pady=(12, 6))

        # Filters row (single line)
        self.frm_filters = ttk.Frame(self, style="TFrame")
        self.frm_filters.pack(fill="x", padx=14, pady=(0, 6))

        row1 = ttk.Frame(self.frm_filters, style="TFrame")
        row1.pack(fill="x")

        period_box = ttk.LabelFrame(row1, text="Period", style="TLabelframe")
        period_box.pack(side="left", padx=(0, 10), pady=6)

        for g, lab in (("Daily", "Day"), ("Weekly", "Week"), ("Monthly", "Month")):
            ttk.Radiobutton(period_box, text=lab, value=g, variable=self.period_var).pack(
                side="left", padx=6, pady=6
            )

        self.frm_dd = ttk.Frame(row1, style="TFrame")
        self.frm_dd.pack(side="left", fill="x", expand=True)

        btns = ttk.Frame(row1, style="TFrame")
        btns.pack(side="right", padx=(10, 0), pady=6)
        self.btn_apply = ttk.Button(btns, text="Apply", command=self._on_apply, width=10)
        self.btn_reset = ttk.Button(btns, text="Reset", command=self._on_reset, width=10)
        self.btn_apply.pack(side="left", padx=(0, 8))
        self.btn_reset.pack(side="left")

        # KPIs row: grid so it *always* fits available width
        self.frm_kpis = ttk.Frame(self, style="TFrame")
        self.frm_kpis.pack(fill="x", padx=14, pady=(0, 6))

        # each KPI gets 1/5 of width
        for i in range(5):
            self.frm_kpis.columnconfigure(i, weight=1, uniform="kpi")

        k1, self.lbl_notional = self._build_kpi(self.frm_kpis, "Notional")
        k2, self.lbl_flow = self._build_kpi(self.frm_kpis, "Flow")
        k3, self.lbl_trades = self._build_kpi(self.frm_kpis, "Trades")
        k4, self.lbl_callpct = self._build_kpi(self.frm_kpis, "Call %")
        k5, self.lbl_buypct = self._build_kpi(self.frm_kpis, "Buy %")

        k1.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=4)
        k2.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=4)
        k3.grid(row=0, column=2, sticky="nsew", padx=(0, 8), pady=4)
        k4.grid(row=0, column=3, sticky="nsew", padx=(0, 8), pady=4)
        k5.grid(row=0, column=4, sticky="nsew", pady=4)

        # Main content: plot + table (plot wider)
        content = ttk.Frame(self, style="TFrame")
        content.pack(fill="both", expand=True, padx=14, pady=(6, 14))
        content.columnconfigure(0, weight=1)  # PLOT wider
        content.columnconfigure(1, weight=4)  # TABLE narrower
        content.rowconfigure(0, weight=1)

        plot_frame = ttk.Frame(content, style="TFrame")
        plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(7.6, 4.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        table_frame = ttk.Frame(content, style="TFrame")
        table_frame.grid(row=0, column=1, sticky="nsew")

        cols = ["Period", "Total Flow", "Total Notional", "Total Trades"]
        self.table = FitTable(table_frame, columns=cols, height=18)
        self.table.pack(fill="both", expand=True)

    # ---------- public ----------

    def render(self, df_clean: pd.DataFrame | None, flow_puro_pack: dict | None = None):
        self._flow_puro_pack = flow_puro_pack
        self._cache.clear()
        self._build_dropdowns()
        self._apply_and_refresh(use_cache=True)

    # ---------- dropdowns ----------

    def _build_dropdowns(self):
        for w in self.frm_dd.winfo_children():
            w.destroy()

        self.filter_vars.clear()
        self.filter_boxes.clear()

        pack = self._flow_puro_pack
        if not pack:
            return

        values = pack.get("values", {})

        mapping = []
        if "CALL_OPTION" in values:
            mapping.append(("C/P:", "CALL_OPTION"))
        if "Counterparty" in values:
            mapping.append(("CP:", "Counterparty"))
        elif "CP" in values:
            mapping.append(("CP:", "CP"))
        if "UND_NAME" in values:
            mapping.append(("UND:", "UND_NAME"))
        elif "UND" in values:
            mapping.append(("UND:", "UND"))

        for (label, col) in mapping:
            var = tk.StringVar(value="ALL")
            self.filter_vars[col] = var

            box = ttk.Frame(self.frm_dd, style="TFrame")
            box.pack(side="left", padx=(0, 12), pady=6)

            ttk.Label(box, text=label, style="Muted.TLabel").pack(side="left", padx=(0, 6))

            cb = ttk.Combobox(
                box,
                textvariable=var,
                state="readonly",
                width=8 if col == "CALL_OPTION" else 14,
                values=["ALL"] + list(values.get(col, [])),
            )
            cb.pack(side="left")
            self.filter_boxes[col] = cb

    # ---------- actions ----------

    def _on_apply(self):
        self._apply_and_refresh(use_cache=True)

    def _on_reset(self):
        self.period_var.set("Daily")
        for var in self.filter_vars.values():
            var.set("ALL")
        self._apply_and_refresh(use_cache=True)

    # ---------- compute ----------

    def _selection_key(self) -> tuple:
        items = [("Period", self.period_var.get())]
        for k in sorted(self.filter_vars.keys()):
            items.append((k, self.filter_vars[k].get()))
        return tuple(items)

    def _apply_filters(self, base: pd.DataFrame) -> pd.DataFrame:
        df = base
        for col, var in self.filter_vars.items():
            v = var.get()
            if v and v != "ALL" and col in df.columns:
                df = df[df[col] == v]
        return df

    def _compute_agg(self) -> pd.DataFrame:
        pack = self._flow_puro_pack
        if not pack:
            return pd.DataFrame(columns=["Period", "Total Flow", "Total Notional", "Total Trades"])

        period = self.period_var.get()
        pre_aggs = pack.get("aggs", {})
        base = pack.get("base")

        all_all = all(v.get() == "ALL" for v in self.filter_vars.values())
        if all_all and period in pre_aggs:
            return pre_aggs[period].copy()

        if base is None or len(base) == 0:
            return pd.DataFrame(columns=["Period", "Total Flow", "Total Notional", "Total Trades"])

        gcol = {"Daily": "_Day", "Weekly": "_Week", "Monthly": "_Month"}[period]
        if gcol not in base.columns:
            return pd.DataFrame(columns=["Period", "Total Flow", "Total Notional", "Total Trades"])

        df = self._apply_filters(base)
        if df.empty:
            return pd.DataFrame(columns=["Period", "Total Flow", "Total Notional", "Total Trades"])

        out = (
            df.groupby(gcol, as_index=False, observed=False)[["Flow", "Notional", "_Trades"]]
            .sum()
            .rename(
                columns={
                    gcol: "Period",
                    "Flow": "Total Flow",
                    "Notional": "Total Notional",
                    "_Trades": "Total Trades",
                }
            )
            .sort_values("Period")
            .reset_index(drop=True)
        )
        return out

    def _apply_and_refresh(self, use_cache: bool):
        key = self._selection_key()

        if use_cache and key in self._cache:
            agg = self._cache[key].copy()
        else:
            agg = self._compute_agg()
            if use_cache:
                self._cache[key] = agg.copy()

        self._refresh_kpis(agg)
        self._refresh_plot(agg)
        self._refresh_table(agg)

    # ---------- render parts ----------

    def _refresh_kpis(self, agg: pd.DataFrame):
        if agg is None or agg.empty:
            self.lbl_notional.config(text="0")
            self.lbl_flow.config(text="0")
            self.lbl_trades.config(text="0")
            self.lbl_callpct.config(text="0%")
            self.lbl_buypct.config(text="0%")
            return

        tot_notional = float(agg["Total Notional"].sum())
        tot_flow = float(agg["Total Flow"].sum())
        tot_trades = float(agg["Total Trades"].sum())

        call_pct = 0.0
        buy_pct = 0.0
        pack = self._flow_puro_pack
        if pack and pack.get("base") is not None:
            base = pack["base"]
            df = self._apply_filters(base)

            if "CALL_OPTION" in df.columns and len(df) > 0:
                co = df["CALL_OPTION"].astype(str).str.strip().str.upper()
                call_pct = 100.0 * float((co == "C").mean())

            if "b/s" in df.columns and len(df) > 0:
                bs = df["b/s"].astype(str).str.strip().str.lower()
                buy_pct = 100.0 * float((bs == "buy").mean())

        self.lbl_notional.config(text=self._fmt_kmb(tot_notional))
        self.lbl_flow.config(text=self._fmt_kmb(tot_flow))
        self.lbl_trades.config(text=f"{int(round(tot_trades)):,}")
        self.lbl_callpct.config(text=f"{call_pct:.0f}%")
        self.lbl_buypct.config(text=f"{buy_pct:.0f}%")

    def _refresh_plot(self, agg: pd.DataFrame):
        self.ax.clear()

        if agg is None or agg.empty:
            self.ax.set_title("No data")
            self.canvas.draw()
            return

        y = agg["Total Flow"].to_numpy(dtype=float)
        x = np.arange(len(agg))

        colors = np.where(y >= 0, "green", "red")
        self.ax.bar(x, y, color=colors)
        self.ax.axhline(0.0, linewidth=1.0)

        cum = np.cumsum(y)
        self.ax.plot(x, cum, linewidth=1.6)

        self.ax.set_title(f"Flow by Period ({self.period_var.get()})")
        self.ax.set_ylabel("Flow")
        self.ax.yaxis.set_major_formatter(FuncFormatter(self._fmt_kmb))

        n = len(x)
        if n <= 0:
            self.ax.set_xticks([])
        elif n == 1:
            self.ax.set_xticks([0])
            self.ax.set_xticklabels([str(agg["Period"].iloc[0])])
        else:
            idxs = [0]
            if n >= 4:
                idxs += [n // 3, (2 * n) // 3]
            idxs += [n - 1]
            idxs = sorted(set(max(0, min(n - 1, i)) for i in idxs))
            self.ax.set_xticks(idxs)
            self.ax.set_xticklabels([str(agg["Period"].iloc[i]) for i in idxs], rotation=0)

        self.canvas.draw()

    def _refresh_table(self, agg: pd.DataFrame):
        if agg is None or agg.empty:
            self.table.set_dataframe(pd.DataFrame(columns=["Period", "Total Flow", "Total Notional", "Total Trades"]))
            return

        df = agg.copy()
        df["Total Flow"] = df["Total Flow"].map(lambda v: self._fmt_kmb(v))
        df["Total Notional"] = df["Total Notional"].map(lambda v: self._fmt_kmb(v))
        df["Total Trades"] = df["Total Trades"].map(lambda v: f"{int(round(float(v))):,}")

        self.table.set_dataframe(df[["Period", "Total Flow", "Total Notional", "Total Trades"]])

        try:
            self.table.autofit_columns()
        except Exception:
            pass
