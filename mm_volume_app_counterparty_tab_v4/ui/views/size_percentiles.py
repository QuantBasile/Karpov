# ui/views/size_percentiles.py
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ui.widgets.table import DataTable


class FitTable(DataTable):
    """
    No horizontal scroll; columns always fit available width.
    First column left aligned, others right aligned.
    """
    def __init__(self, master, columns, height=14):
        super().__init__(master, columns=columns, height=height)
        try:
            self.hsb.grid_remove()
        except Exception:
            pass
        try:
            self.tree.configure(xscrollcommand="")
        except Exception:
            pass

        self._cols = list(columns)
        self._apply_alignment()

        self.autofit_columns()
        self.tree.bind("<Configure>", lambda _e: self.autofit_columns())

    def _apply_alignment(self):
        if not self._cols:
            return
        first = self._cols[0]
        for c in self._cols:
            anchor = "w" if c == first else "e"
            try:
                self.tree.column(c, anchor=anchor)
            except Exception:
                pass

    def _setup_columns(self, cols):
        super()._setup_columns(cols)
        self._cols = list(cols)
        self._apply_alignment()

    def autofit_columns(self):
        w = self.tree.winfo_width()
        if w <= 80:
            return

        avail = max(320, w - 18)
        weights = {c: 1.0 for c in self._cols}

        if "Bin" in weights: weights["Bin"] = 1.15
        if "Range" in weights: weights["Range"] = 1.35
        if "%Flow" in weights: weights["%Flow"] = 0.9
        if "%Trades" in weights: weights["%Trades"] = 0.9
        if "Period" in weights: weights["Period"] = 1.05

        sw = sum(weights.get(c, 1.0) for c in self._cols)
        for c in self._cols:
            cw = int(avail * (weights.get(c, 1.0) / sw))
            cw = max(80, cw)
            if c in ("Bin", "Period"):
                cw = max(110, cw)
            try:
                self.tree.column(c, width=cw, stretch=True)
            except Exception:
                pass

        self._apply_alignment()


class SizePercentilesView(ttk.Frame):
    """
    Size / Percentiles (3 subtabs):
      1) Distribution: histogram (left) + histogram table (right) 50/50
      2) Contribution: contribution plot (left) + bins table (right) 50/50
      3) Heatmap: ONLY heatmap full width (no table)

    Top controls are global and apply everywhere.
    """

    def __init__(self, master):
        super().__init__(master, style="TFrame")

        self._pack = None
        self._cache: dict[tuple, dict] = {}

        # controls
        self.period_var = tk.StringVar(value="Daily")         # for heatmap columns
        self.size_field_var = tk.StringVar(value="Notional")  # Notional | Quantity
        self.scheme_var = tk.StringVar(value="Whales")        # Whales | Deciles | Quartiles | Custom
        self.custom_edges_var = tk.StringVar(value="0,50,80,90,95,99,100")
        self.metric_var = tk.StringVar(value="Net Flow")      # Net Flow | Notional | Trades

        self.normalize_var = tk.BooleanVar(value=False)       # show % on contribution
        self.quantiles_after_filters_var = tk.BooleanVar(value=False)
        self.log_hist_var = tk.BooleanVar(value=True)
        self.top_trades_n_var = tk.IntVar(value=25)

        # filters
        self.filter_vars: dict[str, tk.StringVar] = {}
        self.filter_boxes: dict[str, ttk.Combobox] = {}

        # KPIs (only 3)
        self.lbl_total_trades = None
        self.lbl_total_notional = None
        self.lbl_total_flow = None

        # Heatmap fixed colorbar axis
        self.ax_heat_cbar = None

        self._build()

    # ---------- formatting ----------

    def _fmt_kmb(self, x, _pos=None):
        try:
            x = float(x)
        except Exception:
            return "0"
        ax = abs(x)
        if ax >= 1e9: return f"{x/1e9:.2f}B"
        if ax >= 1e6: return f"{x/1e6:.2f}M"
        if ax >= 1e3: return f"{x/1e3:.2f}k"
        return f"{x:.0f}"

    def _fmt_pct(self, x):
        try:
            return f"{float(x):.1f}%"
        except Exception:
            return "—"

    def _build_kpi(self, parent, title: str):
        box = ttk.Frame(parent, style="Card.TFrame")
        ttk.Label(box, text=f"{title}:", style="Muted.TLabel").grid(row=0, column=0, sticky="w", padx=(8, 4), pady=4)
        val = ttk.Label(box, text="—", style="KPI.TLabel")
        val.grid(row=0, column=1, sticky="e", padx=(0, 8), pady=4)
        box.columnconfigure(0, weight=1)
        box.columnconfigure(1, weight=0)
        return box, val

    # ---------- UI ----------

    def _build(self):
        ttk.Label(self, text="Size / Percentiles", style="Title.TLabel").pack(anchor="w", padx=14, pady=(12, 6))

        # Filters row
        self.frm_filters = ttk.Frame(self, style="TFrame")
        self.frm_filters.pack(fill="x", padx=14, pady=(0, 6))

        row1 = ttk.Frame(self.frm_filters, style="TFrame")
        row1.pack(fill="x")

        period_box = ttk.LabelFrame(row1, text="Period (heatmap)", style="TLabelframe")
        period_box.pack(side="left", padx=(0, 10), pady=6)
        for g, lab in (("Daily", "Day"), ("Weekly", "Week"), ("Monthly", "Month")):
            ttk.Radiobutton(period_box, text=lab, value=g, variable=self.period_var, command=self._on_apply).pack(
                side="left", padx=6, pady=6
            )

        self.frm_dd = ttk.Frame(row1, style="TFrame")
        self.frm_dd.pack(side="left", fill="x", expand=True)

        btns = ttk.Frame(row1, style="TFrame")
        btns.pack(side="right", padx=(10, 0), pady=6)
        ttk.Button(btns, text="Apply", command=self._on_apply, width=10).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Reset", command=self._on_reset, width=10).pack(side="left")

        # Controls row
        ctrl = ttk.Frame(self, style="TFrame")
        ctrl.pack(fill="x", padx=14, pady=(0, 6))

        lf = ttk.LabelFrame(ctrl, text="Size bins", style="TLabelframe")
        lf.pack(side="left", padx=(0, 10), pady=6)

        ttk.Label(lf, text="Field:", style="Muted.TLabel").pack(side="left", padx=(8, 4), pady=6)
        ttk.Radiobutton(lf, text="Notional", value="Notional", variable=self.size_field_var, command=self._on_apply).pack(side="left", padx=6)
        ttk.Radiobutton(lf, text="Quantity", value="Quantity", variable=self.size_field_var, command=self._on_apply).pack(side="left", padx=6)

        ttk.Label(lf, text="Scheme:", style="Muted.TLabel").pack(side="left", padx=(10, 4), pady=6)
        self.cmb_scheme = ttk.Combobox(lf, textvariable=self.scheme_var, state="readonly", width=10,
                                       values=["Whales", "Deciles", "Quartiles", "Custom"])
        self.cmb_scheme.pack(side="left", padx=(0, 6), pady=6)
        self.cmb_scheme.bind("<<ComboboxSelected>>", lambda _e: self._on_scheme_changed())

        self.ent_custom = ttk.Entry(lf, textvariable=self.custom_edges_var, width=24)
        self.ent_custom.pack(side="left", padx=(0, 8), pady=6)

        rf = ttk.LabelFrame(ctrl, text="Explain metric", style="TLabelframe")
        rf.pack(side="left", padx=(0, 10), pady=6)

        ttk.Label(rf, text="Metric:", style="Muted.TLabel").pack(side="left", padx=(8, 4), pady=6)
        self.cmb_metric = ttk.Combobox(rf, textvariable=self.metric_var, state="readonly", width=12,
                                       values=["Net Flow", "Notional", "Trades"])
        self.cmb_metric.pack(side="left", padx=(0, 10), pady=6)
        self.cmb_metric.bind("<<ComboboxSelected>>", lambda _e: self._on_apply())

        ttk.Checkbutton(rf, text="Show %", variable=self.normalize_var, command=self._on_apply).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(rf, text="Quantiles after filters", variable=self.quantiles_after_filters_var, command=self._on_apply).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(rf, text="Log hist", variable=self.log_hist_var, command=self._on_apply).pack(side="left", padx=(0, 10))

        ttk.Label(rf, text="Top trades:", style="Muted.TLabel").pack(side="left", padx=(10, 4))
        ttk.Spinbox(rf, from_=10, to=200, textvariable=self.top_trades_n_var, width=5, command=self._on_apply).pack(
            side="left", padx=(0, 8)
        )

        # KPIs row (only 3)
        self.frm_kpis = ttk.Frame(self, style="TFrame")
        self.frm_kpis.pack(fill="x", padx=14, pady=(0, 6))
        for i in range(3):
            self.frm_kpis.columnconfigure(i, weight=1, uniform="kpi")

        k1, self.lbl_total_trades = self._build_kpi(self.frm_kpis, "Total Trades")
        k2, self.lbl_total_notional = self._build_kpi(self.frm_kpis, "Total Notional")
        k3, self.lbl_total_flow = self._build_kpi(self.frm_kpis, "Total Net Flow")
        k1.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=4)
        k2.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=4)
        k3.grid(row=0, column=2, sticky="nsew", pady=4)

        # Notebook (3 subtabs)
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=14, pady=(6, 14))
        self.nb.bind("<<NotebookTabChanged>>", lambda _e: self._on_tab_changed())

        self.tab_dist = ttk.Frame(self.nb, style="TFrame")
        self.tab_contrib = ttk.Frame(self.nb, style="TFrame")
        self.tab_heat = ttk.Frame(self.nb, style="TFrame")

        self.nb.add(self.tab_dist, text="Distribution")
        self.nb.add(self.tab_contrib, text="Contribution")
        self.nb.add(self.tab_heat, text="Heatmap")

        self._build_distribution_tab()
        self._build_contribution_tab()
        self._build_heatmap_tab()

        self._on_scheme_changed(init=True)

    def _build_distribution_tab(self):
        # enforce 50/50 always via uniform columns
        self.tab_dist.columnconfigure(0, weight=1, uniform="half")
        self.tab_dist.columnconfigure(1, weight=1, uniform="half")
        self.tab_dist.rowconfigure(0, weight=1)

        self.fig_hist = Figure(figsize=(6.0, 3.8), dpi=100)
        self.ax_hist = self.fig_hist.add_subplot(111)
        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, master=self.tab_dist)
        w = self.canvas_hist.get_tk_widget()
        w.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)

        cols = ["Bin", "From", "To", "Trades", "%Trades"]
        self.tbl_hist = FitTable(self.tab_dist, columns=cols, height=18)
        self.tbl_hist.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)

    def _build_contribution_tab(self):
        self.tab_contrib.columnconfigure(0, weight=1, uniform="half")
        self.tab_contrib.columnconfigure(1, weight=1, uniform="half")
        self.tab_contrib.rowconfigure(0, weight=1)

        self.fig_bins = Figure(figsize=(6.0, 3.8), dpi=100)
        self.ax_bins = self.fig_bins.add_subplot(111)
        self.canvas_bins = FigureCanvasTkAgg(self.fig_bins, master=self.tab_contrib)
        w = self.canvas_bins.get_tk_widget()
        w.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)

        cols = ["Bin", "Range", "Trades", "%Trades", "Notional", "Flow", "%Flow"]
        self.tbl_bins = FitTable(self.tab_contrib, columns=cols, height=18)
        self.tbl_bins.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)

    def _build_heatmap_tab(self):
        # heatmap only, full width
        self.tab_heat.columnconfigure(0, weight=1)
        self.tab_heat.rowconfigure(0, weight=1)

        self.fig_heat = Figure(figsize=(10.0, 3.8), dpi=100)
        self.ax_heat = self.fig_heat.add_subplot(111)
        divider = make_axes_locatable(self.ax_heat)
        self.ax_heat_cbar = divider.append_axes("right", size="2.8%", pad=0.10)

        self.canvas_heat = FigureCanvasTkAgg(self.fig_heat, master=self.tab_heat)
        w = self.canvas_heat.get_tk_widget()
        w.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

    # ---------- public ----------

    def render(self, df_clean: pd.DataFrame | None, size_percentiles_pack: dict | None = None):
        self._pack = size_percentiles_pack
        self._cache.clear()
        self._build_dropdowns()
        self._apply_and_refresh(use_cache=True, force_draw=True)

    # ---------- dropdowns ----------

    def _build_dropdowns(self):
        for w in self.frm_dd.winfo_children():
            w.destroy()
        self.filter_vars.clear()
        self.filter_boxes.clear()

        pack = self._pack
        if not pack:
            return

        values = pack.get("values", {})
        mapping = []
        if "CALL_OPTION" in values: mapping.append(("C/P:", "CALL_OPTION"))
        if "b/s" in values: mapping.append(("B/S:", "b/s"))
        if "Counterparty" in values: mapping.append(("CP:", "Counterparty"))
        if "UND_NAME" in values: mapping.append(("UND:", "UND_NAME"))

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
                width=8 if col in ("CALL_OPTION", "b/s") else 14,
                values=["ALL"] + list(values.get(col, [])),
            )
            cb.pack(side="left")
            cb.bind("<<ComboboxSelected>>", lambda _e: self._on_apply())
            self.filter_boxes[col] = cb

    # ---------- events ----------

    def _on_scheme_changed(self, init: bool = False):
        is_custom = (self.scheme_var.get() == "Custom")
        self.ent_custom.configure(state=("normal" if is_custom else "disabled"))
        if not init:
            self._on_apply()

    def _on_tab_changed(self):
        # Ensure plots are correct when arriving at a tab (hidden canvases sometimes don't redraw properly)
        self._apply_and_refresh(use_cache=True, force_draw=True)

    def _on_apply(self):
        self._apply_and_refresh(use_cache=True, force_draw=False)

    def _on_reset(self):
        self.period_var.set("Daily")
        self.size_field_var.set("Notional")
        self.scheme_var.set("Whales")
        self.metric_var.set("Net Flow")
        self.normalize_var.set(False)
        self.quantiles_after_filters_var.set(False)
        self.log_hist_var.set(True)
        self.top_trades_n_var.set(25)

        for var in self.filter_vars.values():
            var.set("ALL")

        self._on_scheme_changed()
        self._apply_and_refresh(use_cache=True, force_draw=True)

    # ---------- compute ----------

    def _selection_key(self) -> tuple:
        items = [
            ("Period", self.period_var.get()),
            ("SizeField", self.size_field_var.get()),
            ("Scheme", self.scheme_var.get()),
            ("Custom", self.custom_edges_var.get()),
            ("Metric", self.metric_var.get()),
            ("Norm", "1" if self.normalize_var.get() else "0"),
            ("QAfter", "1" if self.quantiles_after_filters_var.get() else "0"),
            ("LogHist", "1" if self.log_hist_var.get() else "0"),
            ("TopN", int(self.top_trades_n_var.get())),
        ]
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

    def _get_edges(self, df: pd.DataFrame) -> np.ndarray:
        field = self.size_field_var.get()
        scheme = self.scheme_var.get()

        if scheme == "Custom":
            try:
                pts = [float(x.strip()) for x in self.custom_edges_var.get().split(",") if x.strip() != ""]
                pts = [p for p in pts if 0.0 <= p <= 100.0]
                pts = sorted(set(pts))
                if not pts:
                    pts = [0, 50, 80, 90, 95, 99, 100]
                if pts[0] != 0.0: pts = [0.0] + pts
                if pts[-1] != 100.0: pts = pts + [100.0]
                probs = [p / 100.0 for p in pts]
            except Exception:
                probs = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
        else:
            probs = None
            if self._pack and "edges" in self._pack:
                probs = self._pack["edges"]["schemes"].get(scheme)
            if probs is None:
                probs = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]

        s = pd.to_numeric(df["_AbsNotional"], errors="coerce") if field == "Notional" else pd.to_numeric(df["_AbsQty"], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) == 0:
            return np.array([0.0, 1.0], dtype=float)

        if self.quantiles_after_filters_var.get() or (scheme == "Custom"):
            edges = np.quantile(s.to_numpy(dtype=float), probs)
        else:
            if self._pack and "edges" in self._pack and scheme != "Custom":
                edges = self._pack["edges"][field].get(scheme)
                if edges is None:
                    edges = np.quantile(s.to_numpy(dtype=float), probs)
            else:
                edges = np.quantile(s.to_numpy(dtype=float), probs)

        edges = np.asarray(edges, dtype=float)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-12
        return edges

    def _compute_hist_table(self, sizes: np.ndarray) -> pd.DataFrame:
        sizes = sizes[np.isfinite(sizes)]
        sizes = sizes[sizes >= 0]
        if len(sizes) == 0:
            return pd.DataFrame(columns=["Bin", "From", "To", "Trades", "%Trades"])

        if self.log_hist_var.get():
            s2 = sizes[sizes > 0]
            if len(s2) == 0:
                return pd.DataFrame(columns=["Bin", "From", "To", "Trades", "%Trades"])
            lo, hi = np.percentile(s2, 1), np.percentile(s2, 99.5)
            lo = max(lo, 1e-12)
            edges = np.logspace(np.log10(lo), np.log10(max(hi, lo * 10)), 21)
            counts, _ = np.histogram(s2, bins=edges)
        else:
            edges = np.linspace(np.min(sizes), np.max(sizes), 21)
            counts, _ = np.histogram(sizes, bins=edges)

        total = counts.sum() if counts.sum() > 0 else 1
        rows = []
        for i in range(len(counts)):
            a = float(edges[i])
            b = float(edges[i + 1])
            c = int(counts[i])
            rows.append({
                "Bin": f"{i+1}",
                "From": self._fmt_kmb(a),
                "To": self._fmt_kmb(b),
                "Trades": f"{c:,}",
                "%Trades": self._fmt_pct(100.0 * c / total),
            })
        return pd.DataFrame(rows)

    def _compute(self) -> dict:
        pack = self._pack
        if not pack or pack.get("base") is None:
            return {"df": pd.DataFrame(), "bins": pd.DataFrame(), "heat": None, "hist_table": pd.DataFrame()}

        base = pack["base"]
        df = self._apply_filters(base)
        if df.empty:
            return {"df": df, "bins": pd.DataFrame(), "heat": None, "hist_table": pd.DataFrame()}

        edges = self._get_edges(df)

        # sizes
        if self.size_field_var.get() == "Notional":
            sizes = pd.to_numeric(df["_AbsNotional"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            sizes = pd.to_numeric(df["_AbsQty"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        # bin assignment
        bin_id = np.searchsorted(edges, sizes, side="right") - 1
        bin_id = np.clip(bin_id, 0, len(edges) - 2)

        df2 = df.copy()
        df2["_bin"] = bin_id

        # bins summary
        g = df2.groupby("_bin", observed=False).agg(
            Trades=("_Trades", "sum"),
            Notional=("Notional", "sum"),
            Flow=("Flow", "sum"),
        ).reset_index()

        def rng(i):
            return float(edges[i]), float(edges[i + 1])

        g["Range"] = g["_bin"].map(lambda i: f"{self._fmt_kmb(rng(i)[0])} – {self._fmt_kmb(rng(i)[1])}")
        g["Bin"] = g["_bin"].map(
            lambda i: f"P{int(round(100 * (i / (len(edges) - 1))))}–P{int(round(100 * ((i + 1) / (len(edges) - 1))))}"
        )

        tot_tr = float(g["Trades"].sum()) if len(g) else 0.0
        tot_metric = float(g["Flow"].sum()) if self.metric_var.get() == "Net Flow" else \
                     float(g["Notional"].sum()) if self.metric_var.get() == "Notional" else \
                     float(g["Trades"].sum())

        g["%Trades"] = np.where(tot_tr > 0, 100.0 * g["Trades"] / tot_tr, 0.0)
        g["%Flow"] = np.where(abs(tot_metric) > 0, 100.0 * g["Flow"] / tot_metric, 0.0)

        # heatmap pivot
        period = self.period_var.get()
        gcol = {"Daily": "_Day", "Weekly": "_Week", "Monthly": "_Month"}[period]
        heat = None
        if gcol in df2.columns:
            metric = self.metric_var.get()
            val_col = "Flow" if metric == "Net Flow" else "Notional" if metric == "Notional" else "_Trades"
            heat = df2.pivot_table(index="_bin", columns=gcol, values=val_col, aggfunc="sum", fill_value=0.0).sort_index()

        # histogram table
        hist_table = self._compute_hist_table(sizes)

        return {"df": df2, "bins": g, "heat": heat, "hist_table": hist_table}

    def _apply_and_refresh(self, use_cache: bool, force_draw: bool):
        key = self._selection_key()
        if use_cache and key in self._cache:
            res = self._cache[key]
        else:
            res = self._compute()
            if use_cache:
                self._cache[key] = res

        self._refresh_all(res, force_draw=force_draw)

    # ---------- refresh ----------

    def _refresh_all(self, res: dict, force_draw: bool):
        df = res.get("df", pd.DataFrame())
        bins = res.get("bins", pd.DataFrame())
        heat = res.get("heat")
        hist_table = res.get("hist_table", pd.DataFrame())

        self._refresh_kpis(df)
        self._refresh_distribution(df, hist_table, force_draw=force_draw)
        self._refresh_contribution(bins, force_draw=force_draw)
        self._refresh_heatmap(heat, bins, force_draw=force_draw)

        # ensure tables fit always
        try:
            self.tbl_hist.autofit_columns()
        except Exception:
            pass
        try:
            self.tbl_bins.autofit_columns()
        except Exception:
            pass

    def _refresh_kpis(self, df: pd.DataFrame):
        if df is None or df.empty:
            self.lbl_total_trades.config(text="0")
            self.lbl_total_notional.config(text="0")
            self.lbl_total_flow.config(text="0")
            return

        tot_trades = int(round(float(df["_Trades"].sum())))
        tot_notional = float(pd.to_numeric(df["Notional"], errors="coerce").fillna(0.0).sum())
        tot_flow = float(pd.to_numeric(df["Flow"], errors="coerce").fillna(0.0).sum())

        self.lbl_total_trades.config(text=f"{tot_trades:,}")
        self.lbl_total_notional.config(text=self._fmt_kmb(tot_notional))
        self.lbl_total_flow.config(text=self._fmt_kmb(tot_flow))

    # ---- Distribution tab ----

    def _refresh_distribution(self, df: pd.DataFrame, hist_table: pd.DataFrame, force_draw: bool):
        self.ax_hist.clear()

        if df is None or df.empty:
            self.ax_hist.set_title("No data")
            self._draw_canvas(self.canvas_hist, force_draw)
            self.tbl_hist.set_dataframe(pd.DataFrame(columns=["Bin", "From", "To", "Trades", "%Trades"]))
            return

        if self.size_field_var.get() == "Notional":
            s = pd.to_numeric(df["_AbsNotional"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            xlabel = "Abs Notional"
        else:
            s = pd.to_numeric(df["_AbsQty"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            xlabel = "Abs Quantity"

        s = s[np.isfinite(s)]
        s = s[s >= 0]

        if len(s) == 0:
            self.ax_hist.set_title("No data")
            self._draw_canvas(self.canvas_hist, force_draw)
            self.tbl_hist.set_dataframe(pd.DataFrame(columns=["Bin", "From", "To", "Trades", "%Trades"]))
            return

        if self.log_hist_var.get():
            s2 = s[s > 0]
            if len(s2) == 0:
                self.ax_hist.set_title("No positive sizes for log hist")
            else:
                lo, hi = np.percentile(s2, 1), np.percentile(s2, 99.5)
                lo = max(lo, 1e-12)
                bins = np.logspace(np.log10(lo), np.log10(max(hi, lo * 10)), 40)
                self.ax_hist.hist(s2, bins=bins)
                self.ax_hist.set_xscale("log")
        else:
            self.ax_hist.hist(s, bins=40)

        self.ax_hist.set_title("Distribution of trade sizes")
        self.ax_hist.set_xlabel(xlabel)
        self.ax_hist.set_ylabel("Trades")
        self.ax_hist.yaxis.set_major_formatter(FuncFormatter(self._fmt_kmb))

        try:
            self.fig_hist.tight_layout()
        except Exception:
            pass
        self._draw_canvas(self.canvas_hist, force_draw)

        if hist_table is None or hist_table.empty:
            self.tbl_hist.set_dataframe(pd.DataFrame(columns=["Bin", "From", "To", "Trades", "%Trades"]))
        else:
            self.tbl_hist.set_dataframe(hist_table[["Bin", "From", "To", "Trades", "%Trades"]])

    # ---- Contribution tab ----

    def _refresh_contribution(self, bins: pd.DataFrame, force_draw: bool):
        self.ax_bins.clear()

        if bins is None or bins.empty:
            self.ax_bins.set_title("No bins")
            self._draw_canvas(self.canvas_bins, force_draw)
            self.tbl_bins.set_dataframe(pd.DataFrame(columns=["Bin", "Range", "Trades", "%Trades", "Notional", "Flow", "%Flow"]))
            return

        metric = self.metric_var.get()
        if metric == "Net Flow":
            y = bins["Flow"].to_numpy(dtype=float)
            title = "Net Flow contribution by size bin"
        elif metric == "Notional":
            y = bins["Notional"].to_numpy(dtype=float)
            title = "Notional contribution by size bin"
        else:
            y = bins["Trades"].to_numpy(dtype=float)
            title = "Trades by size bin"

        x = np.arange(len(bins))

        if self.normalize_var.get():
            denom = float(np.sum(np.abs(y))) if metric == "Net Flow" else float(np.sum(y))
            denom = denom if denom != 0 else 1.0
            y_plot = 100.0 * (y / denom)
            self.ax_bins.set_ylabel("%")
        else:
            y_plot = y
            self.ax_bins.set_ylabel("Flow" if metric == "Net Flow" else metric)

        colors = np.where(y_plot >= 0, "green", "red") if metric == "Net Flow" else None
        self.ax_bins.bar(x, y_plot, color=colors)
        self.ax_bins.axhline(0.0, linewidth=1.0)

        cum = np.cumsum(y_plot)
        self.ax_bins.plot(x, cum, linewidth=1.6)

        self.ax_bins.set_title(title)
        self.ax_bins.set_xticks(x)
        self.ax_bins.set_xticklabels(bins["Bin"].tolist(), rotation=30, ha="right")  # ALWAYS 30 deg

        if not self.normalize_var.get():
            self.ax_bins.yaxis.set_major_formatter(FuncFormatter(self._fmt_kmb))

        try:
            self.fig_bins.tight_layout()
        except Exception:
            pass
        self._draw_canvas(self.canvas_bins, force_draw)

        # table
        df = bins.copy().sort_values("_bin").reset_index(drop=True)
        out = pd.DataFrame({
            "Bin": df["Bin"].astype(str),
            "Range": df["Range"].astype(str),
            "Trades": df["Trades"].map(lambda v: f"{int(round(float(v))):,}"),
            "%Trades": df["%Trades"].map(self._fmt_pct),
            "Notional": df["Notional"].map(self._fmt_kmb),
            "Flow": df["Flow"].map(self._fmt_kmb),
            "%Flow": df["%Flow"].map(self._fmt_pct),
        })
        self.tbl_bins.set_dataframe(out[["Bin", "Range", "Trades", "%Trades", "Notional", "Flow", "%Flow"]])

    # ---- Heatmap tab ----

    def _refresh_heatmap(self, heat, bins: pd.DataFrame, force_draw: bool):
        self.ax_heat.clear()
        if self.ax_heat_cbar is not None:
            self.ax_heat_cbar.clear()

        if heat is None or not isinstance(heat, pd.DataFrame) or heat.empty:
            self.ax_heat.set_title("No heatmap data")
            self._draw_canvas(self.canvas_heat, force_draw)
            return

        mat = heat.to_numpy(dtype=float)
        metric = self.metric_var.get()

        if metric == "Net Flow":
            vmax = float(np.nanmax(np.abs(mat))) if mat.size else 1.0
            vmax = max(vmax, 1.0)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            im = self.ax_heat.imshow(mat, aspect="auto", interpolation="nearest", cmap="RdYlGn", norm=norm)
        else:
            im = self.ax_heat.imshow(mat, aspect="auto", interpolation="nearest")

        self.ax_heat.set_title(f"Heatmap: {metric} by bin vs {self.period_var.get()}")

        # y labels
        ylabels = []
        for i in heat.index.tolist():
            if bins is not None and not bins.empty:
                r = bins[bins["_bin"] == i]
                ylabels.append(str(r["Bin"].iloc[0]) if len(r) else str(i))
            else:
                ylabels.append(str(i))
        self.ax_heat.set_yticks(range(len(ylabels)))
        self.ax_heat.set_yticklabels(ylabels)

        # x labels always 30 degrees
        cols = [str(c) for c in heat.columns.tolist()]
        n = len(cols)
        if n <= 1:
            self.ax_heat.set_xticks([0] if n == 1 else [])
            self.ax_heat.set_xticklabels([cols[0]] if n == 1 else [], rotation=30, ha="right")
        else:
            idxs = [0, n // 3, (2 * n) // 3, n - 1]
            idxs = sorted(set(max(0, min(n - 1, i)) for i in idxs))
            self.ax_heat.set_xticks(idxs)
            self.ax_heat.set_xticklabels([cols[i] for i in idxs], rotation=30, ha="right")

        # colorbar fixed axis (always visible)
        try:
            cb = self.fig_heat.colorbar(im, cax=self.ax_heat_cbar)
            cb.formatter = FuncFormatter(self._fmt_kmb)
            cb.update_ticks()
        except Exception:
            pass

        try:
            self.fig_heat.tight_layout()
        except Exception:
            pass
        self._draw_canvas(self.canvas_heat, force_draw)

    def _draw_canvas(self, canvas: FigureCanvasTkAgg, force: bool):
        # hidden canvases sometimes do not redraw correctly with draw_idle.
        if force:
            canvas.draw()
        else:
            canvas.draw_idle()
