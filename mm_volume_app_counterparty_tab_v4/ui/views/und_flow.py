# ui/views/und_flow.py
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


class UndFlowView(ttk.Frame):
    """
    2.3 UND Flow — analog of 1.3 CP Flow but grouped by UND_NAME.
    Uses signed Flow (sell positive, buy negative) — assumes df_clean already provides Flow.

    Supports precomputed aggs (recommended):
      aggs["Daily"/"Weekly"/"Monthly"] with columns: UND, Period, Notional, Flow, Trades
    """

    def __init__(self, parent):
        super().__init__(parent, style="TFrame")

        self.df: pd.DataFrame | None = None
        self._precomputed_aggs: dict[str, pd.DataFrame] | None = None
        self._all_unds: list[str] | None = None

        self.gran_var = tk.StringVar(value="Daily")        # Daily | Weekly | Monthly
        self.auto_topn_var = tk.BooleanVar(value=True)
        self.topn_var = tk.IntVar(value=12)
        self.cum_var = tk.BooleanVar(value=False)          # cumulative toggle

        # Manual selection state (Auto OFF)
        self._manual_selected: list[str] = []

        # Focus table UND selector
        self.focus_und_var = tk.StringVar(value="")
        self._focus_unds: list[str] = []

        # Matplotlib fixed colorbar axis for heatmap
        self.ax_heat_cbar = None

        self._build()

    # ---------------- Formatting helpers ----------------

    def _fmt_axis_kmb(self, x, _pos=None):
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

    def _ellipsize(self, s: str, n: int) -> str:
        s = str(s)
        return s if len(s) <= n else (s[: n - 1] + "…")

    def _fmt_trades_cell(self, v):
        try:
            return f"{int(round(float(v))):,}"
        except Exception:
            return "0"

    # ---------------- UI ----------------

    def _build(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, style="Sidebar.TFrame", width=380)
        left.grid(row=0, column=0, sticky="nsw")
        left.grid_propagate(False)

        right = ttk.Frame(self, style="TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        padx = 12

        ttk.Label(left, text="UND Flow", style="Title.TLabel").pack(anchor="w", padx=padx, pady=(12, 6))

        # Granularity
        box_g = ttk.LabelFrame(left, text="Granularity", style="TLabelframe")
        box_g.pack(fill="x", padx=padx, pady=(6, 6))
        rowg = ttk.Frame(box_g)
        rowg.pack(fill="x", padx=8, pady=6)
        for g in ("Daily", "Weekly", "Monthly"):
            ttk.Radiobutton(rowg, text=g, variable=self.gran_var, value=g, command=self._refresh).pack(
                side="left", padx=(0, 10)
            )

        # TopN + Auto + Cumulative
        box_top = ttk.LabelFrame(left, text="Controls", style="TLabelframe")
        box_top.pack(fill="x", padx=padx, pady=(0, 6))

        row_top = ttk.Frame(box_top)
        row_top.pack(fill="x", padx=8, pady=(8, 4))

        ttk.Checkbutton(row_top, text="Auto Top N", variable=self.auto_topn_var, command=self._on_auto_toggle).pack(
            side="left"
        )
        ttk.Label(row_top, text="N:", style="Muted.TLabel").pack(side="left", padx=(12, 4))
        spn = ttk.Spinbox(row_top, from_=5, to=30, textvariable=self.topn_var, width=5, command=self._refresh)
        spn.pack(side="left")

        ttk.Checkbutton(
            box_top,
            text="Cumulative line",
            variable=self.cum_var,
            command=self._refresh,
        ).pack(anchor="w", padx=8, pady=(0, 8))

        # Underlying selector (dual list; only active when Auto OFF)
        box_und = ttk.LabelFrame(left, text="Underlyings (manual)", style="TLabelframe")
        box_und.pack(fill="x", expand=False, padx=padx, pady=(0, 6))
        box_und.configure(height=200)
        box_und.pack_propagate(False)

        row_dual = ttk.Frame(box_und)
        row_dual.pack(fill="both", expand=True, padx=8, pady=(8, 8))
        row_dual.columnconfigure(0, weight=1)
        row_dual.columnconfigure(1, weight=0)
        row_dual.columnconfigure(2, weight=1)
        row_dual.rowconfigure(0, weight=1)

        self.lst_avail = tk.Listbox(row_dual, selectmode="extended", height=6)
        self.lst_avail.grid(row=0, column=0, sticky="nsew")
        sb_a = ttk.Scrollbar(row_dual, orient="vertical", command=self.lst_avail.yview)
        sb_a.grid(row=0, column=0, sticky="nse", padx=(0, 2))
        self.lst_avail.configure(yscrollcommand=sb_a.set)

        mid = ttk.Frame(row_dual)
        mid.grid(row=0, column=1, sticky="ns", padx=6)
        ttk.Button(mid, text=">", width=4, command=self._move_to_selected).pack(pady=(30, 8))
        ttk.Button(mid, text="<", width=4, command=self._move_to_available).pack(pady=(0, 8))
        ttk.Button(mid, text="Clear", width=6, command=self._clear_manual).pack(pady=(10, 0))

        self.lst_sel = tk.Listbox(row_dual, selectmode="extended", height=6)
        self.lst_sel.grid(row=0, column=2, sticky="nsew")
        sb_s = ttk.Scrollbar(row_dual, orient="vertical", command=self.lst_sel.yview)
        sb_s.grid(row=0, column=2, sticky="nse")
        self.lst_sel.configure(yscrollcommand=sb_s.set)
        self.lst_sel.bind("<<ListboxSelect>>", lambda _e: self._on_manual_selection_changed())

        # Ranking (FULL always)
        ttk.Label(left, text="Ranking (range)", style="Muted.TLabel").pack(anchor="w", padx=padx, pady=(6, 4))

        self.rank = ttk.Treeview(left, columns=("und", "last", "delta", "sum"), show="headings", height=12)
        self.rank.pack(fill="x", padx=padx, pady=(0, 12))

        self.rank.heading("und", text="UND")
        self.rank.heading("last", text="Last")
        self.rank.heading("delta", text="Δ")
        self.rank.heading("sum", text="Sum")

        self.rank.column("und", width=170, anchor="w")
        self.rank.column("last", width=70, anchor="e")
        self.rank.column("delta", width=60, anchor="e")
        self.rank.column("sum", width=70, anchor="e")

        # Right notebook
        self.nb = ttk.Notebook(right)
        self.nb.grid(row=0, column=0, sticky="nsew")

        self.tab_overview = ttk.Frame(self.nb, style="TFrame")
        self.tab_focus = ttk.Frame(self.nb, style="TFrame")
        self.nb.add(self.tab_overview, text="Overview")
        self.nb.add(self.tab_focus, text="Focus")

        self._build_overview()
        self._build_focus()

        self._on_auto_toggle()

    def _build_overview(self):
        self.tab_overview.rowconfigure(1, weight=1)
        self.tab_overview.columnconfigure(0, weight=1)

        self.fig_total = Figure(figsize=(6, 2.2), dpi=100)
        self.ax_total = self.fig_total.add_subplot(111)
        self.canvas_total = FigureCanvasTkAgg(self.fig_total, master=self.tab_overview)
        self.canvas_total.get_tk_widget().grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))

        self.fig_heat = Figure(figsize=(6, 4.2), dpi=100)
        self.ax_heat = self.fig_heat.add_subplot(111)

        divider = make_axes_locatable(self.ax_heat)
        self.ax_heat_cbar = divider.append_axes("right", size="3%", pad=0.08)

        self.canvas_heat = FigureCanvasTkAgg(self.fig_heat, master=self.tab_overview)
        self.canvas_heat.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=12, pady=(6, 12))

    def _build_focus(self):
        self.tab_focus.rowconfigure(2, weight=1)
        self.tab_focus.columnconfigure(0, weight=1)

        top = ttk.Frame(self.tab_focus)
        top.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 2))
        ttk.Label(top, text="Table UND:", style="Muted.TLabel").pack(side="left", padx=(0, 6))
        self.cmb_focus_und = ttk.Combobox(top, textvariable=self.focus_und_var, state="readonly", width=32)
        self.cmb_focus_und.pack(side="left")
        self.cmb_focus_und.bind("<<ComboboxSelected>>", lambda _e: self._refresh_focus_table_only())

        self.fig_focus = Figure(figsize=(6, 3.0), dpi=100)
        self.ax_focus = self.fig_focus.add_subplot(111)
        self.canvas_focus = FigureCanvasTkAgg(self.fig_focus, master=self.tab_focus)
        self.canvas_focus.get_tk_widget().grid(row=1, column=0, sticky="ew", padx=12, pady=(6, 6))

        self.detail = ttk.Treeview(self.tab_focus, columns=("period", "flow", "trades", "delta"), show="headings")
        self.detail.grid(row=2, column=0, sticky="nsew", padx=12, pady=(6, 12))

        self.detail.heading("period", text="Period")
        self.detail.heading("flow", text="Flow")
        self.detail.heading("trades", text="Trades")
        self.detail.heading("delta", text="Δ (Flow)")

        self.detail.column("period", width=120, anchor="w")
        self.detail.column("flow", width=120, anchor="e")
        self.detail.column("trades", width=80, anchor="e")
        self.detail.column("delta", width=90, anchor="e")

    # ---------------- Public API ----------------

    def render(
        self,
        df_clean: pd.DataFrame,
        aggs: dict[str, pd.DataFrame] | None = None,
        und_list: list[str] | None = None,
    ):
        self.df = df_clean
        self._precomputed_aggs = aggs
        self._all_unds = und_list

        # Important: ALWAYS rebuild lists on render so 2.3 doesn't depend on other tabs
        self._rebuild_manual_lists()
        self._refresh()

    # ---------------- Data ----------------

    def _get_agg(self, granularity: str) -> pd.DataFrame | None:
        if self._precomputed_aggs is not None:
            df = self._precomputed_aggs.get(granularity)
            if df is None or len(df) == 0:
                return None
            out = df.copy()
            if "Flow" not in out.columns:
                return None
            if "Trades" not in out.columns:
                out["Trades"] = 0.0
            out["Flow"] = pd.to_numeric(out["Flow"], errors="coerce").fillna(0.0).astype(float)
            out["Trades"] = pd.to_numeric(out["Trades"], errors="coerce").fillna(0.0).astype(float)
            out["UND"] = out["UND"].astype(str)
            out["Period"] = out["Period"].astype(str)
            return out

        # fallback
        if self.df is None or len(self.df) == 0:
            return None

        df = self.df
        needed = ["UND_NAME", "_Day", "_Week", "_Month", "Flow"]
        for c in needed:
            if c not in df.columns:
                return None

        gcol = {"Daily": "_Day", "Weekly": "_Week", "Monthly": "_Month"}[granularity]
        base = df[["UND_NAME", gcol, "Flow"]].copy()
        base["UND_NAME"] = base["UND_NAME"].astype(str)
        base["Flow"] = pd.to_numeric(base["Flow"], errors="coerce").fillna(0.0).astype(float)
        base["_Trades"] = 1.0

        agg = (
            base.groupby(["UND_NAME", gcol], as_index=False, observed=False)[["Flow", "_Trades"]]
            .sum()
            .rename(columns={"UND_NAME": "UND", gcol: "Period", "_Trades": "Trades"})
            .sort_values(["UND", "Period"])
            .reset_index(drop=True)
        )
        return agg

    # ---------------- Manual selector ----------------

    def _on_auto_toggle(self):
        auto = bool(self.auto_topn_var.get())
        state = "disabled" if auto else "normal"

        self.lst_avail.configure(state=state)
        self.lst_sel.configure(state=state)

        if not auto:
            self._rebuild_manual_lists()

        self._refresh()

    def _rebuild_manual_lists(self):
        self.lst_avail.delete(0, tk.END)
        self.lst_sel.delete(0, tk.END)

        if self._all_unds is not None:
            all_unds = list(self._all_unds)
        else:
            if self.df is None or "UND_NAME" not in self.df.columns:
                self._manual_selected = []
                self._sync_focus_und_choices([])
                return
            all_unds = sorted(self.df["UND_NAME"].astype(str).unique().tolist())

        sel_set = set(self._manual_selected)
        for und in all_unds:
            if und in sel_set:
                self.lst_sel.insert(tk.END, und)
            else:
                self.lst_avail.insert(tk.END, und)

        if not self.auto_topn_var.get():
            self._sync_focus_und_choices(self._manual_selected)

    def _move_to_selected(self):
        if self.auto_topn_var.get():
            return
        idxs = list(self.lst_avail.curselection())
        if not idxs:
            return
        vals = [self.lst_avail.get(i) for i in idxs]
        for i in reversed(idxs):
            self.lst_avail.delete(i)
        for v in vals:
            self.lst_sel.insert(tk.END, v)
        self._update_manual_selected_from_list()
        self._refresh()

    def _move_to_available(self):
        if self.auto_topn_var.get():
            return
        idxs = list(self.lst_sel.curselection())
        if not idxs:
            return
        vals = [self.lst_sel.get(i) for i in idxs]
        for i in reversed(idxs):
            self.lst_sel.delete(i)
        for v in vals:
            self.lst_avail.insert(tk.END, v)
        self._update_manual_selected_from_list()
        self._refresh()

    def _clear_manual(self):
        if self.auto_topn_var.get():
            return
        vals = [self.lst_sel.get(i) for i in range(self.lst_sel.size())]
        self.lst_sel.delete(0, tk.END)
        for v in vals:
            self.lst_avail.insert(tk.END, v)
        self._update_manual_selected_from_list()
        self._refresh()

    def _on_manual_selection_changed(self):
        if self.auto_topn_var.get():
            return
        self._update_manual_selected_from_list()
        self._refresh_focus_only()

    def _update_manual_selected_from_list(self):
        self._manual_selected = [self.lst_sel.get(i) for i in range(self.lst_sel.size())]
        self._sync_focus_und_choices(self._manual_selected)

    # ---------------- Focus helpers ----------------

    def _sync_focus_und_choices(self, unds: list[str]):
        unds = list(dict.fromkeys([str(x) for x in unds]))
        self._focus_unds = unds
        self.cmb_focus_und["values"] = unds
        if not unds:
            self.focus_und_var.set("")
        else:
            cur = self.focus_und_var.get()
            if cur not in unds:
                self.focus_und_var.set(unds[0])

    def _get_focus_unds(self, agg: pd.DataFrame) -> list[str]:
        if self.auto_topn_var.get():
            topn = int(self.topn_var.get())
            s = (
                agg.groupby("UND", as_index=False, observed=False)["Flow"]
                .sum()
                .sort_values("Flow", ascending=False)
            )
            return s["UND"].head(topn).tolist()
        return list(self._manual_selected)

    # ---------------- Refresh ----------------

    def _set_time_xticks(self, ax, n: int):
        if n <= 0:
            ax.set_xticks([])
            return
        if n == 1:
            ax.set_xticks([0])
            return
        idxs = [0]
        if n >= 4:
            idxs += [n // 3, (2 * n) // 3]
        idxs += [n - 1]
        idxs = sorted(set(max(0, min(n - 1, i)) for i in idxs))
        ax.set_xticks(idxs)

    def _refresh(self):
        self._refresh_overview()
        self._refresh_focus_only()

    def _refresh_overview(self):
        self.ax_total.clear()
        self.ax_heat.clear()
        if self.ax_heat_cbar is not None:
            self.ax_heat_cbar.clear()

        agg = self._get_agg(self.gran_var.get())
        if agg is None or len(agg) == 0:
            self.ax_total.set_title("No data")
            self.ax_heat.set_title("No data")
            self.canvas_total.draw()
            self.canvas_heat.draw()
            self._refresh_ranking_empty()
            return

        topn = int(self.topn_var.get())

        # UNDs for heatmap
        if self.auto_topn_var.get():
            sums = (
                agg.groupby("UND", as_index=False, observed=False)["Flow"]
                .sum()
                .sort_values("Flow", ascending=False)
            )
            show_unds = sums["UND"].head(topn).tolist()
        else:
            show_unds = list(self._manual_selected)
            if not show_unds:
                sums = (
                    agg.groupby("UND", as_index=False, observed=False)["Flow"]
                    .sum()
                    .sort_values("Flow", ascending=False)
                )
                show_unds = sums["UND"].head(topn).tolist()

        # Total plot (bars) — green/red by sign
        total = agg.groupby("Period", as_index=False, observed=False)["Flow"].sum().sort_values("Period")
        y = total["Flow"].to_numpy(dtype=float)
        x = np.arange(len(total))
        colors = np.where(y >= 0, "green", "red")
        self.ax_total.bar(x, y, color=colors)
        self.ax_total.axhline(0.0, linewidth=1.0)

        # Optional cumulative line overlay
        if self.cum_var.get():
            cum = np.cumsum(y)
            self.ax_total.plot(x, cum, linewidth=1.6)

        self.ax_total.set_title(f"Total Flow ({self.gran_var.get()})")
        self.ax_total.set_ylabel("Flow")
        self.ax_total.yaxis.set_major_formatter(FuncFormatter(self._fmt_axis_kmb))

        self._set_time_xticks(self.ax_total, len(x))
        tick_idx = self.ax_total.get_xticks().astype(int) if len(x) else []
        labels = [str(total["Period"].iloc[i]) if 0 <= i < len(total) else "" for i in tick_idx]
        self.ax_total.set_xticklabels(labels, rotation=0)

        # Heatmap diverging around 0
        sub = agg[agg["UND"].isin(show_unds)].copy()
        if len(sub) == 0:
            self.ax_heat.set_title("No data for selection")
            self.canvas_total.draw()
            self.canvas_heat.draw()
            self._refresh_ranking(agg)
            return

        sums2 = (
            sub.groupby("UND", as_index=False, observed=False)["Flow"]
            .sum()
            .sort_values("Flow", ascending=False)
        )
        und_order = sums2["UND"].tolist()
        periods = sorted(sub["Period"].unique().tolist())

        pivot = sub.pivot_table(index="UND", columns="Period", values="Flow", aggfunc="sum", fill_value=0.0)
        pivot = pivot.reindex(index=und_order, columns=periods, fill_value=0.0)

        mat = pivot.to_numpy(dtype=float)
        vmax = float(np.nanmax(np.abs(mat))) if mat.size else 1.0
        vmax = max(vmax, 1.0)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        im = self.ax_heat.imshow(mat, aspect="auto", interpolation="nearest", cmap="RdYlGn", norm=norm)
        self.ax_heat.set_title(f"Heatmap: Flow ({len(und_order)} UNDs)")
        self.ax_heat.set_yticks(range(len(und_order)))
        self.ax_heat.set_yticklabels([self._ellipsize(c, 26) for c in und_order])

        self._set_time_xticks(self.ax_heat, len(periods))
        xt = self.ax_heat.get_xticks().astype(int) if len(periods) else []
        xlabels = [str(periods[i]) if 0 <= i < len(periods) else "" for i in xt]
        self.ax_heat.set_xticklabels(xlabels, rotation=0)

        cbar = self.fig_heat.colorbar(im, cax=self.ax_heat_cbar)
        cbar.formatter = FuncFormatter(self._fmt_axis_kmb)
        cbar.update_ticks()

        # Ranking FULL (always)
        self._refresh_ranking(agg)

        self.canvas_total.draw()
        self.canvas_heat.draw()

    def _refresh_ranking_empty(self):
        for iid in self.rank.get_children():
            self.rank.delete(iid)

    def _refresh_ranking(self, agg: pd.DataFrame):
        for iid in self.rank.get_children():
            self.rank.delete(iid)

        def last_delta(g: pd.DataFrame) -> pd.Series:
            g = g.sort_values("Period")
            last = float(g["Flow"].iloc[-1]) if len(g) else 0.0
            prev = float(g["Flow"].iloc[-2]) if len(g) >= 2 else 0.0
            return pd.Series({"Last": last, "Delta": last - prev, "Sum": float(g["Flow"].sum())})

        r = agg.groupby("UND", observed=False).apply(last_delta).reset_index()
        r = r.sort_values("Sum", ascending=False)

        for _, row in r.iterrows():
            und_full = str(row["UND"])
            self.rank.insert(
                "",
                tk.END,
                values=(
                    self._ellipsize(und_full, 28),
                    self._fmt_axis_kmb(row["Last"]),
                    self._fmt_axis_kmb(row["Delta"]),
                    self._fmt_axis_kmb(row["Sum"]),
                ),
                tags=(und_full,),
            )

    def _refresh_focus_only(self):
        self.ax_focus.clear()
        for iid in self.detail.get_children():
            self.detail.delete(iid)

        agg = self._get_agg(self.gran_var.get())
        if agg is None or len(agg) == 0:
            self.ax_focus.set_title("No data")
            self.canvas_focus.draw()
            self._sync_focus_und_choices([])
            return

        unds = self._get_focus_unds(agg)
        if not unds:
            self.ax_focus.set_title(f"Focus: Flow ({self.gran_var.get()}) — no UND selected")
            self.ax_focus.set_ylabel("Flow")
            self.ax_focus.set_xticks([])
            self.canvas_focus.draw()
            self._sync_focus_und_choices([])
            return

        self._sync_focus_und_choices(unds)

        for und in unds:
            g = agg[agg["UND"] == und].sort_values("Period")
            vals = g["Flow"].to_numpy(dtype=float)
            if self.cum_var.get():
                vals = np.cumsum(vals)
            self.ax_focus.plot(range(len(g)), vals, label=self._ellipsize(und, 30))

        self.ax_focus.axhline(0.0, linewidth=1.0)
        title_mode = "Cumulative Flow" if self.cum_var.get() else "Flow"
        self.ax_focus.set_title(f"Focus: {title_mode} ({self.gran_var.get()})")
        self.ax_focus.set_ylabel("Flow")
        self.ax_focus.yaxis.set_major_formatter(FuncFormatter(self._fmt_axis_kmb))

        g0 = agg[agg["UND"] == unds[0]].sort_values("Period")
        periods0 = g0["Period"].tolist()
        self._set_time_xticks(self.ax_focus, len(periods0))
        xt = self.ax_focus.get_xticks().astype(int) if periods0 else []
        labels = [str(periods0[i]) if 0 <= i < len(periods0) else "" for i in xt]
        self.ax_focus.set_xticklabels(labels, rotation=0)

        self.ax_focus.legend(loc="upper left", frameon=False)

        self._refresh_focus_table_only(draw=False)
        self.canvas_focus.draw()

    def _refresh_focus_table_only(self, draw: bool = True):
        for iid in self.detail.get_children():
            self.detail.delete(iid)

        agg = self._get_agg(self.gran_var.get())
        if agg is None or len(agg) == 0:
            if draw:
                self.canvas_focus.draw()
            return

        und0 = self.focus_und_var.get().strip()
        if not und0:
            if draw:
                self.canvas_focus.draw()
            return

        g0 = agg[agg["UND"] == und0].sort_values("Period").copy()
        if len(g0) == 0:
            if draw:
                self.canvas_focus.draw()
            return

        g0["_DeltaFlow"] = g0["Flow"].diff().fillna(0.0)

        for _, row in g0.iterrows():
            period = str(row["Period"])
            flow = self._fmt_axis_kmb(row["Flow"])
            trades = self._fmt_trades_cell(row.get("Trades", 0.0))
            delta = self._fmt_axis_kmb(row["_DeltaFlow"])
            self.detail.insert("", tk.END, values=(period, flow, trades, delta))

        if draw:
            self.canvas_focus.draw()
