# ui/views/counterparty_evolution.py
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import pandas as pd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


class CounterpartyEvolutionView(ttk.Frame):
    """
    CP Evolution tab (Tkinter):
      - Left: Granularity, Metric, Auto TopN, Counterparty selector (dual list when Auto OFF), Ranking
      - Right: Notebook with Overview (Total histogram + Heatmap) and Focus (Series + Detail table + CP selector)

    Expects df_clean to contain:
      - Counterparty
      - _Day / _Week / _Month
      - Notional (preferred). If missing, fallback computes abs(Quantity)*Price.
    Trades are counted as number of rows.
    """

    def __init__(self, parent):
        super().__init__(parent, style="TFrame")

        self.df: pd.DataFrame | None = None
        self._agg_cache: dict[tuple[int, str], pd.DataFrame] = {}

        self.metric_var = tk.StringVar(value="Notional")   # Notional | Trades
        self.gran_var = tk.StringVar(value="Daily")        # Daily | Weekly | Monthly
        self.auto_topn_var = tk.BooleanVar(value=True)
        self.topn_var = tk.IntVar(value=12)

        # Manual selection state (Auto OFF)
        self._manual_selected: list[str] = []

        # Focus table CP selector
        self.focus_cp_var = tk.StringVar(value="")
        self._focus_cps: list[str] = []

        # Fixed colorbar axis for heatmap (prevents shrinking)
        self.ax_heat_cbar = None

        self._build()

    # ---------------- Formatting helpers ----------------

    def _fmt_axis_kmb(self, x, _pos=None):
        # Applies to BOTH Notional and Trades (Trades also shows 1k / 1M)
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

    def _fmt_cell(self, metric: str, v):
        try:
            x = float(v)
        except Exception:
            return "0"
        if metric == "Trades":
            # Keep integers in table, but still compact in axis (formatter handles axis)
            return f"{int(round(x)):,}"
        return self._fmt_axis_kmb(x)

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

        ttk.Label(left, text="Counterparty Evolution", style="Title.TLabel").pack(anchor="w", padx=padx, pady=(12, 6))

        # Granularity
        box_g = ttk.LabelFrame(left, text="Granularity", style="TLabelframe")
        box_g.pack(fill="x", padx=padx, pady=(6, 6))
        rowg = ttk.Frame(box_g)
        rowg.pack(fill="x", padx=8, pady=6)
        for g in ("Daily", "Weekly", "Monthly"):
            ttk.Radiobutton(rowg, text=g, variable=self.gran_var, value=g, command=self._refresh).pack(
                side="left", padx=(0, 10)
            )

        # Metric
        box_m = ttk.LabelFrame(left, text="Metric", style="TLabelframe")
        box_m.pack(fill="x", padx=padx, pady=(0, 6))
        rowm = ttk.Frame(box_m)
        rowm.pack(fill="x", padx=8, pady=6)
        for m in ("Notional", "Trades"):
            ttk.Radiobutton(rowm, text=m, variable=self.metric_var, value=m, command=self._refresh).pack(
                side="left", padx=(0, 10)
            )

        # TopN + Auto
        box_top = ttk.LabelFrame(left, text="Top N", style="TLabelframe")
        box_top.pack(fill="x", padx=padx, pady=(0, 6))
        row_top = ttk.Frame(box_top)
        row_top.pack(fill="x", padx=8, pady=6)

        ttk.Checkbutton(row_top, text="Auto Top N", variable=self.auto_topn_var, command=self._on_auto_toggle).pack(
            side="left"
        )
        ttk.Label(row_top, text="N:", style="Muted.TLabel").pack(side="left", padx=(12, 4))
        spn = ttk.Spinbox(row_top, from_=5, to=30, textvariable=self.topn_var, width=5, command=self._refresh)
        spn.pack(side="left")

        # Counterparty selector (dual list; ONLY active when Auto OFF)
        box_cp = ttk.LabelFrame(left, text="Counterparties (manual)", style="TLabelframe")
        box_cp.pack(fill="x", expand=False, padx=padx, pady=(0, 6))
        box_cp.configure(height=200)  # smaller vertically
        box_cp.pack_propagate(False)

        row_dual = ttk.Frame(box_cp)
        row_dual.pack(fill="both", expand=True, padx=8, pady=(8, 8))
        row_dual.columnconfigure(0, weight=1)
        row_dual.columnconfigure(1, weight=0)
        row_dual.columnconfigure(2, weight=1)
        row_dual.rowconfigure(0, weight=1)

        # Available list
        self.lst_avail = tk.Listbox(row_dual, selectmode="extended", height=6)
        self.lst_avail.grid(row=0, column=0, sticky="nsew")
        sb_a = ttk.Scrollbar(row_dual, orient="vertical", command=self.lst_avail.yview)
        sb_a.grid(row=0, column=0, sticky="nse", padx=(0, 2))
        self.lst_avail.configure(yscrollcommand=sb_a.set)

        # Buttons
        mid = ttk.Frame(row_dual)
        mid.grid(row=0, column=1, sticky="ns", padx=6)
        ttk.Button(mid, text=">", width=4, command=self._move_to_selected).pack(pady=(30, 8))
        ttk.Button(mid, text="<", width=4, command=self._move_to_available).pack(pady=(0, 8))
        ttk.Button(mid, text="Clear", width=6, command=self._clear_manual).pack(pady=(10, 0))

        # Selected list
        self.lst_sel = tk.Listbox(row_dual, selectmode="extended", height=6)
        self.lst_sel.grid(row=0, column=2, sticky="nsew")
        sb_s = ttk.Scrollbar(row_dual, orient="vertical", command=self.lst_sel.yview)
        sb_s.grid(row=0, column=2, sticky="nse")
        self.lst_sel.configure(yscrollcommand=sb_s.set)
        self.lst_sel.bind("<<ListboxSelect>>", lambda _e: self._on_manual_selection_changed())

        # Ranking
        ttk.Label(left, text="Ranking (range)", style="Muted.TLabel").pack(anchor="w", padx=padx, pady=(6, 4))

        self.rank = ttk.Treeview(left, columns=("cp", "last", "delta", "sum"), show="headings", height=12)
        self.rank.pack(fill="x", padx=padx, pady=(0, 12))

        self.rank.heading("cp", text="CP")
        self.rank.heading("last", text="Last")
        self.rank.heading("delta", text="Δ")
        self.rank.heading("sum", text="Sum")

        self.rank.column("cp", width=170, anchor="w")
        self.rank.column("last", width=70, anchor="e")
        self.rank.column("delta", width=60, anchor="e")
        self.rank.column("sum", width=70, anchor="e")

        # IMPORTANT: no double-click behavior on ranking (user requested)
        # (No binding)

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

        # Total plot (histogram / bars)
        self.fig_total = Figure(figsize=(6, 2.2), dpi=100)
        self.ax_total = self.fig_total.add_subplot(111)
        self.canvas_total = FigureCanvasTkAgg(self.fig_total, master=self.tab_overview)
        self.canvas_total.get_tk_widget().grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))

        # Heatmap with FIXED colorbar axis (no shrinking ever)
        self.fig_heat = Figure(figsize=(6, 4.2), dpi=100)
        self.ax_heat = self.fig_heat.add_subplot(111)

        divider = make_axes_locatable(self.ax_heat)
        self.ax_heat_cbar = divider.append_axes("right", size="3%", pad=0.08)

        self.canvas_heat = FigureCanvasTkAgg(self.fig_heat, master=self.tab_overview)
        self.canvas_heat.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=12, pady=(6, 12))

    def _build_focus(self):
        self.tab_focus.rowconfigure(2, weight=1)
        self.tab_focus.columnconfigure(0, weight=1)

        # Focus control row: choose which CP to show in table
        top = ttk.Frame(self.tab_focus)
        top.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 2))
        ttk.Label(top, text="Table CP:", style="Muted.TLabel").pack(side="left", padx=(0, 6))
        self.cmb_focus_cp = ttk.Combobox(top, textvariable=self.focus_cp_var, state="readonly", width=32)
        self.cmb_focus_cp.pack(side="left")
        self.cmb_focus_cp.bind("<<ComboboxSelected>>", lambda _e: self._refresh_focus_table_only())

        # Focus plot
        self.fig_focus = Figure(figsize=(6, 3.0), dpi=100)
        self.ax_focus = self.fig_focus.add_subplot(111)
        self.canvas_focus = FigureCanvasTkAgg(self.fig_focus, master=self.tab_focus)
        self.canvas_focus.get_tk_widget().grid(row=1, column=0, sticky="ew", padx=12, pady=(6, 6))

        # Detail table
        self.detail = ttk.Treeview(self.tab_focus, columns=("period", "notional", "trades", "delta"), show="headings")
        self.detail.grid(row=2, column=0, sticky="nsew", padx=12, pady=(6, 12))

        self.detail.heading("period", text="Period")
        self.detail.heading("notional", text="Notional")
        self.detail.heading("trades", text="Trades")
        self.detail.heading("delta", text="Δ (metric)")

        self.detail.column("period", width=120, anchor="w")
        self.detail.column("notional", width=120, anchor="e")
        self.detail.column("trades", width=80, anchor="e")
        self.detail.column("delta", width=90, anchor="e")

    # ---------------- Public API ----------------  
    
    def render(self, df_clean: pd.DataFrame, aggs: dict[str, pd.DataFrame] | None = None, cp_list: list[str] | None = None):
        self.df = df_clean  # solo por si necesitas algo residual; idealmente ni lo usas
        self._agg_cache.clear()
    
        if aggs is not None:
            self._precomputed_aggs = aggs
        else:
            self._precomputed_aggs = None
    
        if cp_list is not None:
            self._all_cps = cp_list
        else:
            self._all_cps = sorted(df_clean["Counterparty"].astype(str).unique().tolist())
    
        self._rebuild_manual_lists()

        self._refresh()

    

    # ---------------- Data ----------------

    def _prepare_df(self) -> pd.DataFrame | None:
        if self.df is None or len(self.df) == 0:
            return None

        df = self.df
        needed = ["Counterparty", "_Day", "_Week", "_Month"]
        for c in needed:
            if c not in df.columns:
                return None

        out = df.copy()

        out["_Trades"] = 1

        if "Notional" in out.columns:
            out["_Notional"] = pd.to_numeric(out["Notional"], errors="coerce").fillna(0.0).astype(float)
        else:
            q = pd.to_numeric(out.get("Quantity", 0.0), errors="coerce").fillna(0.0).abs().astype(float)
            p = pd.to_numeric(out.get("Price", 0.0), errors="coerce").fillna(0.0).astype(float)
            out["_Notional"] = (q * p).astype(float)

        out["_CP"] = out["Counterparty"].astype(str)

        out["_Day"] = out["_Day"].astype("string")
        out["_Week"] = out["_Week"].astype("string")
        out["_Month"] = out["_Month"].astype("string")

        return out


    def _get_agg(self, granularity: str) -> pd.DataFrame | None:
        if getattr(self, "_precomputed_aggs", None) is not None:
            return self._precomputed_aggs.get(granularity)

        if self.df is None:
            return None

        key = (id(self.df), granularity)
        if key in self._agg_cache:
            return self._agg_cache[key]

        base = self._prepare_df()
        if base is None or len(base) == 0:
            return None

        gcol = {"Daily": "_Day", "Weekly": "_Week", "Monthly": "_Month"}[granularity]

        agg = (
            base.dropna(subset=[gcol, "_CP"])
            .groupby(["_CP", gcol], as_index=False, observed=False)[["_Notional", "_Trades"]]
            .sum()
            .rename(columns={"_CP": "CP", gcol: "Period", "_Notional": "Notional", "_Trades": "Trades"})
            .sort_values(["CP", "Period"])
            .reset_index(drop=True)
        )

        agg["Notional"] = pd.to_numeric(agg["Notional"], errors="coerce").fillna(0.0).astype(float)
        agg["Trades"] = pd.to_numeric(agg["Trades"], errors="coerce").fillna(0.0).astype(float)

        self._agg_cache[key] = agg
        return agg

    # ---------------- Manual selector (dual list) ----------------

    def _on_auto_toggle(self):
        auto = bool(self.auto_topn_var.get())
        state = "disabled" if auto else "normal"

        # When Auto is ON, manual selector "does nothing" (disabled)
        self.lst_avail.configure(state=state)
        self.lst_sel.configure(state=state)

        # Also disable buttons inside the middle frame by walking children
        # (safe even if styles differ)
        # mid buttons are the 2nd child of the dual row frame; easiest: set by widget state if present
        # We'll just store state and rely on commands no-op if disabled listboxes.
        if auto:
            # keep selected list content but ignore it
            pass

        self._refresh()

    def _rebuild_manual_lists(self):
        self.lst_avail.delete(0, tk.END)
        self.lst_sel.delete(0, tk.END)
    
        # Prefer cp_list precalculada si existe
        if hasattr(self, "_all_cps") and self._all_cps is not None:
            all_cps = list(self._all_cps)
        else:
            base = self._prepare_df()
            if base is None:
                self._manual_selected = []
                self._sync_focus_cp_choices([])
                return
            all_cps = sorted(base["_CP"].unique().tolist())
    
        sel_set = set(self._manual_selected)
    
        for cp in all_cps:
            if cp in sel_set:
                self.lst_sel.insert(tk.END, cp)
            else:
                self.lst_avail.insert(tk.END, cp)
    
        if not self.auto_topn_var.get():
            self._sync_focus_cp_choices(self._manual_selected)


    def _move_to_selected(self):
        if self.auto_topn_var.get():
            return
        idxs = list(self.lst_avail.curselection())
        if not idxs:
            return
        vals = [self.lst_avail.get(i) for i in idxs]
        # remove from avail (reverse to keep indices valid)
        for i in reversed(idxs):
            self.lst_avail.delete(i)
        # add to selected
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
        # move everything back to available
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
        # Focus plot should reflect all activated CPs (selected list content),
        # not only current selection highlight.
        self._refresh_focus_only()

    def _update_manual_selected_from_list(self):
        self._manual_selected = [self.lst_sel.get(i) for i in range(self.lst_sel.size())]
        self._sync_focus_cp_choices(self._manual_selected)

    # ---------------- Focus helpers ----------------

    def _sync_focus_cp_choices(self, cps: list[str]):
        """Update combobox choices for the Focus table."""
        cps = list(dict.fromkeys([str(x) for x in cps]))  # stable unique
        self._focus_cps = cps
        self.cmb_focus_cp["values"] = cps
        if not cps:
            self.focus_cp_var.set("")
        else:
            # keep current if possible
            cur = self.focus_cp_var.get()
            if cur not in cps:
                self.focus_cp_var.set(cps[0])

    def _get_focus_cps(self, agg: pd.DataFrame, metric: str) -> list[str]:
        """Which CPs should be plotted in Focus plot."""
        if self.auto_topn_var.get():
            topn = int(self.topn_var.get())
            s = agg.groupby("CP", as_index=False, observed=False)[metric].sum().sort_values(metric, ascending=False)
            return s["CP"].head(topn).tolist()
        # manual: all activated (right list)
        return list(self._manual_selected)

    # ---------------- Refresh ----------------

    def _refresh(self):
        # If df has changed or first render, keep manual lists sane
        if self.df is not None and (self.lst_avail.size() == 0 and self.lst_sel.size() == 0):
            self._rebuild_manual_lists()

        self._refresh_overview()
        self._refresh_focus_only()

    def _set_time_xticks(self, ax, n: int):
        """Set x ticks at start, ~1/3, ~2/3, end."""
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
        # unique + sorted
        idxs = sorted(set(max(0, min(n - 1, i)) for i in idxs))
        ax.set_xticks(idxs)

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

        metric = self.metric_var.get()
        topn = int(self.topn_var.get())

        # Determine CPs to show in heatmap
        if self.auto_topn_var.get():
            sums = agg.groupby("CP", as_index=False, observed=False)[metric].sum().sort_values(metric, ascending=False)
            show_cps = sums["CP"].head(topn).tolist()
        else:
            # manual: show activated CPs; if none, fall back to topN so overview never empty
            show_cps = list(self._manual_selected)
            if not show_cps:
                sums = agg.groupby("CP", as_index=False, observed=False)[metric].sum().sort_values(metric, ascending=False)
                show_cps = sums["CP"].head(topn).tolist()

        # Total plot as histogram/bars by period
        total = agg.groupby("Period", as_index=False, observed=False)[metric].sum().sort_values("Period")
        y = total[metric].to_numpy(dtype=float)
        x = list(range(len(total)))
        self.ax_total.bar(x, y)
        self.ax_total.set_title(f"Total {metric} ({self.gran_var.get()})")
        self.ax_total.set_ylabel(metric)

        # x ticks: start, 1/3, 2/3, end
        self._set_time_xticks(self.ax_total, len(x))
        # label those ticks with period strings
        tick_idx = self.ax_total.get_xticks().astype(int) if len(x) else []
        labels = []
        for i in tick_idx:
            if 0 <= i < len(total):
                labels.append(str(total["Period"].iloc[i]))
            else:
                labels.append("")
        self.ax_total.set_xticklabels(labels, rotation=0)

        # y formatter for both metrics (Trades also uses 1k/1M)
        self.ax_total.yaxis.set_major_formatter(FuncFormatter(self._fmt_axis_kmb))

        # Heatmap
        sub = agg[agg["CP"].isin(show_cps)].copy()
        if len(sub) == 0:
            self.ax_heat.set_title("No data for selection")
            self.canvas_total.draw()
            self.canvas_heat.draw()
            self._refresh_ranking(agg, metric, topn=None)
            return

        sums2 = sub.groupby("CP", as_index=False, observed=False)[metric].sum().sort_values(metric, ascending=False)
        cp_order = sums2["CP"].tolist()

        periods = sorted(sub["Period"].unique().tolist())
        pivot = sub.pivot_table(index="CP", columns="Period", values=metric, aggfunc="sum", fill_value=0.0)
        pivot = pivot.reindex(index=cp_order, columns=periods, fill_value=0.0)

        mat = pivot.to_numpy(dtype=float)

        im = self.ax_heat.imshow(mat, aspect="auto", interpolation="nearest")
        self.ax_heat.set_title(f"Heatmap: {metric} ({len(cp_order)} CPs)")
        self.ax_heat.set_yticks(range(len(cp_order)))
        self.ax_heat.set_yticklabels([self._ellipsize(c, 26) for c in cp_order])

        # x ticks: start, 1/3, 2/3, end with period labels
        ncol = len(periods)
        self._set_time_xticks(self.ax_heat, ncol)
        xt = self.ax_heat.get_xticks().astype(int) if ncol else []
        xlabels = []
        for i in xt:
            xlabels.append(str(periods[i]) if 0 <= i < ncol else "")
        self.ax_heat.set_xticklabels(xlabels, rotation=0)

        # Fixed-axis colorbar (no shrinking)
        cbar = self.fig_heat.colorbar(im, cax=self.ax_heat_cbar)
        cbar.formatter = FuncFormatter(self._fmt_axis_kmb)
        cbar.update_ticks()

        self._refresh_ranking(agg, metric, topn=None)

        self.canvas_total.draw()
        self.canvas_heat.draw()

    def _refresh_ranking_empty(self):
        for iid in self.rank.get_children():
            self.rank.delete(iid)

    def _refresh_ranking(self, agg: pd.DataFrame, metric: str, topn: int | None):
        for iid in self.rank.get_children():
            self.rank.delete(iid)
    
        def last_delta(g: pd.DataFrame) -> pd.Series:
            g = g.sort_values("Period")
            last = float(g[metric].iloc[-1]) if len(g) else 0.0
            prev = float(g[metric].iloc[-2]) if len(g) >= 2 else 0.0
            return pd.Series({"Last": last, "Delta": last - prev, "Sum": float(g[metric].sum())})
    
        r = agg.groupby("CP", observed=False).apply(last_delta).reset_index()
        r = r.sort_values("Sum", ascending=False)
    
        # si topn=None -> TODO
        if topn is not None:
            r = r.head(int(topn))
    
        for _, row in r.iterrows():
            cp_full = str(row["CP"])
            self.rank.insert(
                "",
                tk.END,
                values=(
                    self._ellipsize(cp_full, 28),
                    self._fmt_cell(metric, row["Last"]),
                    self._fmt_cell(metric, row["Delta"]),
                    self._fmt_cell(metric, row["Sum"]),
                ),
                tags=(cp_full,),
            )


    def _refresh_focus_only(self):
        self.ax_focus.clear()
        for iid in self.detail.get_children():
            self.detail.delete(iid)

        agg = self._get_agg(self.gran_var.get())
        if agg is None or len(agg) == 0:
            self.ax_focus.set_title("No data")
            self.canvas_focus.draw()
            self._sync_focus_cp_choices([])
            return

        metric = self.metric_var.get()

        cps = self._get_focus_cps(agg, metric)

        # If manual and nothing selected: keep focus empty but usable (no crash)
        if not cps:
            self.ax_focus.set_title(f"Focus: {metric} ({self.gran_var.get()}) — no CP selected")
            self.ax_focus.set_ylabel(metric)
            self.ax_focus.set_xticks([])
            self.canvas_focus.draw()
            self._sync_focus_cp_choices([])
            return

        # Update combobox choices (for table)
        self._sync_focus_cp_choices(cps)

        # Plot ALL active CPs
        for cp in cps:
            g = agg[agg["CP"] == cp].sort_values("Period")
            self.ax_focus.plot(range(len(g)), g[metric].to_numpy(dtype=float), label=self._ellipsize(cp, 30))

        self.ax_focus.set_title(f"Focus: {metric} ({self.gran_var.get()})")
        self.ax_focus.set_ylabel(metric)

        # x ticks: start, 1/3, 2/3, end
        # Use periods from first CP that exists (they may differ; this is “good enough” and stable)
        g0 = agg[agg["CP"] == cps[0]].sort_values("Period")
        periods0 = g0["Period"].tolist()
        n0 = len(periods0)
        self._set_time_xticks(self.ax_focus, n0)
        xt = self.ax_focus.get_xticks().astype(int) if n0 else []
        labels = []
        for i in xt:
            labels.append(str(periods0[i]) if 0 <= i < n0 else "")
        self.ax_focus.set_xticklabels(labels, rotation=0)

        # y formatter for both metrics
        self.ax_focus.yaxis.set_major_formatter(FuncFormatter(self._fmt_axis_kmb))

        self.ax_focus.legend(loc="upper left", frameon=False)

        # Fill table for selected CP in combobox
        self._refresh_focus_table_only(draw=False)

        self.canvas_focus.draw()

    def _refresh_focus_table_only(self, draw: bool = True):
        # Called when combobox changes
        for iid in self.detail.get_children():
            self.detail.delete(iid)

        agg = self._get_agg(self.gran_var.get())
        if agg is None or len(agg) == 0:
            if draw:
                self.canvas_focus.draw()
            return

        metric = self.metric_var.get()
        cp0 = self.focus_cp_var.get().strip()
        if not cp0:
            if draw:
                self.canvas_focus.draw()
            return

        g0 = agg[agg["CP"] == cp0].sort_values("Period").copy()
        if len(g0) == 0:
            if draw:
                self.canvas_focus.draw()
            return

        g0["_DeltaMetric"] = g0[metric].diff().fillna(0.0)

        for _, row in g0.iterrows():
            period = str(row["Period"])
            notional = self._fmt_cell("Notional", row["Notional"])
            trades = self._fmt_cell("Trades", row["Trades"])
            delta = self._fmt_cell(metric, row["_DeltaMetric"])
            self.detail.insert("", tk.END, values=(period, notional, trades, delta))

        if draw:
            self.canvas_focus.draw()
