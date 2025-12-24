from __future__ import annotations

from tkinter import ttk
import pandas as pd


class DataTable(ttk.Frame):
    def __init__(self, master, columns=None, height: int = 18):
        super().__init__(master, style="TFrame")
        self.columns = columns  # None => dynamic

        # IMPORTANT: if columns provided, pass them to Treeview immediately.
        init_cols = tuple(columns) if columns is not None else tuple()

        self.tree = ttk.Treeview(self, columns=init_cols, show="headings", height=height)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.hsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.hsb.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        if columns is not None:
            self._setup_columns(list(columns))

    def _setup_columns(self, cols):
        self.tree["columns"] = cols
        for c in cols:
            self.tree.heading(c, text=c)
            w = 90
            if c == "TradeTime": w = 170
            elif c in ("ISIN",): w = 140
            elif c in ("Counterparty","UND_NAME"): w = 130
            elif c in ("Expiry","Knock_Date"): w = 110
            self.tree.column(c, width=w, minwidth=60, stretch=False, anchor="w")

    def set_dataframe(self, df: pd.DataFrame):
        # clear
        for iid in self.tree.get_children():
            self.tree.delete(iid)

        if df is None or df.empty:
            return

        cols = self.columns if self.columns is not None else list(df.columns)
        self._setup_columns(cols)

        sub = df.copy()
        if "TradeTime" in sub.columns:
            sub["TradeTime"] = pd.to_datetime(sub["TradeTime"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

        for _, row in sub.iterrows():
            vals = []
            for c in cols:
                v = row.get(c, "")
                try:
                    if v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NA:
                        vals.append("")
                    else:
                        vals.append(str(v))
                except Exception:
                    vals.append(str(v))
            self.tree.insert("", "end", values=vals)
