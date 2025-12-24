from __future__ import annotations
from tkinter import ttk
import pandas as pd

from ui.widgets.table import DataTable

class OverviewView(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, style="TFrame")
        header = ttk.Label(self, text="Overview", style="Title.TLabel")
        header.pack(anchor="w", padx=14, pady=(12, 6))

        self.table = DataTable(self, columns=None, height=20)
        self.table.pack(fill="both", expand=True, padx=14, pady=14)

    def render(self, df: pd.DataFrame | None):
        if df is None:
            self.table.set_dataframe(pd.DataFrame())
        else:
            self.table.set_dataframe(df.head(100))
