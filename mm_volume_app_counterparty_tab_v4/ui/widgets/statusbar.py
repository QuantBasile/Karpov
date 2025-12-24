from __future__ import annotations
from tkinter import ttk

class StatusBar(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, style="Topbar.TFrame")
        self.lbl = ttk.Label(self, text="Ready.", style="Muted.TLabel")
        self.lbl.pack(side="left", padx=12, pady=6)

    def set(self, text: str):
        self.lbl.config(text=text)
