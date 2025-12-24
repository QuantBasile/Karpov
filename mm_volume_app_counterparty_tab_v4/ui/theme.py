# Python 3.12
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class Theme:
    """Light theme tuned to reduce eye strain (no external libs)."""

    BG = "#f5f7fb"
    PANEL = "#ffffff"
    BORDER = "#d7dde8"
    TEXT = "#111827"
    MUTED = "#6b7280"
    ACCENT = "#2563eb"

    FONT_UI = ("Segoe UI", 10)
    FONT_UI_BOLD = ("Segoe UI", 10, "bold")
    FONT_TITLE = ("Segoe UI", 12, "bold")
    FONT_NUM = ("Consolas", 10)

    @staticmethod
    def apply(root: tk.Tk) -> ttk.Style:
        style = ttk.Style(root)
        try:
            style.theme_use(style.theme_use())
        except tk.TclError:
            pass

        root.configure(bg=Theme.BG)

        style.configure(".", font=Theme.FONT_UI, background=Theme.BG, foreground=Theme.TEXT)
        style.configure("TFrame", background=Theme.BG)
        style.configure("Panel.TFrame", background=Theme.PANEL)

        style.configure("TLabel", background=Theme.BG, foreground=Theme.TEXT)
        style.configure("Muted.TLabel", foreground=Theme.MUTED)
        style.configure("Title.TLabel", font=Theme.FONT_TITLE)
        style.configure("KPI.TLabel", font=("Segoe UI", 16, "bold"), foreground=Theme.ACCENT)
        style.configure("KPIPos.TLabel", font=("Segoe UI", 12, "bold"), foreground="#0f766e")
        style.configure("KPINeg.TLabel", font=("Segoe UI", 12, "bold"), foreground="#b91c1c")
        style.configure("KPIBase.TLabel", font=("Segoe UI", 12, "bold"), foreground=Theme.TEXT)

        style.configure("Topbar.TFrame", background=Theme.PANEL)
        style.configure("Sidebar.TFrame", background=Theme.PANEL)
        style.configure("Sidebar.TButton", anchor="w", padding=(12, 8))

            # KPI cards
        style.configure("KpiCard.TFrame", background=Theme.PANEL, relief="solid", borderwidth=1)
        style.configure("KpiTitle.TLabel", background=Theme.PANEL, foreground=Theme.MUTED, font=("Segoe UI", 9))
        style.configure("KpiValue.TLabel", background=Theme.PANEL, foreground=Theme.TEXT, font=("Segoe UI", 12, "bold"))
        style.configure("KpiValuePos.TLabel", background=Theme.PANEL, foreground="#0f766e", font=("Segoe UI", 12, "bold"))
        style.configure("KpiValueNeg.TLabel", background=Theme.PANEL, foreground="#b91c1c", font=("Segoe UI", 12, "bold"))

        style.configure("TEntry", padding=(6, 4))
        style.configure("TCombobox", padding=(6, 4))
        style.configure("TSpinbox", padding=(6, 4))
        style.configure("TButton", padding=(10, 6))

        # Treeview
        style.configure("Treeview",
                        background=Theme.PANEL,
                        fieldbackground=Theme.PANEL,
                        foreground=Theme.TEXT,
                        rowheight=24,
                        bordercolor=Theme.BORDER,
                        borderwidth=1)
        style.configure("Treeview.Heading",
                        background=Theme.BG,
                        foreground=Theme.TEXT,
                        font=Theme.FONT_UI_BOLD)
        style.map("Treeview",
                  background=[("selected", "#e5edff")],
                  foreground=[("selected", Theme.TEXT)])
        return style
