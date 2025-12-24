from __future__ import annotations

import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox

from controller.state import AppState
from controller import actions
from ui.theme import Theme
from ui.widgets.statusbar import StatusBar
from ui.views.overview import OverviewView
from ui.views.counterparty import CounterpartyView


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("MM Volume Analyzer (Tkinter) — CP v4")
        self.geometry("1300x780")
        # Crucial: do NOT touch overrideredirect/toolwindow/fullscreen.
        # Let the OS/window-manager decorate the window normally.
        self.resizable(True, True)

        Theme.apply(self)

        self.state = AppState()
        self.state.app = self

        self._q: queue.Queue[tuple[str, object]] = queue.Queue()
        self._worker: threading.Thread | None = None

        self.topbar = ttk.Frame(self, style="Topbar.TFrame")
        self.topbar.pack(side="top", fill="x")

        self.body = ttk.Frame(self, style="TFrame")
        self.body.pack(side="top", fill="both", expand=True)

        self.status = StatusBar(self)
        self.status.pack(side="bottom", fill="x")

        self._build_topbar()
        self._build_body()

        self.after(100, self._poll_queue)

    def _build_topbar(self):
        pad_y = 8

        ttk.Label(self.topbar, text="Start:", style="Muted.TLabel").pack(side="left", padx=(12, 6), pady=pad_y)
        self.ent_start = ttk.Entry(self.topbar, width=12)
        self.ent_start.pack(side="left", pady=pad_y)

        ttk.Label(self.topbar, text="End:", style="Muted.TLabel").pack(side="left", padx=(10, 6), pady=pad_y)
        self.ent_end = ttk.Entry(self.topbar, width=12)
        self.ent_end.pack(side="left", pady=pad_y)

        # Default: last 10 days
        import datetime as _dt
        today = _dt.date.today()
        st = today - _dt.timedelta(days=10)
        self.ent_start.insert(0, st.strftime("%Y-%m-%d"))
        self.ent_end.insert(0, today.strftime("%Y-%m-%d"))

        self.btn_load = ttk.Button(self.topbar, text="Load", command=self.on_load)
        self.btn_load.pack(side="left", padx=(18, 6), pady=pad_y)

        ttk.Label(self.topbar, text=" ", style="Muted.TLabel").pack(side="right", padx=10)

    def _build_body(self):
        self.sidebar = ttk.Frame(self.body, style="Sidebar.TFrame", width=150)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self.content = ttk.Frame(self.body, style="TFrame")
        self.content.pack(side="left", fill="both", expand=True)

        self.views: dict[str, ttk.Frame] = {}
        self.views["overview"] = OverviewView(self.content)
        self.views["counterparty"] = CounterpartyView(self.content)

        for v in self.views.values():
            v.place(relx=0, rely=0, relwidth=1, relheight=1)

        ttk.Label(self.sidebar, text="Navigation", style="Title.TLabel").pack(anchor="w", padx=12, pady=(12, 6))
        ttk.Button(self.sidebar, text="Overview", style="Sidebar.TButton",
                   command=lambda: self.show("overview")).pack(fill="x", padx=10, pady=4)
        ttk.Button(self.sidebar, text="1.1 Counterparty", style="Sidebar.TButton",
                   command=lambda: self.show("counterparty")).pack(fill="x", padx=10, pady=4)

        self.show("overview")

    def show(self, key: str):
        self.views[key].tkraise()
        # render current data
        if key == "overview":
            self.views["overview"].render(self.state.df_clean)
        elif key == "counterparty":
            self.views["counterparty"].render(self.state.df_clean)

    def _set_busy(self, busy: bool, msg: str = ""):
        if busy:
            self.btn_load.config(state="disabled")
            self.status.set(msg or "Working...")
        else:
            self.btn_load.config(state="normal")
            self.status.set(msg or "Ready.")

    def _sync_filters_from_ui(self):
        self.state.filters.start_date = self.ent_start.get().strip()
        self.state.filters.end_date = self.ent_end.get().strip()

    def on_load(self):
        self._sync_filters_from_ui()
        self._run_bg("load")

    def _run_bg(self, kind: str):
        if self._worker is not None and self._worker.is_alive():
            return

        self._set_busy(True, "Starting...")

        def progress(msg: str):
            self._q.put(("status", msg))

        def job():
            try:
                if kind == "load":
                    actions.load_data(self.state, progress=progress)
                    self._q.put(("loaded", None))
                else:
                    raise ValueError("Unknown job")
            except Exception as e:
                self._q.put(("error", e))

        self._worker = threading.Thread(target=job, daemon=True)
        self._worker.start()

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "status":
                    self.status.set(str(payload))
                elif kind == "loaded":
                    self._after_loaded()
                elif kind == "error":
                    self._set_busy(False, "Error.")
                    messagebox.showerror("Error", str(payload))
        except queue.Empty:
            pass
        finally:
            self.after(120, self._poll_queue)

    def _after_loaded(self):
        n = 0 if self.state.df_clean is None else len(self.state.df_clean)
        self._set_busy(False, f"Loaded {n:,} trades.")
        self.show("overview")
