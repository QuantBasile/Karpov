# controller/warmup.py
from __future__ import annotations

def warmup_after_load(app) -> None:
    """
    Precompute/render heavy tabs right after load so tab navigation is instant.
    Runs in small steps via Tkinter's event loop to keep UI responsive.
    """
    df = app.state.df_clean if hasattr(app.state, "df_clean") else None
    if df is None:
        return

    # Add here future tabs you want to warm
    steps = [
        lambda: app.views["overview"].render(df),
        lambda: app.views["counterparty"].render(df),
        # Ensure default 'ALL' state is computed (KPIs + plot + table)
        lambda: app.views["counterparty"].reset_filters(),
    ]

    def run_next(i: int = 0):
        if i >= len(steps):
            return
        steps[i]()
        # tiny delay so UI stays responsive
        app.after(1, lambda: run_next(i + 1))

    run_next()
