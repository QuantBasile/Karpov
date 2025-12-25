from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

@dataclass
class Filters:
    start_date: str = ""
    end_date: str = ""


@dataclass
class AppState:
    # Python 3.11+ dataclasses disallow mutable defaults; use default_factory
    filters: Filters = field(default_factory=Filters)
    df_raw: Optional[pd.DataFrame] = None
    df_clean: Optional[pd.DataFrame] = None
    app: Optional[object] = None   # 👈 añade esto
    cp_time_aggs: Optional[dict] = None
    cp_list: Optional[list[str]] = None
    


