import numpy as np
import pandas as pd
import string


# -------------------------
# Helpers
# -------------------------

def _random_isin(n: int, rng: np.random.Generator, country_weights=None) -> np.ndarray:
    countries = np.array(["DE", "ES", "FR", "IT", "NL", "BE", "CH", "GB", "US"])
    if country_weights is None:
        country_weights = np.array([0.30, 0.15, 0.10, 0.10, 0.08, 0.07, 0.06, 0.07, 0.07])
        country_weights = country_weights / country_weights.sum()

    cc = rng.choice(countries, size=n, p=country_weights)
    alphabet = np.array(list(string.ascii_uppercase + string.digits))
    body = rng.choice(alphabet, size=(n, 9))
    check = rng.integers(0, 10, size=n).astype(str)
    return np.array([cc[i] + "".join(body[i]) + check[i] for i in range(n)])


def _third_friday(ts: pd.Timestamp) -> pd.Timestamp:
    base = ts.replace(day=1)
    fridays = pd.date_range(base, base + pd.offsets.MonthEnd(0), freq="W-FRI")
    return fridays[2] if len(fridays) >= 3 else fridays[-1]


def _sample_intraday_seconds(n: int, rng: np.random.Generator, start_sec: int, end_sec: int) -> np.ndarray:
    """
    (1) Intraday clusters: open / lunch / close
    Returns offsets (seconds) in [start_sec, end_sec).
    """
    mu = np.array([
        8 * 3600 + 25 * 60,   # 08:25
        13 * 3600 + 10 * 60,  # 13:10
        17 * 3600 +  5 * 60,  # 17:05
    ])
    sigma = np.array([35 * 60, 55 * 60, 30 * 60])
    w = np.array([0.42, 0.18, 0.40])

    comp = rng.choice(3, size=n, p=w)
    sec = rng.normal(loc=mu[comp], scale=sigma[comp], size=n)
    sec = np.clip(sec, start_sec, end_sec - 1).astype(np.int64)
    return sec - start_sec


def _build_counterparty_weights(counterparties: np.ndarray, rng: np.random.Generator):
    """
    (3) Counterparty skew by UND_TYPE (FUT/STO/COM)
    """
    und_types = np.array(["FUT", "STO", "COM"])
    n_cp = len(counterparties)

    cp_pref_type = rng.choice(und_types, size=n_cp, p=[0.25, 0.65, 0.10])
    activity = rng.lognormal(mean=0.0, sigma=0.8, size=n_cp)
    activity = activity / activity.sum()

    weights = {}
    for t in und_types:
        boost = np.where(cp_pref_type == t, 6.0, 1.0)
        w = activity * boost
        weights[t] = w / w.sum()
    return weights


def _markov_sides_for_groups(isin_codes: np.ndarray,
                             base_buy_prob: np.ndarray,
                             rng: np.random.Generator,
                             persistence: float = 0.93) -> np.ndarray:
    """
    (2) Book pressure: buy/sell streaks per ISIN (simple Markov).
    Input must be sorted by (ISIN, TradeTime).
    """
    n = len(isin_codes)
    sides = np.empty(n, dtype=object)
    last_isin = None
    last_is_buy = None

    for i in range(n):
        isin = isin_codes[i]
        p0 = base_buy_prob[i]
        if isin != last_isin:
            last_is_buy = (rng.random() < p0)
            last_isin = isin
        else:
            p_buy = persistence * (1.0 if last_is_buy else 0.0) + (1.0 - persistence) * p0
            last_is_buy = (rng.random() < p_buy)
        sides[i] = "buy" if last_is_buy else "sell"
    return sides


def _dt64D_to_str(arr_dt64D: np.ndarray) -> np.ndarray:
    # Safe for 9999-12-31 because dtype is [D], not [ns]
    return arr_dt64D.astype("datetime64[D]").astype(str)


def _make_toxic_pairs_for_day(
    *,
    rng,
    d0: pd.Timestamp,
    start_sec: int,
    end_sec: int,
    alive_idx: np.ndarray,
    isins: np.ndarray,
    call_put: np.ndarray,
    expiry_D: np.ndarray,       # datetime64[D]
    knock_date_D: np.ndarray,   # datetime64[D]
    knock_ts_ns: np.ndarray,    # datetime64[ns]
    ratio: np.ndarray,
    strike: np.ndarray,
    inst_und_idx: np.ndarray,
    und_names: np.ndarray,
    und_type_map: np.ndarray,
    cp_weights_by_type: dict,
    counterparties: np.ndarray,
    daily_factor_row: np.ndarray,
    und_base_ref: np.ndarray,
    n_trades_day: int,
    pair_rate_60: float,
    pair_rate_600: float,
) -> dict:
    """
    Inject sell->buy pairs with same (ISIN, CP, Qty, day), within <=60s or <=600s bands.
    """

    pair_rate_60 = float(np.clip(pair_rate_60, 0.0, 0.90))
    pair_rate_600 = float(np.clip(pair_rate_600, 0.0, 0.90))
    pair_rate_600 = max(pair_rate_600, pair_rate_60)

    paired_trades_60 = int(np.floor(n_trades_day * pair_rate_60))
    paired_trades_600 = int(np.floor(n_trades_day * pair_rate_600))

    n_pairs_short = paired_trades_60 // 2
    n_pairs_total_600 = paired_trades_600 // 2
    n_pairs_medium = max(0, n_pairs_total_600 - n_pairs_short)

    n_pairs = n_pairs_short + n_pairs_medium
    if n_pairs <= 0 or alive_idx.size == 0:
        return {"n": 0}

    # cap
    n_pairs = min(n_pairs, n_trades_day // 2)

    inst_idx = rng.choice(alive_idx, size=n_pairs, replace=True)

    # sell times (uniform-ish) with margin for 10min
    base_sec = rng.integers(start_sec, end_sec - 601, size=n_pairs)
    base_ns = rng.integers(0, 1_000_000_000, size=n_pairs)
    sell_time = (d0 + pd.to_timedelta(base_sec, unit="s") + pd.to_timedelta(base_ns, unit="ns")).to_numpy("datetime64[ns]")

    # delays: short 1..60s, medium 61..600s
    short = rng.integers(1, 61, size=n_pairs_short) if n_pairs_short > 0 else np.array([], dtype=int)
    med = rng.integers(61, 601, size=n_pairs_medium) if n_pairs_medium > 0 else np.array([], dtype=int)
    delay_s = np.concatenate([short, med])
    if delay_s.size < n_pairs:
        delay_s = np.pad(delay_s, (0, n_pairs - delay_s.size), mode="wrap")
    delay_s = delay_s[:n_pairs]
    rng.shuffle(delay_s)

    buy_time = sell_time + delay_s.astype("timedelta64[s]")
    
    # jitter independiente para que no coincidan los nanosegundos
    # (<= 500 ms para no romper ventanas de 60s)
    jitter_ns = rng.integers(0, 500_000_000, size=buy_time.size, dtype=np.int64)  # 0..500ms
    buy_time = buy_time + jitter_ns.astype("timedelta64[ns]")

    # KO intraday constraint
    kt = knock_ts_ns[inst_idx]
    ok = (sell_time <= kt) & (buy_time <= kt)
    inst_idx = inst_idx[ok]
    sell_time = sell_time[ok]
    buy_time = buy_time[ok]
    n_pairs_eff = inst_idx.size
    if n_pairs_eff == 0:
        return {"n": 0}

    # instrument attrs
    isin_t = isins[inst_idx]
    cp_flag = call_put[inst_idx]
    exp_D_t = expiry_D[inst_idx]
    kd_D_t = knock_date_D[inst_idx]
    ratio_t = ratio[inst_idx]
    strike_t = strike[inst_idx]
    und_idx_t = inst_und_idx[inst_idx]
    und_name_t = und_names[und_idx_t]
    und_type_t = und_type_map[und_idx_t]

    # CP skew, same CP for both legs
    cp_out = np.empty(n_pairs_eff, dtype=object)
    for t in ["FUT", "STO", "COM"]:
        m = (und_type_t == t)
        if np.any(m):
            cp_out[m] = rng.choice(counterparties, size=m.sum(), p=cp_weights_by_type[t])

    # repeated quantities (more matchable)
    qty_choices = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000], dtype=np.int64)
    qty_probs = np.array([0.05, 0.12, 0.12, 0.14, 0.20, 0.18, 0.12, 0.07])
    qty = rng.choice(qty_choices, size=n_pairs_eff, p=qty_probs)

    # Ref coherent OTM
    base_today = und_base_ref[und_idx_t] * daily_factor_row[und_idx_t]
    intraday_noise = rng.normal(0.0, 0.006, size=n_pairs_eff)
    ref = np.clip(base_today * (1.0 + intraday_noise), 0.01, None)

    is_call = (cp_flag == "C")
    ref[is_call] = np.minimum(ref[is_call], strike_t[is_call] * 0.995)
    ref[~is_call] = np.maximum(ref[~is_call], strike_t[~is_call] * 1.005)

    # Prices: "toxic" pattern (MM sells worse, buys back better) => tends negative pnl for MM
    mid = 0.01 + (np.clip((ref / strike_t), 0, 1) ** 3.0) * 0.8
    spread = np.abs(rng.normal(0.0, 0.008, size=n_pairs_eff)) + 0.002
    price_sell = np.clip(mid - 0.5 * spread, 0.001, None)
    price_buy = np.clip(mid + 0.5 * spread, 0.001, None)

    return {
        "n": n_pairs_eff * 2,
        "sell_time": sell_time,
        "buy_time": buy_time,
        "isin": isin_t,
        "cp_flag": cp_flag,
        "expiry_D": exp_D_t,
        "knock_date_D": kd_D_t,
        "ratio": ratio_t,
        "strike": strike_t,
        "und_name": und_name_t,
        "und_type": und_type_t,
        "counterparty": cp_out,
        "qty": qty,
        "ref": ref,
        "price_sell": price_sell,
        "price_buy": price_buy,
    }


# -------------------------
# Main generator
# -------------------------

def generate_mm_logs(
    start_date: str,
    end_date: str,
    dummy_input=None,
    trades_per_day: int = 30_000,
    n_underlyings: int = 400,
    n_counterparties: int = 20,
    n_instruments: int = 2500,
    seed: int = 42,
    # Pair-rate controls (used as ranges if pair_rates_mode="random_daily")
    pair_rate_60: float = 0.03,
    pair_rate_600: float = 0.15,
    pair_rates_mode: str = "random_daily",     # "fixed" or "random_daily"
    pair_rate_60_range: tuple[float, float] = (0.01, 0.05),
    pair_rate_600_range: tuple[float, float] = (0.05, 0.25),
) -> pd.DataFrame:
    """
    Fake logs of a market maker.

    OUTPUT requirement:
      - TradeTime: datetime64[ns]
      - everything else: strings (Expiry, Knock_Date included)

    Includes:
      (1) intraday clusters
      (2) book pressure on random block
      (3) counterparty skew
      (4) intraday KO (no trades after KO time)
      + injected toxic pairs with daily-varying rates
    """

    rng = np.random.default_rng(seed)

    # Calendar
    days = pd.date_range(start=start_date, end=end_date, freq="D")
    biz_days = days[days.dayofweek < 5]
    if len(biz_days) == 0:
        return pd.DataFrame(columns=[
            "TradeTime", "TradeNo", "ISIN", "b/s", "Quantity", "Price", "Counterparty",
            "Ref", "CALL_OPTION", "Expiry", "Knock_Date", "Ratio", "Strike", "UND_NAME", "UND_TYPE"
        ])

    # Trading hours
    start_sec = 8 * 3600
    end_sec = 17 * 3600 + 30 * 60

    # Underlyings
    und_names = np.array([f"UND_{i:04d}" for i in range(1, n_underlyings + 1)])
    und_types = np.array(["FUT", "STO", "COM"])
    und_type_map = rng.choice(und_types, size=n_underlyings, p=[0.30, 0.60, 0.10])

    und_base_ref = rng.lognormal(mean=4.2, sigma=0.55, size=n_underlyings)
    und_base_ref = np.clip(und_base_ref, 5, 800)

    # Instruments universe
    isins = _random_isin(n_instruments, rng=rng)
    inst_und_idx = rng.integers(0, n_underlyings, size=n_instruments)
    call_put = rng.choice(np.array(["C", "P"]), size=n_instruments, p=[0.55, 0.45])
    ratio = rng.choice(np.array([1.0, 0.1, 0.001]), size=n_instruments, p=[0.12, 0.70, 0.18])

    # Expiry as datetime64[D] (safe for 9999-12-31)
    start_ts = pd.Timestamp(start_date)
    expiry_list = []
    for _ in range(n_instruments):
        if rng.random() < 0.22:
            expiry_list.append(np.datetime64("9999-12-31", "D"))
        else:
            months_ahead = int(rng.integers(1, 25))
            exp_dt = _third_friday(start_ts + pd.DateOffset(months=months_ahead))
            expiry_list.append(np.datetime64(pd.Timestamp(exp_dt).normalize(), "D"))
    expiry_D = np.array(expiry_list, dtype="datetime64[D]")

    # (4) KO intraday
    knock_date_D = np.empty(n_instruments, dtype="datetime64[D]")
    knock_ts_ns = np.empty(n_instruments, dtype="datetime64[ns]")
    MAX_NS = np.datetime64("2262-04-11T23:47:16.854775807")

    for i in range(n_instruments):
        if rng.random() < 0.15:
            kd_raw = rng.choice(biz_days)
            kd = pd.Timestamp(kd_raw).normalize()

            ko_start = 9 * 3600
            ko_end = 16 * 3600 + 45 * 60
            ko_sec = int(rng.integers(ko_start, ko_end))
            ko_ns = int(rng.integers(0, 1_000_000_000))
            kt = kd + pd.to_timedelta(ko_sec, unit="s") + pd.to_timedelta(ko_ns, unit="ns")

            knock_date_D[i] = np.datetime64(kd, "D")
            knock_ts_ns[i] = np.datetime64(kt, "ns")
        else:
            knock_date_D[i] = np.datetime64("9999-12-31", "D")
            knock_ts_ns[i] = MAX_NS

    # Strike OTM by construction
    margin = rng.uniform(0.05, 0.22, size=n_instruments)
    base_ref_inst = und_base_ref[inst_und_idx]
    strike = np.empty(n_instruments, dtype=float)
    is_call_inst = (call_put == "C")
    strike[is_call_inst] = base_ref_inst[is_call_inst] * (1.0 + margin[is_call_inst])
    strike[~is_call_inst] = base_ref_inst[~is_call_inst] * (1.0 - margin[~is_call_inst])
    strike = np.clip(strike, 0.01, None)

    # Counterparties + skew
    counterparties = np.array([f"CP_{i:02d}" for i in range(1, n_counterparties + 1)])
    cp_weights_by_type = _build_counterparty_weights(counterparties, rng)

    # Daily drift per underlying
    daily_steps = rng.normal(0.0, 0.004, size=(len(biz_days), n_underlyings))
    daily_factor = np.exp(np.cumsum(daily_steps, axis=0))

    trade_no = 1
    frames = []

    for day_idx, d_raw in enumerate(biz_days):
        d = pd.Timestamp(d_raw)
        d0 = d.normalize()
        trade_day_D = np.datetime64(d0, "D")

        # alive date-level
        alive_day_mask = knock_date_D >= trade_day_D
        alive_idx = np.where(alive_day_mask)[0]
        if alive_idx.size == 0:
            continue

        # --- daily-varying pair rates ---
        if pair_rates_mode == "random_daily":
            pr60 = float(rng.uniform(pair_rate_60_range[0], pair_rate_60_range[1]))
            pr600 = float(rng.uniform(pair_rate_600_range[0], pair_rate_600_range[1]))
            pr600 = max(pr600, pr60)
        elif pair_rates_mode == "fixed":
            pr60 = float(pair_rate_60)
            pr600 = float(max(pair_rate_600, pr60))
        else:
            raise ValueError("pair_rates_mode must be 'random_daily' or 'fixed'")

        # -------------------------
        # A) Inject toxic pairs
        # -------------------------
        pairs = _make_toxic_pairs_for_day(
            rng=rng,
            d0=d0,
            start_sec=start_sec,
            end_sec=end_sec,
            alive_idx=alive_idx,
            isins=isins,
            call_put=call_put,
            expiry_D=expiry_D,
            knock_date_D=knock_date_D,
            knock_ts_ns=knock_ts_ns,
            ratio=ratio,
            strike=strike,
            inst_und_idx=inst_und_idx,
            und_names=und_names,
            und_type_map=und_type_map,
            cp_weights_by_type=cp_weights_by_type,
            counterparties=counterparties,
            daily_factor_row=daily_factor[day_idx],
            und_base_ref=und_base_ref,
            n_trades_day=trades_per_day,
            pair_rate_60=pr60,
            pair_rate_600=pr600,
        )

        paired_n = int(pairs.get("n", 0))
        paired_n = min(paired_n, trades_per_day - (trades_per_day % 2))
        remaining_trades = trades_per_day - paired_n

        day_frames = []

        if paired_n > 0:
            n_pairs_eff = paired_n // 2

            tt = np.concatenate([pairs["sell_time"], pairs["buy_time"]]).astype("datetime64[ns]")

            isin_t = np.concatenate([pairs["isin"], pairs["isin"]]).astype(str)
            cp_flag = np.concatenate([pairs["cp_flag"], pairs["cp_flag"]]).astype(str)
            exp_D_t = np.concatenate([pairs["expiry_D"], pairs["expiry_D"]])
            kd_D_t = np.concatenate([pairs["knock_date_D"], pairs["knock_date_D"]])
            ratio_t = np.concatenate([pairs["ratio"], pairs["ratio"]]).astype(float)
            strike_t = np.concatenate([pairs["strike"], pairs["strike"]]).astype(float)
            und_name_t = np.concatenate([pairs["und_name"], pairs["und_name"]]).astype(str)
            und_type_t = np.concatenate([pairs["und_type"], pairs["und_type"]]).astype(str)
            cp_out = np.concatenate([pairs["counterparty"], pairs["counterparty"]]).astype(str)
            qty = np.concatenate([pairs["qty"], pairs["qty"]]).astype(np.int64)
            ref = np.concatenate([pairs["ref"], pairs["ref"]]).astype(float)

            side = np.concatenate([np.array(["sell"] * n_pairs_eff), np.array(["buy"] * n_pairs_eff)]).astype(str)
            price = np.concatenate([pairs["price_sell"], pairs["price_buy"]]).astype(float)

            trade_nos = np.arange(trade_no, trade_no + paired_n, dtype=np.int64)
            trade_no += paired_n

            df_pairs = pd.DataFrame({
                "TradeTime": tt,
                "TradeNo": trade_nos.astype(str),
                "ISIN": isin_t,
                "b/s": side,
                "Quantity": qty.astype(str),
                "Price": np.round(price, 6).astype(str),
                "Counterparty": cp_out,
                "Ref": np.round(ref, 6).astype(str),
                "CALL_OPTION": cp_flag,
                "Expiry": _dt64D_to_str(exp_D_t),
                "Knock_Date": _dt64D_to_str(kd_D_t),
                "Ratio": np.where(np.isclose(ratio_t, 1.0), "1", np.where(np.isclose(ratio_t, 0.1), "0.1", "0.001")),
                "Strike": np.round(strike_t, 6).astype(str),
                "UND_NAME": und_name_t,
                "UND_TYPE": und_type_t,
            })
            day_frames.append(df_pairs)

        # -------------------------
        # B) Random remainder
        # -------------------------
        if remaining_trades > 0:
            needed = remaining_trades
            inst_parts = []
            time_parts = []

            while needed > 0:
                batch = int(np.ceil(needed * 1.25))
                inst_idx = rng.choice(alive_idx, size=batch, replace=True)

                sec_offset = _sample_intraday_seconds(batch, rng, start_sec, end_sec)
                ns_offset = rng.integers(0, 1_000_000_000, size=batch)

                times_ns = (d0
                            + pd.to_timedelta(start_sec + sec_offset, unit="s")
                            + pd.to_timedelta(ns_offset, unit="ns")).to_numpy(dtype="datetime64[ns]")

                ok = times_ns <= knock_ts_ns[inst_idx]
                if not np.any(ok):
                    continue

                inst_ok = inst_idx[ok]
                times_ok = times_ns[ok]

                take = min(len(times_ok), needed)
                inst_parts.append(inst_ok[:take])
                time_parts.append(times_ok[:take])
                needed -= take

            inst_idx = np.concatenate(inst_parts)
            times_ns = np.concatenate(time_parts)

            order = np.argsort(times_ns, kind="mergesort")
            inst_idx = inst_idx[order]
            times_ns = times_ns[order]

            isin_t = isins[inst_idx].astype(str)
            cp_flag = call_put[inst_idx].astype(str)
            exp_D_t = expiry_D[inst_idx]
            kd_D_t = knock_date_D[inst_idx]
            ratio_t = ratio[inst_idx].astype(float)
            strike_t = strike[inst_idx].astype(float)
            und_idx_t = inst_und_idx[inst_idx]
            und_name_t = und_names[und_idx_t].astype(str)
            und_type_t = und_type_map[und_idx_t].astype(str)

            # Ref
            base_today = und_base_ref[und_idx_t] * daily_factor[day_idx, und_idx_t]
            intraday_noise = rng.normal(0.0, 0.010, size=remaining_trades)
            ref = np.clip(base_today * (1.0 + intraday_noise), 0.01, None)

            # OTM constraint
            is_call = (cp_flag == "C")
            ref[is_call] = np.minimum(ref[is_call], strike_t[is_call] * 0.995)
            ref[~is_call] = np.maximum(ref[~is_call], strike_t[~is_call] * 1.005)

            # CP skew
            cp_out = np.empty(remaining_trades, dtype=object)
            for t in ["FUT", "STO", "COM"]:
                mask = (und_type_t == t)
                if np.any(mask):
                    cp_out[mask] = rng.choice(counterparties, size=mask.sum(), p=cp_weights_by_type[t])

            # sides with book pressure
            type_buy_bias = {"FUT": 0.52, "STO": 0.50, "COM": 0.48}
            base_p = np.vectorize(type_buy_bias.get)(und_type_t).astype(float)
            base_p = np.clip(base_p + rng.normal(0.0, 0.03, size=remaining_trades), 0.05, 0.95)

            idx_sort = np.lexsort((times_ns, isin_t))  # by ISIN then time
            idx_unsort = np.empty_like(idx_sort)
            idx_unsort[idx_sort] = np.arange(len(idx_sort))
            sides_sorted = _markov_sides_for_groups(isin_t[idx_sort], base_p[idx_sort], rng=rng, persistence=0.93)
            sides = sides_sorted[idx_unsort].astype(str)

            # Quantity
            qty = rng.lognormal(mean=5.4, sigma=0.65, size=remaining_trades)
            qty = np.clip(np.round(qty).astype(np.int64), 1, 50_000)

            # Time to expiry for pricing (date-space)
            trade_days_D = times_ns.astype("datetime64[D]")
            dt_days = (exp_D_t - trade_days_D).astype("timedelta64[D]").astype(np.int64)
            open_end = (exp_D_t == np.datetime64("9999-12-31", "D"))
            dt_days[open_end] = int(2.5 * 365)
            T = np.clip(dt_days / 365.0, 0.0, 2.5)

            # Price
            closeness = np.empty(remaining_trades, dtype=float)
            closeness[is_call] = ref[is_call] / strike_t[is_call]
            closeness[~is_call] = strike_t[~is_call] / ref[~is_call]
            closeness = np.clip(closeness, 0.0, 1.0)

            vol = np.where(und_type_t == "COM", 0.35, np.where(und_type_t == "FUT", 0.28, 0.22))
            vol = vol * np.clip(1.0 + rng.normal(0.0, 0.10, size=remaining_trades), 0.6, 1.6)

            noise = rng.normal(0.0, 0.015, size=remaining_trades)
            premium = 0.01 + (closeness ** 3.2) * (0.35 + 0.65 * np.sqrt(T + 1e-9)) * (0.6 + 0.9 * vol) + noise
            premium *= (0.9 + 0.1 * np.log10(1.0 / np.clip(ratio_t, 1e-6, None)))
            price = np.clip(premium, 0.001, None)

            # TradeNo
            trade_nos = np.arange(trade_no, trade_no + remaining_trades, dtype=np.int64)
            trade_no += remaining_trades

            df_rest = pd.DataFrame({
                "TradeTime": times_ns.astype("datetime64[ns]"),
                "TradeNo": trade_nos.astype(str),
                "ISIN": isin_t,
                "b/s": sides,
                "Quantity": qty.astype(str),
                "Price": np.round(price, 6).astype(str),
                "Counterparty": cp_out.astype(str),
                "Ref": np.round(ref, 6).astype(str),
                "CALL_OPTION": cp_flag,
                "Expiry": _dt64D_to_str(exp_D_t),
                "Knock_Date": _dt64D_to_str(kd_D_t),
                "Ratio": np.where(np.isclose(ratio_t, 1.0), "1", np.where(np.isclose(ratio_t, 0.1), "0.1", "0.001")),
                "Strike": np.round(strike_t, 6).astype(str),
                "UND_NAME": und_name_t,
                "UND_TYPE": und_type_t,
            })
            day_frames.append(df_rest)

        df_day = pd.concat(day_frames, ignore_index=True)
        df_day = df_day.sort_values(["TradeTime", "TradeNo"], kind="mergesort").reset_index(drop=True)
        frames.append(df_day)

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["TradeTime", "TradeNo"], kind="mergesort").reset_index(drop=True)
    return df


