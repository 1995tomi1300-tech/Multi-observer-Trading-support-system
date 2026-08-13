#!/usr/bin/env python3
"""
Coin-to-coin (BTC-denominated) backtester.

A "csak coin-to-coin trade lehetseges" modellben a toke mindig valamilyen
coinban tartva van (soha nem USDT/flat, soha nem short). Ezert a megfelelo
baseline nem a "Buy & Hold BTC fiat ara", hanem a **BTC-denominalt** viszonyitas:
ha 1 BTC-t inditasz, a strategia vegen mennyi BTC-d lesz (rotacios strategia),
ehhez kepest mennyi lenne, ha vegig csak a BTC-t tartottad volna (Hold BTC).

Strategia: tobb coin kozul (pl. BTC, ETH, SOL) a momentum alapjan a legjobb
teljesitmenyuben tartjuk a toket. Csak long poziciok, coin-valtasokkal.
Tranzakcios koltseg (fee) es slippage modellezve.

Hasznalat:
    pip install yfinance pandas matplotlib numpy
    python backtest.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # fejkent mentunk, nem interaktiv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class BacktestConfig:
    coins: tuple[str, ...] = ("BTC-USD", "ETH-USD", "SOL-USD")
    btc_ticker: str = "BTC-USD"
    start: str = "2020-01-01"
    end: str = "2026-08-13"
    # Strategiavalaszto: "momentum" (eredeti) vagy "rsi_ma" (RSI + MA szuro).
    strategy: str = "momentum"
    # Momentum ablak (napok): ez hatarozza meg, melyik coint tartjuk.
    momentum_window: int = 20
    # RSI parameterek (rsi_ma strategia).
    rsi_period: int = 14
    rsi_oversold: float = 30.0    # RSI ez alatt -> gyenge (long kandidatus ha visszater)
    rsi_overbought: float = 70.0  # RSI e felett -> tulvett (long-ot keruljuk)
    # Mozgo atlagok (rsi_ma strategia): gyors > lassu -> uptrend szuro.
    ma_short: int = 20
    ma_long: int = 50
    # Stop-loss (BTC-denominalt): ha az aktualis pozicio erteke ennyit esik
    # a belepessel osszevetve (pl. -0.10 = -10%), akkor BTC-be valt (vedelem).
    stop_loss: float = 0.0
    # Koltseg egy coin->coin valtasnál (pl. 0.001 = 0.1% oldalankent).
    fee: float = 0.001
    # Slippage: az arfolyamhoz kepest rosszabb teljesitesi ar (0.0005 = 0.05%).
    slippage: float = 0.0005


def download_prices(coins: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Letolti a napi zarorarokat es egy (datum x coin) DataFrame-et ad vissza.

    Az ujabb yfinance verziok MultiIndexet adnak vissza meg egyetlen ticker
    eseten is; ezt itt egy plain oszlopokkal rendelkezo DataFrame-e simitjuk.
    """
    raw = yf.download(list(coins), start=start, end=end, progress=False)
    # MultiIndex: (mezo, ticker) -> csak a Close-t vesszuk
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": coins[0]})

    close.index = pd.to_datetime(close.index)
    close = close.sort_index().dropna(how="all")
    # Biztositsuk a sorrendet es a neveket
    close = close[list(coins)]
    return close


def momentum_signal(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Momentum rangsor: minden napra mekkora a window-os hozam coinkent.

    A legnagyobb momentumu coin kap 1-es sulyt (abban tartjuk a toket).
    """
    ret = prices.pct_change(window)
    # Minden sorban a legnagyobb erteku coin kap 1-est (one-hot).
    # Ha minden NaN (ablak elotti idoszak), akkor nincs pozicio.
    ranks = ret.rank(axis=1, method="first", ascending=False)
    weights = (ranks == 1).astype(float)
    weights[ret.isna().all(axis=1)] = np.nan
    return weights


def rsi(prices: pd.DataFrame, period: int) -> pd.DataFrame:
    """Relative Strength Index coinkent (Wilder-fele simitas)."""
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder atlag (exponencialis), ugralva az elso periodon atlagot hasznalunk.
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return rsi_val


def rsi_ma_signal(prices: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """RSI + MA szurovel mukodo long rotacios strategia.

    Logika coinkent:
      - Uptrend szuro: MA(rovid) > MA(hosszu).
      - Belepesi feltetel: RSI < oversold OR (RSI allit vissza az oversold ala
        es az uptrend meg all).
      - Kilepesi (tulvett): RSI > overbought -> BTC-be valt.
    Minden napra a legjobb kandidatus kap 1-es sulyt (csak-long rotacio).
    Ha nincs ervenyes kandidatus, a BTC (baseline) kap sulyt.
    """
    btc = cfg.btc_ticker
    coins = [c for c in prices.columns]
    ma_short = prices.rolling(cfg.ma_short).mean()
    ma_long = prices.rolling(cfg.ma_long).mean()
    rsi_val = rsi(prices, cfg.rsi_period)

    uptrend = ma_short > ma_long  # bool DataFrame

    # Long kandidatusok: uptrend es NEM tulvett (RSI < overbought).
    # Elonyben a nem-tulvett uptrend coinek; koztuk a legalacsonyabb RSI-uek
    # (meg mindig feltolthetoek).
    candidate = uptrend & (rsi_val < cfg.rsi_overbought)
    # Rang: az alacsonyabb RSI jobb (tobb or a novesig).
    # A jelolt coinek kozul a legkisebb RSI-uet valasztjuk.
    # Ha egyetlen jelolt sincs, BTC-t tartunk (flat a baseline-ban).
    rsi_ranked = rsi_val.where(candidate, np.nan)
    # Ahol legalabb egy ervenyes kandidatus van, a legkisebb RSI-uet valasztjuk;
    # ahol minden NaN, ott BTC-t (flat baseline) tartunk.
    has_candidate = rsi_ranked.notna().any(axis=1)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights.loc[~has_candidate, btc] = 1.0
    if has_candidate.any():
        sub = rsi_ranked.loc[has_candidate]
        chosen_idx = sub.idxmin(axis=1)
        for ts, coin in chosen_idx.items():
            if pd.notna(coin):
                weights.at[ts, coin] = 1.0
    # Ablak elotti idoszak: nincs jel -> BTC (flat baseline).
    warmup = ma_long.isna().any(axis=1) | rsi_val.isna().any(axis=1)
    weights.loc[warmup] = 0.0
    weights.loc[warmup, btc] = 1.0
    return weights


def _apply_stop_loss(
    pos: pd.DataFrame, rel_ret: pd.DataFrame, stop_loss: float, btc_ticker: str
) -> pd.DataFrame:
    """BTC-denominalt stop-loss: ha egy nem-BTC pozicio tobbet esik a belepessel
    osszevetve, mint stop_loss (pl. -0.10), akkor BTC-be valt.

    A szamitas BTC-denominalt: a rel_ret mar BTC-hez viszonyitott, tehat a
    kumulalt (1+rel_ret).cumprod() adja meg, hogy 1 BTC indulassal mennyi
    BTC-t er a pozicio. Ha ez alamesik (1 + stop_loss) alatt, stop.
    A stop esemeny utan a pozicio BTC lesz, amig a stratégia újra jelez.
    A belepés reference-jet minden poziciovaltasnal ujra kell merjük.
    """
    pos = pos.copy().fillna(0.0)
    # Kumulalt BTC-denominalt hozam a JELszerinti pozicioval (stop nelkul).
    # Ez a "hypothetical" kumulalt, amihez a stop-et hasonlitjuk.
    # Egyszerusitett modell: az aktualis pozicio BTC-denominalt P&L-jet
    # a poziciovaltas ota kumuláljuk.
    pos_changed = (pos.diff().abs().sum(axis=1) > 0)
    # Cumulativ BTC-denominalt hozam szakaszonkent (poziciovaltasok kozott).
    cum = pd.DataFrame(index=rel_ret.index, columns=rel_ret.columns, dtype=float)
    running = pd.Series(1.0, index=rel_ret.index)
    last_change_idx = rel_ret.index[0]
    for i, ts in enumerate(rel_ret.index):
        if i == 0:
            running.loc[ts] = 1.0
            continue
        prev = rel_ret.index[i - 1]
        # A tegnapi pozicio * mai rel_ret -> mai BTC-denom hozam
        r = (pos.loc[prev] * rel_ret.loc[ts]).sum()
        running.loc[ts] = running.loc[prev] * (1.0 + r)
        # Poziciovaltasnal ujra merjük a referenciat
        if pos_changed.loc[ts]:
            running.loc[ts] = 1.0
        # Stop: ha az aktualis pozicio NEM BTC es a running < 1 + stop_loss
        cur_coin = pos.loc[ts].idxmax() if pos.loc[ts].sum() > 0 else None
        if cur_coin is not None and cur_coin != btc_ticker and running.loc[ts] < (1.0 + stop_loss):
            pos.loc[ts] = 0.0
            pos.loc[ts, btc_ticker] = 1.0
            running.loc[ts] = 1.0
    return pos


def backtest_coin_to_coin(
    prices: pd.DataFrame, weights: pd.DataFrame, cfg: BacktestConfig
) -> pd.DataFrame:
    """BTC-denominalt backtest.

    Logika:
      - A toke mindig valamilyen coinban van. Ha a kivalasztott coin valtozik,
        tortenik egy valtas (fee + slippage levonva).
      - A napi hozam: a kivalasztott coin arfolyamvaltozasa BTC-hez viszonyitva.
        Mivel a baseline is BTC, a "BTC arfolyam valtozasa BTC-hez viszonyitva"
        erteke 0 -> ha a kivalasztott coin BTC, a hozam 0 BTC-szinten (helyes).
        Egyeb coin eseten: coin_nap_percent - btc_nap_percent.
      - Az equity BTC-ben kifejezve (1 BTC indul).
    """
    btc = prices[cfg.btc_ticker]
    # Napi szazalekos valtozas coinkent
    daily_pct = prices.pct_change()

    # Eltolt sulyok: a tegnapi zarooratarol szolo jel alapjan ma tartjuk a poz.
    # Ez realisztikus: a zaroorat utan kapott jel a kovetkezo napra ervenyes.
    pos = weights.shift(1)

    # BTC-denominalt napi hozam: ha a kivalasztott coin nem BTC, akkor a
    # coin %-os valtozasa MINUSZ a BTC %-os valtozasa (mert BTC-ben merunk).
    # Ha a kivalasztott coin BTC -> hozam 0 (toke BTC-ben all, nincs valtozas).
    non_btc_coins = [c for c in prices.columns if c != cfg.btc_ticker]
    # (daily_pct - btc_pct) csak a nem-BTC coinokra, BTC-re 0
    rel_ret = daily_pct[non_btc_coins].sub(daily_pct[cfg.btc_ticker], axis=0)
    rel_ret[cfg.btc_ticker] = 0.0

    # --- Stop-loss (BTC-denominalt) ---
    # Ha az aktualis pozicio BTC-denominalt erteke a belepessel osszevetve
    # tobbet esik mint cfg.stop_loss, akkor a toke BTC-be valt (vedelem).
    # A stop csak akkor ervenyesul, ha a pozicio NEM mar BTC (kulonben nincs mit
    # vedezni, a baseline mar is BTC).
    if cfg.stop_loss and cfg.stop_loss < 0:
        pos = _apply_stop_loss(pos, rel_ret, cfg.stop_loss, cfg.btc_ticker)

    # A valaszott coin hozama (BTC-denominalt), NaN hol nem volt pozicio
    selected_ret = (pos * rel_ret).sum(axis=1)

    # Valtasi koltseg: ha a pozicio valtozott egynaprol a masikra.
    turnover = (pos.fillna(0).diff().abs()).sum(axis=1)
    cost = turnover * (cfg.fee + cfg.slippage)

    net_ret = selected_ret - cost

    # BTC-denominalt equity: 1 BTC indul, az ujBTC = regiBTC * (1 + net_ret)
    equity = (1 + net_ret).cumprod()
    # Hold BTC baseline: mindig 1 BTC (definicio szerint)
    hold_btc = pd.Series(1.0, index=equity.index)

    out = pd.DataFrame(
        {
            "equity_btc": equity,
            "hold_btc": hold_btc,
            "net_return": net_ret,
            "turnover": turnover,
        }
    )
    return out


def summarize(result: pd.DataFrame, prices: pd.DataFrame, cfg: BacktestConfig) -> dict:
    eq = result["equity_btc"]
    n_years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    final = float(eq.iloc[-1])
    hold = float(result["hold_btc"].iloc[-1])
    # CAGR (BTC-denominalt)
    cagr = final ** (1 / n_years) - 1
    hold_cagr = hold ** (1 / n_years) - 1
    # Max drawdown (BTC-ben)
    peak = eq.cummax()
    drawdown = eq / peak - 1
    max_dd = float(drawdown.min())
    # Volatilitas (napi -> annualizalt)
    vol = float(result["net_return"].std() * np.sqrt(365))
    sharpe = float(result["net_return"].mean() / (result["net_return"].std() + 1e-12) * np.sqrt(365))
    total_turnover = float(result["turnover"].sum())
    return {
        "final_equity_btc": final,
        "hold_btc": hold,
        "outperformance_btc": final - hold,
        "strategy_cagr": cagr,
        "hold_cagr": hold_cagr,
        "max_drawdown": max_dd,
        "annual_vol": vol,
        "sharpe": sharpe,
        "total_turnover": total_turnover,
        "n_days": len(eq),
    }


def plot(result: pd.DataFrame, prices: pd.DataFrame, cfg: BacktestConfig, out_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Felsö: BTC-denominalt equity (strategia vs Hold BTC)
    ax = axes[0]
    ax.plot(result.index, result["equity_btc"], label="Strategia (rotacio)", color="green")
    ax.axhline(1.0, color="blue", linestyle="--", label="Hold BTC (baseline)")
    ax.set_title("BTC-denominalt egyenleg (1 BTC indul) — coin-to-coin rotacios strategia")
    ax.set_ylabel("Egyenleg [BTC]")
    ax.legend()
    ax.grid()

    # Also: valtasi forgalom
    ax = axes[1]
    ax.plot(result.index, result["turnover"], color="orange", alpha=0.7)
    ax.set_title("Napi valtasi forgalom (coin-valtasok szama)")
    ax.set_ylabel("Turnover")
    ax.set_xlabel("Datum")
    ax.grid()

    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Coin-to-coin BTC-denominalt backtester")
    parser.add_argument("--coins", nargs="+", default=list(BacktestConfig.coins))
    parser.add_argument("--btc", default=BacktestConfig.btc_ticker)
    parser.add_argument("--start", default=BacktestConfig.start)
    parser.add_argument("--end", default=BacktestConfig.end)
    parser.add_argument("--strategy", choices=["momentum", "rsi_ma"], default=BacktestConfig.strategy)
    parser.add_argument("--window", type=int, default=BacktestConfig.momentum_window)
    parser.add_argument("--rsi-period", dest="rsi_period", type=int, default=BacktestConfig.rsi_period)
    parser.add_argument("--rsi-oversold", dest="rsi_oversold", type=float, default=BacktestConfig.rsi_oversold)
    parser.add_argument("--rsi-overbought", dest="rsi_overbought", type=float, default=BacktestConfig.rsi_overbought)
    parser.add_argument("--ma-short", dest="ma_short", type=int, default=BacktestConfig.ma_short)
    parser.add_argument("--ma-long", dest="ma_long", type=int, default=BacktestConfig.ma_long)
    parser.add_argument("--stop-loss", dest="stop_loss", type=float, default=BacktestConfig.stop_loss,
                        help="BTC-denominalt stop-loss (negativ, pl. -0.10 = -10%%). 0 = kikapcsolva.")
    parser.add_argument("--fee", type=float, default=BacktestConfig.fee)
    parser.add_argument("--slippage", type=float, default=BacktestConfig.slippage)
    parser.add_argument("--out", default="btc_denom_backtest.png")
    args = parser.parse_args()

    cfg = BacktestConfig(
        coins=tuple(args.coins),
        btc_ticker=args.btc,
        start=args.start,
        end=args.end,
        strategy=args.strategy,
        momentum_window=args.window,
        rsi_period=args.rsi_period,
        rsi_oversold=args.rsi_oversold,
        rsi_overbought=args.rsi_overbought,
        ma_short=args.ma_short,
        ma_long=args.ma_long,
        stop_loss=args.stop_loss,
        fee=args.fee,
        slippage=args.slippage,
    )

    if cfg.btc_ticker not in cfg.coins:
        raise SystemExit(f"A baseline ({cfg.btc_ticker}) szerepelnie kell a --coins kozt.")

    print(f"Adatok letoltve: {cfg.coins} ({cfg.start} .. {cfg.end}) ...")
    prices = download_prices(cfg.coins, cfg.start, cfg.end)
    print(f"Letoltve {len(prices)} nap, oszlopok: {list(prices.columns)}")

    if cfg.strategy == "rsi_ma":
        weights = rsi_ma_signal(prices, cfg)
    elif cfg.strategy == "momentum":
        weights = momentum_signal(prices, cfg.momentum_window)
    else:
        raise SystemExit(f"Ismeretlen strategia: {cfg.strategy}")
    result = backtest_coin_to_coin(prices, weights, cfg)
    stats = summarize(result, prices, cfg)

    print("\n--- EREDMENYEK (BTC-denominalt, coin-to-coin) ---")
    print(f"Coins:                        {cfg.coins}")
    print(f"Strategia:                    {cfg.strategy}")
    print(f"Momentum ablak:               {cfg.momentum_window} nap")
    print(f"RSI / MA:                     period={cfg.rsi_period}, oversold={cfg.rsi_oversold}, overbought={cfg.rsi_overbought}, MA={cfg.ma_short}/{cfg.ma_long}")
    print(f"Stop-loss:                    {cfg.stop_loss:.1%}" if cfg.stop_loss else "Stop-loss:                    kikapcsolva")
    print(f"Fee + slippage / valtas:      {cfg.fee + cfg.slippage:.4%}")
    print(f"Idoszak (napok):              {stats['n_days']}")
    print(f"Strategia vegso egyenleg:     {stats['final_equity_btc']:.4f} BTC (1 BTC indul)")
    print(f"Hold BTC baseline:            {stats['hold_btc']:.4f} BTC")
    print(f"Tobblet (strat - hold):       {stats['outperformance_btc']:+.4f} BTC")
    print(f"Strategia CAGR (BTC):         {stats['strategy_cagr']:.2%}")
    print(f"Hold BTC CAGR (definicio):    {stats['hold_cagr']:.2%}")
    print(f"Max drawdown:                 {stats['max_drawdown']:.2%}")
    print(f"Eves volatilitas:             {stats['annual_vol']:.2%}")
    print(f"Sharpe (BTC-denom):           {stats['sharpe']:.2f}")
    print(f"Osszes turnover:             {stats['total_turnover']:.1f} valtas")

    plot(result, prices, cfg, args.out)
    print(f"\nGrafikon mentve: {args.out}")


if __name__ == "__main__":
    main()
