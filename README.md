# Multi-observer Trading support system

Minimalis, de mukodokeppes **coin-to-coin (BTC-denominalt) backtester**.

## Modell

A rendszerben **csak coin-to-coin trade lehetseges**: a toke mindig valamilyen
coinban tartva van (soha nem USDT/flat, soha nem short). Ezert a megfelelo
baseline nem a "Buy & Hold fiat ara", hanem a **BTC-denominalt** viszonyitas:

> Ha `1 BTC`-t inditasz, a strategia vegen mennyi BTC-d lesz a coin-kozi rotacioval,
> ahhoz kepest, ha vegig csak a BTC-t tartottad volna (`Hold BTC = 1 BTC`).

### Strategia

Tobb coin (alapbeallitas: `BTC-USD`, `ETH-USD`, `SOL-USD`) kozul minden nap a
legnagyobb **momentumu** coinben tartjuk a toket. Csak long poziciok, coin-valtasokkal.
A jel a tegnapi zaroorat alapjan epul (nincs look-ahead bias).

### Modellzett koltsegek

- **Tranzakcios dij (fee)**: `0.1%` oldalankent (`--fee`)
- **Slippage**: `0.05%` (`--slippage`)

## Telepites

```bash
pip install -r requirements.txt
```

## Futtatas

```bash
python backtest.py
```

Grafikon: `btc_denom_backtest.png`

### Osszehasonlitas az eredeti (fiat-alapu) scripttel

Az eredeti pelda scripttel szemben:

- Nincs `Signal = -1` (short) — a modellben csak long rotacio van.
- A baseline nem "Buy & Hold BTC ara", hanem **1 BTC** (BTC-denominalt).
- A hozam BTC-hez viszonyitott (`coin_return - btc_return`), nem fiat %-.
- Modellzett **fee + slippage**.
- Megbízható az ujabb `yfinance`/`pandas` MultiIndex kimenettel.

### Peldak

Mas coinok / parameterek:

```bash
python backtest.py --coins BTC-USD ETH-USD SOL-USD ADA-USD --window 30 --fee 0.0015
```

## Fontos

A backtest eredmenyei (pl. `+88%` CAGR) **nem garancia** a jovo teljesitmenyere.
A crypto piacokon a momentum strategiak erosen tuliljak a tortenelemi adatokat.
A maximalis drawdown (`-75%`+) es a magas volatilitas realis kockazatokat mutat.
**Sajat felelossegre hasznald, ez nem penzugyi tanacs.**
