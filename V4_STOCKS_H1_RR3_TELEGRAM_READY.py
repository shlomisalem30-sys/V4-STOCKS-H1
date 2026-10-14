"""V4 stock adaptation of the user's original H1 forex reversal logic.

No scoring, RR gate, additional trade filters, or position execution.
Research scanner, not a validated 1:1 TradingView Pine backtest.
"""
from __future__ import annotations
import os
import time
import logging
from io import StringIO
from datetime import datetime, time as clocktime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import yfinance as yf

LOG = logging.getLogger('V4_STOCKS')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
NY = ZoneInfo('America/New_York')
SR_BARS = 24
SR_ATR_TOLERANCE = 0.35
VOL_LEN = 20
VOL_MULTIPLE = 1.30
RSI_LEN = 14
TREND_BARS = 8
FIB_LOOKBACK = 60
PIVOT_LEN = 3
# Forex 3-pip offsets are replaced by per-stock ATR-scaled offsets.
# These are offsets, not extra entry filters.
STOP_ATR_BUFFER = float(os.getenv('STOP_ATR_BUFFER', '0.10'))
TARGET_ATR_BUFFER = float(os.getenv('TARGET_ATR_BUFFER', '0.10'))


def get_symbols():
    # Load the complete S&P 500 universe; no ticker screening or strategy changes.
    # Try Wikipedia with browser-like headers; fall back to a public CSV mirror.
    sources = (
        ('Wikipedia', 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 'html'),
        ('GitHub CSV mirror', 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv', 'csv'),
    )
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; V4StocksH1Research/1.0)',
               'Accept': 'text/html,text/csv,application/xhtml+xml,*/*'}
    for name, url, fmt in sources:
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            if fmt == 'html':
                tables = pd.read_html(StringIO(response.text))
                table = next((t for t in tables if 'Symbol' in t.columns), None)
                if table is None:
                    raise ValueError('No S&P 500 constituent table with Symbol column')
            else:
                table = pd.read_csv(StringIO(response.text))
            if 'Symbol' not in table.columns:
                raise ValueError('Missing Symbol column')
            symbols = table['Symbol'].dropna().astype(str).str.strip().str.replace('.', '-', regex=False).tolist()
            symbols = list(dict.fromkeys(s for s in symbols if s))
            if len(symbols) < 450:
                raise ValueError(f'Incomplete S&P 500 universe: only {len(symbols)} tickers')
            LOG.info('Loaded %d S&P 500 symbols from %s', len(symbols), name)
            return symbols
        except Exception as exc:
            LOG.warning('S&P 500 list source %s failed: %s', name, exc)
    raise RuntimeError('Unable to load a complete S&P 500 constituent list from either source')


def clean(raw):
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df = df[['open', 'high', 'low', 'close', 'volume']].dropna(subset=['open', 'high', 'low', 'close'])
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert(NY)
    return df.sort_index()


def get_data(symbol, interval, period):
    return clean(yf.download(symbol, interval=interval, period=period, auto_adjust=False,
                             progress=False, threads=False, prepost=False))


def rsi(x, n):
    d = x.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100/(1+rs)


def atr(df, n=14):
    prev = df.close.shift()
    tr = pd.concat([(df.high-df.low), (df.high-prev).abs(), (df.low-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def h4_from_h1(h1):
    # US cash session has 6.5 hours, so exchange-session H4 groups differ from 24h FX H4.
    # Group first 4 hourly candles, then remaining session candles; only closed bars used.
    frames = []
    for _, day in h1.groupby(h1.index.date):
        for i in range(0, len(day), 4):
            part = day.iloc[i:i+4]
            if len(part) == 0:
                continue
            frames.append((part.index[-1], part.open.iloc[0], part.high.max(),
                           part.low.min(), part.close.iloc[-1], part.volume.sum()))
    if not frames:
        return pd.DataFrame(columns=['open','high','low','close','volume'])
    return pd.DataFrame(frames, columns=['date','open','high','low','close','volume']).set_index('date')


def pivot_levels(h4):
    highs, lows = [], []
    # Pivot confirmed only after 3 bars to its right; no future leak.
    for i in range(PIVOT_LEN, len(h4)-PIVOT_LEN):
        window = h4.iloc[i-PIVOT_LEN:i+PIVOT_LEN+1]
        confirmed_at = h4.index[i+PIVOT_LEN]
        if h4.high.iloc[i] == window.high.max():
            highs.append((confirmed_at, float(h4.high.iloc[i])))
        if h4.low.iloc[i] == window.low.min():
            lows.append((confirmed_at, float(h4.low.iloc[i])))
    return highs, lows


def setup_flags(h1):
    d = h1.copy()
    o,h,l,c = d.open,d.high,d.low,d.close
    body = (c-o).abs()
    realbody = body.clip(lower=0.000001)
    lower = pd.concat([o,c],axis=1).min(axis=1)-l
    upper = h-pd.concat([o,c],axis=1).max(axis=1)
    bull_engulf = (c>o)&(c.shift(1)<o.shift(1))&(c>=o.shift(1))&(o<=c.shift(1))
    bear_engulf = (c<o)&(c.shift(1)>o.shift(1))&(c<=o.shift(1))&(o>=c.shift(1))
    bull_pin = (c>o)&(lower>=body*1.8)&(upper<=body)
    bear_pin = (c<o)&(upper>=body*1.8)&(lower<=body)
    hammer = (lower>=realbody*2)&(upper<=realbody)
    shooting = (upper>=realbody*2)&(lower<=realbody)
    midbody = body.shift(1)
    firstbody = body.shift(2)
    morning = (c.shift(2)<o.shift(2))&(midbody<firstbody*.5)&(c>o)&(c>=(o.shift(2)+c.shift(2))/2)
    evening = (c.shift(2)>o.shift(2))&(midbody<firstbody*.5)&(c<o)&(c<=(o.shift(2)+c.shift(2))/2)
    # Equity tick size is normally $0.01; exchange-specific tick may vary.
    tweezer_b = (c.shift(1)<o.shift(1))&(c>o)&((l-l.shift(1)).abs()<=.02)
    tweezer_t = (c.shift(1)>o.shift(1))&(c<o)&((h-h.shift(1)).abs()<=.02)
    soldiers = (c>o)&(c.shift(1)>o.shift(1))&(c.shift(2)>o.shift(2))&(c>c.shift(1))&(c.shift(1)>c.shift(2))
    crows = (c<o)&(c.shift(1)<o.shift(1))&(c.shift(2)<o.shift(2))&(c<c.shift(1))&(c.shift(1)<c.shift(2))
    bull = bull_engulf|bull_pin|hammer|morning|tweezer_b|soldiers
    bear = bear_engulf|bear_pin|shooting|evening|tweezer_t|crows
    a = atr(d)
    rs = rsi(c, RSI_LEN)
    support = l.shift(1).rolling(SR_BARS).min()
    resistance = h.shift(1).rolling(SR_BARS).max()
    near_sup = (l<=support+a*SR_ATR_TOLERANCE)&(c>=support-a*SR_ATR_TOLERANCE)
    near_res = (h>=resistance-a*SR_ATR_TOLERANCE)&(c<=resistance+a*SR_ATR_TOLERANCE)
    rsi_bull = (c<c.shift(TREND_BARS))&(rs<=30)
    rsi_bear = (c>c.shift(TREND_BARS))&(rs>=70)
    sh = h.shift(1).rolling(FIB_LOOKBACK).max()
    sl = l.shift(1).rolling(FIB_LOOKBACK).min()
    rng = sh-sl
    fib_long = (rng>a)&(l<=sh-rng*.382)&(l>=sh-rng*.618-a*SR_ATR_TOLERANCE)
    fib_short = (rng>a)&(h>=sl+rng*.382)&(h<=sl+rng*.618+a*SR_ATR_TOLERANCE)
    d['long_setup_prev'] = (bull&(near_sup|rsi_bull|fib_long)).shift(1).fillna(False)
    d['short_setup_prev'] = (bear&(near_res|rsi_bear|fib_short)).shift(1).fillna(False)
    avgvol = d.volume.rolling(VOL_LEN).mean()
    vol_ok = d.volume>avgvol*VOL_MULTIPLE
    d['confirm_long'] = (c>o)&(h>h.shift(1))&vol_ok
    d['confirm_short'] = (c<o)&(l<l.shift(1))&vol_ok
    d['atr'] = a
    return d


def last_closed_asof(frame, instant):
    # Data bars are stamped by start time (Yahoo). Scanner only evaluates at next hourly close.
    # Return the latest row that is safely in the past; excludes incomplete current bar.
    eligible = frame.loc[frame.index < instant]
    return eligible.iloc[-1] if len(eligible) else None


def run_symbol(symbol, spy_h4, spy_day, now):
    h1 = get_data(symbol, '60m', '60d')
    if len(h1)<FIB_LOOKBACK+10:
        return None
    h1 = h1.between_time('09:30', '16:00')
    if len(h1)<FIB_LOOKBACK+10:
        return None
    h4 = h4_from_h1(h1)
    daily = get_data(symbol, '1d', '1y')
    # Evaluate the latest fully CLOSED regular-session 60-minute bar.
    # Yahoo timestamps hourly bars at their OPEN. The 15:30 bar is only
    # 30 minutes long and is intentionally not treated as a full H1 bar.
    closed = h1.loc[
        (h1.index.date == now.date())
        & (h1.index.minute == 30)
        & (h1.index.hour >= 9)
        & (h1.index.hour <= 14)
        & (h1.index + pd.Timedelta(hours=1) <= now)
    ]
    if closed.empty:
        return None
    bar_start = closed.index[-1]
    # Never reuse an old signal if a scheduled run was delayed substantially.
    if now > bar_start + pd.Timedelta(minutes=115):
        return None
    # H4 is assembled from H1, so use only groups whose end candle has closed.
    h4_closed = h4.loc[h4.index + pd.Timedelta(hours=1) <= bar_start]
    spy_h4_closed = spy_h4.loc[spy_h4.index + pd.Timedelta(hours=1) <= bar_start]
    if h4_closed.empty or spy_h4_closed.empty:
        return None
    signals = setup_flags(h1)
    s = signals.loc[bar_start]
    direction = None
    if s.long_setup_prev and s.confirm_long:
        direction='LONG'
    elif s.short_setup_prev and s.confirm_short:
        direction='SHORT'
    if direction is None:
        return None
    entry=float(s.close)
    stop_buffer=float(s.atr)*STOP_ATR_BUFFER
    if direction=='LONG':
        stop=float(signals.low.shift(1).loc[bar_start])-stop_buffer
        if stop>=entry: return None
        target=entry+3.0*(entry-stop)
    else:
        stop=float(signals.high.shift(1).loc[bar_start])+stop_buffer
        if stop<=entry: return None
        target=entry-3.0*(stop-entry)
    rr=3.0
    return {'ticker':symbol,'direction':direction,'bar_time':bar_start.isoformat(),
            'entry':round(entry,4),'stop':round(stop,4),'target':round(target,4),
            'rr':round(rr,2),'atr':round(float(s.atr),4)}


def notify(signal):
    token=os.getenv('TELEGRAM_BOT_TOKEN')
    chat=os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat:
        LOG.warning('No Telegram secrets. Signal: %s',signal)
        return
    msg=(f"V4 STOCKS H1 REVERSAL | RESEARCH\n{signal['ticker']} {signal['direction']}\n"
         f"H1 confirmation: {signal['bar_time']}\nEntry: {signal['entry']}\n"
         f"SL: {signal['stop']} | TP (RR 1:3): {signal['target']}\n"
         f"RR: {signal['rr']} | ATR: {signal['atr']}")
    response=requests.post(f'https://api.telegram.org/bot{token}/sendMessage',
                           data={'chat_id':chat,'text':msg},timeout=20)
    response.raise_for_status()


def main():
    now=datetime.now(NY)
    # RTH only: operational market-session check, not a trading-signal filter.
    if now.weekday()>=5 or not (clocktime(10,40)<=now.time()<=clocktime(16,0)):
        LOG.info('Outside H1 RTH close window, no scan: %s',now)
        return
    spy_h1=get_data('SPY','60m','60d').between_time('09:30','16:00')
    spy_day=get_data('SPY','1d','1y')
    if spy_h1.empty or spy_day.empty:
        raise RuntimeError('SPY market data missing')
    spy_h4=h4_from_h1(spy_h1)
    symbols=get_symbols()
    LOG.info('Scanning %d S&P 500 stocks',len(symbols))
    matches=[]
    for n,symbol in enumerate(symbols,1):
        try:
            result=run_symbol(symbol,spy_h4,spy_day,now)
            if result:
                matches.append(result)
                notify(result)
        except Exception as exc:
            LOG.warning('Skipped %s: %s',symbol,exc)
        if n%50==0:
            LOG.info('Scanned %d/%d; signals=%d',n,len(symbols),len(matches))
        time.sleep(.1)
    LOG.info('Finished: %d matches',len(matches))

if __name__=='__main__':
    main()
