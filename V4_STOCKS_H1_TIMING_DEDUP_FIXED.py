"""V4 stock adaptation of the user's original H1 forex reversal logic.

No scoring, RR gate, additional trade filters, or position execution.
Research scanner, not a validated 1:1 TradingView Pine backtest.
"""
from __future__ import annotations
import os
import json
import base64
import math
import time
import logging
from io import StringIO
from datetime import datetime, time as clocktime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import yfinance as yf

LOG = logging.getLogger('V4_STOCKS_FINAL')
VERSION = 'V4 STOCKS H1 RR3 ANY COLOR SR TIMING FIX'
STATE_PATH = '.github/v4_sent_signals.json'
MAX_SIGNAL_AGE_MINUTES = 25
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
    d['long_setup_prev'] = (bull & near_sup).shift(1).fillna(False).astype(bool)
    d['short_setup_prev'] = (bear & near_res).shift(1).fillna(False).astype(bool)
    avgvol = d.volume.rolling(VOL_LEN).mean()
    vol_ok = d.volume>avgvol*VOL_MULTIPLE
    d['confirm_long'] = (h > h.shift(1)) & vol_ok  # Any candle color; wick break is sufficient
    d['confirm_short'] = (l < l.shift(1)) & vol_ok  # Any candle color; wick break is sufficient
    d['atr'] = a
    return d


def last_closed_asof(frame, instant):
    # Data bars are stamped by start time (Yahoo). Scanner only evaluates at next hourly close.
    # Return the latest row that is safely in the past; excludes incomplete current bar.
    eligible = frame.loc[frame.index < instant]
    return eligible.iloc[-1] if len(eligible) else None


def run_symbol(symbol, now):
    h1 = get_data(symbol, '60m', '60d')
    if len(h1) < FIB_LOOKBACK + 10:
        return None
    h1 = h1.between_time('09:30', '16:00')
    if len(h1) < FIB_LOOKBACK + 10:
        return None
    # Yahoo H1 timestamps are the OPEN of the bar. Use fully closed RTH H1 bars.
    # The final 15:30-16:00 candle is 30 minutes, not a full H1 candle.
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
    bar_close = bar_start + pd.Timedelta(hours=1)
    age_minutes = (now - bar_close).total_seconds() / 60
    if not (0 <= age_minutes <= MAX_SIGNAL_AGE_MINUTES):
        return None
    # Prevent look-ahead: indicators are computed only through the confirmed bar.
    signals = setup_flags(h1.loc[:bar_start])
    s = signals.iloc[-1]
    direction = None
    if bool(s.long_setup_prev) and bool(s.confirm_long):
        direction = 'LONG'
    elif bool(s.short_setup_prev) and bool(s.confirm_short):
        direction = 'SHORT'
    if direction is None:
        return None
    entry = float(s.close)
    buffer = float(s.atr) * STOP_ATR_BUFFER
    if not all(math.isfinite(x) for x in (entry, buffer)) or buffer < 0:
        return None
    # Structural stop: beyond BOTH setup and confirmation candles, plus ATR buffer.
    setup = signals.iloc[-2]
    if direction == 'LONG':
        stop = min(float(setup.low), float(s.low)) - buffer
        if not math.isfinite(stop) or stop >= entry:
            return None
        target = entry + 3.0 * (entry - stop)
    else:
        stop = max(float(setup.high), float(s.high)) + buffer
        if not math.isfinite(stop) or stop <= entry:
            return None
        target = entry - 3.0 * (stop - entry)
    if not math.isfinite(target):
        return None
    # TP formula is UNCHANGED: exactly 3x initial stop distance.
    return {'ticker': symbol, 'direction': direction,
            'bar_time': bar_start.isoformat(), 'bar_close': bar_close.isoformat(),
            'entry': round(entry, 4), 'stop': round(stop, 4),
            'target': round(target, 4), 'rr': 3.0,
            'atr': round(float(s.atr), 4), 'age_minutes': round(age_minutes, 1)}


def _github_api(path):
    repo = os.getenv('GITHUB_REPOSITORY', '')
    token = os.getenv('GH_STATE_TOKEN') or os.getenv('GITHUB_TOKEN', '')
    if not repo or not token:
        raise RuntimeError('GITHUB_REPOSITORY and GITHUB_TOKEN required for persistent dedup')
    return f'https://api.github.com/repos/{repo}/contents/{path}', {
        'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28'}


def load_state():
    url, headers = _github_api(STATE_PATH)
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code == 404:
        return {}, None
    r.raise_for_status()
    body = r.json()
    raw = base64.b64decode(body['content']).decode('utf-8')
    return json.loads(raw), body['sha']


def save_state(state, sha):
    url, headers = _github_api(STATE_PATH)
    # Keep recent signal IDs only; prevents repository state from growing indefinitely.
    cutoff = (datetime.now(NY) - timedelta(days=10)).date().isoformat()
    state = {k:v for k,v in state.items() if v >= cutoff}
    content = base64.b64encode((json.dumps(state, sort_keys=True, indent=2) + '\n').encode()).decode()
    body = {'message': 'Record sent V4 stock alert IDs', 'content': content,
            'branch': os.getenv('GITHUB_REF_NAME', 'main')}
    if sha:
        body['sha'] = sha
    r = requests.put(url, headers=headers, json=body, timeout=30)
    r.raise_for_status()
    return r.json()['content']['sha'], state


def notify(signal):
    token=os.getenv('TELEGRAM_BOT_TOKEN')
    chat=os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat:
        raise RuntimeError('Missing Telegram secrets; refusing to mark signal sent')
    msg=(f"{VERSION} | RESEARCH\n{signal['ticker']} {signal['direction']}\n"
         f"H1 confirmation: {signal['bar_time']}\nEntry: {signal['entry']}\n"
         f"SL: {signal['stop']} | TP (RR 1:3): {signal['target']}\n"
         f"RR: {signal['rr']} | ATR: {signal['atr']}")
    response=requests.post(f'https://api.telegram.org/bot{token}/sendMessage',
                           data={'chat_id':chat,'text':msg},timeout=20)
    response.raise_for_status()
    if not response.json().get('ok'):
        raise RuntimeError('Telegram API did not confirm message')


def main():
    now = datetime.now(NY)
    # Every 10 minutes UTC via workflow; NY timezone automatically handles DST.
    # Only scan in the 25 minutes following a full RTH H1 candle close.
    if now.weekday() >= 5 or not (clocktime(10, 30) <= now.time() <= clocktime(15, 55)):
        LOG.info('Outside H1 RTH window: %s', now.isoformat())
        return
    # H1 RTH bars close at xx:30 NY time. Skip scans outside the freshness window.
    if not (now.minute >= 30 and now.minute <= 55):
        LOG.info('Not within 25 minutes of H1 candle close: %s', now.isoformat())
        return
    state, sha = load_state()  # Fail closed if dedup store cannot be accessed.
    symbols = get_symbols()
    LOG.info('%s: scanning %d symbols; NY=%s', VERSION, len(symbols), now.isoformat())
    matches = sent = duplicates = errors = 0
    for n, symbol in enumerate(symbols, 1):
        try:
            result = run_symbol(symbol, now)
            if result is not None:
                matches += 1
                key = '|'.join([result['ticker'], result['direction'], result['bar_time']])
                if key in state:
                    duplicates += 1
                else:
                    notify(result)
                    sent += 1
                    state[key] = now.date().isoformat()
                    # Persist each confirmed send so later scheduled runs won't resend.
                    sha, state = save_state(state, sha)
                    LOG.info('SENT %s %s bar=%s SL=%s TP=%s', symbol,
                             result['direction'], result['bar_time'], result['stop'], result['target'])
        except Exception as exc:
            errors += 1
            LOG.warning('Skipped %s: %s', symbol, exc)
        if n % 50 == 0:
            LOG.info('Progress %d/%d; matches=%d sent=%d dup=%d errors=%d',
                     n, len(symbols), matches, sent, duplicates, errors)
        time.sleep(.1)
    LOG.info('FINISHED matches=%d sent=%d duplicate_suppressed=%d errors=%d',
             matches, sent, duplicates, errors)
    if errors > 0:
        LOG.warning('Some symbols failed; see per-symbol warnings above')


if __name__ == '__main__':
    main()
