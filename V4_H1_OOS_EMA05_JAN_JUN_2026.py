"""V4 H1 FULL DIAGNOSTIC: same V4 entry setup, compare fixed 1R, 1.5R, 2R, 3R and H4 pivot TP.
Not investment advice. Yahoo 60m history is limited; historical S&P500 membership is not reconstructed.
"""
import os, csv, logging
from datetime import timedelta
import pandas as pd
import importlib.util
SOURCE='v4_stocks_h1-2-1_FIXED.py'
spec=importlib.util.spec_from_file_location('v4_live_source',SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f'Cannot load existing scanner: {SOURCE}')
v4=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)
get_symbols=v4.get_symbols
get_data=v4.get_data
setup_flags=v4.setup_flags
h4_from_h1=v4.h4_from_h1
pivot_levels=v4.pivot_levels
STOP_ATR_BUFFER=v4.STOP_ATR_BUFFER
TARGET_ATR_BUFFER=v4.TARGET_ATR_BUFFER
NY=v4.NY

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
LOG=logging.getLogger('V4_BACKTEST')
MAX_HOLD_BARS=int(os.getenv('BT_MAX_HOLD_BARS','12'))
MAX_SYMBOLS=int(os.getenv('BT_MAX_SYMBOLS','503'))
TEST_START=pd.Timestamp('2026-01-01',tz=NY)
TEST_END=pd.Timestamp('2026-07-01',tz=NY)
DATA_START='2025-10-01'  # warmup for H4 pivots/EMA, outside test
DATA_END='2026-07-15'    # sufficient exit bars after test window
EMA_LIMIT=0.5
MIN_WARMUP=75



def diagnostic_features(h1):
    """Causal H1 indicators: values at bar i use bars <= i only."""
    d=h1.copy()
    c,h,l=d.close,d.high,d.low
    prev=c.shift(1)
    tr=pd.concat([(h-l),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    up=h.diff(); down=-l.diff()
    plus=up.where((up>down)&(up>0),0.0)
    minus=down.where((down>up)&(down>0),0.0)
    period=14
    smooth=lambda x:x.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
    atr=smooth(tr); pdi=100*smooth(plus)/atr.replace(0,float('nan'))
    mdi=100*smooth(minus)/atr.replace(0,float('nan'))
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,float('nan'))
    d['diag_adx14']=smooth(dx)
    d['diag_plus_di14']=pdi;d['diag_minus_di14']=mdi
    d['diag_ema21']=c.ewm(span=21,adjust=False,min_periods=21).mean()
    d['diag_ema50']=c.ewm(span=50,adjust=False,min_periods=50).mean()
    d['diag_rvol20']=d.volume/d.volume.shift(1).rolling(20,min_periods=20).mean().replace(0,float('nan'))
    d['diag_atr14']=atr
    return d


def as_float(value):
    return round(float(value),6) if pd.notna(value) else None


def diagnostic_snapshot(diag,h4,t,side,entry):
    row=diag.loc[t]
    # H4 bars are stamped at their START; +1h is the end of their last H1 component.
    # This matches the existing backtest's H4 availability rule.
    closed=h4.loc[h4.index+pd.Timedelta(hours=1)<=t]
    h4c=closed.close
    h4ema9=h4c.ewm(span=9,adjust=False,min_periods=9).mean()
    h4ema21=h4c.ewm(span=21,adjust=False,min_periods=21).mean()
    h4trend='UNKNOWN'
    if len(h4c)>=21 and pd.notna(h4ema21.iloc[-1]):
        h4trend='BULL' if h4c.iloc[-1]>h4ema21.iloc[-1] and h4ema9.iloc[-1]>h4ema21.iloc[-1] else ('BEAR' if h4c.iloc[-1]<h4ema21.iloc[-1] and h4ema9.iloc[-1]<h4ema21.iloc[-1] else 'MIXED')
    a=row.diag_atr14
    return dict(adx14=as_float(row.diag_adx14),plus_di14=as_float(row.diag_plus_di14),minus_di14=as_float(row.diag_minus_di14),
        rvol20=as_float(row.diag_rvol20),volume=as_float(row.volume),atr14=as_float(a),
        ema21=as_float(row.diag_ema21),ema50=as_float(row.diag_ema50),
        signal_close=as_float(row.close),signal_body_atr=as_float(abs(row.close-row.open)/a) if a>0 else None,
        signal_range_atr=as_float((row.high-row.low)/a) if a>0 else None,
        entry_ema21_atr=as_float(abs(entry-row.diag_ema21)/a) if a>0 else None,
        signal_ema21_atr=as_float(abs(row.close-row.diag_ema21)/a) if a>0 else None,
        h4_trend=h4trend,h4_close=as_float(h4c.iloc[-1]) if len(h4c) else None,
        h4_ema9=as_float(h4ema9.iloc[-1]) if len(h4c) else None,
        h4_ema21=as_float(h4ema21.iloc[-1]) if len(h4c) else None,
        h4_aligned=int((side=='LONG' and h4trend=='BULL') or (side=='SHORT' and h4trend=='BEAR')),
        signal_hour_ny=t.hour)

def historical_signals(h1):
    """Evaluate only historical fully closed 60-minute H1 bars. No future pivots."""
    flags=setup_flags(h1)
    h4=h4_from_h1(h1)
    diag=diagnostic_features(h1)
    for i in range(MIN_WARMUP,len(flags)-1):
        s=flags.iloc[i]
        t=flags.index[i]
        if t.minute!=30 or not (9<=t.hour<=14):
            continue
        if pd.isna(s.atr) or not s.atr or s.atr<=0:
            continue
        if bool(s.long_setup_prev) and bool(s.confirm_long):
            side='LONG'
        elif bool(s.short_setup_prev) and bool(s.confirm_short):
            side='SHORT'
        else:
            continue
        # Original live scanner selects H4 groups with end H1 bar already closed
        # before the current confirmation bar began.
        closed_h4=h4.loc[h4.index+pd.Timedelta(hours=1)<=t]
        if len(closed_h4)<2*3+1:
            continue
        highs,lows=pivot_levels(closed_h4)
        prior=flags.iloc[i-1]
        sb=float(s.atr)*STOP_ATR_BUFFER
        tb=float(s.atr)*TARGET_ATR_BUFFER
        # Model an actionable trade at the NEXT H1 candle open (not at the already closed signal candle).
        nextbar=flags.iloc[i+1]
        entry=float(nextbar.open)
        if side=='LONG':
            stop=float(prior.low)-sb
            candidates=[p-tb for known,p in highs if known<=t and p-tb>entry]
            target=min(candidates) if candidates else None
            risk=entry-stop
        else:
            stop=float(prior.high)+sb
            candidates=[p+tb for known,p in lows if known<=t and p+tb<entry]
            target=max(candidates) if candidates else None
            risk=stop-entry
        if target is None or risk<=0:
            continue
        rr=abs(target-entry)/risk
        yield i, {'signal_bar':t.isoformat(),'direction':side,'entry_time':flags.index[i+1].isoformat(),
                  'entry':round(entry,5),'stop':round(stop,5),'h4_target':round(target,5),
                  'h4_rr':round(rr,4),'risk':risk,**diagnostic_snapshot(diag,h4,t,side,entry)}

def simulate(h1,idx,signal,rr_target):
    entry=signal['entry']; stop=signal['stop']; side=signal['direction']; risk=signal['risk']
    target=signal['h4_target'] if rr_target is None else entry+(risk*rr_target if side=='LONG' else -risk*rr_target)
    end=min(len(h1),idx+1+MAX_HOLD_BARS)
    for j in range(idx+1,end):
        bar=h1.iloc[j]
        if side=='LONG':
            hit_sl=bar.low<=stop; hit_tp=bar.high>=target
        else:
            hit_sl=bar.high>=stop; hit_tp=bar.low<=target
        # Conservative if both are hit in same 60m candle; gap through stop is not modeled.
        if hit_sl: return -1.0,'SL',j
        if hit_tp: return float((target-entry)/risk if side=='LONG' else (entry-target)/risk),'TP',j
    if end<=idx+1:return None,'NO_DATA',idx
    close=float(h1.close.iloc[end-1])
    r=(close-entry)/risk if side=='LONG' else (entry-close)/risk
    return float(r),'TIME_EXIT',end-1

def main():
    rows=[]; symbols=get_symbols()[:MAX_SYMBOLS]
    LOG.info('Starting backtest on %s symbols, maximum %s bars held',len(symbols),MAX_HOLD_BARS)
    for n,sym in enumerate(symbols,1):
        try:
            # Explicit historical dates; Yahoo 1h history is typically available up to 730 days.
            # Use the scanner's clean() to preserve the original timezone and OHLC conventions.
            raw=v4.yf.download(sym,interval='1h',start=DATA_START,end=DATA_END,
                               auto_adjust=False,progress=False,threads=False,prepost=False)
            h1=v4.clean(raw).between_time('09:30','16:00')
            if len(h1)<MIN_WARMUP+MAX_HOLD_BARS+1: continue
            busy_until=-1
            for idx,s in historical_signals(h1):
                signal_time=pd.Timestamp(s['signal_bar'])
                if not (TEST_START<=signal_time<TEST_END): continue
                if idx<=busy_until:continue  # no overlapping signals in the same ticker
                outcomes={}
                ends=[]
                for label,multiple in [('R1',1.0),('R1_5',1.5),('R2',2.0),('R3',3.0),('H4',None)]:
                    r,why,end=simulate(h1,idx,s,multiple)
                    outcomes[label+'_R']=r
                    outcomes[label+'_exit']=why
                    ends.append(end)
                # Same set of trades for all approaches; conservative avoid overlap until latest exit.
                busy_until=max(ends)
                eligible=(s['direction']=='SHORT' or (s['entry_ema21_atr'] is not None and s['entry_ema21_atr']<=EMA_LIMIT))
                rows.append({'ticker':sym,**{k:v for k,v in s.items() if k!='risk'},'ema05_eligible':int(eligible),**outcomes})
        except Exception as exc:
            LOG.warning('%s failed: %s',sym,exc)
        if n%25==0:LOG.info('Processed %d/%d symbols; %d historical setups',n,len(symbols),len(rows))
    os.makedirs('backtest_results',exist_ok=True)
    output='backtest_results/V4_H1_OOS_JAN_JUN_2026_TRADES.csv'
    if rows:
        pd.DataFrame(rows).to_csv(output,index=False)
    else:
        with open(output,'w') as f:f.write('ticker,signal_bar,direction,entry,stop,h4_target,h4_rr\n')
    print('\n=== V4 H1 HISTORICAL BACKTEST ===')
    print('Setups:',len(rows),'\nOUT OF SAMPLE: 2026-01-01 through 2026-06-30 NY (signal date).')
    print('Same historical signals and next-bar entry for all five methods; no fees/slippage.')
    print('Time exit after',MAX_HOLD_BARS,'hourly bars; same-bar TP/SL collision counted as SL.')
    for label in ('R1','R1_5','R2','R3','H4'):
        vals=[r[label+'_R'] for r in rows if r[label+'_R'] is not None]
        wins=sum(v>0 for v in vals)
        print(f'{label:6s} trades={len(vals):4d} wins={wins:4d} win_rate={wins/len(vals):.1%} total_R={sum(vals):.2f} avg_R={sum(vals)/len(vals):.3f}' if vals else f'{label}: no trades')
    print('Output:',output)
    # Matched-sample comparison: SAME trades, SHORT unchanged, LONG only if EMA21 distance <= 0.5 ATR.
    if rows:
        d=pd.DataFrame(rows)
        summary=[]
        for label in ('ALL','LONG','SHORT'):
            group=d if label=='ALL' else d[d.direction==label]
            for variant,subset in [('BASELINE',group),('LONG_EMA05',group[group.ema05_eligible==1])]:
                for month,part in [('TOTAL',subset)]+[(str(m),g) for m,g in subset.groupby(subset.signal_bar.str.slice(0,7))]:
                    vals=pd.to_numeric(part.R3_R,errors='coerce').dropna()
                    summary.append(dict(side=label,variant=variant,month=month,trades=len(vals),total_R=round(vals.sum(),4),avg_R=round(vals.mean(),5) if len(vals) else None,win_rate=round((vals>0).mean(),4) if len(vals) else None))
        summary_path='backtest_results/V4_H1_OOS_JAN_JUN_2026_COMPARISON.csv'
        pd.DataFrame(summary).to_csv(summary_path,index=False)
        print('=== OUT-OF-SAMPLE 3R / 12 BARS: MATCHED COMPARISON ===')
        for rec in summary:
            if rec['side']=='ALL' and rec['month']=='TOTAL':
                print(rec)
        print('Comparison:',summary_path)
    print('DIAGNOSTIC: ADX14, DI14, RVOL20, H4 trend, EMA distances and candle extensions recorded at signal time.')
    print('NOTE: fixed targets do NOT impose a minimum H4 RR gate; this compares exits, not filters.')
    print('NOTE: This is research only; Yahoo OHLC bars cannot resolve intrabar order or realistic fills.')
    print('NOTE: Historical S&P500 membership is NOT reconstructed; current members introduce survivorship bias.')
    print('NOTE: Selection of trades uses original baseline no-overlap schedule; filtered version does not re-enter freed slots.')

if __name__=='__main__':main()
