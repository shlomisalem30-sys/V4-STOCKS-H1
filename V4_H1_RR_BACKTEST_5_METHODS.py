"""V4 H1 research backtest: same V4 entry setup, compare fixed 1R, 1.5R, 2R, 3R and H4 pivot TP.
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
MIN_WARMUP=75

def historical_signals(h1):
    """Evaluate only historical fully closed 60-minute H1 bars. No future pivots."""
    flags=setup_flags(h1)
    h4=h4_from_h1(h1)
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
                  'h4_rr':round(rr,4),'risk':risk}

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
            h1=get_data(sym,'60m','60d').between_time('09:30','16:00')
            if len(h1)<MIN_WARMUP+MAX_HOLD_BARS+1: continue
            busy_until=-1
            for idx,s in historical_signals(h1):
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
                rows.append({'ticker':sym,**{k:v for k,v in s.items() if k!='risk'},**outcomes})
        except Exception as exc:
            LOG.warning('%s failed: %s',sym,exc)
        if n%25==0:LOG.info('Processed %d/%d symbols; %d historical setups',n,len(symbols),len(rows))
    os.makedirs('backtest_results',exist_ok=True)
    output='backtest_results/V4_H1_RR_comparison.csv'
    if rows:
        pd.DataFrame(rows).to_csv(output,index=False)
    else:
        with open(output,'w') as f:f.write('ticker,signal_bar,direction,entry,stop,h4_target,h4_rr\n')
    print('\n=== V4 H1 HISTORICAL BACKTEST ===')
    print('Setups:',len(rows),'\nPeriod: last ~60 calendar days of Yahoo 60m history, where available.')
    print('Same historical signals and next-bar entry for all five methods; no fees/slippage.')
    print('Time exit after',MAX_HOLD_BARS,'hourly bars; same-bar TP/SL collision counted as SL.')
    for label in ('R1','R1_5','R2','R3','H4'):
        vals=[r[label+'_R'] for r in rows if r[label+'_R'] is not None]
        wins=sum(v>0 for v in vals)
        print(f'{label:6s} trades={len(vals):4d} wins={wins:4d} win_rate={wins/len(vals):.1%} total_R={sum(vals):.2f} avg_R={sum(vals)/len(vals):.3f}' if vals else f'{label}: no trades')
    print('Output:',output)
    print('NOTE: fixed targets do NOT impose a minimum H4 RR gate; this compares exits, not filters.')
    print('NOTE: This is research only; Yahoo OHLC bars cannot resolve intrabar order or realistic fills.')

if __name__=='__main__':main()
