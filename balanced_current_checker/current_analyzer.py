from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from dico_reader import MachineGroup

try:
    from can_mux_reader import CanSignal
except ImportError:
    CanSignal = None


@dataclass
class ImbalanceEvent:
    machine: str
    start_time: float
    end_time: float
    duration_s: float
    max_abs_sum: float
    mean_abs_sum: float
    n_samples: int
    can_snapshot: Dict[str, float] = field(default_factory=dict)


@dataclass
class MachineAnalysis:
    machine: str
    threshold_a: float
    window_ms: float
    n_samples: int
    total_samples: int
    compensated_samples: int
    uncompensated_samples: int
    pct_compensated: float
    pct_uncompensated: float
    events: List[ImbalanceEvent] = field(default_factory=list)
    sum_series: Optional[pd.Series] = None
    smooth_series: Optional[pd.Series] = None


def _can_snapshot(df: pd.DataFrame, t: float, can_cols: List[str]) -> Dict[str, float]:
    snap = {}
    if not can_cols:
        return snap
    idx = df.index.searchsorted(t, side='left')
    if idx >= len(df):
        idx = len(df) - 1
    for col in can_cols:
        try:
            snap[col] = round(float(df[col].iloc[idx]), 4)
        except (IndexError, ValueError):
            pass
    return snap


def analyze_machine(
    df: pd.DataFrame,
    mg: MachineGroup,
    threshold_a: float,
    window_ms: float,
    min_event_ms: float = 10.0,
    can_columns: Optional[List[str]] = None,
) -> MachineAnalysis:
    ch1, ch2, ch3 = mg.channels
    raster = mg.raster
    window_samples = max(1, int(round(window_ms / 1000.0 / raster)))
    min_event_samples = max(1, int(round(min_event_ms / 1000.0 / raster)))

    sum_inst = df[ch1] + df[ch2] + df[ch3]
    smooth = sum_inst.rolling(window=window_samples, center=True, min_periods=1).mean()
    smooth = smooth.bfill().ffill()

    uncompensated = smooth.abs() > threshold_a

    total = len(uncompensated)
    uncomp_samples = int(uncompensated.sum())
    comp_samples = total - uncomp_samples

    can_cols = can_columns or []

    # Detect events: state machine
    events = []
    in_event = False
    event_start = 0
    event_sum_abs_max = 0.0
    event_sum_abs_sum = 0.0
    event_count = 0

    for i in range(total):
        if uncompensated.iloc[i]:
            if not in_event:
                in_event = True
                event_start = i
                event_sum_abs_max = abs(smooth.iloc[i])
                event_sum_abs_sum = abs(smooth.iloc[i])
                event_count = 1
            else:
                val = abs(smooth.iloc[i])
                if val > event_sum_abs_max:
                    event_sum_abs_max = val
                event_sum_abs_sum += val
                event_count += 1
        else:
            if in_event:
                dur_samples = i - event_start
                if dur_samples >= min_event_samples:
                    mid_t = (df.index[event_start] + df.index[i - 1]) / 2
                    events.append(ImbalanceEvent(
                        machine=mg.machine,
                        start_time=df.index[event_start],
                        end_time=df.index[i - 1],
                        duration_s=df.index[i - 1] - df.index[event_start],
                        max_abs_sum=event_sum_abs_max,
                        mean_abs_sum=event_sum_abs_sum / event_count,
                        n_samples=event_count,
                        can_snapshot=_can_snapshot(df, mid_t, can_cols),
                    ))
                in_event = False

    # Handle event still open at end
    if in_event:
        dur_samples = total - event_start
        if dur_samples >= min_event_samples:
            mid_t = (df.index[event_start] + df.index[total - 1]) / 2
            events.append(ImbalanceEvent(
                machine=mg.machine,
                start_time=df.index[event_start],
                end_time=df.index[total - 1],
                duration_s=df.index[total - 1] - df.index[event_start],
                max_abs_sum=event_sum_abs_max,
                mean_abs_sum=event_sum_abs_sum / event_count,
                n_samples=event_count,
                can_snapshot=_can_snapshot(df, mid_t, can_cols),
            ))

    return MachineAnalysis(
        machine=mg.machine,
        threshold_a=threshold_a,
        window_ms=window_ms,
        n_samples=total,
        total_samples=total,
        compensated_samples=comp_samples,
        uncompensated_samples=uncomp_samples,
        pct_compensated=comp_samples / total * 100 if total else 0.0,
        pct_uncompensated=uncomp_samples / total * 100 if total else 0.0,
        events=events,
        sum_series=sum_inst,
        smooth_series=smooth,
    )


def summarize_analysis(analyses: List[MachineAnalysis]) -> str:
    lines = []
    for a in analyses:
        lines.append(f"[{a.machine}] Threshold: ±{a.threshold_a} A | Window: {a.window_ms} ms")
        lines.append(f"  Total muestras: {a.n_samples}")
        lines.append(f"  Compensado: {a.compensated_samples} ({a.pct_compensated:.1f}%)")
        lines.append(f"  Descompensado: {a.uncompensated_samples} ({a.pct_uncompensated:.1f}%)")
        lines.append(f"  Eventos de descompensación: {len(a.events)}")
        if a.events:
            # Top 5 longest events
            top5 = sorted(a.events, key=lambda e: e.duration_s, reverse=True)[:5]
            lines.append(f"  Top 5 por duración:")
            for i, e in enumerate(top5, 1):
                lines.append(f"    {i}. t={e.start_time:.3f}s -> t={e.end_time:.3f}s "
                             f"(dur={e.duration_s:.3f}s, max|sum|={e.max_abs_sum:.2f}A)")
        lines.append("")
    return "\n".join(lines)


def events_to_dataframe(events: List[ImbalanceEvent]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=[
            'machine', 'start_time_s', 'end_time_s', 'duration_s',
            'max_abs_sum_A', 'mean_abs_sum_A', 'n_samples'
        ])
    rows = []
    for e in events:
        row = {
            'machine': e.machine,
            'start_time_s': round(e.start_time, 4),
            'end_time_s': round(e.end_time, 4),
            'duration_s': round(e.duration_s, 4),
            'max_abs_sum_A': round(e.max_abs_sum, 4),
            'mean_abs_sum_A': round(e.mean_abs_sum, 4),
            'n_samples': e.n_samples,
        }
        # Add CAN snapshot columns flattened
        for k, v in e.can_snapshot.items():
            row[f'can_{k}'] = round(v, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def stats_to_dataframe(analyses: List[MachineAnalysis]) -> pd.DataFrame:
    rows = []
    for a in analyses:
        rows.append({
            'machine': a.machine,
            'threshold_A': a.threshold_a,
            'window_ms': a.window_ms,
            'total_samples': a.total_samples,
            'compensated_samples': a.compensated_samples,
            'uncompensated_samples': a.uncompensated_samples,
            'pct_compensated': round(a.pct_compensated, 2),
            'pct_uncompensated': round(a.pct_uncompensated, 2),
            'num_events': len(a.events),
            'total_uncompensated_time_s': round(
                sum(e.duration_s for e in a.events), 4),
        })
    return pd.DataFrame(rows)
