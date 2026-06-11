from typing import Dict, List, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from thd_analyzer import ThdResult, ThdMeasurement

PWM_FREQ_MAP = {1: 10000, 2: 5000, 3: 2000}


@dataclass
class SyncRiskPoint:
    machine: str
    speed_rpm: float
    torque_nm: float
    f_elec_hz: float
    f_sw_hz: int
    carrier_ratio: float
    thd_avg: float
    i5_pct: float
    i7_pct: float
    i11_pct: float
    i13_pct: float
    risk_level: str
    dominant_harmonics: str


@dataclass
class SyncRiskResult:
    me: List[SyncRiskPoint] = field(default_factory=list)
    hsg: List[SyncRiskPoint] = field(default_factory=list)


def _get_pwm_freq_at_segment(df: pd.DataFrame, pwm_ch: str, start_idx: int, end_idx: int) -> int:
    if pwm_ch not in df.columns:
        return 0
    vals = df[pwm_ch].iloc[start_idx:end_idx].dropna()
    if vals.empty:
        return 0
    mode_val = vals.mode()
    if mode_val.empty:
        return 0
    return PWM_FREQ_MAP.get(int(mode_val.iloc[0]), 0)


def _assess_risk(thd_avg: float, harmonics: Dict[int, float], carrier_ratio: float) -> tuple:
    i5 = harmonics.get(5, 0)
    i7 = harmonics.get(7, 0)
    i11 = harmonics.get(11, 0)
    i13 = harmonics.get(13, 0)
    harm_high = [h for h, v in zip(['I5', 'I7', 'I11', 'I13'], [i5, i7, i11, i13]) if v > 3.0]

    if thd_avg > 8.0 or len(harm_high) >= 2 or carrier_ratio < 15:
        risk = 'high'
    elif thd_avg > 3.0 or len(harm_high) >= 1 or carrier_ratio < 21:
        risk = 'medium'
    else:
        risk = 'low'

    dominant = ', '.join(harm_high) if harm_high else '-'
    return risk, dominant


def analyze_sync_risk(
    thd_result: ThdResult,
    df: pd.DataFrame,
    pwm_me: str,
    pwm_hsg: str,
    speed_me: str = 'Wxx_emot_n',
    speed_hsg: str = 'Wxx_emot_n_emot2',
    poles_me: int = 8,
    poles_hsg: int = 4,
) -> SyncRiskResult:
    result = SyncRiskResult()

    for label, mthd, pwm_ch, speed_ch, poles in [
        ('ME', thd_result.me, pwm_me, speed_me, poles_me),
        ('HSG', thd_result.hsg, pwm_hsg, speed_hsg, poles_hsg),
    ]:
        if not mthd or not mthd.measurements:
            continue

        for m in mthd.measurements:
            f_elec = m.speed_rpm * poles / 120.0
            f_sw = _get_pwm_freq_at_segment(df, pwm_ch, m.start_idx, m.end_idx)
            if f_sw == 0:
                continue
            carrier_ratio = f_sw / f_elec if f_elec > 0 else 0
            i5 = round(m.harmonics.get(5, 0), 2) if m.harmonics else 0.0
            i7 = round(m.harmonics.get(7, 0), 2) if m.harmonics else 0.0
            i11 = round(m.harmonics.get(11, 0), 2) if m.harmonics else 0.0
            i13 = round(m.harmonics.get(13, 0), 2) if m.harmonics else 0.0
            risk, dominant = _assess_risk(m.thd_avg, m.harmonics, carrier_ratio)

            pt = SyncRiskPoint(
                machine=label,
                speed_rpm=m.speed_rpm,
                torque_nm=m.torque_nm,
                f_elec_hz=round(f_elec, 1),
                f_sw_hz=f_sw,
                carrier_ratio=round(carrier_ratio, 1),
                thd_avg=m.thd_avg,
                i5_pct=i5,
                i7_pct=i7,
                i11_pct=i11,
                i13_pct=i13,
                risk_level=risk,
                dominant_harmonics=dominant,
            )
            if label == 'ME':
                result.me.append(pt)
            else:
                result.hsg.append(pt)

    return result


def print_sync_risk(result: SyncRiskResult):
    for label in ('ME', 'HSG'):
        points = result.me if label == 'ME' else result.hsg
        if not points:
            continue
        print(f"\n=== Riesgo de sincronía PWM - {label} ===")
        print(f"{'Vel(rpm)':>9s} | {'Par(Nm)':>8s} | {'f_elec':>6s} | {'f_sw':>5s} | {'m_f':>5s} | "
              f"{'THD%':>5s} | {'I5%':>5s} | {'I7%':>5s} | {'I11%':>5s} | {'I13%':>5s} | Riesgo | Armónicos")
        print("-" * 90)
        high = medium = 0
        for p in points:
            print(f"{p.speed_rpm:9.0f} | {p.torque_nm:8.1f} | {p.f_elec_hz:6.1f} | "
                  f"{p.f_sw_hz:5d} | {p.carrier_ratio:5.1f} | {p.thd_avg:5.2f} | "
                  f"{p.i5_pct:5.2f} | {p.i7_pct:5.2f} | {p.i11_pct:5.2f} | {p.i13_pct:5.2f} | "
                  f"{p.risk_level:>7s} | {p.dominant_harmonics}")
            if p.risk_level == 'high':
                high += 1
            elif p.risk_level == 'medium':
                medium += 1
        total = len(points)
        print(f"  Total: {total} | Alto riesgo: {high} | Medio: {medium} | Bajo: {total - high - medium}")


def sync_risk_to_dataframe(result: SyncRiskResult) -> pd.DataFrame:
    rows = []
    for points, label in [(result.me, 'ME'), (result.hsg, 'HSG')]:
        for p in points:
            rows.append({
                'Máquina': p.machine,
                'Vel (rpm)': p.speed_rpm,
                'Par (Nm)': p.torque_nm,
                'f_elec (Hz)': p.f_elec_hz,
                'f_sw (Hz)': p.f_sw_hz,
                'm_f': p.carrier_ratio,
                'THD avg (%)': p.thd_avg,
                'I5 (%)': p.i5_pct,
                'I7 (%)': p.i7_pct,
                'I11 (%)': p.i11_pct,
                'I13 (%)': p.i13_pct,
                'Riesgo': p.risk_level,
                'Armónicos dom.': p.dominant_harmonics,
            })
    return pd.DataFrame(rows)
