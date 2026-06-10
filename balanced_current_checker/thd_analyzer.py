from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from dico_reader import MachineGroup, TorqueChannels


@dataclass
class ThdMeasurement:
    speed_rpm: float
    torque_nm: float
    n_windows: int
    thd_u: float
    thd_v: float
    thd_w: float
    thd_avg: float
    harmonics: Dict[int, float] = field(default_factory=dict)


@dataclass
class MachineThdResult:
    machine: str
    measurements: List[ThdMeasurement]


@dataclass
class ThdResult:
    me: Optional[MachineThdResult] = None
    hsg: Optional[MachineThdResult] = None


# ── Steady-state detection ──

def detect_steady_state(
    torque_arr: np.ndarray,
    time_arr: np.ndarray,
    window_s: float = 60.0,
    torque_tol: float = 5.0,
    raster: float = 0.0001,
) -> List[Tuple[int, int]]:
    """Find contiguous 60s windows where torque varies < 5 Nm.
    Returns list of (start_idx, end_idx) into the ORIGINAL arrays (not decimated).
    """
    dec_step = max(1, int(0.1 / raster))
    n_full = (len(torque_arr) // dec_step) * dec_step
    if n_full < dec_step * 2:
        return []
    t_dec = torque_arr[:n_full].reshape(-1, dec_step).mean(axis=1)
    win_dec = int(window_s / 0.1)
    if len(t_dec) < win_dec:
        return []

    s = pd.Series(t_dec)
    rmin = s.rolling(win_dec, min_periods=win_dec).min()
    rmax = s.rolling(win_dec, min_periods=win_dec).max()
    steady = (rmax - rmin) < torque_tol

    segments = []
    in_seg = False
    seg_start = 0
    for i in range(len(steady)):
        if steady.iloc[i] and not in_seg:
            in_seg = True
            seg_start = i
        elif not steady.iloc[i] and in_seg:
            in_seg = False
            orig_start = (seg_start - win_dec + 1) * dec_step
            orig_end = i * dec_step
            if orig_end > orig_start:
                segments.append((max(0, orig_start), min(len(torque_arr), orig_end)))
    if in_seg:
        orig_start = (seg_start - win_dec + 1) * dec_step
        segments.append((max(0, orig_start), len(torque_arr)))

    return segments


# ── FFT-based THD ──

def _compute_thd_for_window(
    signal: np.ndarray,
    fs: float,
    fund_hz: float,
    harmonics: List[int] = None,
) -> Tuple[float, Dict[int, float]]:
    """Compute THD for one signal window. Returns (thd_pct, {h: mag_pct})."""
    if harmonics is None:
        harmonics = [1, 3, 5, 7, 11, 13]
    n = len(signal)
    if n < 10:
        return 0.0, {}
    # Hanning window
    window = np.hanning(n)
    sig = signal * window
    # FFT
    spectrum = np.fft.rfft(sig) / n
    mag = np.abs(spectrum) * 2  # single-sided amplitude
    freq = np.fft.rfftfreq(n, d=1.0 / fs)

    # Find fundamental peak near expected frequency
    fund_idx = np.argmax(mag)
    # Refine: search within ±5% of expected fund_hz
    f_lo = fund_hz * 0.95
    f_hi = fund_hz * 1.05
    mask = (freq >= f_lo) & (freq <= f_hi)
    if mask.any():
        fund_idx = np.argmax(mag * mask)
    else:
        # fall back to global peak
        fund_idx = np.argmax(mag)

    if fund_idx == 0 or mag[fund_idx] < 1e-9:
        return 0.0, {}

    i1 = mag[fund_idx]
    h_data: Dict[int, float] = {}
    sq_sum = 0.0
    for h in harmonics:
        if h == 1:
            continue
        h_idx = int(round(h * fund_idx))
        if h_idx < len(mag):
            val = mag[h_idx]
            h_data[h] = val / i1 * 100.0
            sq_sum += val ** 2
        else:
            h_data[h] = 0.0

    thd = np.sqrt(sq_sum) / i1 * 100.0
    return thd, h_data


# ── Main analysis ──

def analyze_thd(
    df: pd.DataFrame,
    mg: MachineGroup,
    tc: Optional[TorqueChannels],
    machine: str,
    poles: int,
    raster: float = 0.0001,
) -> Optional[MachineThdResult]:
    """Analyze THD for one machine. Returns MachineThdResult or None."""
    if not mg or len(mg.channels) < 3:
        print(f"[WARN] {machine}: no hay 3 canales de fase")
        return None
    if tc is None or not tc.tq_est or not tc.speed:
        print(f"[WARN] {machine}: faltan canales de par/velocidad")
        return None
    if tc.tq_est not in df.columns or tc.speed not in df.columns:
        print(f"[WARN] {machine}: canales de par/velocidad no estan en datos")
        return None

    ph_u, ph_v, ph_w = mg.channels[0], mg.channels[1], mg.channels[2]
    for ch in [ph_u, ph_v, ph_w]:
        if ch not in df.columns:
            print(f"[WARN] {machine}: falta canal {ch}")
            return None

    torque = df[tc.tq_est].astype(float).values
    speed = df[tc.speed].astype(float).values
    time_idx = df.index.values
    fs = 1.0 / raster

    segments = detect_steady_state(torque, time_idx, raster=raster)
    if not segments:
        print(f"[INFO] {machine}: no se detectaron segmentos estables")
        return None

    measurements: List[ThdMeasurement] = []
    for seg_idx, (s_start, s_end) in enumerate(segments):
        n_samples = s_end - s_start
        if n_samples < int(10 * fs / 10):  # less than ~10 cycles at 10Hz
            print(f"[WARN] {machine} segmento {seg_idx}: demasiado corto ({n_samples} muestras)")
            continue

        avg_speed = float(np.nanmean(speed[s_start:s_end]))
        avg_torque = float(np.nanmean(torque[s_start:s_end]))
        fund_hz = avg_speed * poles / 120.0
        if fund_hz < 1:
            continue

        n_cycle = int(fs / fund_hz)
        win_size = n_cycle * 10  # 10 cycles
        if win_size < 10 or win_size > n_samples:
            print(f"[WARN] {machine} seg {seg_idx}: ventana {win_size} > segmento {n_samples}")
            continue

        step = max(1, win_size // 2)
        n_wins = 0
        thd_u_list, thd_v_list, thd_w_list = [], [], []
        h_acc: Dict[int, float] = {}

        for w_start in range(s_start, s_end - win_size + 1, step):
            w_end = w_start + win_size
            sig_u = df[ph_u].values[w_start:w_end].astype(float)
            sig_v = df[ph_v].values[w_start:w_end].astype(float)
            sig_w = df[ph_w].values[w_start:w_end].astype(float)

            for sig, lst in [(sig_u, thd_u_list), (sig_v, thd_v_list), (sig_w, thd_w_list)]:
                try:
                    thd_v, h_data = _compute_thd_for_window(sig, fs, fund_hz)
                    lst.append(thd_v)
                except Exception as e:
                    print(f"[WARN] {machine} seg {seg_idx}: FFT fallo: {e}")
                    continue

            # Accumulate harmonics from phase U
            if thd_u_list:
                _, h_data = _compute_thd_for_window(sig_u, fs, fund_hz)
                for hk, hv in h_data.items():
                    h_acc[hk] = h_acc.get(hk, 0.0) + hv

            n_wins += 1
            if n_wins >= 50:  # limit to avoid excessive computation
                break

        if n_wins == 0:
            print(f"[WARN] {machine} seg {seg_idx}: 0 ventanas procesadas")
            continue

        avg_u = float(np.mean(thd_u_list)) if thd_u_list else 0.0
        avg_v = float(np.mean(thd_v_list)) if thd_v_list else 0.0
        avg_w = float(np.mean(thd_w_list)) if thd_w_list else 0.0
        avg_all = float(np.mean(thd_u_list + thd_v_list + thd_w_list)) if (thd_u_list or thd_v_list or thd_w_list) else 0.0
        h_avg = {k: v / n_wins for k, v in h_acc.items()}

        measurements.append(ThdMeasurement(
            speed_rpm=round(avg_speed, 0),
            torque_nm=round(avg_torque, 1),
            n_windows=n_wins,
            thd_u=round(avg_u, 2),
            thd_v=round(avg_v, 2),
            thd_w=round(avg_w, 2),
            thd_avg=round(avg_all, 2),
            harmonics=h_avg,
        ))
        print(f"[THD] {machine} seg {seg_idx}: {avg_speed:.0f}rpm {avg_torque:.1f}Nm -> THD={avg_all:.2f}% ({n_wins} wins)")

    if not measurements:
        print(f"[INFO] {machine}: no se generaron mediciones THD")
        return None

    return MachineThdResult(machine=machine, measurements=measurements)


# ── Print ──

def print_thd_result(result: ThdResult):
    has_data = False
    for label, mthd in [('ME', result.me), ('HSG', result.hsg)]:
        if not mthd or not mthd.measurements:
            continue
        has_data = True
        print(f"\n=== THD {label} ({len(mthd.measurements)} puntos) ===")
        print(f"{'#':>3s} | {'Vel (rpm)':>9s} | {'Par (Nm)':>8s} | "
              f"{'THD_U%':>7s} | {'THD_V%':>7s} | {'THD_W%':>7s} | {'THD_avg%':>8s} | "
              f"{'I3%':>5s} | {'I5%':>5s} | {'I7%':>5s} | {'I11%':>5s} | {'I13%':>5s}")
        print("-" * 90)
        for i, m in enumerate(mthd.measurements, 1):
            i3 = f"{m.harmonics.get(3, 0):.1f}" if m.harmonics else '-'
            i5 = f"{m.harmonics.get(5, 0):.1f}" if m.harmonics else '-'
            i7 = f"{m.harmonics.get(7, 0):.1f}" if m.harmonics else '-'
            i11 = f"{m.harmonics.get(11, 0):.1f}" if m.harmonics else '-'
            i13 = f"{m.harmonics.get(13, 0):.1f}" if m.harmonics else '-'
            print(f"{i:3d} | {m.speed_rpm:9.0f} | {m.torque_nm:8.1f} | "
                  f"{m.thd_u:7.2f} | {m.thd_v:7.2f} | {m.thd_w:7.2f} | {m.thd_avg:8.2f} | "
                  f"{i3:>5s} | {i5:>5s} | {i7:>5s} | {i11:>5s} | {i13:>5s}")
    if not has_data:
        print("\n=== THD: NO DISPONIBLE ===")


# ── DataFrames ──

def thd_to_dataframe(result: ThdResult) -> pd.DataFrame:
    rows = []
    for label, mthd in [('ME', result.me), ('HSG', result.hsg)]:
        if not mthd or not mthd.measurements:
            continue
        for m in mthd.measurements:
            rows.append({
                'Maquina': label,
                'Vel (rpm)': m.speed_rpm,
                'Par (Nm)': m.torque_nm,
                'THD U (%)': m.thd_u,
                'THD V (%)': m.thd_v,
                'THD W (%)': m.thd_w,
                'THD avg (%)': m.thd_avg,
                'I3 (%)': round(m.harmonics.get(3, 0), 2) if m.harmonics else None,
                'I5 (%)': round(m.harmonics.get(5, 0), 2) if m.harmonics else None,
                'I7 (%)': round(m.harmonics.get(7, 0), 2) if m.harmonics else None,
                'I11 (%)': round(m.harmonics.get(11, 0), 2) if m.harmonics else None,
                'I13 (%)': round(m.harmonics.get(13, 0), 2) if m.harmonics else None,
                'Ventanas': m.n_windows,
            })
    return pd.DataFrame(rows)
