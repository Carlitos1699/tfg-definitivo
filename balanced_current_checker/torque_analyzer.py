from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from dico_reader import TorqueChannels
from torque_reader import TorquePerfDatabase
from calibrables_reader import CalibrationData


@dataclass
class TorquePrecisionEvent:
    machine: str
    start_time: float
    end_time: float
    duration_s: float
    max_error_nm: float
    mean_error_nm: float
    n_samples: int
    model_violations: int = 0
    cmin_at_event: Optional[float] = None
    cmax_at_event: Optional[float] = None
    mode: str = 'motor'
    avg_tq_est: float = 0.0
    limit_used: Optional[float] = None
    within_limits: bool = True
    avg_voltage: float = 0.0
    avg_speed: float = 0.0
    avg_rotor_temp: float = 0.0
    mismatch_samples: int = 0
    derating_samples: int = 0
    derating_min_samples: int = 0
    derating_max_samples: int = 0
    derating_cause: str = ''
    active_flags_desc: str = ''
    eajs_filtered_samples: int = 0
    avg_vd: float = 0.0
    avg_vq: float = 0.0
    rgl_sat_samples: int = 0
    saturation_state: str = ''
    avg_i_d: float = 0.0
    avg_i_q: float = 0.0
    avg_i_d_req: float = 0.0
    avg_i_q_req: float = 0.0
    avg_i_d_error: float = 0.0
    avg_i_q_error: float = 0.0
    phase_unbalanced: bool = False
    phase_imbalance_index: float = 0.0


@dataclass
class TorqueAnalysis:
    machine: str
    threshold_nm: float
    total_samples: int
    error_samples: int
    pct_error: float
    model_violation_samples: int
    pct_violation: float
    max_error_nm: float
    mean_error_nm: float
    events: List[TorquePrecisionEvent] = field(default_factory=list)
    error_series: Optional[pd.Series] = None
    outside_model: Optional[pd.Series] = None
    calibration: Optional[CalibrationData] = None


# Variables que alimentan el bloque MIN del path MIN (Vxx_norm_tq_min):
#   min(Vxx_norm_tq_max_temp, Vxx_norm_tq_min_udc, Vxx_norm_tq_max_stop,
#       Vxx_norm_tq_cur_sat_max, Vxx_norm_tq_max_asc_oc, Vxx_norm_tq_min_back)
_MIN_CAUSE_LABELS = {
    'norm_tq_max_temp': 'max_temp',
    'norm_tq_min_udc': 'min_udc',
    'norm_tq_max_stop': 'max_stop',
    'norm_tq_cur_sat_max': 'cur_sat',
    'norm_tq_max_asc_oc': 'asc_oc',
    'norm_tq_min_back': 'min_back',
}

# Variables que alimentan el bloque MIN del path MAX (Vxx_norm_tq_max):
#   min(Vxx_norm_tq_max_temp, Vxx_norm_tq_max_udc, Vxx_norm_tq_max_stop,
#       Vxx_norm_tq_cur_sat_max, Vxx_norm_tq_max_asc_oc)
_MAX_CAUSE_LABELS = {
    'norm_tq_max_temp': 'max_temp',
    'norm_tq_max_udc': 'max_udc',
    'norm_tq_max_stop': 'max_stop',
    'norm_tq_cur_sat_max': 'cur_sat',
    'norm_tq_max_asc_oc': 'asc_oc',
}


# TorqueChannels attribute names for Vbx_* derating activation flags
# (both aggregated and individual)
_DERATING_FLAG_ATTRS = (
    'spt_der_act',
    'spt_der_act_tqc',
    'invt_stop',
    'spt_der_v_oor',
    'spt_der_ovh',
    'max_cur_sat_act',
    'asc_oc_der_act',
    'invt_ztq_reg_act',
    'dm_ov_sfty',
    'back_der_act',
)

_PHASE_IMBALANCE_THRESHOLD_A = 2.0


def _determine_saturation(
    df: pd.DataFrame,
    tc: TorqueChannels,
    sat_series: Dict[str, Optional[pd.Series]],
    idx: int,
    rgl_sat_count: int,
    total_samples: int,
) -> str:
    """Determine saturation state at a given sample index.

    Returns 'saturating', 'harmonic_control', or 'unknown'.
    """
    if not sat_series:
        return ''
    try:
        vd = float(sat_series.get('v_d').iloc[idx]) if sat_series.get('v_d') is not None else None
        vq = float(sat_series.get('v_q').iloc[idx]) if sat_series.get('v_q') is not None else None
        vd_ang = float(sat_series.get('v_d_ang').iloc[idx]) if sat_series.get('v_d_ang') is not None else None
        vd_harm = float(sat_series.get('v_d_harm').iloc[idx]) if sat_series.get('v_d_harm') is not None else None
        vq_ang = float(sat_series.get('v_q_ang').iloc[idx]) if sat_series.get('v_q_ang') is not None else None
        vq_harm = float(sat_series.get('v_q_harm').iloc[idx]) if sat_series.get('v_q_harm') is not None else None
    except (ValueError, TypeError, IndexError):
        return 'unknown'

    vd_ok = vd is not None and vd_ang is not None and vd_harm is not None
    vq_ok = vq is not None and vq_ang is not None and vq_harm is not None

    if not vd_ok and not vq_ok:
        return ''

    vd_closer_to_ang = abs(vd - vd_ang) < abs(vd - vd_harm) if vd_ok else None
    vq_closer_to_ang = abs(vq - vq_ang) < abs(vq - vq_harm) if vq_ok else None

    rgl_sat_active = rgl_sat_count > 0

    if vd_closer_to_ang is True and vq_closer_to_ang is True and rgl_sat_active:
        return 'saturating'
    if vd_closer_to_ang is True and vq_closer_to_ang is False and rgl_sat_active:
        return 'saturating'
    if vd_closer_to_ang is False and vq_closer_to_ang is True and rgl_sat_active:
        return 'saturating'

    if vd_closer_to_ang is False and vq_closer_to_ang is False and not rgl_sat_active:
        return 'harmonic_control'
    if vd_closer_to_ang is False and vq_closer_to_ang is True and not rgl_sat_active:
        return 'harmonic_control'
    if vd_closer_to_ang is True and vq_closer_to_ang is False and not rgl_sat_active:
        return 'harmonic_control'

    return 'unknown'


def _classify_derating(
    tq_sp: float, tq_min: float, tq_max: float,
    min_specific: Dict[str, Optional[float]],
    max_specific: Dict[str, Optional[float]],
) -> Tuple[str, str]:
    """Returns (derating_type, cause_label).

    derating_type: 'none', 'min_limit', 'max_limit', 'both'
    cause_label: e.g. 'max_temp', 'min_udc', 'max_temp+min_udc', ''

    Identifies the binding variable by finding which specific variable
    equals the raw min of its path's MIN block (before multiplication
    by calibratables and cell_ov_fac).
    """
    after_max = max(tq_sp, tq_min)
    is_min = tq_min > tq_sp
    is_max = tq_max < after_max

    if not is_min and not is_max:
        return 'none', ''

    cause = ''

    if is_max:
        valid = {k: v for k, v in max_specific.items() if v is not None}
        if valid:
            raw_min_val = min(valid.values())
            for attr, label in _MAX_CAUSE_LABELS.items():
                v = max_specific.get(attr)
                if v is not None and abs(v - raw_min_val) < 1e-6:
                    cause = label
                    break

    if is_min:
        valid = {k: v for k, v in min_specific.items() if v is not None}
        if valid:
            raw_min_val = min(valid.values())
            for attr, label in _MIN_CAUSE_LABELS.items():
                v = min_specific.get(attr)
                if v is not None and abs(v - raw_min_val) < 1e-6:
                    cause = f'{cause}+{label}' if cause else label
                    break

    if is_min and is_max:
        return 'both', cause
    elif is_min:
        return 'min_limit', cause
    else:
        return 'max_limit', cause


def analyze_torque_precision(
    df: pd.DataFrame,
    tc: TorqueChannels,
    perf_db: Optional[TorquePerfDatabase] = None,
    threshold_nm: float = 5.0,
    min_event_ms: float = 5000.0,
    calibration: Optional[CalibrationData] = None,
    eajs_column: Optional[str] = None,
    phase_channels: Optional[Tuple[str, str, str]] = None,
) -> TorqueAnalysis:
    raster = _infer_raster(df)
    min_event_samples = max(1, int(round(min_event_ms / 1000.0 / raster)))

    tq_cmd = df[tc.tq_cmd]
    tq_est = df[tc.tq_est]

    error = tq_cmd - tq_est
    is_error_raw = error.abs() > threshold_nm

    # Sync mode: only check torque error when sync_mode == 0 (torque mode)
    if tc.sync_mode and tc.sync_mode in df.columns:
        raw_sync = df[tc.sync_mode]
        if raw_sync.dtype == object:
            sync_series = pd.to_numeric(raw_sync.map(lambda x: x.decode() if isinstance(x, bytes) else x), errors='coerce')
        else:
            sync_series = raw_sync
        is_error = is_error_raw & (sync_series == 0)
    else:
        sync_series = None
        is_error = is_error_raw

    total = len(error)
    err_samples = int(is_error.sum())

    # Load derating channels
    has_tq_sp = tc.norm_tq_sp and tc.norm_tq_sp in df.columns
    has_tq_min = tc.norm_tq_min and tc.norm_tq_min in df.columns
    has_tq_max = tc.norm_tq_max and tc.norm_tq_max in df.columns
    has_derating = has_tq_sp and has_tq_min and has_tq_max

    tq_sp_series = df[tc.norm_tq_sp] if has_tq_sp else None
    tq_min_series = df[tc.norm_tq_min] if has_tq_min else None
    tq_max_series = df[tc.norm_tq_max] if has_tq_max else None

    # Load specific derating variables
    min_specific_series: Dict[str, Optional[pd.Series]] = {}
    for attr in _MIN_CAUSE_LABELS:
        ch = getattr(tc, attr, '')
        if ch and ch in df.columns:
            min_specific_series[attr] = df[ch]
        else:
            min_specific_series[attr] = None

    max_specific_series: Dict[str, Optional[pd.Series]] = {}
    for attr in _MAX_CAUSE_LABELS:
        ch = getattr(tc, attr, '')
        if ch and ch in df.columns:
            max_specific_series[attr] = df[ch]
        else:
            max_specific_series[attr] = None

    # Load Vbx_* derating activation flag series
    from dico_reader import DICO_DESCRIPTIONS
    flag_series: Dict[str, Optional[pd.Series]] = {}
    for attr in _DERATING_FLAG_ATTRS:
        ch = getattr(tc, attr, '')
        if ch and ch in df.columns:
            flag_series[attr] = df[ch]
        else:
            flag_series[attr] = None

    # EAJS (Curative Antijerk) activation — only for ME
    eajs_series: Optional[pd.Series] = None
    if tc.machine == 'ME':
        eajs_dico = getattr(tc, 'eajs_cor_act', '')
        if eajs_dico and eajs_dico in df.columns:
            eajs_series = df[eajs_dico]
        elif eajs_column and eajs_column in df.columns:
            eajs_series = df[eajs_column]

    # Saturation voltage channels (Vd, Vq, references, and rgl_sat flag)
    _SAT_ATTRS = ('v_d', 'v_q', 'v_d_ang', 'v_d_harm', 'v_q_ang', 'v_q_harm', 'rgl_sat')
    sat_series: Dict[str, Optional[pd.Series]] = {}
    for attr in _SAT_ATTRS:
        ch = getattr(tc, attr, '')
        if ch and ch in df.columns:
            sat_series[attr] = df[ch]
        else:
            sat_series[attr] = None

    # Current channels (Id, Iq measured and required)
    has_i_d = tc.i_d and tc.i_d in df.columns
    has_i_q = tc.i_q and tc.i_q in df.columns
    has_i_d_req = tc.i_d_req and tc.i_d_req in df.columns
    has_i_q_req = tc.i_q_req and tc.i_q_req in df.columns
    has_currents = has_i_d and has_i_q and has_i_d_req and has_i_q_req
    i_d_series = df[tc.i_d] if has_i_d else None
    i_q_series = df[tc.i_q] if has_i_q else None
    i_d_req_series = df[tc.i_d_req] if has_i_d_req else None
    i_q_req_series = df[tc.i_q_req] if has_i_q_req else None

    # Three-phase current channels for imbalance detection
    has_phases = phase_channels is not None and all(ch in df.columns for ch in phase_channels)
    if has_phases:
        ph1_series, ph2_series, ph3_series = (df[ch] for ch in phase_channels)
    else:
        ph1_series = ph2_series = ph3_series = None

    # State machine for events
    events = []
    in_event = False
    event_start = 0
    event_max_err = 0.0
    event_err_sum = 0.0
    event_count = 0
    event_violations = 0
    event_tq_sum = 0.0
    event_volt_sum = 0.0
    event_speed_sum = 0.0
    event_temp_sum = 0.0
    event_mismatch = 0
    event_derating = 0
    event_derating_min = 0
    event_derating_max = 0
    event_cause_counts: Dict[str, int] = {}
    event_active_flags: set = set()
    event_eajs_filtered = 0
    event_vd_sum = 0.0
    event_vq_sum = 0.0
    event_rgl_sat = 0
    event_i_d_sum = 0.0
    event_i_q_sum = 0.0
    event_i_d_req_sum = 0.0
    event_i_q_req_sum = 0.0
    event_imbalance_sum = 0.0

    has_voltage = tc.voltage and tc.voltage in df.columns
    has_speed = tc.speed and tc.speed in df.columns
    has_temp = tc.rotor_temp and tc.rotor_temp in df.columns
    has_tq_cons = tc.tq_cons and tc.tq_cons in df.columns
    tq_cons_series = df[tc.tq_cons].ffill().fillna(0) if has_tq_cons else None

    outside_model = pd.Series(False, index=df.index)

    def _sample_flags(i: int):
        nonlocal event_active_flags
        for attr in _DERATING_FLAG_ATTRS:
            series = flag_series.get(attr)
            if series is not None:
                try:
                    val = float(series.iloc[i])
                    if val != 0.0:
                        ch_name = getattr(tc, attr, '')
                        if ch_name:
                            event_active_flags.add(ch_name)
                except (ValueError, TypeError):
                    pass

    def _accum_saturation(i: int):
        nonlocal event_vd_sum, event_vq_sum, event_rgl_sat
        for attr in ('v_d', 'v_q'):
            series = sat_series.get(attr)
            if series is not None:
                try:
                    val = float(series.iloc[i])
                    if attr == 'v_d':
                        event_vd_sum += val
                    else:
                        event_vq_sum += val
                except (ValueError, TypeError):
                    pass
        rgl = sat_series.get('rgl_sat')
        if rgl is not None:
            try:
                if float(rgl.iloc[i]) != 0.0:
                    event_rgl_sat += 1
            except (ValueError, TypeError):
                pass

    def _accum_currents(i: int):
        nonlocal event_i_d_sum, event_i_q_sum, event_i_d_req_sum, event_i_q_req_sum, event_imbalance_sum
        if has_currents:
            try:
                event_i_d_sum += float(i_d_series.iloc[i])
                event_i_q_sum += float(i_q_series.iloc[i])
                event_i_d_req_sum += float(i_d_req_series.iloc[i])
                event_i_q_req_sum += float(i_q_req_series.iloc[i])
            except (ValueError, TypeError):
                pass
        if has_phases:
            try:
                i1 = float(ph1_series.iloc[i])
                i2 = float(ph2_series.iloc[i])
                i3 = float(ph3_series.iloc[i])
                event_imbalance_sum += abs(i1 + i2 + i3)
            except (ValueError, TypeError):
                pass

    def _sample_derating(i: int):
        nonlocal event_derating, event_derating_min, event_derating_max
        tq_sp_val = float(tq_sp_series.iloc[i])
        tq_min_val = float(tq_min_series.iloc[i])
        tq_max_val = float(tq_max_series.iloc[i])

        min_specific = {}
        for attr, series in min_specific_series.items():
            if series is not None:
                try:
                    min_specific[attr] = float(series.iloc[i])
                except (ValueError, TypeError):
                    pass
        max_specific = {}
        for attr, series in max_specific_series.items():
            if series is not None:
                try:
                    max_specific[attr] = float(series.iloc[i])
                except (ValueError, TypeError):
                    pass

        dr_type, cause = _classify_derating(
            tq_sp_val, tq_min_val, tq_max_val,
            min_specific, max_specific,
        )

        if dr_type != 'none':
            event_derating += 1
            if 'min_limit' in dr_type:
                event_derating_min += 1
            if 'max_limit' in dr_type:
                event_derating_max += 1
            if cause:
                event_cause_counts[cause] = event_cause_counts.get(cause, 0) + 1

    for i in range(total):
        in_torque_mode = True
        if sync_series is not None:
            sv = sync_series.iloc[i]
            if isinstance(sv, bytes):
                sv = sv.decode()
            in_torque_mode = (sv == 0 or sv == '0')

        # EAJS filter: for ME, if error + mismatch + EAJS active → discard sample
        is_eajs_filter = False
        if tc.machine == 'ME' and eajs_series is not None and is_error.iloc[i] and in_torque_mode:
            try:
                if has_tq_cons and tq_cons_series.iloc[i] != tq_est.iloc[i]:
                    if float(eajs_series.iloc[i]) != 0.0:
                        is_eajs_filter = True
            except (ValueError, TypeError):
                pass

        if is_error.iloc[i] and in_torque_mode and not is_eajs_filter:
            if not in_event:
                in_event = True
                event_start = i
                event_max_err = abs(error.iloc[i])
                event_err_sum = abs(error.iloc[i])
                event_count = 1
                event_violations = 1 if _check_violation(df, tc, perf_db, i) else 0
                event_tq_sum = tq_est.iloc[i]
                event_volt_sum = df[tc.voltage].iloc[i] if has_voltage else 0.0
                event_speed_sum = abs(df[tc.speed].iloc[i]) if has_speed else 0.0
                event_temp_sum = df[tc.rotor_temp].iloc[i] if has_temp else 0.0
                event_mismatch = 1 if (has_tq_cons and tq_cons_series.iloc[i] != tq_est.iloc[i]) else 0
                event_vd_sum = 0.0
                event_vq_sum = 0.0
                event_rgl_sat = 0
                event_i_d_sum = 0.0
                event_i_q_sum = 0.0
                event_i_d_req_sum = 0.0
                event_i_q_req_sum = 0.0
                event_imbalance_sum = 0.0
                _accum_saturation(i)
                _accum_currents(i)
                event_derating = 0
                event_derating_min = 0
                event_derating_max = 0
                event_cause_counts = {}
                event_active_flags = set()
                event_eajs_filtered = 0
                event_vd_sum = 0.0
                event_vq_sum = 0.0
                event_rgl_sat = 0
                event_i_d_sum = 0.0
                event_i_q_sum = 0.0
                event_i_d_req_sum = 0.0
                event_i_q_req_sum = 0.0
                event_imbalance_sum = 0.0
                if has_derating:
                    _sample_derating(i)
                    _sample_flags(i)
                    _accum_saturation(i)
                _accum_currents(i)
            else:
                val = abs(error.iloc[i])
                if val > event_max_err:
                    event_max_err = val
                event_err_sum += val
                event_count += 1
                event_tq_sum += tq_est.iloc[i]
                if has_voltage:
                    event_volt_sum += df[tc.voltage].iloc[i]
                if has_speed:
                    event_speed_sum += abs(df[tc.speed].iloc[i])
                if has_temp:
                    event_temp_sum += df[tc.rotor_temp].iloc[i]
                if has_tq_cons and tq_cons_series.iloc[i] != tq_est.iloc[i]:
                    event_mismatch += 1
                if _check_violation(df, tc, perf_db, i):
                    event_violations += 1
                    outside_model.iloc[i] = True
                if has_derating:
                    _sample_derating(i)
                    _sample_flags(i)
                _accum_saturation(i)
                _accum_currents(i)
        else:
            if in_event:
                dur_samples = i - event_start
                if dur_samples >= min_event_samples:
                    mid = (event_start + i - 1) // 2
                    cmin_v, cmax_v = _get_range_at(df, tc, perf_db, mid)
                    saturation_state = _determine_saturation(df, tc, sat_series, mid, event_rgl_sat, event_count)
                    avg_tq = event_tq_sum / event_count
                    mode = 'motor' if avg_tq >= 0 else 'generator'
                    limit_used = cmax_v if mode == 'motor' else cmin_v
                    if limit_used is not None:
                        if mode == 'motor':
                            within = avg_tq <= limit_used
                        else:
                            within = avg_tq >= limit_used
                    else:
                        within = True
                    avg_v = event_volt_sum / event_count if has_voltage else 0.0
                    avg_s = event_speed_sum / event_count if has_speed else 0.0
                    avg_t = event_temp_sum / event_count if has_temp else 0.0

                    if has_currents:
                        avg_i_d = event_i_d_sum / event_count
                        avg_i_q = event_i_q_sum / event_count
                        avg_i_d_req = event_i_d_req_sum / event_count
                        avg_i_q_req = event_i_q_req_sum / event_count
                        avg_i_d_error = abs(avg_i_d_req - avg_i_d)
                        avg_i_q_error = abs(avg_i_q_req - avg_i_q)
                    else:
                        avg_i_d = avg_i_q = avg_i_d_req = avg_i_q_req = 0.0
                        avg_i_d_error = avg_i_q_error = 0.0

                    if has_phases:
                        phase_imbalance_index = event_imbalance_sum / event_count
                        phase_unbalanced = phase_imbalance_index > _PHASE_IMBALANCE_THRESHOLD_A
                    else:
                        phase_imbalance_index = 0.0
                        phase_unbalanced = False

                    dominant_cause = max(event_cause_counts, key=event_cause_counts.get) if event_cause_counts else ''
                    _active_descs = []
                    for ch in sorted(event_active_flags):
                        _active_descs.append(DICO_DESCRIPTIONS.get(ch, ch))
                    active_flags_desc = '; '.join(_active_descs)
                    events.append(TorquePrecisionEvent(
                        machine=tc.machine,
                        start_time=df.index[event_start],
                        end_time=df.index[i - 1],
                        duration_s=df.index[i - 1] - df.index[event_start],
                        max_error_nm=event_max_err,
                        mean_error_nm=event_err_sum / event_count,
                        n_samples=event_count,
                        model_violations=event_violations,
                        cmin_at_event=cmin_v,
                        cmax_at_event=cmax_v,
                        mode=mode,
                        avg_tq_est=avg_tq,
                        limit_used=limit_used,
                        within_limits=within,
                        avg_voltage=avg_v,
                        avg_speed=avg_s,
                        avg_rotor_temp=avg_t,
                        mismatch_samples=event_mismatch,
                        derating_samples=event_derating,
                        derating_min_samples=event_derating_min,
                        derating_max_samples=event_derating_max,
                        derating_cause=dominant_cause,
                        active_flags_desc=active_flags_desc,
                        eajs_filtered_samples=event_eajs_filtered,
                        avg_vd=event_vd_sum / event_count,
                        avg_vq=event_vq_sum / event_count,
                        rgl_sat_samples=event_rgl_sat,
                        saturation_state=saturation_state,
                        avg_i_d=avg_i_d, avg_i_q=avg_i_q,
                        avg_i_d_req=avg_i_d_req, avg_i_q_req=avg_i_q_req,
                        avg_i_d_error=avg_i_d_error, avg_i_q_error=avg_i_q_error,
                        phase_unbalanced=phase_unbalanced,
                        phase_imbalance_index=phase_imbalance_index,
                    ))
                in_event = False
                if is_eajs_filter:
                    event_eajs_filtered += 1

    if in_event:
        dur_samples = total - event_start
        if dur_samples >= min_event_samples:
            mid = (event_start + total - 1) // 2
            cmin_v, cmax_v = _get_range_at(df, tc, perf_db, mid)
            saturation_state = _determine_saturation(df, tc, sat_series, mid, event_rgl_sat, event_count)
            avg_tq = event_tq_sum / event_count
            mode = 'motor' if avg_tq >= 0 else 'generator'
            limit_used = cmax_v if mode == 'motor' else cmin_v
            if limit_used is not None:
                if mode == 'motor':
                    within = avg_tq <= limit_used
                else:
                    within = avg_tq >= limit_used
            else:
                within = True
            avg_v = event_volt_sum / event_count if has_voltage else 0.0
            avg_s = event_speed_sum / event_count if has_speed else 0.0
            avg_t = event_temp_sum / event_count if has_temp else 0.0

            if has_currents:
                avg_i_d = event_i_d_sum / event_count
                avg_i_q = event_i_q_sum / event_count
                avg_i_d_req = event_i_d_req_sum / event_count
                avg_i_q_req = event_i_q_req_sum / event_count
                avg_i_d_error = abs(avg_i_d_req - avg_i_d)
                avg_i_q_error = abs(avg_i_q_req - avg_i_q)
            else:
                avg_i_d = avg_i_q = avg_i_d_req = avg_i_q_req = 0.0
                avg_i_d_error = avg_i_q_error = 0.0

            if has_phases:
                phase_imbalance_index = event_imbalance_sum / event_count
                phase_unbalanced = phase_imbalance_index > _PHASE_IMBALANCE_THRESHOLD_A
            else:
                phase_imbalance_index = 0.0
                phase_unbalanced = False

            dominant_cause = max(event_cause_counts, key=event_cause_counts.get) if event_cause_counts else ''
            _active_descs = []
            for ch in sorted(event_active_flags):
                _active_descs.append(DICO_DESCRIPTIONS.get(ch, ch))
            active_flags_desc = '; '.join(_active_descs)
            events.append(TorquePrecisionEvent(
                machine=tc.machine,
                start_time=df.index[event_start],
                end_time=df.index[total - 1],
                duration_s=df.index[total - 1] - df.index[event_start],
                max_error_nm=event_max_err,
                mean_error_nm=event_err_sum / event_count,
                n_samples=event_count,
                model_violations=event_violations,
                cmin_at_event=cmin_v,
                cmax_at_event=cmax_v,
                mode=mode,
                avg_tq_est=avg_tq,
                limit_used=limit_used,
                within_limits=within,
                avg_voltage=avg_v,
                avg_speed=avg_s,
                avg_rotor_temp=avg_t,
                mismatch_samples=event_mismatch,
                derating_samples=event_derating,
                derating_min_samples=event_derating_min,
                derating_max_samples=event_derating_max,
                derating_cause=dominant_cause,
                active_flags_desc=active_flags_desc,
                eajs_filtered_samples=event_eajs_filtered,
                avg_vd=event_vd_sum / event_count,
                avg_vq=event_vq_sum / event_count,
                rgl_sat_samples=event_rgl_sat,
                saturation_state=saturation_state,
                avg_i_d=avg_i_d, avg_i_q=avg_i_q,
                avg_i_d_req=avg_i_d_req, avg_i_q_req=avg_i_q_req,
                avg_i_d_error=avg_i_d_error, avg_i_q_error=avg_i_q_error,
                phase_unbalanced=phase_unbalanced,
                phase_imbalance_index=phase_imbalance_index,
            ))

    viol_samples = int(outside_model.sum())
    max_err = float(error.abs().max()) if total else 0.0
    mean_err = float(error.abs().mean()) if total else 0.0

    # ── Merge events with gap < 5 seconds ──
    if len(events) > 1:
        _merged = [events[0]]
        for ev in events[1:]:
            last = _merged[-1]
            gap = ev.start_time - last.end_time
            if gap < 5.0:
                n = last.n_samples + ev.n_samples
                def _w(v1, v2):
                    return (v1 * last.n_samples + v2 * ev.n_samples) / n

                def _s(v1, v2):
                    if v1 is None:
                        return v2
                    if v2 is None:
                        return v1
                    if v1 is not None and v2 is not None:
                        return _w(v1, v2)
                    return None

                def _cat(a, b):
                    if not a:
                        return b
                    if not b:
                        return a
                    return f"{a} / {b}"

                sat_order = {'saturating': 2, 'harmonic_control': 1, 'unknown': 0}
                best_sat = max(last.saturation_state, ev.saturation_state,
                               key=lambda s: sat_order.get(s, 0))

                e = TorquePrecisionEvent(
                    machine=last.machine,
                    start_time=last.start_time,
                    end_time=ev.end_time,
                    duration_s=ev.end_time - last.start_time,
                    max_error_nm=max(last.max_error_nm, ev.max_error_nm),
                    mean_error_nm=_w(last.mean_error_nm, ev.mean_error_nm),
                    n_samples=n,
                    model_violations=last.model_violations + ev.model_violations,
                    cmin_at_event=_s(last.cmin_at_event, ev.cmin_at_event),
                    cmax_at_event=_s(last.cmax_at_event, ev.cmax_at_event),
                    mode=last.mode if last.mode == ev.mode else 'mixed',
                    avg_tq_est=_w(last.avg_tq_est, ev.avg_tq_est),
                    limit_used=_s(last.limit_used, ev.limit_used),
                    within_limits=last.within_limits and ev.within_limits,
                    avg_voltage=_w(last.avg_voltage, ev.avg_voltage),
                    avg_speed=_w(last.avg_speed, ev.avg_speed),
                    avg_rotor_temp=_w(last.avg_rotor_temp, ev.avg_rotor_temp),
                    mismatch_samples=last.mismatch_samples + ev.mismatch_samples,
                    derating_samples=last.derating_samples + ev.derating_samples,
                    derating_min_samples=last.derating_min_samples + ev.derating_min_samples,
                    derating_max_samples=last.derating_max_samples + ev.derating_max_samples,
                    derating_cause=_cat(last.derating_cause, ev.derating_cause),
                    active_flags_desc=_cat(last.active_flags_desc, ev.active_flags_desc),
                    eajs_filtered_samples=last.eajs_filtered_samples + ev.eajs_filtered_samples,
                    avg_vd=_w(last.avg_vd, ev.avg_vd),
                    avg_vq=_w(last.avg_vq, ev.avg_vq),
                    rgl_sat_samples=last.rgl_sat_samples + ev.rgl_sat_samples,
                    saturation_state=best_sat,
                    avg_i_d=_w(last.avg_i_d, ev.avg_i_d),
                    avg_i_q=_w(last.avg_i_q, ev.avg_i_q),
                    avg_i_d_req=_w(last.avg_i_d_req, ev.avg_i_d_req),
                    avg_i_q_req=_w(last.avg_i_q_req, ev.avg_i_q_req),
                    avg_i_d_error=_w(last.avg_i_d_error, ev.avg_i_d_error),
                    avg_i_q_error=_w(last.avg_i_q_error, ev.avg_i_q_error),
                    phase_unbalanced=last.phase_unbalanced or ev.phase_unbalanced,
                    phase_imbalance_index=max(last.phase_imbalance_index, ev.phase_imbalance_index),
                )
                print(f"[MERGE] {last.machine} Eventos fusionados (gap={gap:.3f}s) -> "
                      f"error medio={e.mean_error_nm:.2f}Nm, duracion={e.duration_s:.2f}s")
                _merged[-1] = e
            else:
                _merged.append(ev)
        events = _merged

    # Descartar eventos con duración < 5 segundos (tras fusión)
    events = [e for e in events if e.duration_s >= 5.0]

    return TorqueAnalysis(
        machine=tc.machine,
        threshold_nm=threshold_nm,
        total_samples=total,
        error_samples=err_samples,
        pct_error=err_samples / total * 100 if total else 0.0,
        model_violation_samples=viol_samples,
        pct_violation=viol_samples / total * 100 if total else 0.0,
        max_error_nm=max_err,
        mean_error_nm=mean_err,
        events=events,
        error_series=error,
        outside_model=outside_model,
        calibration=calibration,
    )


def print_torque_events(torque_analyses: list, top_n: int = 20):
    for ta in torque_analyses:
        if not ta.events:
            continue
        sorted_events = sorted(ta.events, key=lambda e: e.duration_s, reverse=True)
        print(f"\n=== {ta.machine}: {len(ta.events)} eventos de precisión de par ===")
        for i, e in enumerate(sorted_events[:top_n], 1):
            limit_label = "Cmax" if e.mode == 'motor' else "Cmin"
            within_str = "DENTRO del límite" if e.within_limits else "FUERA del límite"
            mismatch_str = f" | {e.mismatch_samples}/{e.n_samples} mismatch CsTq-EST" if e.mismatch_samples > 0 else ""
            derating_str = f" | {e.derating_samples}/{e.n_samples} derating" if e.derating_samples > 0 else ""
            cause_str = f" ({e.derating_cause})" if e.derating_cause else ""
            flags_str = f" | Flags: {e.active_flags_desc}" if e.active_flags_desc else ""
            eajs_str = f" | EAJS: {e.eajs_filtered_samples}" if e.eajs_filtered_samples > 0 else ""
            sat_str = f" | sat: {e.saturation_state}" if e.saturation_state else ""
            current_str = ""
            if e.avg_i_d_error > 0 or e.avg_i_q_error > 0:
                sat_tag = "[SATURADO]" if e.rgl_sat_samples > 0 else "[NO SAT]"
                current_str = f" | Id_err={e.avg_i_d_error:.2f}A Iq_err={e.avg_i_q_error:.2f}A {sat_tag}"
            fases_str = ""
            if e.phase_unbalanced:
                fases_str = f" | Fases: DESCOMPENSADAS (idx={e.phase_imbalance_index:.2f}A)"
            else:
                fases_str = " | Fases: OK"
            limit_str = f"{e.limit_used:.1f}" if e.limit_used is not None else "N/A"
            print(
                f"  #{i} ({e.mode}) t={e.start_time:.2f}s->{e.end_time:.2f}s "
                f"(dur={e.duration_s:.3f}s) | "
                f"Par medio={e.avg_tq_est:.2f} Nm vs {limit_label}={limit_str} Nm -> {within_str} | "
                f"Udc={e.avg_voltage:.1f} V N={e.avg_speed:.0f} rpm T={e.avg_rotor_temp:.1f} C"
                f"{mismatch_str}{derating_str}{cause_str}{flags_str}{eajs_str}{sat_str}{current_str}{fases_str}"
            )


def _get_range_at(df: pd.DataFrame, tc: TorqueChannels,
                  perf_db: Optional[TorquePerfDatabase], idx: int
                  ) -> Tuple[Optional[float], Optional[float]]:
    if perf_db is None:
        return None, None
    try:
        voltage = float(df[tc.voltage].iloc[idx]) if tc.voltage and tc.voltage in df.columns else None
        speed = abs(float(df[tc.speed].iloc[idx])) if tc.speed and tc.speed in df.columns else None
        temp = float(df[tc.rotor_temp].iloc[idx]) if tc.rotor_temp and tc.rotor_temp in df.columns else 40.0
        if voltage is None or speed is None:
            return None, None
        return perf_db.interpolate(tc.machine, voltage, speed, temp)
    except (IndexError, ValueError, TypeError):
        return None, None


def _check_violation(df: pd.DataFrame, tc: TorqueChannels,
                     perf_db: Optional[TorquePerfDatabase], idx: int) -> bool:
    if perf_db is None:
        return False
    tq_est = float(df[tc.tq_est].iloc[idx])
    cmin, cmax = _get_range_at(df, tc, perf_db, idx)
    if cmin is None or cmax is None:
        return False
    if tq_est >= 0:
        return tq_est > cmax
    else:
        return tq_est < cmin


def _infer_raster(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.001
    return float(df.index[1] - df.index[0])


def torque_events_to_dataframe(events: List[TorquePrecisionEvent]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=[
            'machine', 'start_time_s', 'end_time_s', 'duration_s',
            'max_error_Nm', 'mean_error_Nm', 'n_samples', 'model_violations',
            'Cmin_Nm', 'Cmax_Nm', 'mode', 'avg_tq_est_Nm', 'limit_used_Nm',
            'within_limits', 'avg_voltage_V', 'avg_speed_rpm', 'avg_rotor_temp_C',
            'mismatch_samples', 'derating_samples', 'derating_min_samples',
            'derating_max_samples', 'derating_cause',
            'active_flags_desc', 'eajs_filtered_samples',
            'avg_vd', 'avg_vq', 'rgl_sat_samples', 'saturation_state',
            'avg_i_d', 'avg_i_q', 'avg_i_d_req', 'avg_i_q_req',
            'avg_i_d_error', 'avg_i_q_error',
            'phase_unbalanced', 'phase_imbalance_index',
        ])
    rows = []
    for e in events:
        rows.append({
            'machine': e.machine,
            'start_time_s': round(e.start_time, 4),
            'end_time_s': round(e.end_time, 4),
            'duration_s': round(e.duration_s, 4),
            'max_error_Nm': round(e.max_error_nm, 4),
            'mean_error_Nm': round(e.mean_error_nm, 4),
            'n_samples': e.n_samples,
            'model_violations': e.model_violations,
            'Cmin_Nm': round(e.cmin_at_event, 2) if e.cmin_at_event is not None else '',
            'Cmax_Nm': round(e.cmax_at_event, 2) if e.cmax_at_event is not None else '',
            'mode': e.mode,
            'avg_tq_est_Nm': round(e.avg_tq_est, 2),
            'limit_used_Nm': round(e.limit_used, 2) if e.limit_used is not None else '',
            'within_limits': 'Yes' if e.within_limits else 'No',
            'avg_voltage_V': round(e.avg_voltage, 1),
            'avg_speed_rpm': round(e.avg_speed, 0),
            'avg_rotor_temp_C': round(e.avg_rotor_temp, 1),
            'mismatch_samples': e.mismatch_samples,
            'derating_samples': e.derating_samples,
            'derating_min_samples': e.derating_min_samples,
            'derating_max_samples': e.derating_max_samples,
            'derating_cause': e.derating_cause,
            'active_flags_desc': e.active_flags_desc,
            'eajs_filtered_samples': e.eajs_filtered_samples,
            'avg_vd': round(e.avg_vd, 2),
            'avg_vq': round(e.avg_vq, 2),
            'rgl_sat_samples': e.rgl_sat_samples,
            'saturation_state': e.saturation_state,
            'avg_i_d': round(e.avg_i_d, 2),
            'avg_i_q': round(e.avg_i_q, 2),
            'avg_i_d_req': round(e.avg_i_d_req, 2),
            'avg_i_q_req': round(e.avg_i_q_req, 2),
            'avg_i_d_error': round(e.avg_i_d_error, 2),
            'avg_i_q_error': round(e.avg_i_q_error, 2),
            'phase_unbalanced': 'Sí' if e.phase_unbalanced else 'No',
            'phase_imbalance_index': round(e.phase_imbalance_index, 2),
        })
    return pd.DataFrame(rows)


def torque_stats_to_dataframe(analyses: List[TorqueAnalysis]) -> pd.DataFrame:
    rows = []
    for a in analyses:
        rows.append({
            'machine': a.machine,
            'threshold_Nm': a.threshold_nm,
            'total_samples': a.total_samples,
            'error_samples': a.error_samples,
            'pct_error': round(a.pct_error, 2),
            'model_violation_samples': a.model_violation_samples,
            'pct_violation': round(a.pct_violation, 2),
            'max_error_Nm': round(a.max_error_nm, 4),
            'mean_error_Nm': round(a.mean_error_nm, 4),
            'num_events': len(a.events),
        })
    return pd.DataFrame(rows)
