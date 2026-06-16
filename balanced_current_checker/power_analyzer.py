from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from dico_reader import TorqueChannels

_POWER_IDLE_KW = 1.0
_BALANCE_ERR_KW = 2.0
_RPM_TO_RAD_S = np.pi / 30.0


# ── Internal column names (added to DataFrame by compute_power_channels) ──
_COL_ME_ELEC_HEVC = '_ME_ELEC_HEVC'
_COL_HSG_ELEC_HEVC = '_HSG_ELEC_HEVC'
_COL_INV_CURR_ME = '_INV_CURR_ME'
_COL_INV_CURR_HSG = '_INV_CURR_HSG'
_COL_ME_ELEC_INV = '_ME_ELEC_INV'
_COL_HSG_ELEC_INV = '_HSG_ELEC_INV'
_COL_ME_MECH_HEVC = '_ME_MECH_HEVC'
_COL_HSG_MECH_HEVC = '_HSG_MECH_HEVC'
_COL_ME_MECH_INV = '_ME_MECH_INV'
_COL_HSG_MECH_INV = '_HSG_MECH_INV'
_COL_ICE_POWER = '_ICE_POWER'
_COL_HVB_POW_KW = '_HVB_POW_KW'
_COL_P_CONS_KW = '_P_CONS_KW'
_COL_TQ_WHEEL = '_TQ_WHEEL'
_COL_POT_RUEDA = '_POT_RUEDA'


@dataclass
class CasePowerFlow:
    case_name: str
    n_samples: int
    pct_time: float
    me_elec_hevc_kw: float
    hsg_elec_hevc_kw: float
    me_elec_inv_kw: float
    hsg_elec_inv_kw: float
    me_elec_offset_kw: float
    hsg_elec_offset_kw: float
    me_mech_hevc_kw: float
    hsg_mech_hevc_kw: float
    me_mech_inv_kw: float
    hsg_mech_inv_kw: float
    me_mech_offset_kw: float
    hsg_mech_offset_kw: float
    me_inv_losses_kw: float
    hsg_inv_losses_kw: float
    me_efficiency: float
    hsg_efficiency: float
    p_cons_kw: float
    p_bat_kw: float
    ice_power_kw: float
    tq_wheel_nm: float
    pot_rueda_kw: float
    eff_ice_hsg: float
    eff_me_hevc: float
    eff_me_inv: float
    eff_hsg_hevc: float
    eff_hsg_inv: float
    balance_error_kw: float
    pct_balance_ok: float


@dataclass
class EfficiencyEntry:
    machine: str
    mode: str
    efficiency: float
    n_samples: int
    mean_p_mech_kw: float
    mean_p_elec_kw: float


@dataclass
class DlsEffEntry:
    dls_code: int
    dls_name: str
    n_samples: int
    eff_me_motor_hevc: float
    eff_me_motor_inv: float
    eff_me_gen_hevc: float
    eff_me_gen_inv: float
    eff_hsg_motor_hevc: float
    eff_hsg_motor_inv: float
    eff_hsg_gen_hevc: float
    eff_hsg_gen_inv: float
    eff_ice_hsg: float


@dataclass
class BalanceEvent:
    start_time: float
    end_time: float
    duration_s: float
    max_error_kw: float
    mean_error_kw: float
    n_samples: int


@dataclass
class PowerBalanceResult:
    cases: List[CasePowerFlow]
    efficiencies: List[EfficiencyEntry]
    dls_effs: List[DlsEffEntry]
    balance_events: List[BalanceEvent]
    total_samples: int
    sign_validation: Optional[Dict] = None


# ── Helper: classify operation case ──

def _classify_case(me_pow_kw: float, hsg_pow_kw: float, p_bat_net_kw: float) -> str:
    me_active = abs(me_pow_kw) >= _POWER_IDLE_KW
    hsg_active = abs(hsg_pow_kw) >= _POWER_IDLE_KW

    if not me_active and not hsg_active:
        return 'Dual idle'

    if not me_active:
        if hsg_pow_kw > 0:
            return 'ICE puro (HSG gen)'
        else:
            return 'ICE puro (HSG motor)'

    if not hsg_active:
        if me_pow_kw > 0:
            return 'EV puro (ME gen)'
        else:
            return 'EV puro (ME motor)'

    if me_pow_kw < 0 and hsg_pow_kw < 0:
        return 'Dual motor'
    elif me_pow_kw > 0 and hsg_pow_kw > 0:
        return 'Dual gen'
    elif me_pow_kw < 0 and hsg_pow_kw > 0:
        return 'ME motor + HSG gen'
    else:
        if abs(p_bat_net_kw) < _POWER_IDLE_KW:
            return 'Motoring (ME gen + HSG motor)'
        else:
            return 'ME gen + HSG motor'


# ── Compute all derived power channels ──

def compute_power_channels(
    df: pd.DataFrame,
    col_hvbus_v_inv1: str, col_inv1_i: str,
    col_hvbus_v_inv2: str, col_inv2_i: str,
    col_hvbus_aftr_rly: str,
    col_i_d_me: str, col_i_q_me: str, col_v_d_me: str, col_v_q_me: str,
    col_i_d_hsg: str, col_i_q_hsg: str, col_v_d_hsg: str, col_v_q_hsg: str,
    col_est_tq_me_hevc: str, col_est_tq_hsg_hevc: str,
    col_spd_me_hevc: str, col_spd_hsg_hevc: str,
    col_ice_tq: str, col_ice_spd: str,
    col_hvb_pow: str, col_p_cons: str,
    col_ice_tq_mech: str = '', col_dls: str = '',
    dls_table: Optional[Dict[int, tuple]] = None,
    col_whl_spd: str = '', wheel_radius_m: float = 0.0,
    tc_me: Optional[TorqueChannels] = None,
    tc_hsg: Optional[TorqueChannels] = None,
) -> Dict[str, bool]:
    """Add derived power columns to df in-place. Returns dict of which columns were created."""
    created: Dict[str, bool] = {k: False for k in [
        _COL_ME_ELEC_HEVC, _COL_HSG_ELEC_HEVC,
        _COL_INV_CURR_ME, _COL_INV_CURR_HSG,
        _COL_ME_ELEC_INV, _COL_HSG_ELEC_INV,
        _COL_ME_MECH_HEVC, _COL_HSG_MECH_HEVC,
        _COL_ME_MECH_INV, _COL_HSG_MECH_INV,
        _COL_ICE_POWER, _COL_HVB_POW_KW, _COL_P_CONS_KW, _COL_TQ_WHEEL, _COL_POT_RUEDA,
    ]}
    astype = 'float64'

    def _has(c: str) -> bool:
        return bool(c) and c in df.columns

    # ME_ELEC_HEVC (negado: signo desde batería, positivo = genera, negativo = motor)
    if _has(col_hvbus_v_inv1) and _has(col_inv1_i):
        v = df[col_hvbus_v_inv1].astype(astype)
        i = df[col_inv1_i].astype(astype)
        df[_COL_ME_ELEC_HEVC] = -v * i / 1000.0
        created[_COL_ME_ELEC_HEVC] = True

    # HSG_ELEC_HEVC (negado: mismo criterio)
    if _has(col_hvbus_v_inv2) and _has(col_inv2_i):
        v = df[col_hvbus_v_inv2].astype(astype)
        i = df[col_inv2_i].astype(astype)
        df[_COL_HSG_ELEC_HEVC] = -v * i / 1000.0
        created[_COL_HSG_ELEC_HEVC] = True

    # INV_Current_ME and ME_ELEC_INV
    if (_has(col_i_d_me) and _has(col_v_d_me) and _has(col_i_q_me) and _has(col_v_q_me)
            and _has(col_hvbus_aftr_rly)):
        id_ = df[col_i_d_me].astype(astype)
        vd = df[col_v_d_me].astype(astype)
        iq = df[col_i_q_me].astype(astype)
        vq = df[col_v_q_me].astype(astype)
        vbus = df[col_hvbus_aftr_rly].astype(astype)
        i_dc = 1.5 * (id_ * vd + iq * vq) / vbus.clip(lower=1.0)
        df[_COL_INV_CURR_ME] = i_dc
        df[_COL_ME_ELEC_INV] = vbus * i_dc / 1000.0
        created[_COL_INV_CURR_ME] = True
        created[_COL_ME_ELEC_INV] = True

    # INV_Current_HSG and HSG_ELEC_INV
    if (_has(col_i_d_hsg) and _has(col_v_d_hsg) and _has(col_i_q_hsg) and _has(col_v_q_hsg)
            and _has(col_hvbus_aftr_rly)):
        id_ = df[col_i_d_hsg].astype(astype)
        vd = df[col_v_d_hsg].astype(astype)
        iq = df[col_i_q_hsg].astype(astype)
        vq = df[col_v_q_hsg].astype(astype)
        vbus = df[col_hvbus_aftr_rly].astype(astype)
        i_dc = 1.5 * (id_ * vd + iq * vq) / vbus.clip(lower=1.0)
        df[_COL_INV_CURR_HSG] = i_dc
        df[_COL_HSG_ELEC_INV] = vbus * i_dc / 1000.0
        created[_COL_INV_CURR_HSG] = True
        created[_COL_HSG_ELEC_INV] = True

    # ME_MECH_HEVC
    if _has(col_est_tq_me_hevc) and _has(col_spd_me_hevc):
        tq = df[col_est_tq_me_hevc].astype(astype)
        spd = df[col_spd_me_hevc].astype(astype).abs()
        df[_COL_ME_MECH_HEVC] = tq * spd * _RPM_TO_RAD_S / 1000.0
        created[_COL_ME_MECH_HEVC] = True

    # HSG_MECH_HEVC
    if _has(col_est_tq_hsg_hevc) and _has(col_spd_hsg_hevc):
        tq = df[col_est_tq_hsg_hevc].astype(astype)
        spd = df[col_spd_hsg_hevc].astype(astype).abs()
        df[_COL_HSG_MECH_HEVC] = tq * spd * _RPM_TO_RAD_S / 1000.0
        created[_COL_HSG_MECH_HEVC] = True

    # ME_MECH_INV (from TorqueChannels — INV signals)
    has_me_inv = (tc_me and tc_me.tq_est and tc_me.speed
                  and tc_me.tq_est in df.columns and tc_me.speed in df.columns)
    if has_me_inv:
        tq = df[tc_me.tq_est].astype(astype)
        spd = df[tc_me.speed].astype(astype).abs()
        df[_COL_ME_MECH_INV] = tq * spd * _RPM_TO_RAD_S / 1000.0
        created[_COL_ME_MECH_INV] = True

    # HSG_MECH_INV
    has_hsg_inv = (tc_hsg and tc_hsg.tq_est and tc_hsg.speed
                   and tc_hsg.tq_est in df.columns and tc_hsg.speed in df.columns)
    if has_hsg_inv:
        tq = df[tc_hsg.tq_est].astype(astype)
        spd = df[tc_hsg.speed].astype(astype).abs()
        df[_COL_HSG_MECH_INV] = tq * spd * _RPM_TO_RAD_S / 1000.0
        created[_COL_HSG_MECH_INV] = True

    # ICE_POWER
    if _has(col_ice_tq) and _has(col_ice_spd):
        tq = df[col_ice_tq].astype(astype)
        spd = df[col_ice_spd].astype(astype).abs()
        df[_COL_ICE_POWER] = tq * spd * _RPM_TO_RAD_S / 1000.0
        created[_COL_ICE_POWER] = True

    # HVB_POW_KW
    if _has(col_hvb_pow):
        df[_COL_HVB_POW_KW] = df[col_hvb_pow].astype(astype) / 1000.0
        created[_COL_HVB_POW_KW] = True

    # P_CONS_KW
    if _has(col_p_cons):
        df[_COL_P_CONS_KW] = df[col_p_cons].astype(astype) / 1000.0
        created[_COL_P_CONS_KW] = True

    # Wheel torque via DLS lookup
    if dls_table and _has(col_dls) and _has(col_ice_tq) and _has(col_est_tq_me_hevc):
        dls = df[col_dls].astype(astype)
        t_ice = df[col_ice_tq].astype(astype)
        t_me = df[col_est_tq_me_hevc].astype(astype)
        t_wheel = pd.Series(np.nan, index=df.index)
        for code, (_, ri, rm) in dls_table.items():
            mask = dls == code
            if mask.any():
                t_wheel.loc[mask] = t_ice.loc[mask] * ri + t_me.loc[mask] * rm
        df[_COL_TQ_WHEEL] = t_wheel
        created[_COL_TQ_WHEEL] = True

    # Wheel power (needs T_wheel + wheel speed)
    if created.get(_COL_TQ_WHEEL, False) and _has(col_whl_spd) and wheel_radius_m > 0:
        whl_spd_kmh = df[col_whl_spd].astype(astype).abs()
        omega_radps = (whl_spd_kmh / 3.6) / wheel_radius_m
        df[_COL_POT_RUEDA] = df[_COL_TQ_WHEEL] * omega_radps / 1000.0
        created[_COL_POT_RUEDA] = True

    return created


# ── Main analysis ──

def _efficiency(p_elec: float, p_mech: float) -> float:
    if abs(p_elec) < 0.01 or abs(p_mech) < 0.01:
        return 0.0
    if (p_elec > 0 and p_mech > 0) or (p_elec < 0 and p_mech < 0):
        return abs(p_mech) / abs(p_elec)
    return 0.0


def analyze_power_balance(
    df: pd.DataFrame,
    tc_me: Optional[TorqueChannels] = None,
    tc_hsg: Optional[TorqueChannels] = None,
    col_hvbus_v_inv1: str = '',
    col_inv1_i: str = '',
    col_hvbus_v_inv2: str = '',
    col_inv2_i: str = '',
    col_hvbus_aftr_rly: str = '',
    col_i_d_me: str = '',
    col_i_q_me: str = '',
    col_v_d_me: str = '',
    col_v_q_me: str = '',
    col_i_d_hsg: str = '',
    col_i_q_hsg: str = '',
    col_v_d_hsg: str = '',
    col_v_q_hsg: str = '',
    col_est_tq_me_hevc: str = '',
    col_est_tq_hsg_hevc: str = '',
    col_spd_me_hevc: str = '',
    col_spd_hsg_hevc: str = '',
    col_ice_tq: str = '',
    col_ice_spd: str = '',
    col_hvb_pow: str = '',
    col_p_cons: str = '',
    col_dls: str = '',
    dls_table: Optional[Dict[int, tuple]] = None,
    col_whl_spd: str = '',
    wheel_radius_m: float = 0.0,
) -> PowerBalanceResult:
    # Compute derived channels
    created = compute_power_channels(
        df,
        col_hvbus_v_inv1, col_inv1_i,
        col_hvbus_v_inv2, col_inv2_i,
        col_hvbus_aftr_rly,
        col_i_d_me, col_i_q_me, col_v_d_me, col_v_q_me,
        col_i_d_hsg, col_i_q_hsg, col_v_d_hsg, col_v_q_hsg,
        col_est_tq_me_hevc, col_est_tq_hsg_hevc,
        col_spd_me_hevc, col_spd_hsg_hevc,
        col_ice_tq, col_ice_spd,
        col_hvb_pow, col_p_cons,
        col_dls=col_dls,
        dls_table=dls_table,
        col_whl_spd=col_whl_spd, wheel_radius_m=wheel_radius_m,
        tc_me=tc_me, tc_hsg=tc_hsg,
    )

    has_me_elec = created.get(_COL_ME_ELEC_HEVC, False)
    has_hsg_elec = created.get(_COL_HSG_ELEC_HEVC, False)
    has_cons = created.get(_COL_P_CONS_KW, False)
    has_bat = created.get(_COL_HVB_POW_KW, False)

    if not (has_me_elec and has_hsg_elec):
        return PowerBalanceResult(cases=[], efficiencies=[], dls_effs=[], balance_events=[], total_samples=len(df))

    # ── Convert all Series to numpy arrays for vectorized ops ──
    me_elec = df[_COL_ME_ELEC_HEVC].astype('float64').values
    hsg_elec = df[_COL_HSG_ELEC_HEVC].astype('float64').values
    cons = df[_COL_P_CONS_KW].astype('float64').values if has_cons else np.zeros(len(df))
    bat = df[_COL_HVB_POW_KW].astype('float64').values if has_bat else np.zeros(len(df))
    me_elec_inv = df[_COL_ME_ELEC_INV].astype('float64').values if _COL_ME_ELEC_INV in df.columns else np.full(len(df), np.nan)
    hsg_elec_inv = df[_COL_HSG_ELEC_INV].astype('float64').values if _COL_HSG_ELEC_INV in df.columns else np.full(len(df), np.nan)
    me_mech_hevc_arr = df[_COL_ME_MECH_HEVC].astype('float64').values if _COL_ME_MECH_HEVC in df.columns else np.full(len(df), np.nan)
    hsg_mech_hevc_arr = df[_COL_HSG_MECH_HEVC].astype('float64').values if _COL_HSG_MECH_HEVC in df.columns else np.full(len(df), np.nan)
    me_mech_inv = df[_COL_ME_MECH_INV].astype('float64').values if _COL_ME_MECH_INV in df.columns else np.full(len(df), np.nan)
    hsg_mech_inv = df[_COL_HSG_MECH_INV].astype('float64').values if _COL_HSG_MECH_INV in df.columns else np.full(len(df), np.nan)
    ice_pow = df[_COL_ICE_POWER].astype('float64').values if _COL_ICE_POWER in df.columns else np.full(len(df), np.nan)
    tq_wheel = df[_COL_TQ_WHEEL].astype('float64').values if _COL_TQ_WHEEL in df.columns else np.full(len(df), np.nan)
    pot_rueda = df[_COL_POT_RUEDA].astype('float64').values if _COL_POT_RUEDA in df.columns else np.full(len(df), np.nan)

    time_idx = df.index.values
    total = len(df)

    # ── Vectorized case classification ──
    me_active = np.abs(me_elec) >= _POWER_IDLE_KW
    hsg_active = np.abs(hsg_elec) >= _POWER_IDLE_KW
    conds = [
        ~me_active & ~hsg_active,
        ~me_active & (hsg_elec > 0),
        ~me_active & ~(hsg_elec > 0),
        me_active & ~hsg_active & (me_elec > 0),
        me_active & ~hsg_active & ~(me_elec > 0),
        (me_elec < 0) & (hsg_elec < 0),
        (me_elec > 0) & (hsg_elec > 0),
        (me_elec < 0) & (hsg_elec > 0),
    ]
    choices = [
        'Dual idle', 'ICE puro (HSG gen)', 'ICE puro (HSG motor)',
        'EV puro (ME gen)', 'EV puro (ME motor)',
        'Dual motor', 'Dual gen', 'ME motor + HSG gen',
    ]
    case_names = np.select(conds, choices, default='ME gen + HSG motor')
    motoring = (me_elec > 0) & (hsg_elec < 0) & (np.abs(bat) < _POWER_IDLE_KW)
    case_names[motoring] = 'Motoring (ME gen + HSG motor)'

    bal_err = np.abs(cons + bat - me_elec - hsg_elec)
    ok_flag = bal_err < _BALANCE_ERR_KW

    # ── Efficiency mode masks ──
    me_hevc_motor = ~np.isnan(me_mech_hevc_arr) & (me_elec < -_POWER_IDLE_KW) & (me_mech_hevc_arr > _POWER_IDLE_KW)
    me_hevc_gen   = ~np.isnan(me_mech_hevc_arr) & (me_elec > _POWER_IDLE_KW) & (me_mech_hevc_arr < -_POWER_IDLE_KW)
    hsg_hevc_motor = ~np.isnan(hsg_mech_hevc_arr) & (hsg_elec < -_POWER_IDLE_KW) & (hsg_mech_hevc_arr > _POWER_IDLE_KW)
    hsg_hevc_gen   = ~np.isnan(hsg_mech_hevc_arr) & (hsg_elec > _POWER_IDLE_KW) & (hsg_mech_hevc_arr < -_POWER_IDLE_KW)
    # INV: P_elec_INV > 0 = motoring (dq convention, opposite to HEVC/battery)
    me_inv_motor = ~np.isnan(me_elec_inv) & ~np.isnan(me_mech_inv) & (me_elec_inv > _POWER_IDLE_KW) & (me_mech_inv > _POWER_IDLE_KW)
    me_inv_gen   = ~np.isnan(me_elec_inv) & ~np.isnan(me_mech_inv) & (me_elec_inv < -_POWER_IDLE_KW) & (me_mech_inv < -_POWER_IDLE_KW)
    hsg_inv_motor = ~np.isnan(hsg_elec_inv) & ~np.isnan(hsg_mech_inv) & (hsg_elec_inv > _POWER_IDLE_KW) & (hsg_mech_inv > _POWER_IDLE_KW)
    hsg_inv_gen   = ~np.isnan(hsg_elec_inv) & ~np.isnan(hsg_mech_inv) & (hsg_elec_inv < -_POWER_IDLE_KW) & (hsg_mech_inv < -_POWER_IDLE_KW)
    ice_hsg_mask = (ice_pow > _POWER_IDLE_KW) & ~np.isnan(pot_rueda) & ~np.isnan(hsg_mech_hevc_arr)

    # ── Build accumulator DataFrame (lightweight, no copy of arrays) ──
    df_acc = pd.DataFrame({
        'case': case_names, 'mh': me_elec, 'hh': hsg_elec,
        'co': cons, 'ba': bat, 'be': bal_err, 'ok': ok_flag,
        'mhi': me_elec_inv, 'hhi': hsg_elec_inv,
        'mmh': me_mech_hevc_arr, 'hmh': hsg_mech_hevc_arr,
        'mmi': me_mech_inv, 'hmi': hsg_mech_inv,
        'ip': ice_pow, 'tq': tq_wheel, 'pr': pot_rueda,
    })
    has_dls = bool(col_dls and col_dls in df.columns and dls_table)
    if has_dls:
        dls_arr = df[col_dls].astype(int).values
        df_acc['dls'] = dls_arr

    # ── Per-case aggregation via groupby ──
    g = df_acc.groupby('case', sort=True)

    def _agg_sum(col, mask=None):
        if mask is None:
            return g[col].sum()
        return g[col].apply(lambda s: s.loc[mask.loc[s.index]].sum() if mask.loc[s.index].any() else 0.0)

    case_data = {}
    for cn, grp in g:
        n = len(grp)
        idx = grp.index
        m_ok = ok_flag[idx]
        def _s(col, mask=None):
            if mask is None:
                return float(grp[col].sum(skipna=True))
            return float(grp.loc[mask, col].sum(skipna=True) if mask.any() else 0.0)
        case_data[cn] = {
            'n': n, 'me_elec_hevc': _s('mh'), 'hsg_elec_hevc': _s('hh'),
            'me_elec_inv': _s('mhi'), 'hsg_elec_inv': _s('hhi'),
            'me_mech_hevc': _s('mmh'), 'hsg_mech_hevc': _s('hmh'),
            'me_mech_inv': _s('mmi'), 'hsg_mech_inv': _s('hmi'),
            'p_cons': _s('co'), 'p_bat': _s('ba'),
            'ice_power': _s('ip'), 'tq_wheel': _s('tq'), 'pot_rueda': _s('pr'),
            'err_sum': _s('be'), 'ok_count': _s('ok'),
        }
        # ICE+HSG efficiency
        ihg = ice_hsg_mask[idx]
        if ihg.any():
            case_data[cn]['ice_hsg_eff_num'] = _s('pr', ihg) + _s('hmh', ihg)
            case_data[cn]['ice_hsg_eff_den'] = _s('ip', ihg)
        else:
            case_data[cn]['ice_hsg_eff_num'] = case_data[cn]['ice_hsg_eff_den'] = 0.0
        # ME HEVC
        mot = me_hevc_motor[idx]; gen = me_hevc_gen[idx]
        d = case_data[cn]
        d['me_hevc_num'] = _s('mmh', mot) + _s('mh', gen)
        d['me_hevc_den'] = (-_s('mh', mot)) + (-_s('mmh', gen))
        # HSG HEVC
        hmot = hsg_hevc_motor[idx]; hgen = hsg_hevc_gen[idx]
        d['hsg_hevc_num'] = _s('hmh', hmot) + _s('hh', hgen)
        d['hsg_hevc_den'] = (-_s('hh', hmot)) + (-_s('hmh', hgen))
        # ME INV (P_INV > 0 = motor en convencion dq, opuesta a HEVC)
        imot = me_inv_motor[idx]; igen = me_inv_gen[idx]
        d['me_inv_num'] = _s('mmi', imot) + (-_s('mhi', igen))  # P_mech + (-P_INV_gen)
        d['me_inv_den'] = _s('mhi', imot) + (-_s('mmi', igen))  # P_INV + (-P_mech_gen)
        # HSG INV
        hmot_i = hsg_inv_motor[idx]; hgen_i = hsg_inv_gen[idx]
        d['hsg_inv_num'] = _s('hmi', hmot_i) + (-_s('hhi', hgen_i))
        d['hsg_inv_den'] = _s('hhi', hmot_i) + (-_s('hmi', hgen_i))

    # ── Efficiency sample collections (vectorized extraction) ──
    def _zip_msk(a, b, mask):
        idx = np.where(mask)[0]
        return list(zip(a[idx], b[idx]))

    me_motor_samples = _zip_msk(me_mech_hevc_arr, -me_elec, me_hevc_motor)
    me_gen_samples   = _zip_msk(me_elec, -me_mech_hevc_arr, me_hevc_gen)
    hsg_motor_samples = _zip_msk(hsg_mech_hevc_arr, -hsg_elec, hsg_hevc_motor)
    hsg_gen_samples   = _zip_msk(hsg_elec, -hsg_mech_hevc_arr, hsg_hevc_gen)
    inv_me_motor = _zip_msk(me_mech_inv, me_elec_inv, me_inv_motor)
    inv_me_gen   = _zip_msk(-me_elec_inv, -me_mech_inv, me_inv_gen)
    inv_hsg_motor = _zip_msk(hsg_mech_inv, hsg_elec_inv, hsg_inv_motor)
    inv_hsg_gen   = _zip_msk(-hsg_elec_inv, -hsg_mech_inv, hsg_inv_gen)

    # ── DLS accumulators (vectorized groupby per DLS) ──
    dls_effs_list = []
    if has_dls:
        for code in sorted(set(dls_arr) & set(dls_table.keys())):
            m = dls_arr == code
            n = int(m.sum())
            if n == 0:
                continue
            def _de(col, mask, neg=False):
                s = df_acc.loc[m, col][mask[m]].sum(skipna=True) if mask[m].any() else 0.0
                return -s if neg else s
            dls_effs_list.append(DlsEffEntry(
                dls_code=int(code), dls_name=dls_table[int(code)][0], n_samples=n,
                eff_me_motor_hevc=(_de('mmh', me_hevc_motor) / max(1e-9, -_de('mh', me_hevc_motor, neg=True))),
                eff_me_motor_inv=(_de('mmi', me_inv_motor) / max(1e-9, _de('mhi', me_inv_motor))),
                eff_me_gen_hevc=(_de('mh', me_hevc_gen) / max(1e-9, -_de('mmh', me_hevc_gen, neg=True))),
                eff_me_gen_inv=(max(0.0, -_de('mhi', me_inv_gen)) / max(1e-9, -_de('mmi', me_inv_gen))),
                eff_hsg_motor_hevc=(_de('hmh', hsg_hevc_motor) / max(1e-9, -_de('hh', hsg_hevc_motor, neg=True))),
                eff_hsg_motor_inv=(_de('hmi', hsg_inv_motor) / max(1e-9, _de('hhi', hsg_inv_motor))),
                eff_hsg_gen_hevc=(_de('hh', hsg_hevc_gen) / max(1e-9, -_de('hmh', hsg_hevc_gen, neg=True))),
                eff_hsg_gen_inv=(max(0.0, -_de('hhi', hsg_inv_gen)) / max(1e-9, -_de('hmi', hsg_inv_gen))),
                eff_ice_hsg=(_de('pr', ice_hsg_mask) + _de('hmh', ice_hsg_mask)) / max(1e-9, _de('ip', ice_hsg_mask)),
            ))

    cases = []
    for cn, d in sorted(case_data.items()):
        n = d['n']
        n_inv = max(1, d['n'])
        me_elec_inv_avg = d['me_elec_inv'] / n_inv
        hsg_elec_inv_avg = d['hsg_elec_inv'] / n_inv
        me_mech_hevc_avg = d['me_mech_hevc'] / n_inv
        hsg_mech_hevc_avg = d['hsg_mech_hevc'] / n_inv
        me_mech_inv_avg = d['me_mech_inv'] / n_inv
        hsg_mech_inv_avg = d['hsg_mech_inv'] / n_inv
        me_elec_hevc_avg = d['me_elec_hevc'] / n
        hsg_elec_hevc_avg = d['hsg_elec_hevc'] / n

        cases.append(CasePowerFlow(
            case_name=cn,
            n_samples=n,
            pct_time=n / total * 100,
            me_elec_hevc_kw=me_elec_hevc_avg,
            hsg_elec_hevc_kw=hsg_elec_hevc_avg,
            me_elec_inv_kw=me_elec_inv_avg,
            hsg_elec_inv_kw=hsg_elec_inv_avg,
            me_elec_offset_kw=me_elec_hevc_avg - me_elec_inv_avg,
            hsg_elec_offset_kw=hsg_elec_hevc_avg - hsg_elec_inv_avg,
            me_mech_hevc_kw=me_mech_hevc_avg,
            hsg_mech_hevc_kw=hsg_mech_hevc_avg,
            me_mech_inv_kw=me_mech_inv_avg,
            hsg_mech_inv_kw=hsg_mech_inv_avg,
            me_mech_offset_kw=me_mech_hevc_avg - me_mech_inv_avg,
            hsg_mech_offset_kw=hsg_mech_hevc_avg - hsg_mech_inv_avg,
            me_inv_losses_kw=me_elec_inv_avg - me_mech_inv_avg,
            hsg_inv_losses_kw=hsg_elec_inv_avg - hsg_mech_inv_avg,
            me_efficiency=_efficiency(me_elec_inv_avg, me_mech_inv_avg),
            hsg_efficiency=_efficiency(hsg_elec_inv_avg, hsg_mech_inv_avg),
            p_cons_kw=d['p_cons'] / n,
            p_bat_kw=d['p_bat'] / n,
            ice_power_kw=d['ice_power'] / n_inv,
            tq_wheel_nm=d['tq_wheel'] / n,
            pot_rueda_kw=d['pot_rueda'] / n,
            eff_ice_hsg=(d['ice_hsg_eff_num'] / d['ice_hsg_eff_den']) if d['ice_hsg_eff_den'] > 0 else 0.0,
            eff_me_hevc=(d['me_hevc_num'] / d['me_hevc_den']) if d['me_hevc_den'] > 0 else 0.0,
            eff_me_inv=(d['me_inv_num'] / d['me_inv_den']) if d['me_inv_den'] > 0 else 0.0,
            eff_hsg_hevc=(d['hsg_hevc_num'] / d['hsg_hevc_den']) if d['hsg_hevc_den'] > 0 else 0.0,
            eff_hsg_inv=(d['hsg_inv_num'] / d['hsg_inv_den']) if d['hsg_inv_den'] > 0 else 0.0,
            balance_error_kw=d['err_sum'] / n,
            pct_balance_ok=d['ok_count'] / n * 100,
        ))

    # Build efficiency entries (HEVC path and INV path)
    effs = []
    for label, samples, machine, mode in [
        ('ME motor', me_motor_samples, 'ME', 'motor'),
        ('ME gen', me_gen_samples, 'ME', 'generator'),
        ('HSG motor', hsg_motor_samples, 'HSG', 'motor'),
        ('HSG gen', hsg_gen_samples, 'HSG', 'generator'),
    ]:
        if samples:
            arr = np.array(samples)
            p_mech = arr[:, 0]
            p_elec = arr[:, 1]
            eff = p_mech / p_elec
            effs.append(EfficiencyEntry(
                machine=machine, mode=mode + ' (HEVC)',
                efficiency=float(np.mean(eff)),
                n_samples=len(samples),
                mean_p_mech_kw=float(np.mean(p_mech)),
                mean_p_elec_kw=float(np.mean(p_elec)),
            ))
    for label, samples, machine, mode in [
        ('ME motor', inv_me_motor, 'ME', 'motor'),
        ('ME gen', inv_me_gen, 'ME', 'generator'),
        ('HSG motor', inv_hsg_motor, 'HSG', 'motor'),
        ('HSG gen', inv_hsg_gen, 'HSG', 'generator'),
    ]:
        if samples:
            arr = np.array(samples)
            p_mech = arr[:, 0]
            p_elec = arr[:, 1]
            eff = p_mech / p_elec
            effs.append(EfficiencyEntry(
                machine=machine, mode=mode + ' (INV)',
                efficiency=float(np.mean(eff)),
                n_samples=len(samples),
                mean_p_mech_kw=float(np.mean(p_mech)),
                mean_p_elec_kw=float(np.mean(p_elec)),
            ))

    dls_effs = dls_effs_list

    # Balance event detection (vectorized: find contiguous regions)
    balance_events = []
    if has_me_elec and has_hsg_elec and total > 0:
        exceed = bal_err > _BALANCE_ERR_KW
        diffs = np.diff(np.concatenate(([False], exceed, [False])).astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        for s, e in zip(starts, ends):
            idx_slice = slice(s, e)
            err_slice = bal_err[idx_slice]
            balance_events.append(BalanceEvent(
                start_time=float(time_idx[s]),
                end_time=float(time_idx[e - 1]),
                duration_s=float(time_idx[e - 1] - time_idx[s]),
                max_error_kw=float(err_slice.max()),
                mean_error_kw=float(err_slice.mean()),
                n_samples=int(e - s),
            ))

    return PowerBalanceResult(cases=cases, efficiencies=effs,
                               dls_effs=dls_effs, balance_events=balance_events,
                               total_samples=total)


# ── Print ──

def print_power_balance(result: PowerBalanceResult):
    if not result.cases:
        print("\n=== Balance de Potencia: NO DISPONIBLE ===")
        return
    print(f"\n=== Balance de Potencia ({result.total_samples} muestras) ===")
    hdr = (f"{'Caso':28s} | {'%t':>4s} | {'P_ME_HEVC':>9s} | {'P_ME_INV':>9s} | "
           f"{'Off_ME':>7s} | {'P_HSG_HEVC':>10s} | {'P_HSG_INV':>10s} | {'Off_HSG':>8s} | "
           f"{'T_wheel':>8s} | {'P_rueda':>8s} | {'Eff_ICE':>7s} | {'Err':>6s} | {'%OK':>4s}")
    print(hdr)
    print("-" * len(hdr))
    for c in result.cases:
        off_me = f"{c.me_elec_offset_kw:+.1f}" if not np.isnan(c.me_elec_offset_kw) else 'N/A'
        off_hsg = f"{c.hsg_elec_offset_kw:+.1f}" if not np.isnan(c.hsg_elec_offset_kw) else 'N/A'
        tq_s = f"{c.tq_wheel_nm:7.1f}" if not np.isnan(c.tq_wheel_nm) else '    N/A'
        pr_s = f"{c.pot_rueda_kw:7.2f}" if not np.isnan(c.pot_rueda_kw) else '    N/A'
        eff_s = f"{c.eff_ice_hsg*100:5.1f}%" if c.eff_ice_hsg > 0 else '  N/A'
        me_h = f"{c.eff_me_hevc*100:4.1f}%" if c.eff_me_hevc > 0 else ' N/A'
        me_i = f"{c.eff_me_inv*100:4.1f}%" if c.eff_me_inv > 0 else ' N/A'
        me_d = f"{'%+.1f' % ((c.eff_me_hevc-c.eff_me_inv)*100):>6s}" if c.eff_me_hevc > 0 and c.eff_me_inv > 0 else '   N/A'
        hs_h = f"{c.eff_hsg_hevc*100:4.1f}%" if c.eff_hsg_hevc > 0 else ' N/A'
        hs_i = f"{c.eff_hsg_inv*100:4.1f}%" if c.eff_hsg_inv > 0 else ' N/A'
        hs_d = f"{'%+.1f' % ((c.eff_hsg_hevc-c.eff_hsg_inv)*100):>6s}" if c.eff_hsg_hevc > 0 and c.eff_hsg_inv > 0 else '   N/A'
        print(f"{c.case_name:28s} | {c.pct_time:3.0f}% | "
              f"{c.me_elec_hevc_kw:8.2f} | {c.me_elec_inv_kw:8.2f} | {off_me:>7s} | "
              f"{c.hsg_elec_hevc_kw:9.2f} | {c.hsg_elec_inv_kw:9.2f} | {off_hsg:>8s} | "
              f"{tq_s:>8s} | {pr_s:>8s} | {eff_s:>7s} | {c.balance_error_kw:5.2f} | {c.pct_balance_ok:3.0f}%")
        print(f"{'':28s}   {'':4s}   {'':9s}   {'':9s}   {'':7s}   {'':10s}   {'':10s}   {'':8s}   "
              f"ME eff: HEVC={me_h}  INV={me_i}  Diff={me_d}   "
              f"HSG eff: HEVC={hs_h}  INV={hs_i}  Diff={hs_d}")

    if any(not np.isnan(c.ice_power_kw) for c in result.cases):
        print(f"\nICE Power (media por caso):")
        for c in result.cases:
            if not np.isnan(c.ice_power_kw) and abs(c.ice_power_kw) > 0.01:
                print(f"  {c.case_name:30s}: {c.ice_power_kw:7.2f} kW")

    if result.efficiencies:
        print(f"\nRendimientos:")
        for e in result.efficiencies:
            print(f"  {e.machine} ({e.mode:15s}): eff={e.efficiency*100:.1f}%  "
                  f"(P_mech={e.mean_p_mech_kw:.2f}kW  P_elec={e.mean_p_elec_kw:.2f}kW  "
                  f"n={e.n_samples})")


def print_dls_report(result: PowerBalanceResult):
    if not result.dls_effs:
        print("\n=== Rendimiento por marcha (DLS): NO DISPONIBLE ===")
        return
    print(f"\n=== Rendimiento por marcha (DLS) ===")
    print(f"{'DLS':>4s} | {'Marcha':12s} | {'Muestras':>8s} | "
          f"{'ME_mot_HEVC':>10s} | {'ME_mot_INV':>10s} | {'ME_gen_HEVC':>10s} | {'ME_gen_INV':>10s} | "
          f"{'HSG_mot_HEVC':>10s} | {'HSG_mot_INV':>10s} | {'HSG_gen_HEVC':>10s} | {'HSG_gen_INV':>10s} | "
          f"{'ICE+HSG':>7s}")
    print("-" * 130)
    for de in result.dls_effs:
        def _p(v):
            return f"{v*100:5.1f}%" if v > 0 else '  N/A'
        print(f"{de.dls_code:4d} | {de.dls_name:12s} | {de.n_samples:8d} | "
              f"{_p(de.eff_me_motor_hevc):>10s} | {_p(de.eff_me_motor_inv):>10s} | "
              f"{_p(de.eff_me_gen_hevc):>10s} | {_p(de.eff_me_gen_inv):>10s} | "
              f"{_p(de.eff_hsg_motor_hevc):>10s} | {_p(de.eff_hsg_motor_inv):>10s} | "
              f"{_p(de.eff_hsg_gen_hevc):>10s} | {_p(de.eff_hsg_gen_inv):>10s} | "
              f"{_p(de.eff_ice_hsg):>7s}")


def print_balance_events(result: PowerBalanceResult):
    if not result.balance_events:
        print("\n=== Eventos de desbalance: NO DETECTADOS ===")
        return
    print(f"\n=== Eventos de desbalance de potencia ({len(result.balance_events)} eventos) ===")
    print(f"{'Evento':>6s} | {'Inicio (s)':>10s} | {'Fin (s)':>10s} | {'Duracion (s)':>12s} | "
          f"{'Error max (kW)':>13s} | {'Error medio (kW)':>15s} | {'Muestras':>8s}")
    print("-" * 85)
    for i, ev in enumerate(result.balance_events, 1):
        print(f"{i:6d} | {ev.start_time:10.4f} | {ev.end_time:10.4f} | {ev.duration_s:12.4f} | "
              f"{ev.max_error_kw:13.3f} | {ev.mean_error_kw:15.3f} | {ev.n_samples:8d}")


# ── DataFrames ──

def power_balance_to_dataframe(result: PowerBalanceResult) -> pd.DataFrame:
    if not result.cases:
        return pd.DataFrame()
    rows = []
    for c in result.cases:
        def _fmt(v):
            return round(v, 2) if not np.isnan(v) else None
        rows.append({
            'Caso': c.case_name,
            'Muestras': c.n_samples,
            '% tiempo': round(c.pct_time, 1),
            'P_ME_HEVC (kW)': _fmt(c.me_elec_hevc_kw),
            'P_ME_INV (kW)': _fmt(c.me_elec_inv_kw),
            'Offset ME (kW)': _fmt(c.me_elec_offset_kw),
            'P_HSG_HEVC (kW)': _fmt(c.hsg_elec_hevc_kw),
            'P_HSG_INV (kW)': _fmt(c.hsg_elec_inv_kw),
            'Offset HSG (kW)': _fmt(c.hsg_elec_offset_kw),
            'P_cons (kW)': _fmt(c.p_cons_kw),
            'P_bat (kW)': _fmt(c.p_bat_kw),
            'ICE (kW)': _fmt(c.ice_power_kw),
            'Par rueda (Nm)': _fmt(c.tq_wheel_nm),
            'Pot rueda (kW)': _fmt(c.pot_rueda_kw),
            'Eff ICE+HSG': round(c.eff_ice_hsg * 100, 1) if not np.isnan(c.eff_ice_hsg) and c.eff_ice_hsg > 0 else None,
            'Eff ME HEVC': round(c.eff_me_hevc * 100, 1) if not np.isnan(c.eff_me_hevc) and c.eff_me_hevc > 0 else None,
            'Eff ME INV': round(c.eff_me_inv * 100, 1) if not np.isnan(c.eff_me_inv) and c.eff_me_inv > 0 else None,
            'Diff ME': round((c.eff_me_hevc - c.eff_me_inv) * 100, 1) if c.eff_me_hevc > 0 and c.eff_me_inv > 0 else None,
            'Eff HSG HEVC': round(c.eff_hsg_hevc * 100, 1) if not np.isnan(c.eff_hsg_hevc) and c.eff_hsg_hevc > 0 else None,
            'Eff HSG INV': round(c.eff_hsg_inv * 100, 1) if not np.isnan(c.eff_hsg_inv) and c.eff_hsg_inv > 0 else None,
            'Diff HSG': round((c.eff_hsg_hevc - c.eff_hsg_inv) * 100, 1) if c.eff_hsg_hevc > 0 and c.eff_hsg_inv > 0 else None,
            'Error balance (kW)': round(c.balance_error_kw, 3),
            '% balance OK': round(c.pct_balance_ok, 1),
        })
    return pd.DataFrame(rows)


def efficiency_to_dataframe(result: PowerBalanceResult) -> pd.DataFrame:
    if not result.efficiencies:
        return pd.DataFrame()
    rows = []
    for e in result.efficiencies:
        rows.append({
            'Máquina': e.machine,
            'Modo': e.mode,
            'Rendimiento (%)': round(e.efficiency * 100, 1),
            'P_mec media (kW)': round(e.mean_p_mech_kw, 2),
            'P_elec media (kW)': round(e.mean_p_elec_kw, 2),
            'Muestras': e.n_samples,
        })
    return pd.DataFrame(rows)


def power_comparison_to_dataframe(result: PowerBalanceResult) -> pd.DataFrame:
    """Global HEVC vs INV comparison per machine (aggregated across all cases)."""
    if not result.cases:
        return pd.DataFrame()
    me_hevc = 0.0
    me_inv = 0.0
    hsg_hevc = 0.0
    hsg_inv = 0.0
    me_mech_h = 0.0
    me_mech_i = 0.0
    hsg_mech_h = 0.0
    hsg_mech_i = 0.0
    n_me = 0
    n_hsg = 0
    for c in result.cases:
        w = c.n_samples
        if not np.isnan(c.me_elec_hevc_kw):
            me_hevc += c.me_elec_hevc_kw * w
            me_inv += c.me_elec_inv_kw * w
            me_mech_h += c.me_mech_hevc_kw * w
            me_mech_i += c.me_mech_inv_kw * w
            n_me += w
        if not np.isnan(c.hsg_elec_hevc_kw):
            hsg_hevc += c.hsg_elec_hevc_kw * w
            hsg_inv += c.hsg_elec_inv_kw * w
            hsg_mech_h += c.hsg_mech_hevc_kw * w
            hsg_mech_i += c.hsg_mech_inv_kw * w
            n_hsg += w

    def _avg(s, n):
        return round(s / n, 2) if n > 0 else None

    rows = []
    if n_me > 0:
        me_inv_avg = me_inv / n_me
        me_mech_i_avg = me_mech_i / n_me
        has_inv_me = not (np.isnan(me_inv) or np.isnan(me_mech_i) or n_me == 0)
        rows.append({
            'Máquina': 'ME',
            'P_HEVC media (kW)': _avg(me_hevc, n_me),
            'P_INV media (kW)': _avg(me_inv, n_me),
            'Offset HEVC-INV (kW)': _avg(me_hevc - me_inv, n_me),
            'P_mec HEVC (kW)': _avg(me_mech_h, n_me),
            'P_mec INV (kW)': _avg(me_mech_i, n_me),
            'Perdidas INV (kW)': round(me_inv_avg - me_mech_i_avg, 2) if has_inv_me else None,
            'Eff INV (%)': round(_efficiency(me_inv_avg, me_mech_i_avg) * 100, 1) if has_inv_me else None,
        })
    if n_hsg > 0:
        hsg_inv_avg = hsg_inv / n_hsg
        hsg_mech_i_avg = hsg_mech_i / n_hsg
        has_inv_hsg = not (np.isnan(hsg_inv) or np.isnan(hsg_mech_i) or n_hsg == 0)
        rows.append({
            'Máquina': 'HSG',
            'P_HEVC media (kW)': _avg(hsg_hevc, n_hsg),
            'P_INV media (kW)': _avg(hsg_inv, n_hsg),
            'Offset HEVC-INV (kW)': _avg(hsg_hevc - hsg_inv, n_hsg),
            'P_mec HEVC (kW)': _avg(hsg_mech_h, n_hsg),
            'P_mec INV (kW)': _avg(hsg_mech_i, n_hsg),
            'Perdidas INV (kW)': round(hsg_inv_avg - hsg_mech_i_avg, 2) if has_inv_hsg else None,
            'Eff INV (%)': round(_efficiency(hsg_inv_avg, hsg_mech_i_avg) * 100, 1) if has_inv_hsg else None,
        })
    return pd.DataFrame(rows)


def dls_eff_to_dataframe(result: PowerBalanceResult) -> pd.DataFrame:
    if not result.dls_effs:
        return pd.DataFrame()
    rows = []
    for de in result.dls_effs:
        def _fmt(v):
            return round(v * 100, 1) if v > 0 else None
        rows.append({
            'DLS': de.dls_code,
            'Marcha': de.dls_name,
            'Muestras': de.n_samples,
            'ME motor HEVC (%)': _fmt(de.eff_me_motor_hevc),
            'ME motor INV (%)': _fmt(de.eff_me_motor_inv),
            'ME gen HEVC (%)': _fmt(de.eff_me_gen_hevc),
            'ME gen INV (%)': _fmt(de.eff_me_gen_inv),
            'HSG motor HEVC (%)': _fmt(de.eff_hsg_motor_hevc),
            'HSG motor INV (%)': _fmt(de.eff_hsg_motor_inv),
            'HSG gen HEVC (%)': _fmt(de.eff_hsg_gen_hevc),
            'HSG gen INV (%)': _fmt(de.eff_hsg_gen_inv),
            'ICE+HSG (%)': _fmt(de.eff_ice_hsg),
        })
    return pd.DataFrame(rows)


def balance_events_to_dataframe(result: PowerBalanceResult) -> pd.DataFrame:
    if not result.balance_events:
        return pd.DataFrame()
    rows = []
    for ev in result.balance_events:
        rows.append({
            'Inicio (s)': round(ev.start_time, 4),
            'Fin (s)': round(ev.end_time, 4),
            'Duracion (s)': round(ev.duration_s, 4),
            'Error max (kW)': round(ev.max_error_kw, 3),
            'Error medio (kW)': round(ev.mean_error_kw, 3),
            'Muestras': ev.n_samples,
        })
    return pd.DataFrame(rows)


# ── Power sign validation ──

def validate_power_signs(
    df: pd.DataFrame,
    col_emot1_pow: str, col_emot2_pow: str,
    col_tq1: str, col_tq2: str,
    col_spd1: str, col_spd2: str,
    col_hvb_pow: str, col_cons: str,
) -> dict:
    """Analyze sign coherence between direct power signals and mechanical power.
    Returns dict with per-machine sign analysis and HVB formula validation."""
    result = {}
    astype = 'float64'

    machines = [
        ('ME', col_emot1_pow, col_tq1, col_spd1),
        ('HSG', col_emot2_pow, col_tq2, col_spd2),
    ]
    total_samples = len(df)

    for machine, col_pow, col_tq, col_spd in machines:
        if col_pow not in df.columns or col_tq not in df.columns or col_spd not in df.columns:
            continue

        p_elec = df[col_pow].astype(astype).values / 1000.0  # W to kW
        tq = df[col_tq].astype(astype).values
        spd = df[col_spd].astype(astype).values
        p_mec = tq * np.abs(spd) * _RPM_TO_RAD_S / 1000.0  # kW

        # Active mask (machine doing significant work)
        active = np.abs(p_mec) > _POWER_IDLE_KW
        if not active.any():
            continue

        motor = active & (p_mec > 0)
        gen = active & (p_mec < 0)

        # Quadrant counts
        motor_pos = int(np.sum(motor & (p_elec > 0)))   # P_mec>0 & P_elec_direct>0
        motor_neg = int(np.sum(motor & (p_elec < 0)))   # P_mec>0 & P_elec_direct<0
        gen_pos = int(np.sum(gen & (p_elec > 0)))       # P_mec<0 & P_elec_direct>0
        gen_neg = int(np.sum(gen & (p_elec < 0)))       # P_mec<0 & P_elec_direct<0

        total_motor = motor_pos + motor_neg
        total_gen = gen_pos + gen_neg

        # Coherent samples: P_mec and P_elec have same sign
        coherent = int(np.sum((motor & (p_elec > 0)) | (gen & (p_elec < 0))))
        total_active = total_motor + total_gen
        coherence_pct = coherent / total_active * 100 if total_active > 0 else 0.0

        # Detect convention
        # Same sign (motor+) / (gen-) → "consumidor positivo"
        # Opposite sign (motor-) / (gen+) → "generador positivo"
        if coherent > total_active / 2:
            detected_convention = 'consumidor+'
            confidence = coherent / total_active * 100
        else:
            detected_convention = 'generador+'
            confidence = (total_active - coherent) / total_active * 100

        mean_p_elec = float(np.mean(p_elec[active]))
        mean_p_mec = float(np.mean(p_mec[active]))

        result[machine] = {
            'total_samples': total_active,
            'motor_samples': total_motor,
            'motor_pos': motor_pos,
            'motor_neg': motor_neg,
            'gen_samples': total_gen,
            'gen_pos': gen_pos,
            'gen_neg': gen_neg,
            'coherence_pct': round(coherence_pct, 1),
            'detected_convention': detected_convention,
            'confidence_pct': round(confidence, 1),
            'mean_p_elec_direct_kw': round(mean_p_elec, 2),
            'mean_p_mec_kw': round(mean_p_mec, 2),
            'offset_kw': round(mean_p_elec - mean_p_mec, 2),
        }

    # HVB formula validation
    if col_hvb_pow in df.columns and col_cons in df.columns:
        hvb = df[col_hvb_pow].astype(astype).values / 1000.0
        cons = df[col_cons].astype(astype).values / 1000.0

        # Use detected conventions to estimate HVB
        estimated_hvb = np.full(len(df), np.nan)

        me_data = result.get('ME', {})
        hsg_data = result.get('HSG', {})

        me_pow = df[col_emot1_pow].astype(astype).values / 1000.0 if col_emot1_pow in df.columns else np.zeros(len(df))
        hsg_pow = df[col_emot2_pow].astype(astype).values / 1000.0 if col_emot2_pow in df.columns else np.zeros(len(df))

        me_sign = 1 if me_data.get('detected_convention') == 'consumidor+' else -1
        hsg_sign = 1 if hsg_data.get('detected_convention') == 'consumidor+' else -1

        # HVB formula: bus equation sum(P) = 0 => P_HVB = -(P_emot1 + P_emot2 + P_cons)
        # cons is always negative (power drawn from bus), so -cons = abs(cons)
        me_pow_signed = me_sign * me_pow
        hsg_pow_signed = hsg_sign * hsg_pow
        estimated_hvb = -(me_pow_signed + hsg_pow_signed + cons)

        valid = ~np.isnan(estimated_hvb) & (np.abs(hvb) > _POWER_IDLE_KW * 0.5)
        if valid.any():
            err = np.abs(estimated_hvb[valid] - hvb[valid])
            result['hvb_validation'] = {
                'hvb_pow_mean_kw': round(float(np.mean(hvb[valid])), 2),
                'estimated_hvb_mean_kw': round(float(np.mean(estimated_hvb[valid])), 2),
                'mean_error_kw': round(float(np.mean(err)), 2),
                'max_error_kw': round(float(np.max(err)), 2),
                'pct_within_2kw': round(float(np.sum(err < 2.0) / len(err) * 100), 1),
                'samples_valid': int(np.sum(valid)),
                'me_sign_coef': me_sign,
                'hsg_sign_coef': hsg_sign,
                'formula': f"HVB = -(({'+' if me_sign > 0 else ''}{me_sign}) * emot1_pow + ({'+' if hsg_sign > 0 else ''}{hsg_sign}) * emot2_pow + cons)",
            }

    return result


def print_power_sign_validation(result: PowerBalanceResult):
    sv = result.sign_validation
    if not sv:
        print("\n=== Validación de signos: NO DISPONIBLE ===")
        return
    print(f"\n=== Validación de signos de potencia ({result.total_samples} muestras) ===")
    for machine in ('ME', 'HSG'):
        d = sv.get(machine)
        if not d:
            continue
        print(f"\n  {machine}:")
        print(f"    Muestras activas: {d['total_samples']}")
        print(f"    Motor (P_mec>0): {d['motor_samples']}  → P_direct>0: {d['motor_pos']}  P_direct<0: {d['motor_neg']}")
        print(f"    Gen   (P_mec<0): {d['gen_samples']}  → P_direct>0: {d['gen_pos']}  P_direct<0: {d['gen_neg']}")
        print(f"    Coincidencia signos: {d['coherence_pct']}%")
        print(f"    Convención detectada: {d['detected_convention']} (confianza: {d['confidence_pct']}%)")
        print(f"    P_elec_direct media: {d['mean_p_elec_direct_kw']:.2f} kW")
        print(f"    P_mec media:          {d['mean_p_mec_kw']:.2f} kW")
        print(f"    Offset (elec - mec):  {d['offset_kw']:.2f} kW")

    hvb = sv.get('hvb_validation')
    if hvb:
        print(f"\n  Validación fórmula HVB:")
        print(f"    Fórmula: {hvb['formula']}")
        print(f"    HVB real media:       {hvb['hvb_pow_mean_kw']:.2f} kW")
        print(f"    HVB estimado media:   {hvb['estimated_hvb_mean_kw']:.2f} kW")
        print(f"    Error medio:          {hvb['mean_error_kw']:.2f} kW")
        print(f"    Error máximo:         {hvb['max_error_kw']:.2f} kW")
        print(f"    % dentro 2 kW:        {hvb['pct_within_2kw']:.1f}%")
        print(f"    Muestras válidas:     {hvb['samples_valid']}")


def power_sign_to_dataframe(result: PowerBalanceResult) -> pd.DataFrame:
    sv = result.sign_validation
    if not sv:
        return pd.DataFrame()
    rows = []
    for machine in ('ME', 'HSG'):
        d = sv.get(machine)
        if not d:
            continue
        rows.append({
            'Máquina': machine,
            'Muestras activas': d['total_samples'],
            'Motor (P_mec>0)': d['motor_samples'],
            'P>0 motor': d['motor_pos'],
            'P<0 motor': d['motor_neg'],
            'Gen (P_mec<0)': d['gen_samples'],
            'P>0 gen': d['gen_pos'],
            'P<0 gen': d['gen_neg'],
            'Coincidencia (%)': d['coherence_pct'],
            'Convención': d['detected_convention'],
            'Confianza (%)': d['confidence_pct'],
            'P_elec (kW)': d['mean_p_elec_direct_kw'],
            'P_mec (kW)': d['mean_p_mec_kw'],
            'Offset (kW)': d['offset_kw'],
        })
    hvb = sv.get('hvb_validation')
    if hvb:
        rows.append({
            'Máquina': 'HVB formula',
            'Muestras activas': hvb['samples_valid'],
            'Motor (P_mec>0)': '',
            'P>0 motor': '',
            'P<0 motor': '',
            'Gen (P_mec<0)': '',
            'P>0 gen': '',
            'P<0 gen': '',
            'Coincidencia (%)': '',
            'Convención': hvb['formula'],
            'Confianza (%)': '',
            'P_elec (kW)': hvb['hvb_pow_mean_kw'],
            'P_mec (kW)': hvb['estimated_hvb_mean_kw'],
            'Offset (kW)': hvb['mean_error_kw'],
        })
    return pd.DataFrame(rows)
