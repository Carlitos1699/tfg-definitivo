from PyQt5.QtCore import QThread, pyqtSignal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dico_reader import read_dico, read_equivalences_from_dico, read_torque_channels
from mf4_loader import load_mf4_channels
from current_analyzer import analyze_machine, MachineAnalysis
from can_mux_reader import read_can_mux
from can_equivalence import EquivalenceDb
from torque_reader import TorquePerfDatabase
from torque_analyzer import analyze_torque_precision, print_torque_events
from power_analyzer import analyze_power_balance, print_power_balance, print_dls_report, print_balance_events
from calibrables_reader import read_calibrations
from thd_analyzer import analyze_thd, print_thd_result, ThdResult
import pandas as pd


class AnalysisWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, object, object, object, object, object)
    error_occurred = pyqtSignal(str)
    warning_occurred = pyqtSignal(str)

    def __init__(self, mf4_path, dico_path, thresholds, window_ms, min_event_ms, raster,
                 architecture=None, can_mux_paths=None, equivalence_path=None,
                 torque_perf_path=None, torque_threshold_nm=5.0, calibration_path=None,
                 analyses_flags=None):
        super().__init__()
        self.torque_threshold_nm = torque_threshold_nm
        self.mf4_path = mf4_path
        self.dico_path = dico_path
        self.thresholds = thresholds
        self.window_ms = window_ms
        self.min_event_ms = min_event_ms
        self.raster = raster
        self.architecture = architecture
        self.can_mux_paths = can_mux_paths or {}
        self.equivalence_path = equivalence_path
        self.torque_perf_path = torque_perf_path
        self.calibration_path = calibration_path
        self.analyses_flags = analyses_flags or {
            'current': True, 'torque': True, 'power': True, 'thd': True,
        }

    def run(self):
        try:
            self.progress.emit(5, "Leyendo DICO...")
            machine_groups = read_dico(self.dico_path)

            equivalences = None
            if self.equivalence_path:
                self.progress.emit(7, "Leyendo equivalencias CAN...")
                try:
                    equivalences = EquivalenceDb.from_excel(self.equivalence_path)
                except Exception as e:
                    self.progress.emit(7, f"Advertencia: no se pudo leer equivalencias ({e})")
            else:
                try:
                    eq = read_equivalences_from_dico(self.dico_path)
                    if eq is not None:
                        equivalences = eq
                        self.progress.emit(7, f"Equivalencias cargadas desde DICO ({len(eq)} entradas)")
                except Exception:
                    pass

            can_signals = None
            if self.architecture and self.can_mux_paths:
                self.progress.emit(10, f"Leyendo CAN MUX ({self.architecture})...")
                can_signals = read_can_mux(self.architecture, self.can_mux_paths)

            # Read torque channels from DICO
            torque_channels = read_torque_channels(self.dico_path)
            extra_columns = []
            _DERATING_ATTRS = (
                'tq_cmd', 'tq_est', 'speed', 'voltage', 'rotor_temp',
                'sync_mode', 'tq_cons', 'norm_tq_der',
                'norm_tq_sp', 'norm_tq_min', 'norm_tq_max',
                'norm_tq_max_temp', 'norm_tq_max_udc', 'norm_tq_max_stop',
                'norm_tq_cur_sat_max', 'norm_tq_max_asc_oc', 'norm_tq_max_cell_ov_fac',
                'norm_tq_min_udc', 'norm_tq_min_back', 'norm_tq_min_cell_ov_fac',
                # Vbx_* derating activation flags
                'spt_der_act', 'spt_der_act_tqc',
                'invt_stop', 'spt_der_v_oor', 'spt_der_ovh',
                'max_cur_sat_act', 'asc_oc_der_act',
                'invt_ztq_reg_act', 'dm_ov_sfty', 'back_der_act',
                # EAJS (Curative Antijerk)
                'eajs_cor_act',
                # Saturation voltage channels
                'v_d', 'v_q', 'v_d_ang', 'v_d_harm', 'v_q_ang', 'v_q_harm',
                'rgl_sat',
                # Current channels (Id, Iq)
                'i_d', 'i_q', 'i_d_req', 'i_q_req',
            )
            for tc in torque_channels.values():
                for attr in _DERATING_ATTRS:
                    ch = getattr(tc, attr)
                    if ch:
                        extra_columns.append(ch)
            extra_columns = list(set(extra_columns))

            # Power balance channels
            from dico_reader import (
                _INV1_I_ME, _HVBUS_V_INV1_ME,
                _INV2_I_HSG, _HVBUS_V_INV2_HSG,
                _HVBUS_AFTR_RLY_V,
                _EST_TQ_EMOT1_ME, _EST_TQ_EMOT2_HSG,
                _SPD_EMOT1_ME, _SPD_EMOT2_HSG,
                _HVB_POW, _HVBUS_POW_CONS_EST,
                _V_D_ME, _V_Q_ME, _I_D_ME, _I_Q_ME,
                _V_D_HSG, _V_Q_HSG, _I_D_HSG, _I_Q_HSG,
                _ICE_TQ, _ICE_SPD,
                _VNX_CRT_VH_DL,
                _WHL_SPD, _WHEEL_RADIUS_M,
            )
            _POWER_CHANNELS = [
                _INV1_I_ME, _HVBUS_V_INV1_ME,
                _INV2_I_HSG, _HVBUS_V_INV2_HSG,
                _HVBUS_AFTR_RLY_V,
                _EST_TQ_EMOT1_ME, _EST_TQ_EMOT2_HSG,
                _SPD_EMOT1_ME, _SPD_EMOT2_HSG,
                _HVB_POW, _HVBUS_POW_CONS_EST,
                _V_D_ME, _V_Q_ME, _I_D_ME, _I_Q_ME,
                _V_D_HSG, _V_Q_HSG, _I_D_HSG, _I_Q_HSG,
                _ICE_TQ, _ICE_SPD,
                _VNX_CRT_VH_DL,
                _WHL_SPD,
            ]
            for ch in _POWER_CHANNELS:
                if ch not in extra_columns:
                    extra_columns.append(ch)

            # Load torque perfos if available
            perf_db = None
            if self.torque_perf_path:
                self.progress.emit(12, "Leyendo tablas de prestaciones de par...")
                try:
                    perf_db = TorquePerfDatabase.from_excel(self.torque_perf_path)
                except Exception as e:
                    self.progress.emit(12, f"Advertencia: no se pudo leer perfos ({e})")

            self.progress.emit(15, f"Cargando MF4 ({Path(self.mf4_path).name})...")
            df = load_mf4_channels(self.mf4_path, machine_groups, raster=self.raster,
                                   can_signals=can_signals,
                                   equivalences=equivalences,
                                   can_mux_signals=can_signals,
                                   extra_channels=extra_columns)

            can_columns_map = {}
            if can_signals:
                for machine, sigs in can_signals.items():
                    cols = []
                    for s in sigs:
                        col_name = None
                        if s.basic_name in df.columns:
                            col_name = s.basic_name
                        elif equivalences and s.basic_name in equivalences:
                            alt = equivalences.get(s.basic_name)
                            if alt and alt in df.columns:
                                col_name = alt
                        if col_name:
                            cols.append(col_name)
                    can_columns_map[machine] = cols
                total_found = sum(len(v) for v in can_columns_map.values())
                self.progress.emit(15, f"Señales CAN cargadas: {total_found}/{sum(len(sigs) for sigs in can_signals.values())}")

            # Check for missing torque-related channels
            if torque_channels:
                for machine, tc in torque_channels.items():
                    if tc.tq_cons and tc.tq_cons not in df.columns:
                        self.warning_occurred.emit(
                            f"{machine}: variable '{tc.tq_cons}' (par consolidado) "
                            f"no encontrada en el archivo MF4"
                        )
                    if tc.norm_tq_sp and tc.norm_tq_sp not in df.columns:
                        self.warning_occurred.emit(
                            f"{machine}: variable '{tc.norm_tq_sp}' (par normalizado sp) "
                            f"no encontrada en el archivo MF4 — se omitirá análisis de derating"
                        )
                    if tc.norm_tq_min and tc.norm_tq_min not in df.columns:
                        self.warning_occurred.emit(
                            f"{machine}: variable '{tc.norm_tq_min}' (par normalizado min) "
                            f"no encontrada en el archivo MF4 — se omitirá análisis de derating"
                        )
                    if tc.norm_tq_max and tc.norm_tq_max not in df.columns:
                        self.warning_occurred.emit(
                            f"{machine}: variable '{tc.norm_tq_max}' (par normalizado max) "
                            f"no encontrada en el archivo MF4 — se omitirá análisis de derating"
                        )

            # Warning agrupado: canales de saturación (Vd, Vq)
            for machine, tc in (torque_channels or {}).items():
                sat_missing = []
                for attr in ('v_d', 'v_q', 'v_d_ang', 'v_d_harm', 'v_q_ang', 'v_q_harm', 'rgl_sat'):
                    ch = getattr(tc, attr, '')
                    if ch and ch not in df.columns:
                        sat_missing.append(ch)
                if sat_missing:
                    self.warning_occurred.emit(
                        f"{machine}: canales de saturación no encontrados: {', '.join(sat_missing)}"
                    )

            # Warning agrupado: canales de corriente (Id, Iq)
            for machine, tc in (torque_channels or {}).items():
                cur_missing = []
                for attr in ('i_d', 'i_q', 'i_d_req', 'i_q_req'):
                    ch = getattr(tc, attr, '')
                    if ch and ch not in df.columns:
                        cur_missing.append(ch)
                if cur_missing:
                    self.warning_occurred.emit(
                        f"{machine}: canales de corriente no encontrados: {', '.join(cur_missing)}"
                    )

            analyses = []
            if self.analyses_flags.get('current'):
                total = len(machine_groups)
                for i, (machine, mg) in enumerate(machine_groups.items()):
                    th = self.thresholds.get(machine, 2.0)
                    pct = 20 + int(70 * (i / total))
                    self.progress.emit(pct, f"Analizando {machine} (threshold=±{th}A)...")
                    analysis = analyze_machine(
                        df, mg,
                        threshold_a=th,
                        window_ms=self.window_ms,
                        min_event_ms=self.min_event_ms,
                        can_columns=can_columns_map.get(machine, []),
                    )
                    analyses.append(analysis)

            # Torque precision analysis
            torque_analyses = []
            if self.analyses_flags.get('torque'):
                self.progress.emit(85, "Leyendo calibración del inversor...")
                calibrations = read_calibrations(self.calibration_path)
                self.progress.emit(86, f"Calibración cargada ({len(calibrations.raw)} valores)")

                # Resolve EAJS column for ME (fallback: CAN signal)
                eajs_column = None
                for machine, tc in (torque_channels or {}).items():
                    if machine == 'ME':
                        eajs_dico = getattr(tc, 'eajs_cor_act', '')
                        if eajs_dico and eajs_dico in df.columns:
                            eajs_column = eajs_dico
                        else:
                            for col in can_columns_map.get('ME', []):
                                if 'CurativeAntiJerk' in col:
                                    eajs_column = col
                                    break
                        break

                if torque_channels and perf_db:
                    total_tq = len(torque_channels)
                    for i, (machine, tc) in enumerate(torque_channels.items()):
                        if tc.tq_cmd in df.columns and tc.tq_est in df.columns:
                            pct = 90 + int(10 * (i / total_tq))
                            self.progress.emit(pct, f"Analizando precisión de par {machine}...")
                            ta = analyze_torque_precision(
                                df, tc, perf_db=perf_db,
                                threshold_nm=self.torque_threshold_nm,
                                min_event_ms=self.min_event_ms,
                                calibration=calibrations,
                                eajs_column=eajs_column if machine == 'ME' else None,
                                phase_channels=tuple(mg.channels) if machine in machine_groups and len(machine_groups[machine].channels) == 3 else None,
                            )
                            torque_analyses.append(ta)

            # Power balance analysis
            power_result = None
            if self.analyses_flags.get('power'):
                self.progress.emit(96, "Analizando balance de potencia...")
                tc_me = torque_channels.get('ME') if torque_channels else None
                tc_hsg = torque_channels.get('HSG') if torque_channels else None

                from dico_reader import read_dls_table
                dls_table = read_dls_table(self.dico_path)
                if dls_table:
                    self.progress.emit(96, f"DLS cargadas ({len(dls_table)} combinaciones)")

                power_result = analyze_power_balance(
                    df, tc_me=tc_me, tc_hsg=tc_hsg,
                    col_hvbus_v_inv1=_HVBUS_V_INV1_ME,
                    col_inv1_i=_INV1_I_ME,
                    col_hvbus_v_inv2=_HVBUS_V_INV2_HSG,
                    col_inv2_i=_INV2_I_HSG,
                    col_hvbus_aftr_rly=_HVBUS_AFTR_RLY_V,
                    col_i_d_me=_I_D_ME, col_i_q_me=_I_Q_ME,
                    col_v_d_me=_V_D_ME, col_v_q_me=_V_Q_ME,
                    col_i_d_hsg=_I_D_HSG, col_i_q_hsg=_I_Q_HSG,
                    col_v_d_hsg=_V_D_HSG, col_v_q_hsg=_V_Q_HSG,
                    col_est_tq_me_hevc=_EST_TQ_EMOT1_ME,
                    col_est_tq_hsg_hevc=_EST_TQ_EMOT2_HSG,
                    col_spd_me_hevc=_SPD_EMOT1_ME,
                    col_spd_hsg_hevc=_SPD_EMOT2_HSG,
                    col_ice_tq=_ICE_TQ, col_ice_spd=_ICE_SPD,
                    col_hvb_pow=_HVB_POW, col_p_cons=_HVBUS_POW_CONS_EST,
                    col_dls=_VNX_CRT_VH_DL,
                    dls_table=dls_table,
                    col_whl_spd=_WHL_SPD,
                    wheel_radius_m=_WHEEL_RADIUS_M,
                )
                print_power_balance(power_result)
                print_dls_report(power_result)
                print_balance_events(power_result)

            # THD analysis
            thd_result = ThdResult(me=None, hsg=None)
            if self.analyses_flags.get('thd'):
                self.progress.emit(97, "Analizando THD de corrientes...")
                for machine, mg in (machine_groups or {}).items():
                    if mg and len(mg.channels) == 3:
                        tc = (torque_channels or {}).get(machine)
                        poles = 8 if machine == 'ME' else 4
                        mthd = analyze_thd(df, mg, tc, machine, poles)
                        if machine == 'ME':
                            thd_result.me = mthd
                        else:
                            thd_result.hsg = mthd
                print_thd_result(thd_result)

            self.progress.emit(100, "Análisis completado.")
            print_torque_events(torque_analyses)
            self.finished.emit(analyses, df, can_signals, torque_analyses, power_result, thd_result)

        except Exception as e:
            self.error_occurred.emit(str(e))
