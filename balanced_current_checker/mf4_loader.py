from typing import Dict, List, Optional, Set
import pandas as pd
from asammdf import MDF

from dico_reader import MachineGroup

try:
    from can_mux_reader import CanSignal
except ImportError:
    CanSignal = None

try:
    from can_equivalence import EquivalenceDb
except ImportError:
    EquivalenceDb = None


def load_mf4_channels(
    mf4_path: str,
    machine_groups: Dict[str, MachineGroup],
    raster: Optional[float] = None,
    can_signals: Optional[Dict[str, List]] = None,
    equivalences: Optional['EquivalenceDb'] = None,
    can_mux_signals: Optional[Dict[str, List]] = None,
    extra_channels: Optional[List[str]] = None,
) -> pd.DataFrame:
    mdf = MDF(mf4_path)

    available: Set[str] = set()
    for gp in mdf.groups:
        for ch in gp.channels:
            try:
                available.add(ch.name)
            except Exception:
                pass

    all_channels = []
    equivalence_resolved: Dict[str, CanSignal] = {}

    # Build lookup: basic_name -> CanSignal across all machines
    basic_to_can: Dict[str, CanSignal] = {}
    if can_mux_signals:
        for sigs in can_mux_signals.values():
            for s in sigs:
                basic_to_can[s.basic_name] = s

    for mg in machine_groups.values():
        for ch in mg.channels:
            if ch in available:
                if ch not in all_channels:
                    all_channels.append(ch)
            elif equivalences:
                nombre_can = equivalences.get_can_name(ch)
                if nombre_can:
                    cs = basic_to_can.get(nombre_can)
                    if cs:
                        signal_name = cs.signal_name
                        if signal_name in available and signal_name not in all_channels:
                            all_channels.append(signal_name)
                            equivalence_resolved[ch] = cs
                    else:
                        if nombre_can in available and nombre_can not in all_channels:
                            all_channels.append(nombre_can)
                            cs_direct = CanSignal(
                                signal_name=nombre_can, basic_name=nombre_can,
                                unit='', resolution=1.0, offset=0.0,
                                min_val=0.0, max_val=0.0, period_ms=0, machine='', direction='')
                            equivalence_resolved[ch] = cs_direct

    if can_signals:
        for machine, sigs in can_signals.items():
            for s in sigs:
                name = s.basic_name
                if name and name not in all_channels:
                    if name in available:
                        all_channels.append(name)

    if extra_channels:
        for ch in extra_channels:
            if ch in available and ch not in all_channels:
                all_channels.append(ch)

    if raster is None:
        raster = min(mg.raster for mg in machine_groups.values())

    df = mdf.to_dataframe(channels=all_channels, raster=raster, time_from_zero=True)
    df.index.name = 'Time_s'

    # Rename CAN columns back to original DICO names
    for ch_dico in equivalence_resolved:
        cs = equivalence_resolved[ch_dico]
        if cs.signal_name in df.columns:
            df.rename(columns={cs.signal_name: ch_dico}, inplace=True)

    # Apply transformations
    for machine, mg in machine_groups.items():
        for ch in mg.channels:
            if ch not in df.columns:
                continue
            if ch in equivalence_resolved:
                cs = equivalence_resolved[ch]
                col = df[ch]
                max_raw = abs(cs.max_val - cs.offset) * 2 if cs.max_val != cs.offset else 0
                if cs.max_val != cs.offset and col.abs().max() > max_raw:
                    df[ch] = col * cs.resolution + cs.offset
                if cs.min_val != cs.max_val:
                    df[ch] = df[ch].clip(lower=cs.min_val, upper=cs.max_val)
            else:
                col = df[ch]
                if col.abs().max() > mg.max_val * 2:
                    df[ch] = col * mg.resolution
                df[ch] = df[ch].clip(lower=mg.min_val, upper=mg.max_val)

    # Apply resolution/offset to remaining CAN signals (non-equivalent)
    if can_signals:
        for machine, sigs in can_signals.items():
            for s in sigs:
                col_name = s.basic_name
                if col_name not in df.columns:
                    continue
                col = df[col_name]
                if col.abs().max() > abs(s.max_val - s.offset) * 2 if s.max_val != s.offset else False:
                    df[col_name] = col * s.resolution + s.offset
                if s.min_val != s.max_val:
                    df[col_name] = df[col_name].clip(lower=s.min_val, upper=s.max_val)


    return df

