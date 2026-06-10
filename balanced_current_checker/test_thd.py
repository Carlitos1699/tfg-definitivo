import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from thd_analyzer import _compute_thd_for_window, detect_steady_state, analyze_thd, print_thd_result, ThdResult
from dico_reader import MachineGroup

# Test 1: FFT-based THD computation
fs = 10000.0
f_fund = 200.0
t = np.arange(0, 0.1, 1/fs)
i1, i3, i5 = 100.0, 5.0, 3.0
i7, i11, i13 = 2.0, 1.0, 0.5
sig = (i1*np.sin(2*np.pi*f_fund*t) + i3*np.sin(2*np.pi*3*f_fund*t) +
       i5*np.sin(2*np.pi*5*f_fund*t) +
       i7*np.sin(2*np.pi*7*f_fund*t) +
       i11*np.sin(2*np.pi*11*f_fund*t) +
       i13*np.sin(2*np.pi*13*f_fund*t))
thd, h = _compute_thd_for_window(sig, fs, f_fund)
exp = np.sqrt(5**2+3**2+2**2+1**2+0.5**2)/100*100
print(f'Test 1 - THD: {thd:.2f}% (exp {exp:.2f}%)')
assert abs(thd - exp) < 1, f'THD mismatch'
print('  PASS')

# Test 2: Steady-state detection
torque = np.concatenate([
    np.random.randn(30000)*0.1 + 50,
    np.random.randn(30000)*3 + 50,
    np.random.randn(30000)*0.1 + 60])
time = np.arange(len(torque)) * 0.0001
segs = detect_steady_state(torque, time, window_s=1.0, torque_tol=1.0)
print(f'Test 2 - Steady segments: {len(segs)}')
assert len(segs) >= 1, f'No segments found'
print('  PASS')

# Test 3: Full pipeline with synthetic 3-phase data
fs = 10000.0
tfull = np.arange(0, 65, 1/fs)
f_fund = 3000*8/120
df = pd.DataFrame(index=tfull)
for phase, off in [('ph_u', 0), ('ph_v', -2*np.pi/3), ('ph_w', 2*np.pi/3)]:
    df[f'Wxx_i_mot_{phase}'] = (100*np.sin(2*np.pi*f_fund*tfull+off) +
                                5*np.sin(2*np.pi*3*f_fund*tfull+off) +
                                3*np.sin(2*np.pi*5*f_fund*tfull+off))
df['Wxx_esti_emot_tq'] = 50.0 + np.random.randn(len(tfull))*0.2
df['Wxx_emot_n'] = 3000.0 + np.random.randn(len(tfull))*1.0

mg = MachineGroup(machine='ME',
    channels=['Wxx_i_mot_ph_u', 'Wxx_i_mot_ph_v', 'Wxx_i_mot_ph_w'],
    units='A', min_val=-500, max_val=500, raster=0.0001, resolution=0.015625)

class FakeTC:
    machine = 'ME'
    tq_est = 'Wxx_esti_emot_tq'
    speed = 'Wxx_emot_n'
    voltage = ''
    rotor_temp = ''

result = analyze_thd(df, mg, FakeTC(), 'ME', poles=8)
print_thd_result(ThdResult(me=result))
assert result and len(result.measurements) > 0, 'No THD measurements'
print('ALL PASS')
