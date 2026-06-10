import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

from dico_reader import read_dico
from mf4_loader import load_mf4_channels
from current_analyzer import (
    analyze_machine,
    summarize_analysis,
    events_to_dataframe,
    stats_to_dataframe,
)


def plot_analysis(df: pd.DataFrame, analyses, output_dir: Path, machine: str):
    """Plot the sum and smooth series with event markers."""
    a = next((x for x in analyses if x.machine == machine), None)
    if a is None or a.sum_series is None:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    time = df.index.values
    ax.plot(time, a.sum_series.values, alpha=0.3, linewidth=0.5, label='Suma instantánea')
    ax.plot(time, a.smooth_series.values, linewidth=1.0, label=f'Media móvil ({a.window_ms}ms)')
    ax.axhline(a.threshold_a, color='r', linestyle='--', linewidth=1, label=f'+{a.threshold_a}A')
    ax.axhline(-a.threshold_a, color='r', linestyle='--', linewidth=1, label=f'-{a.threshold_a}A')

    # Mark events
    for e in a.events:
        ax.axvspan(e.start_time, e.end_time, alpha=0.15, color='red')

    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Suma de corrientes (A)')
    ax.set_title(f'Compensación de corrientes - {machine}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f'grafica_suma_{machine}.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Analizador de compensación de corrientes trifásicas ME/HSG')
    parser.add_argument('-i', '--input', required=True,
                        help='Ruta al fichero .mf4 de entrada')
    parser.add_argument('-d', '--dico', default=None,
                        help='Ruta al fichero DICO_VARIABLES_CSS.xlsx')
    parser.add_argument('--threshold-me', type=float, default=2.0,
                        help='Threshold de compensación para ME (A) [default: 2.0]')
    parser.add_argument('--threshold-hsg', type=float, default=0.5,
                        help='Threshold de compensación para HSG (A) [default: 0.5]')
    parser.add_argument('--window-ms', type=float, default=50.0,
                        help='Ventana de media móvil en ms [default: 50]')
    parser.add_argument('--min-event-ms', type=float, default=10.0,
                        help='Duración mínima de evento en ms [default: 10]')
    parser.add_argument('--output-dir', default=None,
                        help='Directorio de salida (default: ./output)')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Generar gráficas PNG')
    parser.add_argument('--raster', type=float, default=None,
                        help='Raster de remuestreo en segundos (default: el del DICO)')

    args = parser.parse_args()

    mf4_path = Path(args.input)
    if not mf4_path.exists():
        print(f"ERROR: No existe el fichero .mf4: {mf4_path}")
        sys.exit(1)

    # Locate DICO file
    dico_path = args.dico
    if dico_path is None:
        # Search relative to mf4 or project root
        candidates = [
            Path('../DICO_VARIABLES_CSS.xlsx'),
            Path('../../DICO_VARIABLES_CSS.xlsx'),
            Path(__file__).parent.parent / 'DICO_VARIABLES_CSS.xlsx',
            Path(__file__).parent / 'DICO_VARIABLES_CSS.xlsx',
        ]
        for c in candidates:
            if c.exists():
                dico_path = str(c)
                break
    if dico_path is None or not Path(dico_path).exists():
        print("ERROR: No se encontró DICO_VARIABLES_CSS.xlsx. Especifique con --dico")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path('output')
    output_dir = mf4_path.parent / output_dir if not output_dir.is_absolute() else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo DICO: {dico_path}")
    machine_groups = read_dico(str(dico_path))
    print(f"  Máquinas detectadas: {list(machine_groups.keys())}")
    for mg in machine_groups.values():
        print(f"    {mg.machine}: {mg.channels} (raster={mg.raster}s, res={mg.resolution})")

    print(f"\nCargando MF4: {mf4_path}")
    df = load_mf4_channels(str(mf4_path), machine_groups, raster=args.raster)
    print(f"  Shape: {df.shape}")
    print(f"  Rango tiempo: [{df.index.min():.4f}, {df.index.max():.4f}] s")

    print(f"\nAnalizando compensación...")
    thresholds = {
        'ME': args.threshold_me,
        'HSG': args.threshold_hsg,
    }
    analyses = []
    for machine, mg in machine_groups.items():
        th = thresholds.get(machine, 2.0)
        print(f"  {machine}: threshold=±{th}A, window={args.window_ms}ms")
        analysis = analyze_machine(
            df, mg,
            threshold_a=th,
            window_ms=args.window_ms,
            min_event_ms=args.min_event_ms,
        )
        analyses.append(analysis)

    print("\n" + "=" * 65)
    print(summarize_analysis(analyses))
    print("=" * 65)

    # Save events CSV
    for a in analyses:
        edf = events_to_dataframe(a.events)
        path = output_dir / f'informe_compensacion_{a.machine}.csv'
        edf.to_csv(path, sep=';', decimal=',', index=False)
        print(f"Eventos guardados: {path}  ({len(edf)} eventos)")

    # Save stats
    sdf = stats_to_dataframe(analyses)
    spath = output_dir / 'resumen_estadisticas.csv'
    sdf.to_csv(spath, sep=';', decimal=',', index=False)
    print(f"Estadísticas guardadas: {spath}")

    # Plot if requested
    if args.plot:
        print("\nGenerando gráficas...")
        for machine in machine_groups:
            plot_analysis(df, analyses, output_dir, machine)
            print(f"  Gráfica generada: {output_dir / f'grafica_suma_{machine}.png'}")

    print("\nAnálisis completado.")


if __name__ == '__main__':
    main()
