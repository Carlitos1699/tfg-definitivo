import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget


class GraphWindow(QMainWindow):
    def __init__(self, df, analysis, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Gráfica - {analysis.machine}")
        self.resize(1200, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.canvas = FigureCanvasQTAgg(Figure(figsize=(12, 5)))
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas)

        self._plot(df, analysis)

    def _plot(self, df, a):
        fig = self.canvas.figure
        fig.clf()
        ax = fig.add_subplot(111)

        time = df.index.values
        ax.plot(time, a.sum_series.values, alpha=0.3, linewidth=0.5, label='Suma instantánea')
        ax.plot(time, a.smooth_series.values, linewidth=1.0, label=f'Media móvil ({a.window_ms}ms)')
        ax.axhline(a.threshold_a, color='r', linestyle='--', linewidth=1, label=f'+{a.threshold_a}A')
        ax.axhline(-a.threshold_a, color='r', linestyle='--', linewidth=1, label=f'-{a.threshold_a}A')

        for e in a.events:
            ax.axvspan(e.start_time, e.end_time, alpha=0.15, color='red')

        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Suma de corrientes (A)')
        ax.set_title(f'Compensación de corrientes - {a.machine}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.canvas.draw()
