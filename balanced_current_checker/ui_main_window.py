import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QLineEdit, QPushButton, QDoubleSpinBox, QSpinBox, QCheckBox,
    QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QFileDialog, QSplitter, QFrame, QStatusBar, QApplication, QTabWidget,
    QAbstractItemView, QComboBox, QStackedWidget,
)
from PyQt5.QtGui import QFont, QIcon, QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ui_worker import AnalysisWorker
from ui_graph_window import GraphWindow
from current_analyzer import (
    events_to_dataframe, stats_to_dataframe,
)
from torque_analyzer import (
    torque_events_to_dataframe, torque_stats_to_dataframe,
)
from can_mux_reader import ARCHITECTURES


STYLES = """
QMainWindow { background: #f5f5f5; }
QGroupBox {
    font-weight: bold; border: 1px solid #ccc; border-radius: 6px;
    margin-top: 10px; padding-top: 16px;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
QPushButton#btnRun {
    background: #1976D2; color: white; font-size: 14px; font-weight: bold;
    padding: 8px 24px; border-radius: 4px; min-width: 180px;
}
QPushButton#btnRun:hover { background: #1565C0; }
QPushButton#btnRun:disabled { background: #90CAF9; }
QPushButton { padding: 4px 12px; border-radius: 3px; }
QTableWidget { gridline-color: #e0e0e0; }
QHeaderView::section {
    background: #e8e8e8; padding: 4px; border: 1px solid #ddd; font-weight: bold;
}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analizador de Compensación de Corrientes")
        self.resize(1100, 780)
        self.setStyleSheet(STYLES)

        self._analyses = None
        self._df = None
        self._worker = None
        self._current_machine_idx = 0
        self._can_signals = None
        self._torque_analyses = []
        self._power_result = None
        self._sync_result = None
        self._eq_path_used = None

        self._build_ui()
        self._update_mux_visibility()
        self.statusBar().showMessage("Listo.")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)
        vbox.setSpacing(8)

        # --- Config panel ---
        cfg = QGroupBox("Configuración")
        cfg_layout = QVBoxLayout()
        fbox = QFormLayout()

        # Architecture selector
        arch_row = QHBoxLayout()
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(ARCHITECTURES)
        self.arch_combo.currentTextChanged.connect(self._on_arch_changed)
        arch_row.addWidget(self.arch_combo)
        arch_row.addStretch()
        fbox.addRow("Arquitectura:", arch_row)

        row1 = QHBoxLayout()
        self.mf4_path = QLineEdit()
        self.mf4_path.setPlaceholderText("Seleccione fichero .mf4...")
        row1.addWidget(self.mf4_path, 1)
        btn_mf4 = QPushButton("📂")
        btn_mf4.setFixedWidth(36)
        btn_mf4.clicked.connect(lambda: self._browse_file(self.mf4_path, "Ficheros MF4 (*.mf4)"))
        row1.addWidget(btn_mf4)
        fbox.addRow("Fichero .mf4:", row1)

        row2 = QHBoxLayout()
        self.dico_path = QLineEdit()
        self.dico_path.setPlaceholderText("Seleccione DICO_VARIABLES_CSS.xlsx...")
        row2.addWidget(self.dico_path, 1)
        btn_dico = QPushButton("📂")
        btn_dico.setFixedWidth(36)
        btn_dico.clicked.connect(lambda: self._browse_file(self.dico_path, "Excel (*.xlsx)"))
        row2.addWidget(btn_dico)
        fbox.addRow("Fichero DICO:", row2)

        # CAN MUX fields - stacked for different architectures
        self.mux_stack = QStackedWidget()

        # Page 0: Sweet200 (ME + HSG)
        mux_page_200 = QWidget()
        mux_200_layout = QFormLayout(mux_page_200)
        me_row = QHBoxLayout()
        self.mux_me_path = QLineEdit()
        self.mux_me_path.setPlaceholderText("CAN MUX ME Sweet200...")
        me_row.addWidget(self.mux_me_path, 1)
        btn_mux_me = QPushButton("📂")
        btn_mux_me.setFixedWidth(36)
        btn_mux_me.clicked.connect(lambda: self._browse_file(self.mux_me_path, "Excel (*.xlsx)"))
        me_row.addWidget(btn_mux_me)
        mux_200_layout.addRow("CAN MUX ME:", me_row)
        hsg_row = QHBoxLayout()
        self.mux_hsg_path = QLineEdit()
        self.mux_hsg_path.setPlaceholderText("CAN MUX HSG Sweet200...")
        hsg_row.addWidget(self.mux_hsg_path, 1)
        btn_mux_hsg = QPushButton("📂")
        btn_mux_hsg.setFixedWidth(36)
        btn_mux_hsg.clicked.connect(lambda: self._browse_file(self.mux_hsg_path, "Excel (*.xlsx)"))
        hsg_row.addWidget(btn_mux_hsg)
        mux_200_layout.addRow("CAN MUX HSG:", hsg_row)
        self.mux_stack.addWidget(mux_page_200)

        # Page 1: Sweet400 (single file)
        mux_page_400 = QWidget()
        mux_400_layout = QFormLayout(mux_page_400)
        mux_row = QHBoxLayout()
        self.mux_path = QLineEdit()
        self.mux_path.setPlaceholderText("CAN MUX Sweet400...")
        mux_row.addWidget(self.mux_path, 1)
        btn_mux = QPushButton("📂")
        btn_mux.setFixedWidth(36)
        btn_mux.clicked.connect(lambda: self._browse_file(self.mux_path, "Excel (*.xlsx)"))
        mux_row.addWidget(btn_mux)
        mux_400_layout.addRow("CAN MUX:", mux_row)
        self.mux_stack.addWidget(mux_page_400)

        fbox.addRow("Comunicación CAN:", self.mux_stack)

        # Equivalence file
        eq_row = QHBoxLayout()
        self.eq_path = QLineEdit()
        self.eq_path.setPlaceholderText("(Opcional) Excel de equivalencias CAN...")
        eq_row.addWidget(self.eq_path, 1)
        btn_eq = QPushButton("📂")
        btn_eq.setFixedWidth(36)
        btn_eq.clicked.connect(lambda: self._browse_file(self.eq_path, "Excel (*.xlsx)"))
        eq_row.addWidget(btn_eq)
        self.btn_eq_template = QPushButton("Crear plantilla")
        self.btn_eq_template.clicked.connect(self._create_eq_template)
        eq_row.addWidget(self.btn_eq_template)
        fbox.addRow("Equivalencias:", eq_row)

        # Torque perfos file
        perf_row = QHBoxLayout()
        self.perf_path = QLineEdit()
        self.perf_path.setPlaceholderText("(Opcional) Tablas de prestaciones de par...")
        perf_row.addWidget(self.perf_path, 1)
        btn_perf = QPushButton("📂")
        btn_perf.setFixedWidth(36)
        btn_perf.clicked.connect(lambda: self._browse_file(self.perf_path, "Excel (*.xlsx)"))
        perf_row.addWidget(btn_perf)
        fbox.addRow("Perfos Par:", perf_row)

        # Calibration file
        cal_row = QHBoxLayout()
        self.cal_path = QLineEdit()
        self.cal_path.setPlaceholderText("Seleccione Excel de calibración del inversor (E011_5840_6081.xlsx)...")
        cal_row.addWidget(self.cal_path, 1)
        btn_cal = QPushButton("📂")
        btn_cal.setFixedWidth(36)
        btn_cal.clicked.connect(lambda: self._browse_file(self.cal_path, "Excel (*.xlsx)"))
        cal_row.addWidget(btn_cal)
        fbox.addRow("Calibración:", cal_row)

        cfg_layout.addLayout(fbox)

        params_row = QHBoxLayout()
        params_row.setSpacing(16)
        p1 = QHBoxLayout()
        p1.addWidget(QLabel("Threshold ME:"))
        self.spin_th_me = QDoubleSpinBox()
        self.spin_th_me.setRange(0.1, 100.0)
        self.spin_th_me.setSingleStep(0.5)
        self.spin_th_me.setValue(2.0)
        self.spin_th_me.setSuffix(" A")
        p1.addWidget(self.spin_th_me)
        params_row.addLayout(p1)

        p2 = QHBoxLayout()
        p2.addWidget(QLabel("Threshold HSG:"))
        self.spin_th_hsg = QDoubleSpinBox()
        self.spin_th_hsg.setRange(0.1, 100.0)
        self.spin_th_hsg.setSingleStep(0.1)
        self.spin_th_hsg.setValue(0.5)
        self.spin_th_hsg.setSuffix(" A")
        p2.addWidget(self.spin_th_hsg)
        params_row.addLayout(p2)

        p3 = QHBoxLayout()
        p3.addWidget(QLabel("Ventana media:"))
        self.spin_window = QSpinBox()
        self.spin_window.setRange(1, 1000)
        self.spin_window.setValue(50)
        self.spin_window.setSuffix(" ms")
        p3.addWidget(self.spin_window)
        params_row.addLayout(p3)

        p4 = QHBoxLayout()
        p4.addWidget(QLabel("Evento mín:"))
        self.spin_min_event = QSpinBox()
        self.spin_min_event.setRange(1, 500)
        self.spin_min_event.setValue(10)
        self.spin_min_event.setSuffix(" ms")
        p4.addWidget(self.spin_min_event)
        params_row.addLayout(p4)

        p5 = QHBoxLayout()
        p5.addWidget(QLabel("Umbral par:"))
        self.spin_torque_th = QDoubleSpinBox()
        self.spin_torque_th.setRange(0.1, 200.0)
        self.spin_torque_th.setSingleStep(1.0)
        self.spin_torque_th.setValue(5.0)
        self.spin_torque_th.setSuffix(" Nm")
        p5.addWidget(self.spin_torque_th)
        params_row.addLayout(p5)

        cfg_layout.addLayout(params_row)

        # Analysis selection checkboxes
        analysis_row = QHBoxLayout()
        analysis_row.setSpacing(12)
        analysis_row.addWidget(QLabel("Análisis:"))
        self.chk_current = QCheckBox("Corrientes")
        self.chk_current.setChecked(True)
        analysis_row.addWidget(self.chk_current)
        self.chk_torque = QCheckBox("Precisión par")
        self.chk_torque.setChecked(True)
        analysis_row.addWidget(self.chk_torque)
        self.chk_power = QCheckBox("Balance potencia")
        self.chk_power.setChecked(True)
        analysis_row.addWidget(self.chk_power)
        self.chk_thd = QCheckBox("THD")
        self.chk_thd.setChecked(True)
        analysis_row.addWidget(self.chk_thd)
        self.chk_sync = QCheckBox("Sincronía PWM")
        self.chk_sync.setChecked(True)
        analysis_row.addWidget(self.chk_sync)
        analysis_row.addStretch()
        cfg_layout.addLayout(analysis_row)

        opts_row = QHBoxLayout()
        self.chk_plot = QCheckBox("Generar gráfica PNG")
        self.chk_plot.setChecked(True)
        opts_row.addWidget(self.chk_plot)
        opts_row.addStretch()
        cfg_layout.addLayout(opts_row)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_run = QPushButton("▶ Ejecutar Análisis")
        self.btn_run.setObjectName("btnRun")
        self.btn_run.clicked.connect(self._run_analysis)
        btn_row.addWidget(self.btn_run)
        cfg_layout.addLayout(btn_row)

        cfg.setLayout(cfg_layout)
        vbox.addWidget(cfg)

        # --- Progress ---
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.lbl_status = QLabel("")
        self.lbl_status.setVisible(False)
        vbox.addWidget(self.progress)
        vbox.addWidget(self.lbl_status)

        # --- Results tabs ---
        self.tabs = QTabWidget()
        vbox.addWidget(self.tabs, 1)

        # Tab: Summary
        self.tab_summary = QWidget()
        self._build_summary_tab()
        self.tabs.addTab(self.tab_summary, "📊 Resumen")

        # Tab: Events table
        self.tab_events = QWidget()
        self._build_events_tab()
        self.tabs.addTab(self.tab_events, "📋 Eventos")

        # Tab: Graph thumbnail
        self.tab_graph = QWidget()
        self._build_graph_tab()
        self.tabs.addTab(self.tab_graph, "📈 Gráfica")

        # Tab: CAN Bus
        self.tab_can = QWidget()
        self._build_can_tab()
        self.tabs.addTab(self.tab_can, "📡 CAN Bus")

        # Tab: Torque Precision
        self.tab_torque = QWidget()
        self._build_torque_tab()
        self.tabs.addTab(self.tab_torque, "🔧 Precisión Par")

        # Tab: Power Balance
        self.tab_power = QWidget()
        self._build_power_tab()
        self.tabs.addTab(self.tab_power, "⚡ Balance Potencia")

        # Tab: THD
        self.tab_thd = QWidget()
        self._build_thd_tab()
        self.tabs.addTab(self.tab_thd, "📊 THD Corrientes")

        # Tab: Sync Risk
        self.tab_sync = QWidget()
        self._build_sync_tab()
        self.tabs.addTab(self.tab_sync, "🔀 Sincronía PWM")

        # Export row
        export_row = QHBoxLayout()
        export_row.addStretch()
        self.btn_export = QPushButton("💾 Exportar CSV")
        self.btn_export.clicked.connect(self._export_csv)
        self.btn_export.setEnabled(False)
        export_row.addWidget(self.btn_export)
        vbox.addLayout(export_row)

        self._set_tabs_enabled(False)

    def _build_summary_tab(self):
        layout = QVBoxLayout(self.tab_summary)
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(6)
        self.summary_table.setHorizontalHeaderLabels(
            ["Máquina", "Umbral", "Compensado", "Descompensado", "Eventos", "Tiempo afectado"])
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.summary_table.setAlternatingRowColors(True)
        layout.addWidget(self.summary_table)

    def _build_events_tab(self):
        layout = QVBoxLayout(self.tab_events)
        self.events_table = QTableWidget()
        self.events_table.setColumnCount(6)
        self.events_table.setHorizontalHeaderLabels(
            ["Máquina", "Inicio (s)", "Fin (s)", "Duración (s)", "Max |Σ| (A)", "Media |Σ| (A)"])
        self.events_table.horizontalHeader().setStretchLastSection(True)
        self.events_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.events_table.setAlternatingRowColors(True)
        self.events_table.setSortingEnabled(True)
        layout.addWidget(self.events_table)

    def _build_can_tab(self):
        layout = QVBoxLayout(self.tab_can)

        lbl = QLabel("Señales CAN recibidas por INV, capturadas durante cada evento de descompensación.")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        self.can_table = QTableWidget()
        self.can_table.setColumnCount(8)
        self.can_table.setHorizontalHeaderLabels(
            ["Máquina", "Evento #", "Tiempo (s)", "Señal CAN", "Valor", "Unidad", "Res", "Offset"])
        self.can_table.horizontalHeader().setStretchLastSection(True)
        self.can_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.can_table.setAlternatingRowColors(True)
        self.can_table.setSortingEnabled(True)
        layout.addWidget(self.can_table, 1)

        snap_row = QHBoxLayout()
        self.lbl_signal_count = QLabel("Sin datos CAN")
        snap_row.addWidget(self.lbl_signal_count)
        snap_row.addStretch()
        layout.addLayout(snap_row)

    def _build_torque_tab(self):
        layout = QVBoxLayout(self.tab_torque)
        self.torque_summary_table = QTableWidget()
        self.torque_summary_table.setColumnCount(7)
        self.torque_summary_table.setHorizontalHeaderLabels(
            ["Máquina", "Error medio (Nm)", "Error máx (Nm)", "% error",
             "Violaciones modelo", "% violación", "Eventos"])
        self.torque_summary_table.horizontalHeader().setStretchLastSection(True)
        self.torque_summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.torque_summary_table.setAlternatingRowColors(True)
        layout.addWidget(self.torque_summary_table)

        lbl = QLabel("Eventos donde |par_pedido - par_realizado| > 5 Nm ")
        layout.addWidget(lbl)

        self.torque_events_table = QTableWidget()
        self.torque_events_table.setColumnCount(36)
        self.torque_events_table.setHorizontalHeaderLabels(
            ["Máquina", "Inicio (s)", "Fin (s)", "Duración (s)",
             "Max error (Nm)", "Media error (Nm)", "Muestras", "Violaciones",
             "Cmin (Nm)", "Cmax (Nm)", "Modo",
             "Par medio (Nm)", "Límite (Nm)", "Dentro límite?",
             "Udc media (V)", "N media (rpm)", "T rotor media (C)",
             "Mismatch CsTq-Est", "Derating (mstras)", "Derating min",
             "Derating max", "Causa derating", "Flags activos", "EAJS filtered",
             "Vd medio", "Vq medio", "RGL sat cnt", "Estado sat",
             "Id medio", "Iq medio", "Id_req medio", "Iq_req medio",
             "Id error", "Iq error",
             "Fases OK?", "Índice descomp. (A)"])
        self.torque_events_table.horizontalHeader().setStretchLastSection(True)
        self.torque_events_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.torque_events_table.setAlternatingRowColors(True)
        self.torque_events_table.setSortingEnabled(True)
        layout.addWidget(self.torque_events_table, 1)

    def _build_power_tab(self):
        layout = QVBoxLayout(self.tab_power)
        lbl_cases = QLabel("Balance de potencia por caso de operación")
        layout.addWidget(lbl_cases)
        self.power_cases_table = QTableWidget()
        self.power_cases_table.setColumnCount(23)
        self.power_cases_table.setHorizontalHeaderLabels(
            ["Caso", "Muestras", "% tiempo",
             "P_ME_HEVC (kW)", "P_ME_INV (kW)", "Offset ME (kW)",
             "P_HSG_HEVC (kW)", "P_HSG_INV (kW)", "Offset HSG (kW)",
             "P_cons (kW)", "P_bat (kW)", "ICE (kW)", "Par rueda (Nm)",
             "Pot rueda (kW)", "Eff ICE+HSG (%)",
             "Eff ME HEVC (%)", "Eff ME INV (%)", "Diff ME (%)",
             "Eff HSG HEVC (%)", "Eff HSG INV (%)", "Diff HSG (%)",
             "Error balance (kW)", "% balance OK"])
        self.power_cases_table.horizontalHeader().setStretchLastSection(True)
        self.power_cases_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.power_cases_table.setAlternatingRowColors(True)
        layout.addWidget(self.power_cases_table)

        lbl_comp = QLabel("Comparación HEVC vs INV (global)")
        layout.addWidget(lbl_comp)
        self.power_comp_table = QTableWidget()
        self.power_comp_table.setColumnCount(8)
        self.power_comp_table.setHorizontalHeaderLabels(
            ["Máquina", "P_HEVC media", "P_INV media", "Offset HEVC-INV",
             "P_mec HEVC", "P_mec INV", "Perdidas INV", "Eff INV (%)"])
        self.power_comp_table.horizontalHeader().setStretchLastSection(True)
        self.power_comp_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.power_comp_table.setAlternatingRowColors(True)
        layout.addWidget(self.power_comp_table)

        lbl_eff = QLabel("Rendimientos por máquina y modo")
        layout.addWidget(lbl_eff)
        self.power_eff_table = QTableWidget()
        self.power_eff_table.setColumnCount(6)
        self.power_eff_table.setHorizontalHeaderLabels(
            ["Máquina", "Modo", "Rendimiento (%)", "P_mec media (kW)",
             "P_elec media (kW)", "Muestras"])
        self.power_eff_table.horizontalHeader().setStretchLastSection(True)
        self.power_eff_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.power_eff_table.setAlternatingRowColors(True)
        layout.addWidget(self.power_eff_table, 1)

        lbl_dls = QLabel("Rendimiento por marcha (DLS)")
        layout.addWidget(lbl_dls)
        self.power_dls_table = QTableWidget()
        self.power_dls_table.setColumnCount(12)
        self.power_dls_table.setHorizontalHeaderLabels(
            ["DLS", "Marcha", "Muestras",
             "ME motor HEVC (%)", "ME motor INV (%)",
             "ME gen HEVC (%)", "ME gen INV (%)",
             "HSG motor HEVC (%)", "HSG motor INV (%)",
             "HSG gen HEVC (%)", "HSG gen INV (%)",
             "ICE+HSG (%)"])
        self.power_dls_table.horizontalHeader().setStretchLastSection(True)
        self.power_dls_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.power_dls_table.setAlternatingRowColors(True)
        layout.addWidget(self.power_dls_table)

        lbl_be = QLabel("Eventos de desbalance de potencia")
        layout.addWidget(lbl_be)
        self.power_balance_events_table = QTableWidget()
        self.power_balance_events_table.setColumnCount(6)
        self.power_balance_events_table.setHorizontalHeaderLabels(
            ["Inicio (s)", "Fin (s)", "Duracion (s)",
             "Error max (kW)", "Error medio (kW)", "Muestras"])
        self.power_balance_events_table.horizontalHeader().setStretchLastSection(True)
        self.power_balance_events_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.power_balance_events_table.setAlternatingRowColors(True)
        layout.addWidget(self.power_balance_events_table, 1)

    def _build_thd_tab(self):
        layout = QVBoxLayout(self.tab_thd)
        lbl = QLabel("THD de corrientes de fase por punto de operacion")
        layout.addWidget(lbl)
        self.thd_table = QTableWidget()
        self.thd_table.setColumnCount(13)
        self.thd_table.setHorizontalHeaderLabels(
            ["Maquina", "Vel (rpm)", "Par (Nm)",
             "THD U (%)", "THD V (%)", "THD W (%)", "THD avg (%)",
             "I3 (%)", "I5 (%)", "I7 (%)", "I11 (%)", "I13 (%)",
             "Ventanas"])
        self.thd_table.horizontalHeader().setStretchLastSection(True)
        self.thd_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.thd_table.setAlternatingRowColors(True)
        layout.addWidget(self.thd_table, 1)

    def _build_sync_tab(self):
        layout = QVBoxLayout(self.tab_sync)
        lbl = QLabel("Riesgo de sincronía PWM por punto de operación estable")
        layout.addWidget(lbl)
        self.sync_table = QTableWidget()
        self.sync_table.setColumnCount(14)
        self.sync_table.setHorizontalHeaderLabels(
            ["Máquina", "Vel (rpm)", "Par (Nm)", "f_elec (Hz)",
             "f_sw (Hz)", "m_f", "Estrategia", "THD avg (%)", "I5 (%)", "I7 (%)",
             "I11 (%)", "I13 (%)", "Riesgo", "Armónicos dom."])
        self.sync_table.horizontalHeader().setStretchLastSection(True)
        self.sync_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.sync_table.setAlternatingRowColors(True)
        self.sync_table.setSortingEnabled(True)
        layout.addWidget(self.sync_table, 1)

    def _build_graph_tab(self):
        layout = QVBoxLayout(self.tab_graph)
        graph_btn_row = QHBoxLayout()
        self.btn_prev_machine = QPushButton("◀ Anterior")
        self.btn_prev_machine.clicked.connect(self._prev_machine)
        self.btn_prev_machine.setEnabled(False)
        graph_btn_row.addWidget(self.btn_prev_machine)

        self.lbl_machine_title = QLabel("ME")
        self.lbl_machine_title.setAlignment(Qt.AlignCenter)
        self.lbl_machine_title.setFont(QFont("", 12, QFont.Bold))
        graph_btn_row.addWidget(self.lbl_machine_title, 1)

        self.btn_next_machine = QPushButton("Siguiente ▶")
        self.btn_next_machine.clicked.connect(self._next_machine)
        self.btn_next_machine.setEnabled(False)
        graph_btn_row.addWidget(self.btn_next_machine)

        self.btn_open_graph = QPushButton("📈 Abrir gráfica grande")
        self.btn_open_graph.clicked.connect(self._open_big_graph)
        self.btn_open_graph.setEnabled(False)
        graph_btn_row.addWidget(self.btn_open_graph)

        layout.addLayout(graph_btn_row)

        self.graph_canvas = FigureCanvasQTAgg(Figure(figsize=(10, 3.5)))
        layout.addWidget(self.graph_canvas, 1)

    def _on_arch_changed(self, arch):
        self._update_mux_visibility()

    def _update_mux_visibility(self):
        arch = self.arch_combo.currentText()
        if arch == 'Sweet200':
            self.mux_stack.setCurrentIndex(0)
        else:
            self.mux_stack.setCurrentIndex(1)

    def _browse_file(self, line_edit, filter_str):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar fichero", "", filter_str)
        if path:
            line_edit.setText(path)

    def _set_tabs_enabled(self, enabled):
        for i in range(self.tabs.count()):
            self.tabs.setTabEnabled(i, enabled)

    def _create_eq_template(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar plantilla de equivalencias", "equivalencias_can.xlsx",
            "Excel (*.xlsx)")
        if not path:
            return
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Equivalencias"
        ws.append(["nombre_can", "variable_interna", "maquina", "notas"])
        ws.append(["Vxx_cani_UserSOC_HV", "Vxx_hvb_usoc", "ME|HSG",
                    "Variable interna que corresponde a la señal CAN"])
        ws.column_dimensions['A'].width = 40
        ws.column_dimensions['B'].width = 40
        ws.column_dimensions['C'].width = 20
        ws.column_dimensions['D'].width = 60
        wb.save(path)
        self.eq_path.setText(path)
        self.statusBar().showMessage(f"Plantilla creada: {path}")

    def _run_analysis(self):
        mf4 = self.mf4_path.text().strip()
        dico = self.dico_path.text().strip()
        cal_path = self.cal_path.text().strip()
        if not mf4:
            QMessageBox.warning(self, "Error", "Seleccione un fichero .mf4")
            return
        if not dico:
            QMessageBox.warning(self, "Error", "Seleccione el fichero DICO")
            return
        needs_cal = self.chk_torque.isChecked()
        if needs_cal and not cal_path:
            QMessageBox.warning(self, "Error", "Seleccione el fichero de calibración del inversor (necesario para análisis de par)")
            return
        if not Path(mf4).exists():
            QMessageBox.warning(self, "Error", f"No existe: {mf4}")
            return
        if not Path(dico).exists():
            QMessageBox.warning(self, "Error", f"No existe: {dico}")
            return
        if not Path(cal_path).exists():
            QMessageBox.warning(self, "Error", f"No existe: {cal_path}")
            return

        arch = self.arch_combo.currentText()
        can_paths = {}
        if arch == 'Sweet200':
            me = self.mux_me_path.text().strip()
            hsg = self.mux_hsg_path.text().strip()
            if me and Path(me).exists():
                can_paths['me'] = me
            if hsg and Path(hsg).exists():
                can_paths['hsg'] = hsg
        else:  # Sweet400
            mux = self.mux_path.text().strip()
            if mux and Path(mux).exists():
                can_paths['mux'] = mux

        self.btn_run.setEnabled(False)
        self._set_tabs_enabled(False)
        self.btn_export.setEnabled(False)
        self.progress.setVisible(True)
        self.lbl_status.setVisible(True)
        self.progress.setValue(0)

        thresholds = {
            'ME': self.spin_th_me.value(),
            'HSG': self.spin_th_hsg.value(),
        }

        eq_path = self.eq_path.text().strip()
        if eq_path and not Path(eq_path).exists():
            eq_path = None

        perf_path = self.perf_path.text().strip()
        if perf_path and not Path(perf_path).exists():
            perf_path = None

        self._worker = AnalysisWorker(
            mf4_path=mf4,
            dico_path=dico,
            thresholds=thresholds,
            window_ms=self.spin_window.value(),
            min_event_ms=self.spin_min_event.value(),
            raster=None,
            architecture=arch if can_paths else None,
            can_mux_paths=can_paths if can_paths else None,
            equivalence_path=eq_path,
            torque_perf_path=perf_path,
            torque_threshold_nm=self.spin_torque_th.value(),
            calibration_path=cal_path,
            analyses_flags={
                'current': self.chk_current.isChecked(),
                'torque': self.chk_torque.isChecked(),
                'power': self.chk_power.isChecked(),
                'thd': self.chk_thd.isChecked(),
                'sync': self.chk_sync.isChecked(),
            },
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.warning_occurred.connect(self._on_warning)
        self._worker.start()

    def _on_progress(self, pct, msg):
        self.progress.setValue(pct)
        self.lbl_status.setText(msg)

    def _on_finished(self, analyses, df, can_signals, torque_analyses, power_result, thd_result, sync_result):
        self._analyses = analyses
        self._df = df
        self._can_signals = can_signals
        self._torque_analyses = torque_analyses
        self._power_result = power_result
        self._thd_result = thd_result
        self._sync_result = sync_result
        self._current_machine_idx = 0
        self._eq_path_used = self.eq_path.text().strip()

        self._populate_summary(analyses)
        self._populate_events(analyses)
        self._populate_can(analyses, can_signals)
        self._populate_torque(torque_analyses)
        self._populate_power_balance(power_result)
        self._populate_thd(thd_result)
        self._populate_sync_risk(sync_result)
        self._update_graph()

        self._set_tabs_enabled(True)
        self.btn_export.setEnabled(True)
        self.tabs.setCurrentIndex(0)

        # Save CSV if plot checkbox checked
        if self.chk_plot.isChecked():
            output_dir = Path(self.mf4_path.text().strip()).parent / "output"
            output_dir.mkdir(exist_ok=True)
            for a in analyses:
                edf = events_to_dataframe(a.events)
                edf.to_csv(output_dir / f'informe_compensacion_{a.machine}.csv',
                           sep=';', decimal=',', index=False)
            sdf = stats_to_dataframe(analyses)
            sdf.to_csv(output_dir / 'resumen_estadisticas.csv',
                       sep=';', decimal=',', index=False)
            # Torque reports
            if torque_analyses:
                for ta in torque_analyses:
                    tedf = torque_events_to_dataframe(ta.events)
                    tedf.to_csv(output_dir / f'informe_precision_par_{ta.machine}.csv',
                                sep=';', decimal=',', index=False)
                tsdf = torque_stats_to_dataframe(torque_analyses)
                tsdf.to_csv(output_dir / 'resumen_precision_par.csv',
                            sep=';', decimal=',', index=False)
            self.statusBar().showMessage(f"Informes guardados en: {output_dir}")

        self.progress.setVisible(False)
        self.lbl_status.setVisible(False)
        self.btn_run.setEnabled(True)
        self.statusBar().showMessage("Análisis completado.")

    def _on_error(self, msg):
        QMessageBox.critical(self, "Error", f"Error durante el análisis:\n{msg}")
        self.progress.setVisible(False)
        self.lbl_status.setVisible(False)
        self.btn_run.setEnabled(True)
        self.statusBar().showMessage("Error.")

    def _on_warning(self, msg):
        QMessageBox.warning(self, "Advertencia", msg)

    def _populate_summary(self, analyses):
        self.summary_table.setRowCount(0)
        for a in analyses:
            row = self.summary_table.rowCount()
            self.summary_table.insertRow(row)
            self.summary_table.setItem(row, 0, QTableWidgetItem(a.machine))
            self.summary_table.setItem(row, 1, QTableWidgetItem(f"±{a.threshold_a} A"))
            self.summary_table.setItem(row, 2, QTableWidgetItem(f"{a.pct_compensated:.2f}%"))
            self.summary_table.setItem(row, 3, QTableWidgetItem(f"{a.pct_uncompensated:.2f}%"))
            self.summary_table.setItem(row, 4, QTableWidgetItem(str(len(a.events))))
            total_t = sum(e.duration_s for e in a.events)
            self.summary_table.setItem(row, 5, QTableWidgetItem(f"{total_t:.3f} s"))
        self.summary_table.resizeColumnsToContents()

    def _populate_can(self, analyses, can_signals):
        self.can_table.setSortingEnabled(False)
        self.can_table.setRowCount(0)
        total_rows = 0
        sig_names = set()

        # Build lookup: basic_name -> CanSignal (also check equivalence alt name)
        sig_lookup = {}
        if can_signals:
            for sigs in can_signals.values():
                for s in sigs:
                    sig_lookup[s.basic_name] = s
                    if self._eq_path_used:
                        try:
                            from can_equivalence import EquivalenceDb
                            eq = EquivalenceDb.from_excel(self._eq_path_used)
                            if s.basic_name in eq:
                                alt = eq.get(s.basic_name)
                                if alt not in sig_lookup:
                                    sig_lookup[alt] = s
                        except Exception:
                            pass

        for a_idx, a in enumerate(analyses):
            for e_idx, e in enumerate(a.events):
                if not e.can_snapshot:
                    continue
                for sig_name, val in e.can_snapshot.items():
                    row = self.can_table.rowCount()
                    self.can_table.insertRow(row)
                    self.can_table.setItem(row, 0, QTableWidgetItem(e.machine))
                    self.can_table.setItem(row, 1, QTableWidgetItem(str(e_idx + 1)))
                    self.can_table.setItem(row, 2, QTableWidgetItem(f"{e.start_time:.4f}"))
                    self.can_table.setItem(row, 3, QTableWidgetItem(sig_name))
                    text = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
                    self.can_table.setItem(row, 4, QTableWidgetItem(text))
                    s = sig_lookup.get(sig_name)
                    unit = s.unit if s else ''
                    res = str(s.resolution) if s else ''
                    offset = str(s.offset) if s else ''
                    self.can_table.setItem(row, 5, QTableWidgetItem(unit))
                    self.can_table.setItem(row, 6, QTableWidgetItem(res))
                    self.can_table.setItem(row, 7, QTableWidgetItem(offset))
                    sig_names.add(sig_name)
                    total_rows += 1

        self.can_table.setSortingEnabled(True)
        self.can_table.resizeColumnsToContents()
        if total_rows:
            self.lbl_signal_count.setText(f"{total_rows} lecturas CAN de {len(sig_names)} señales distintas")
        else:
            self.lbl_signal_count.setText("Sin datos CAN (no se cargó archivo MUX o no hay señales en el .mf4)")

    def _populate_torque(self, torque_analyses):
        # Summary table
        self.torque_summary_table.setRowCount(0)
        for ta in torque_analyses:
            row = self.torque_summary_table.rowCount()
            self.torque_summary_table.insertRow(row)
            self.torque_summary_table.setItem(row, 0, QTableWidgetItem(ta.machine))
            self.torque_summary_table.setItem(row, 1, QTableWidgetItem(f"{ta.mean_error_nm:.3f}"))
            self.torque_summary_table.setItem(row, 2, QTableWidgetItem(f"{ta.max_error_nm:.3f}"))
            self.torque_summary_table.setItem(row, 3, QTableWidgetItem(f"{ta.pct_error:.2f}%"))
            self.torque_summary_table.setItem(row, 4, QTableWidgetItem(str(ta.model_violation_samples)))
            self.torque_summary_table.setItem(row, 5, QTableWidgetItem(f"{ta.pct_violation:.2f}%"))
            self.torque_summary_table.setItem(row, 6, QTableWidgetItem(str(len(ta.events))))
        self.torque_summary_table.resizeColumnsToContents()

        # Events table
        self.torque_events_table.setSortingEnabled(False)
        self.torque_events_table.setRowCount(0)
        for ta in torque_analyses:
            for e in ta.events:
                row = self.torque_events_table.rowCount()
                self.torque_events_table.insertRow(row)
                self.torque_events_table.setItem(row, 0, QTableWidgetItem(e.machine))
                self.torque_events_table.setItem(row, 1, QTableWidgetItem(f"{e.start_time:.4f}"))
                self.torque_events_table.setItem(row, 2, QTableWidgetItem(f"{e.end_time:.4f}"))
                self.torque_events_table.setItem(row, 3, QTableWidgetItem(f"{e.duration_s:.4f}"))
                self.torque_events_table.setItem(row, 4, QTableWidgetItem(f"{e.max_error_nm:.3f}"))
                self.torque_events_table.setItem(row, 5, QTableWidgetItem(f"{e.mean_error_nm:.3f}"))
                self.torque_events_table.setItem(row, 6, QTableWidgetItem(str(e.n_samples)))
                self.torque_events_table.setItem(row, 7, QTableWidgetItem(str(e.model_violations)))
                cmin_s = f"{e.cmin_at_event:.1f}" if e.cmin_at_event is not None else "-"
                cmax_s = f"{e.cmax_at_event:.1f}" if e.cmax_at_event is not None else "-"
                self.torque_events_table.setItem(row, 8, QTableWidgetItem(cmin_s))
                self.torque_events_table.setItem(row, 9, QTableWidgetItem(cmax_s))
                self.torque_events_table.setItem(row, 10, QTableWidgetItem(e.mode))
                self.torque_events_table.setItem(row, 11, QTableWidgetItem(f"{e.avg_tq_est:.2f}"))
                limit_s = f"{e.limit_used:.1f}" if e.limit_used is not None else "-"
                self.torque_events_table.setItem(row, 12, QTableWidgetItem(limit_s))
                within_s = "Sí" if e.within_limits else "No"
                w_item = QTableWidgetItem(within_s)
                if not e.within_limits:
                    w_item.setForeground(Qt.red)
                else:
                    w_item.setForeground(Qt.darkGreen)
                self.torque_events_table.setItem(row, 13, w_item)
                self.torque_events_table.setItem(row, 14, QTableWidgetItem(f"{e.avg_voltage:.1f}"))
                self.torque_events_table.setItem(row, 15, QTableWidgetItem(f"{e.avg_speed:.0f}"))
                self.torque_events_table.setItem(row, 16, QTableWidgetItem(f"{e.avg_rotor_temp:.1f}"))
                self.torque_events_table.setItem(row, 17, QTableWidgetItem(str(e.mismatch_samples)))
                self.torque_events_table.setItem(row, 18, QTableWidgetItem(str(e.derating_samples)))
                self.torque_events_table.setItem(row, 19, QTableWidgetItem(str(e.derating_min_samples)))
                self.torque_events_table.setItem(row, 20, QTableWidgetItem(str(e.derating_max_samples)))
                self.torque_events_table.setItem(row, 21, QTableWidgetItem(e.derating_cause))
                self.torque_events_table.setItem(row, 22, QTableWidgetItem(e.active_flags_desc))
                self.torque_events_table.setItem(row, 23, QTableWidgetItem(str(e.eajs_filtered_samples)))
                self.torque_events_table.setItem(row, 24, QTableWidgetItem(f"{e.avg_vd:.2f}"))
                self.torque_events_table.setItem(row, 25, QTableWidgetItem(f"{e.avg_vq:.2f}"))
                self.torque_events_table.setItem(row, 26, QTableWidgetItem(str(e.rgl_sat_samples)))
                self.torque_events_table.setItem(row, 27, QTableWidgetItem(e.saturation_state))
                self.torque_events_table.setItem(row, 28, QTableWidgetItem(f"{e.avg_i_d:.2f}"))
                self.torque_events_table.setItem(row, 29, QTableWidgetItem(f"{e.avg_i_q:.2f}"))
                self.torque_events_table.setItem(row, 30, QTableWidgetItem(f"{e.avg_i_d_req:.2f}"))
                self.torque_events_table.setItem(row, 31, QTableWidgetItem(f"{e.avg_i_q_req:.2f}"))
                self.torque_events_table.setItem(row, 32, QTableWidgetItem(f"{e.avg_i_d_error:.2f}"))
                self.torque_events_table.setItem(row, 33, QTableWidgetItem(f"{e.avg_i_q_error:.2f}"))
                fases_ok = "No" if e.phase_unbalanced else "Sí"
                item_fases = QTableWidgetItem(fases_ok)
                if e.phase_unbalanced:
                    item_fases.setForeground(Qt.red)
                else:
                    item_fases.setForeground(Qt.darkGreen)
                self.torque_events_table.setItem(row, 34, item_fases)
                self.torque_events_table.setItem(row, 35, QTableWidgetItem(f"{e.phase_imbalance_index:.2f}"))
        self.torque_events_table.setSortingEnabled(True)
        self.torque_events_table.resizeColumnsToContents()

    def _populate_power_balance(self, power_result):
        from power_analyzer import (power_balance_to_dataframe, efficiency_to_dataframe,
                                     power_comparison_to_dataframe, dls_eff_to_dataframe,
                                     balance_events_to_dataframe)

        if power_result is None:
            for t in (self.power_cases_table, self.power_comp_table,
                       self.power_eff_table, self.power_dls_table,
                       self.power_balance_events_table):
                t.setRowCount(0)
                t.setColumnCount(1)
                t.setHorizontalHeaderLabels(["Análisis no ejecutado"])
                t.setItem(0, 0, QTableWidgetItem("Seleccione 'Balance potencia' en la configuración"))
            return

        # Cases table
        df_cases = power_balance_to_dataframe(power_result)
        self.power_cases_table.setSortingEnabled(False)
        self.power_cases_table.setRowCount(0)
        cols_cases = list(df_cases.columns) if not df_cases.empty else []
        if cols_cases:
            self.power_cases_table.setColumnCount(len(cols_cases))
            self.power_cases_table.setHorizontalHeaderLabels(cols_cases)
        for _, row in df_cases.iterrows():
            r = self.power_cases_table.rowCount()
            self.power_cases_table.insertRow(r)
            for c, col in enumerate(cols_cases):
                val = row[col]
                if val is None:
                    text = '-'
                elif isinstance(val, float):
                    text = f"{val:.2f}"
                else:
                    text = str(val)
                self.power_cases_table.setItem(r, c, QTableWidgetItem(text))
        self.power_cases_table.setSortingEnabled(True)
        self.power_cases_table.resizeColumnsToContents()

        # Comparison table
        df_comp = power_comparison_to_dataframe(power_result)
        self.power_comp_table.setSortingEnabled(False)
        self.power_comp_table.setRowCount(0)
        cols_comp = list(df_comp.columns) if not df_comp.empty else []
        if cols_comp:
            self.power_comp_table.setColumnCount(len(cols_comp))
            self.power_comp_table.setHorizontalHeaderLabels(cols_comp)
        for _, row in df_comp.iterrows():
            r = self.power_comp_table.rowCount()
            self.power_comp_table.insertRow(r)
            for c, col in enumerate(cols_comp):
                val = row[col]
                if val is None:
                    text = '-'
                elif isinstance(val, float):
                    text = f"{val:.2f}"
                else:
                    text = str(val)
                self.power_comp_table.setItem(r, c, QTableWidgetItem(text))
        self.power_comp_table.setSortingEnabled(True)
        self.power_comp_table.resizeColumnsToContents()

        # Efficiency table
        df_eff = efficiency_to_dataframe(power_result)
        self.power_eff_table.setSortingEnabled(False)
        self.power_eff_table.setRowCount(0)
        cols_eff = ['Máquina', 'Modo', 'Rendimiento (%)', 'P_mec media (kW)',
                    'P_elec media (kW)', 'Muestras']
        for _, row in df_eff.iterrows():
            r = self.power_eff_table.rowCount()
            self.power_eff_table.insertRow(r)
            for c, col in enumerate(cols_eff):
                val = row[col]
                text = f"{val:.2f}" if isinstance(val, float) else str(val)
                self.power_eff_table.setItem(r, c, QTableWidgetItem(text))
        self.power_eff_table.setSortingEnabled(True)
        self.power_eff_table.resizeColumnsToContents()

        # DLS efficiency table
        df_dls = dls_eff_to_dataframe(power_result)
        self.power_dls_table.setSortingEnabled(False)
        self.power_dls_table.setRowCount(0)
        cols_dls = list(df_dls.columns) if not df_dls.empty else []
        if cols_dls:
            self.power_dls_table.setColumnCount(len(cols_dls))
            self.power_dls_table.setHorizontalHeaderLabels(cols_dls)
        for _, row in df_dls.iterrows():
            r = self.power_dls_table.rowCount()
            self.power_dls_table.insertRow(r)
            for c, col in enumerate(cols_dls):
                val = row[col]
                if val is None:
                    text = '-'
                elif isinstance(val, float):
                    text = f"{val:.2f}"
                else:
                    text = str(val)
                self.power_dls_table.setItem(r, c, QTableWidgetItem(text))
        self.power_dls_table.setSortingEnabled(True)
        self.power_dls_table.resizeColumnsToContents()

        # Balance events table
        df_be = balance_events_to_dataframe(power_result)
        self.power_balance_events_table.setSortingEnabled(False)
        self.power_balance_events_table.setRowCount(0)
        cols_be = list(df_be.columns) if not df_be.empty else []
        if cols_be:
            self.power_balance_events_table.setColumnCount(len(cols_be))
            self.power_balance_events_table.setHorizontalHeaderLabels(cols_be)
        for _, row in df_be.iterrows():
            r = self.power_balance_events_table.rowCount()
            self.power_balance_events_table.insertRow(r)
            for c, col in enumerate(cols_be):
                val = row[col]
                text = f"{val:.4f}" if isinstance(val, float) else str(val)
                self.power_balance_events_table.setItem(r, c, QTableWidgetItem(text))
        self.power_balance_events_table.setSortingEnabled(True)
        self.power_balance_events_table.resizeColumnsToContents()

    def _populate_thd(self, thd_result):
        from thd_analyzer import thd_to_dataframe
        df = thd_to_dataframe(thd_result)
        self.thd_table.setSortingEnabled(False)
        self.thd_table.setRowCount(0)
        cols = list(df.columns) if not df.empty else []
        if cols:
            self.thd_table.setColumnCount(len(cols))
            self.thd_table.setHorizontalHeaderLabels(cols)
        for _, row in df.iterrows():
            r = self.thd_table.rowCount()
            self.thd_table.insertRow(r)
            for c, col in enumerate(cols):
                val = row[col]
                if val is None:
                    text = '-'
                elif isinstance(val, float):
                    text = f"{val:.2f}"
                else:
                    text = str(val)
                self.thd_table.setItem(r, c, QTableWidgetItem(text))
        self.thd_table.setSortingEnabled(True)
        self.thd_table.resizeColumnsToContents()

    def _populate_sync_risk(self, sync_result):
        from sync_analyzer import sync_risk_to_dataframe
        self.sync_table.setSortingEnabled(False)
        self.sync_table.setRowCount(0)
        if sync_result is None:
            self.sync_table.setColumnCount(1)
            self.sync_table.setHorizontalHeaderLabels(["Análisis no ejecutado"])
            self.sync_table.setItem(0, 0, QTableWidgetItem("Seleccione 'Sincronía PWM' en la configuración"))
            self.sync_table.setSortingEnabled(True)
            return
        df = sync_risk_to_dataframe(sync_result)
        cols = list(df.columns) if not df.empty else []
        if cols:
            self.sync_table.setColumnCount(len(cols))
            self.sync_table.setHorizontalHeaderLabels(cols)
        for _, row in df.iterrows():
            r = self.sync_table.rowCount()
            self.sync_table.insertRow(r)
            for c, col in enumerate(cols):
                val = row[col]
                if val is None:
                    text = '-'
                elif isinstance(val, float):
                    text = f"{val:.2f}"
                else:
                    text = str(val)
                item = QTableWidgetItem(text)
                if col == 'Riesgo':
                    if val == 'high':
                        item.setForeground(Qt.red)
                    elif val == 'medium':
                        item.setForeground(QColor(255, 165, 0))
                    else:
                        item.setForeground(Qt.darkGreen)
                self.sync_table.setItem(r, c, item)
        self.sync_table.setSortingEnabled(True)
        self.sync_table.resizeColumnsToContents()

    def _populate_events(self, analyses):
        self.events_table.setSortingEnabled(False)
        self.events_table.setRowCount(0)
        for a in analyses:
            for e in a.events:
                row = self.events_table.rowCount()
                self.events_table.insertRow(row)
                self.events_table.setItem(row, 0, QTableWidgetItem(e.machine))
                self.events_table.setItem(row, 1, QTableWidgetItem(f"{e.start_time:.4f}"))
                self.events_table.setItem(row, 2, QTableWidgetItem(f"{e.end_time:.4f}"))
                self.events_table.setItem(row, 3, QTableWidgetItem(f"{e.duration_s:.4f}"))
                self.events_table.setItem(row, 4, QTableWidgetItem(f"{e.max_abs_sum:.4f}"))
                self.events_table.setItem(row, 5, QTableWidgetItem(f"{e.mean_abs_sum:.4f}"))
        self.events_table.setSortingEnabled(True)
        self.events_table.resizeColumnsToContents()

    def _update_graph(self):
        if not self._analyses:
            return
        idx = self._current_machine_idx
        if idx >= len(self._analyses):
            idx = 0
        a = self._analyses[idx]
        self.lbl_machine_title.setText(a.machine)
        self.btn_prev_machine.setEnabled(idx > 0)
        self.btn_next_machine.setEnabled(idx < len(self._analyses) - 1)
        self.btn_open_graph.setEnabled(True)

        fig = self.graph_canvas.figure
        fig.clf()
        ax = fig.add_subplot(111)
        time = self._df.index.values
        ax.plot(time, a.sum_series.values, alpha=0.3, linewidth=0.5, label='Suma instantánea')
        ax.plot(time, a.smooth_series.values, linewidth=1.0, label=f'Media móvil ({a.window_ms}ms)')
        ax.axhline(a.threshold_a, color='r', linestyle='--', linewidth=1, label=f'+{a.threshold_a}A')
        ax.axhline(-a.threshold_a, color='r', linestyle='--', linewidth=1, label=f'-{a.threshold_a}A')
        for e in a.events:
            ax.axvspan(e.start_time, e.end_time, alpha=0.15, color='red')
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Suma de corrientes (A)')
        ax.set_title(f'Compensación - {a.machine} (umbral ±{a.threshold_a}A)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.graph_canvas.draw()

    def _prev_machine(self):
        if self._current_machine_idx > 0:
            self._current_machine_idx -= 1
            self._update_graph()

    def _next_machine(self):
        if self._current_machine_idx < len(self._analyses) - 1:
            self._current_machine_idx += 1
            self._update_graph()

    def _open_big_graph(self):
        if not self._analyses:
            return
        idx = self._current_machine_idx
        w = GraphWindow(self._df, self._analyses[idx], self)
        w.show()

    def _export_csv(self):
        if not self._analyses:
            return
        output_dir = Path(self.mf4_path.text().strip()).parent / "output"
        output_dir.mkdir(exist_ok=True)
        for a in self._analyses:
            edf = events_to_dataframe(a.events)
            edf.to_csv(output_dir / f'informe_compensacion_{a.machine}.csv',
                       sep=';', decimal=',', index=False)
        sdf = stats_to_dataframe(self._analyses)
        sdf.to_csv(output_dir / 'resumen_estadisticas.csv',
                   sep=';', decimal=',', index=False)
        if self._torque_analyses:
            for ta in self._torque_analyses:
                tedf = torque_events_to_dataframe(ta.events)
                tedf.to_csv(output_dir / f'informe_precision_par_{ta.machine}.csv',
                            sep=';', decimal=',', index=False)
            tsdf = torque_stats_to_dataframe(self._torque_analyses)
            tsdf.to_csv(output_dir / 'resumen_precision_par.csv',
                        sep=';', decimal=',', index=False)
        self.statusBar().showMessage(f"Exportado a: {output_dir}")
        QMessageBox.information(self, "Exportar", f"Ficheros guardados en:\n{output_dir}")
