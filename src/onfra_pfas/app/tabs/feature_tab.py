"""
Feature Finding tab for ONFRA PFAS application.

Provides UI for mzML input, configuration, and feature finding execution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, Signal, QThread, QObject
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QGroupBox,
    QFormLayout,
    QDoubleSpinBox,
    QSpinBox,
    QComboBox,
    QProgressBar,
    QTextEdit,
    QSplitter,
    QMessageBox,
    QCheckBox,
)

from ...core.config import (
    PipelineConfig,
    FeatureFinderConfig,
    BlankCorrectionPolicy,
    GPUMode,
)
from ...core.io_mzml import load_mzml_meta, MzMLMeta
from ...core.featurefinding import run_feature_finder_metabo, FeatureResult
from ...core.checkpoints import RunManager, RunContext
from ...core.backend import probe_gpu, get_array_backend
from ...core.errors import OnfraError

logger = logging.getLogger(__name__)


class FeatureFinderWorker(QObject):
    """Worker for running feature finding in background thread."""

    progress = Signal(float, str)  # progress percentage, message
    finished = Signal(object)  # FeatureResult or Exception
    log_message = Signal(str)  # Log messages

    def __init__(
        self,
        mzml_path: Path,
        config: FeatureFinderConfig,
        run_context: RunContext,
    ):
        super().__init__()
        self.mzml_path = mzml_path
        self.config = config
        self.run_context = run_context
        self._cancelled = False

    def run(self):
        """Execute feature finding."""
        try:
            self.log_message.emit(f"Starting feature finding on {self.mzml_path.name}")

            def progress_callback(pct: float, msg: str):
                if self._cancelled:
                    raise InterruptedError("Cancelled by user")
                self.progress.emit(pct * 100, msg)
                self.log_message.emit(f"[{pct:.0%}] {msg}")

            result = run_feature_finder_metabo(
                self.mzml_path,
                self.config,
                progress_callback,
            )

            self.log_message.emit(f"Found {len(result.features_df)} features")
            self.finished.emit(result)

        except Exception as e:
            self.log_message.emit(f"Error: {e}")
            self.finished.emit(e)

    def cancel(self):
        """Cancel the operation."""
        self._cancelled = True


class FeatureTab(QWidget):
    """Feature Finding tab widget."""

    # Signal emitted when features are found
    features_found = Signal(object)  # FeatureResult

    def __init__(self, config: PipelineConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._worker = None
        self._thread = None
        self._run_context: RunContext | None = None
        self._mzml_meta: MzMLMeta | None = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Title
        title = QLabel("Feature Finding")
        title.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: #e94560;
            }
        """)
        layout.addWidget(title)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Configuration
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # File input group
        file_group = self._create_file_group()
        left_layout.addWidget(file_group)

        # Parameters group
        params_group = self._create_params_group()
        left_layout.addWidget(params_group)

        # GPU settings
        gpu_group = self._create_gpu_group()
        left_layout.addWidget(gpu_group)

        # Blank correction
        blank_group = self._create_blank_group()
        left_layout.addWidget(blank_group)

        left_layout.addStretch()

        splitter.addWidget(left_panel)

        # Right panel - Progress and Log
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # File info
        self.file_info_label = QLabel("No file selected")
        self.file_info_label.setStyleSheet("color: #808080; font-size: 11px;")
        self.file_info_label.setWordWrap(True)
        right_layout.addWidget(self.file_info_label)

        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #a0a0a0;")
        progress_layout.addWidget(self.status_label)

        right_layout.addWidget(progress_group)

        # Log output
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, monospace;
                font-size: 11px;
            }
        """)
        log_layout.addWidget(self.log_text)

        right_layout.addWidget(log_group)

        splitter.addWidget(right_panel)

        # Set splitter sizes
        splitter.setSizes([400, 400])
        layout.addWidget(splitter)

        # Buttons
        button_layout = QHBoxLayout()

        self.run_button = QPushButton("Run Feature Finding")
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #e94560;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #ff6b6b;
            }
            QPushButton:disabled {
                background-color: #555555;
            }
        """)
        self.run_button.clicked.connect(self._on_run)
        button_layout.addWidget(self.run_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._on_cancel)
        button_layout.addWidget(self.cancel_button)

        button_layout.addStretch()

        layout.addLayout(button_layout)

    def _create_file_group(self) -> QGroupBox:
        """Create file input group."""
        group = QGroupBox("Input Files")
        layout = QVBoxLayout(group)

        # mzML file
        mzml_layout = QHBoxLayout()
        mzml_layout.addWidget(QLabel("Sample mzML:"))

        self.mzml_input = QLineEdit()
        self.mzml_input.setPlaceholderText("Select mzML file...")
        self.mzml_input.textChanged.connect(self._on_mzml_changed)
        mzml_layout.addWidget(self.mzml_input)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_mzml)
        mzml_layout.addWidget(browse_btn)

        layout.addLayout(mzml_layout)

        return group

    def _create_params_group(self) -> QGroupBox:
        """Create parameters group."""
        group = QGroupBox("Parameters")
        layout = QFormLayout(group)

        # m/z tolerance
        self.mz_tol_spin = QDoubleSpinBox()
        self.mz_tol_spin.setRange(0.001, 0.1)
        self.mz_tol_spin.setDecimals(4)
        self.mz_tol_spin.setSingleStep(0.001)
        self.mz_tol_spin.setValue(self.config.feature_finder.mass_trace_mz_tolerance)
        layout.addRow("m/z Tolerance (Da):", self.mz_tol_spin)

        # Noise threshold
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0, 1000000)
        self.noise_spin.setDecimals(0)
        self.noise_spin.setSingleStep(100)
        self.noise_spin.setValue(self.config.feature_finder.noise_threshold_int)
        layout.addRow("Noise Threshold:", self.noise_spin)

        # SNR
        self.snr_spin = QDoubleSpinBox()
        self.snr_spin.setRange(1.0, 20.0)
        self.snr_spin.setDecimals(1)
        self.snr_spin.setSingleStep(0.5)
        self.snr_spin.setValue(self.config.feature_finder.chrom_peak_snr)
        layout.addRow("Peak SNR:", self.snr_spin)

        # FWHM
        self.fwhm_spin = QDoubleSpinBox()
        self.fwhm_spin.setRange(1.0, 300.0)
        self.fwhm_spin.setDecimals(1)
        self.fwhm_spin.setSingleStep(1.0)
        self.fwhm_spin.setValue(self.config.feature_finder.chrom_fwhm)
        layout.addRow("Peak FWHM (s):", self.fwhm_spin)

        return group

    def _create_gpu_group(self) -> QGroupBox:
        """Create GPU settings group."""
        group = QGroupBox("GPU Acceleration")
        layout = QFormLayout(group)

        # GPU mode
        self.gpu_mode_combo = QComboBox()
        self.gpu_mode_combo.addItems(["Auto", "Force GPU", "Force CPU"])
        mode_map = {GPUMode.AUTO: 0, GPUMode.FORCE_GPU: 1, GPUMode.FORCE_CPU: 2}
        self.gpu_mode_combo.setCurrentIndex(mode_map.get(self.config.gpu.mode, 0))
        layout.addRow("GPU Mode:", self.gpu_mode_combo)

        # GPU status
        gpu_info = probe_gpu()
        if gpu_info.available:
            status = f"✓ {gpu_info.device_name} ({gpu_info.free_memory_gb:.1f}GB free)"
            color = "#4caf50"
        else:
            status = f"✗ Not available ({gpu_info.error_message})"
            color = "#ff9800"

        self.gpu_status_label = QLabel(status)
        self.gpu_status_label.setStyleSheet(f"color: {color}; font-size: 11px;")
        layout.addRow("Status:", self.gpu_status_label)

        return group

    def _create_blank_group(self) -> QGroupBox:
        """Create blank correction group."""
        group = QGroupBox("Blank Correction")
        layout = QVBoxLayout(group)

        # Enable checkbox
        self.blank_enable = QCheckBox("Enable blank correction")
        layout.addWidget(self.blank_enable)

        # Policy
        policy_layout = QHBoxLayout()
        policy_layout.addWidget(QLabel("Policy:"))
        self.blank_policy_combo = QComboBox()
        self.blank_policy_combo.addItems(["None", "Subtract", "Fold Change", "Presence"])
        policy_layout.addWidget(self.blank_policy_combo)
        layout.addLayout(policy_layout)

        # Blank files
        blank_layout = QHBoxLayout()
        self.blank_input = QLineEdit()
        self.blank_input.setPlaceholderText("Blank mzML files (optional)")
        self.blank_input.setEnabled(False)
        blank_layout.addWidget(self.blank_input)

        blank_btn = QPushButton("Browse")
        blank_btn.clicked.connect(self._browse_blank)
        blank_btn.setEnabled(False)
        self.blank_browse_btn = blank_btn
        blank_layout.addWidget(blank_btn)

        layout.addLayout(blank_layout)

        # Connect enable checkbox
        self.blank_enable.toggled.connect(lambda checked: (
            self.blank_input.setEnabled(checked),
            self.blank_browse_btn.setEnabled(checked),
        ))

        return group

    def _browse_mzml(self):
        """Open file dialog for mzML selection."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select mzML File",
            "",
            "mzML Files (*.mzML *.mzml);;All Files (*)",
        )
        if file_path:
            self.mzml_input.setText(file_path)

    def _browse_blank(self):
        """Open file dialog for blank mzML selection."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Blank mzML Files",
            "",
            "mzML Files (*.mzML *.mzml);;All Files (*)",
        )
        if file_paths:
            self.blank_input.setText(";".join(file_paths))

    def _on_mzml_changed(self, text: str):
        """Handle mzML path change."""
        if not text:
            self.file_info_label.setText("No file selected")
            self._mzml_meta = None
            return

        path = Path(text)
        if not path.exists():
            self.file_info_label.setText("File not found")
            self._mzml_meta = None
            return

        try:
            self._mzml_meta = load_mzml_meta(path, compute_hash=False)
            info_parts = [
                f"<b>{path.name}</b>",
                f"Size: {self._mzml_meta.file_size_mb:.1f} MB",
                f"Spectra: {self._mzml_meta.total_spectra:,}",
                f"MS1: {self._mzml_meta.ms1_count:,}, MS2: {self._mzml_meta.ms2_count:,}",
                f"RT: {self._mzml_meta.rt_min/60:.1f} - {self._mzml_meta.rt_max/60:.1f} min",
            ]

            if self._mzml_meta.is_centroided is False:
                info_parts.append("<span style='color: #ff9800;'>⚠ Profile mode detected</span>")

            if not self._mzml_meta.is_indexed:
                info_parts.append("<span style='color: #ff9800;'>⚠ Not indexed</span>")

            self.file_info_label.setText("<br>".join(info_parts))

        except Exception as e:
            self.file_info_label.setText(f"Error reading file: {e}")
            self._mzml_meta = None

    def _on_run(self):
        """Start feature finding."""
        mzml_path = self.mzml_input.text().strip()
        if not mzml_path:
            QMessageBox.warning(self, "Error", "Please select an mzML file")
            return

        path = Path(mzml_path)
        if not path.exists():
            QMessageBox.warning(self, "Error", "mzML file not found")
            return

        # Update config from UI
        self.config.feature_finder.mass_trace_mz_tolerance = self.mz_tol_spin.value()
        self.config.feature_finder.noise_threshold_int = self.noise_spin.value()
        self.config.feature_finder.chrom_peak_snr = self.snr_spin.value()
        self.config.feature_finder.chrom_fwhm = self.fwhm_spin.value()

        # GPU mode
        mode_map = {0: GPUMode.AUTO, 1: GPUMode.FORCE_GPU, 2: GPUMode.FORCE_CPU}
        self.config.gpu.mode = mode_map.get(self.gpu_mode_combo.currentIndex(), GPUMode.AUTO)

        # Create run context
        run_manager = RunManager(Path(self.config.output_dir))
        backend = get_array_backend(self.config.gpu.mode)
        gpu_info = probe_gpu()

        self._run_context = run_manager.create_run(
            config=self.config.to_dict(),
            input_files=[path],
            gpu_mode=self.config.gpu.mode.value,
            backend=backend.backend_type.value,
            gpu_info=gpu_info.device_name if gpu_info.available else None,
        )

        # Clear log
        self.log_text.clear()
        self._log(f"Created run: {self._run_context.run_meta.run_id}")

        # Setup worker
        self._worker = FeatureFinderWorker(
            path,
            self.config.feature_finder,
            self._run_context,
        )

        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        # Connect signals
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_message.connect(self._log)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)

        # Update UI
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting...")

        # Start
        self._thread.start()

    def _on_cancel(self):
        """Cancel feature finding."""
        if self._worker:
            self._worker.cancel()
            self._log("Cancellation requested...")

    def _on_progress(self, progress: float, message: str):
        """Handle progress update."""
        self.progress_bar.setValue(int(progress))
        self.status_label.setText(message)

    def _on_finished(self, result):
        """Handle completion."""
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

        if isinstance(result, Exception):
            self.status_label.setText(f"Error: {result}")
            self.progress_bar.setValue(0)

            if isinstance(result, OnfraError):
                QMessageBox.warning(self, "Error", result.user_message)
            else:
                QMessageBox.warning(self, "Error", str(result))

        else:
            self.status_label.setText(f"Complete: {len(result.features_df)} features found")
            self.progress_bar.setValue(100)
            self._log(f"Feature finding complete: {len(result.features_df)} features")

            # Emit signal
            self.features_found.emit(result)

    def _log(self, message: str):
        """Add message to log."""
        self.log_text.append(message)
        # Scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
