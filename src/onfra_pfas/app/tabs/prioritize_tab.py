"""
Prioritization tab for ONFRA PFAS application.

Provides UI for PFAS prioritization, scoring, and report generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSplitter,
    QMessageBox,
    QCheckBox,
    QTabWidget,
)

import pandas as pd

from ...core.config import PipelineConfig, PrioritizationConfig
from ...core.pfas_prioritization import run_prioritization
from ...core.checkpoints import RunManager, RunContext

logger = logging.getLogger(__name__)


class PrioritizationWorker(QObject):
    """Worker for running prioritization in background."""

    progress = Signal(float, str)
    finished = Signal(object)  # DataFrame or Exception
    log_message = Signal(str)

    def __init__(
        self,
        features_df: pd.DataFrame,
        ms2_data: dict | None,
        config: PrioritizationConfig,
        suspect_list_path: str | None,
    ):
        super().__init__()
        self.features_df = features_df
        self.ms2_data = ms2_data
        self.config = config
        self.suspect_list_path = suspect_list_path
        self._cancelled = False

    def run(self):
        """Execute prioritization."""
        try:
            self.log_message.emit(f"Starting prioritization on {len(self.features_df)} features")
            self.progress.emit(10, "Running prioritization...")

            result = run_prioritization(
                self.features_df,
                self.ms2_data,
                self.config,
                self.suspect_list_path,
            )

            self.progress.emit(100, "Complete")
            self.log_message.emit(f"Prioritization complete: {len(result)} scored features")
            self.finished.emit(result)

        except Exception as e:
            self.log_message.emit(f"Error: {e}")
            self.finished.emit(e)


class PrioritizeTab(QWidget):
    """Prioritization tab widget."""

    # Signal when prioritization completes
    prioritization_complete = Signal(object)  # DataFrame

    def __init__(self, config: PipelineConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._features_df: pd.DataFrame | None = None
        self._ms2_data: dict | None = None
        self._result_df: pd.DataFrame | None = None
        self._worker = None
        self._thread = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Title
        title = QLabel("PFAS Prioritization")
        title.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: #e94560;
            }
        """)
        layout.addWidget(title)

        # Main splitter
        splitter = QSplitter(Qt.Vertical)

        # Top panel - Inputs and Config
        top_panel = QWidget()
        top_layout = QHBoxLayout(top_panel)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Left - Input
        input_group = self._create_input_group()
        top_layout.addWidget(input_group)

        # Right - Config
        config_group = self._create_config_group()
        top_layout.addWidget(config_group)

        splitter.addWidget(top_panel)

        # Middle - Progress and Log
        middle_panel = QWidget()
        middle_layout = QHBoxLayout(middle_panel)
        middle_layout.setContentsMargins(0, 0, 0, 0)

        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready - Load features from Feature Finding tab")
        self.status_label.setStyleSheet("color: #a0a0a0;")
        progress_layout.addWidget(self.status_label)

        middle_layout.addWidget(progress_group)

        # Log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                color: #1a1a2e;
                font-family: Consolas, monospace;
                font-size: 12px;
                border: 1px solid #d0d5dd;
                border-radius: 6px;
            }
        """)
        log_layout.addWidget(self.log_text)

        middle_layout.addWidget(log_group)

        splitter.addWidget(middle_panel)

        # Bottom - Results Table
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                alternate-background-color: #f8f9fa;
                color: #1a1a2e;
                gridline-color: #d0d5dd;
                border: 1px solid #d0d5dd;
                border-radius: 6px;
            }
            QHeaderView::section {
                background-color: #e8eaed;
                color: #1a1a2e;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
        """)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)

        splitter.addWidget(results_group)

        # Set splitter sizes
        splitter.setSizes([150, 100, 300])
        layout.addWidget(splitter)

        # Buttons
        button_layout = QHBoxLayout()

        self.run_button = QPushButton("Run Prioritization")
        self.run_button.setEnabled(False)
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

        self.export_button = QPushButton("Export Results")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        button_layout.addWidget(self.export_button)

        self.report_button = QPushButton("Generate Report")
        self.report_button.setEnabled(False)
        self.report_button.clicked.connect(self._on_generate_report)
        button_layout.addWidget(self.report_button)

        # Build Dataset button (NEW)
        self.dataset_button = QPushButton("📦 Build Dataset")
        self.dataset_button.setEnabled(False)
        self.dataset_button.setToolTip("ML 학습용 데이터셋 생성")
        self.dataset_button.setStyleSheet("""
            QPushButton {
                background-color: #5a27a0;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #7a47c0;
            }
            QPushButton:disabled {
                background-color: #555555;
            }
        """)
        self.dataset_button.clicked.connect(self._on_build_dataset)
        button_layout.addWidget(self.dataset_button)

        button_layout.addStretch()

        layout.addLayout(button_layout)

    def _create_input_group(self) -> QGroupBox:
        """Create input group."""
        group = QGroupBox("Input")
        layout = QVBoxLayout(group)

        # Feature count
        self.feature_count_label = QLabel("Features: 0")
        self.feature_count_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.feature_count_label)

        # Load from checkpoint
        load_layout = QHBoxLayout()
        load_layout.addWidget(QLabel("Load from checkpoint:"))

        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.addItem("(Use features from Feature Finding)")
        load_layout.addWidget(self.checkpoint_combo)

        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._load_checkpoint)
        load_layout.addWidget(load_btn)

        layout.addLayout(load_layout)

        # Suspect list
        suspect_layout = QHBoxLayout()
        suspect_layout.addWidget(QLabel("Suspect List:"))

        self.suspect_input = QLineEdit()
        self.suspect_input.setPlaceholderText("Optional CSV/TSV file...")
        suspect_layout.addWidget(self.suspect_input)

        suspect_btn = QPushButton("Browse")
        suspect_btn.clicked.connect(self._browse_suspect_list)
        suspect_layout.addWidget(suspect_btn)

        layout.addLayout(suspect_layout)

        # Built-in NIST suspect list button
        nist_layout = QHBoxLayout()
        self.use_nist_btn = QPushButton("📋 Use Built-in NIST PFAS List (86 compounds)")
        self.use_nist_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d5a27;
                color: white;
                border: none;
                padding: 8px 15px;
                font-size: 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3d7a37;
            }
        """)
        self.use_nist_btn.clicked.connect(self._use_builtin_suspect_list)
        nist_layout.addWidget(self.use_nist_btn)
        nist_layout.addStretch()
        
        layout.addLayout(nist_layout)

        return group

    def _create_config_group(self) -> QGroupBox:
        """Create configuration group."""
        group = QGroupBox("Configuration")
        layout = QFormLayout(group)

        # Enable checkboxes
        self.mdc_enable = QCheckBox("MD/C Filtering")
        self.mdc_enable.setChecked(self.config.prioritization.mdc.enabled)
        layout.addRow(self.mdc_enable)

        self.kmd_enable = QCheckBox("KMD Series Detection")
        self.kmd_enable.setChecked(self.config.prioritization.kmd.enabled)
        layout.addRow(self.kmd_enable)

        self.df_enable = QCheckBox("Diagnostic Fragment Matching")
        self.df_enable.setChecked(self.config.prioritization.diagnostic_fragments.enabled)
        layout.addRow(self.df_enable)

        self.suspect_enable = QCheckBox("Suspect Screening")
        self.suspect_enable.setChecked(self.config.prioritization.suspect_screening.enabled)
        layout.addRow(self.suspect_enable)

        # ppm tolerance
        self.ppm_spin = QDoubleSpinBox()
        self.ppm_spin.setRange(1.0, 50.0)
        self.ppm_spin.setDecimals(1)
        self.ppm_spin.setValue(self.config.prioritization.suspect_screening.ppm_tolerance)
        layout.addRow("Suspect ppm:", self.ppm_spin)

        # Min score
        self.min_score_spin = QDoubleSpinBox()
        self.min_score_spin.setRange(0.0, 20.0)
        self.min_score_spin.setDecimals(1)
        self.min_score_spin.setValue(self.config.prioritization.scoring.min_score_threshold)
        layout.addRow("Min Score:", self.min_score_spin)

        # --- ML Scoring Section ---
        ml_separator = QLabel("── ML Scoring ──")
        ml_separator.setStyleSheet("color: #e94560; font-weight: bold;")
        layout.addRow(ml_separator)

        self.ml_enable = QCheckBox("Enable ML Scoring")
        self.ml_enable.setChecked(False)
        self.ml_enable.setToolTip("AI 기반 PFAS 확률 점수 (ml_score)")
        layout.addRow(self.ml_enable)

        self.ml_uncertainty_enable = QCheckBox("Show Uncertainty")
        self.ml_uncertainty_enable.setChecked(True)
        self.ml_uncertainty_enable.setToolTip("MC Dropout 기반 불확실성 표시")
        layout.addRow(self.ml_uncertainty_enable)

        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 20)
        self.top_k_spin.setValue(5)
        self.top_k_spin.setToolTip("유사 스펙트럼 상위 K개 표시")
        layout.addRow("Top-K Similar:", self.top_k_spin)

        # --- Quantification Section ---
        quant_separator = QLabel("── Quantification ──")
        quant_separator.setStyleSheet("color: #e94560; font-weight: bold;")
        layout.addRow(quant_separator)

        self.quant_mode_combo = QComboBox()
        self.quant_mode_combo.addItems(["None", "Quantitative", "Semi-quantitative"])
        self.quant_mode_combo.setToolTip("농도 추정 모드 선택")
        layout.addRow("Quant Mode:", self.quant_mode_combo)

        self.calibration_input = QLineEdit()
        self.calibration_input.setPlaceholderText("Calibration CSV (정량 모드용)...")
        self.calibration_input.setEnabled(False)
        layout.addRow("Calibration:", self.calibration_input)

        # Connect quant mode to enable/disable calibration input
        self.quant_mode_combo.currentTextChanged.connect(self._on_quant_mode_changed)

        return group

    def _on_quant_mode_changed(self, text: str):
        """Handle quant mode change."""
        self.calibration_input.setEnabled(text == "Quantitative")

    def _browse_suspect_list(self):
        """Browse for suspect list file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Suspect List",
            "",
            "CSV/TSV Files (*.csv *.tsv);;All Files (*)",
        )
        if file_path:
            self.suspect_input.setText(file_path)

    def _use_builtin_suspect_list(self):
        """Use the built-in NIST PFAS suspect list."""
        from ...core.utils import resource_path
        
        # Get the path to the built-in suspect list
        suspect_path = resource_path("assets/nist_pfas_suspect_list.csv")
        
        if suspect_path.exists():
            self.suspect_input.setText(str(suspect_path))
            self.suspect_enable.setChecked(True)
            self._log("✅ Loaded built-in NIST PFAS Suspect List (86 compounds)")
            self._log(f"   Path: {suspect_path}")
        else:
            QMessageBox.warning(
                self,
                "Error",
                f"Built-in suspect list not found at:\n{suspect_path}\n\n"
                "Please check the installation.",
            )

    def _load_checkpoint(self):
        """Load features from checkpoint."""
        # TODO: Implement checkpoint loading UI
        QMessageBox.information(
            self,
            "Load Checkpoint",
            "Use the Feature Finding tab to process an mzML file first, "
            "or implement checkpoint selection.",
        )

    def set_features(self, features_df: pd.DataFrame, ms2_data: dict | None = None):
        """
        Set features from Feature Finding tab.

        Args:
            features_df: Features DataFrame
            ms2_data: Optional MS2 data dictionary
        """
        self._features_df = features_df
        self._ms2_data = ms2_data

        self.feature_count_label.setText(f"Features: {len(features_df):,}")
        self.run_button.setEnabled(len(features_df) > 0)
        self.status_label.setText(f"Ready - {len(features_df)} features loaded")

        self._log(f"Loaded {len(features_df)} features from Feature Finding")

    def _on_run(self):
        """Run prioritization."""
        if self._features_df is None or len(self._features_df) == 0:
            QMessageBox.warning(self, "Error", "No features loaded")
            return

        # Update config
        self.config.prioritization.mdc.enabled = self.mdc_enable.isChecked()
        self.config.prioritization.kmd.enabled = self.kmd_enable.isChecked()
        self.config.prioritization.diagnostic_fragments.enabled = self.df_enable.isChecked()
        self.config.prioritization.suspect_screening.enabled = self.suspect_enable.isChecked()
        self.config.prioritization.suspect_screening.ppm_tolerance = self.ppm_spin.value()
        self.config.prioritization.scoring.min_score_threshold = self.min_score_spin.value()

        suspect_path = self.suspect_input.text().strip() or None

        # Setup worker
        self._worker = PrioritizationWorker(
            self._features_df,
            self._ms2_data,
            self.config.prioritization,
            suspect_path,
        )

        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_message.connect(self._log)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)

        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Running...")

        self._thread.start()

    def _on_progress(self, progress: float, message: str):
        """Handle progress update."""
        self.progress_bar.setValue(int(progress))
        self.status_label.setText(message)

    def _on_finished(self, result):
        """Handle completion."""
        self.run_button.setEnabled(True)

        if isinstance(result, Exception):
            self.status_label.setText(f"Error: {result}")
            QMessageBox.warning(self, "Error", str(result))
        else:
            self._result_df = result
            self.status_label.setText(f"Complete: {len(result)} prioritized features")
            self._update_results_table(result)
            self.export_button.setEnabled(True)
            self.report_button.setEnabled(True)
            self.dataset_button.setEnabled(True)

            self.prioritization_complete.emit(result)

    def _update_results_table(self, df: pd.DataFrame):
        """Update results table with DataFrame."""
        if len(df) == 0:
            self.results_table.setRowCount(0)
            return

        # Select columns to display (expanded for Week 3)
        display_cols = [
            "feature_id", "mz", "rt", "intensity",
            "pfas_score", "ml_score", "confidence_level",
            "evidence_count", "evidence_types",
            "quant_value", "quant_fold_error",
        ]
        display_cols = [c for c in display_cols if c in df.columns]

        self.results_table.setColumnCount(len(display_cols))
        self.results_table.setHorizontalHeaderLabels(display_cols)
        self.results_table.setRowCount(min(len(df), 1000))  # Limit rows

        for i, (_, row) in enumerate(df.head(1000).iterrows()):
            for j, col in enumerate(display_cols):
                value = row[col]
                if isinstance(value, float):
                    if col in ["mz", "rt"]:
                        text = f"{value:.4f}"
                    elif col == "ml_score":
                        text = f"{value:.3f}"
                    elif col == "quant_value":
                        text = f"{value:.1f}" if value else "-"
                    else:
                        text = f"{value:.2f}"
                elif value is None:
                    text = "-"
                else:
                    text = str(value)

                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                # Highlight high scores
                if col == "pfas_score" and isinstance(value, (int, float)) and value >= 5:
                    item.setBackground(Qt.darkGreen)
                elif col == "ml_score" and isinstance(value, (int, float)) and value >= 0.7:
                    item.setBackground(Qt.darkGreen)
                elif col == "confidence_level" and isinstance(value, (int, float)) and value <= 2:
                    item.setBackground(Qt.darkGreen)

                self.results_table.setItem(i, j, item)

    def _on_export(self):
        """Export results to file."""
        if self._result_df is None:
            return

        file_path, file_type = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "pfas_results.xlsx",
            "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)",
        )

        if not file_path:
            return

        try:
            if file_path.endswith(".csv"):
                self._result_df.to_csv(file_path, index=False)
            else:
                self._result_df.to_excel(file_path, index=False)

            self._log(f"Results exported to {file_path}")
            QMessageBox.information(self, "Export", f"Results saved to {file_path}")

        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def _on_generate_report(self):
        """Generate HTML report."""
        if self._result_df is None:
            return

        # TODO: Implement full HTML report generation
        QMessageBox.information(
            self,
            "Report",
            "HTML report generation would be implemented here.\n"
            "For now, use Export Results to save data.",
        )

    def _log(self, message: str):
        """Add message to log."""
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_build_dataset(self):
        """Build ML training dataset from results."""
        if self._result_df is None:
            QMessageBox.warning(
                self,
                "No Results",
                "먼저 Prioritization을 실행하세요.",
            )
            return
        
        QMessageBox.information(
            self,
            "Dataset Builder",
            "ML 탭 (🤖 ML Analysis)으로 이동하여 Dataset Builder를 사용하세요.\n\n"
            "Prioritization 결과가 자동으로 전달됩니다.",
        )

