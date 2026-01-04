"""
ML Analysis tab for ONFRA PFAS application.

Provides UI for ML-based scoring, MS2 similarity search, and dataset building.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal, QRunnable, QThreadPool, QObject
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QComboBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QTextEdit,
    QFileDialog,
    QSpinBox,
    QProgressBar,
    QFormLayout,
    QLineEdit,
    QCheckBox,
)

import pandas as pd
import numpy as np

from ...core.config import PipelineConfig
from ...core.ml_inference import PFASClassifier, MS2Embedder, add_ml_scores_to_dataframe
from ...core.dataset_builder import DatasetBuilder, DatasetConfig, build_dataset_from_results
from ...core.confidence import add_confidence_to_dataframe

logger = logging.getLogger(__name__)


class WorkerSignals(QObject):
    """Signals for worker threads."""
    
    progress = Signal(float, str)
    finished = Signal(object)
    error = Signal(str)


class MLScoringWorker(QRunnable):
    """Worker for ML scoring in background."""
    
    def __init__(self, features_df: pd.DataFrame, classifier: PFASClassifier):
        super().__init__()
        self.features_df = features_df
        self.classifier = classifier
        self.signals = WorkerSignals()
    
    def run(self):
        try:
            self.signals.progress.emit(0.1, "ML 스코어링 시작...")
            
            result_df = add_ml_scores_to_dataframe(
                self.features_df, 
                classifier=self.classifier
            )
            
            self.signals.progress.emit(0.8, "Confidence 레벨 계산...")
            result_df = add_confidence_to_dataframe(result_df)
            
            self.signals.progress.emit(1.0, "완료")
            self.signals.finished.emit(result_df)
            
        except Exception as e:
            logger.exception("ML scoring failed")
            self.signals.error.emit(str(e))


class DatasetBuildWorker(QRunnable):
    """Worker for dataset building."""
    
    def __init__(self, features_df: pd.DataFrame, output_dir: Path, config_kwargs: dict):
        super().__init__()
        self.features_df = features_df
        self.output_dir = output_dir
        self.config_kwargs = config_kwargs
        self.signals = WorkerSignals()
    
    def run(self):
        try:
            self.signals.progress.emit(0.1, "데이터셋 빌드 시작...")
            
            output_path = build_dataset_from_results(
                features_df=self.features_df,
                output_dir=self.output_dir,
                **self.config_kwargs,
            )
            
            self.signals.progress.emit(1.0, "완료")
            self.signals.finished.emit(output_path)
            
        except Exception as e:
            logger.exception("Dataset build failed")
            self.signals.error.emit(str(e))


class MLTab(QWidget):
    """ML Analysis tab widget."""
    
    analysis_complete = Signal(object)  # Emits updated DataFrame
    
    def __init__(self, config: PipelineConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._features_df: pd.DataFrame | None = None
        self._result_df: pd.DataFrame | None = None
        self._classifier = PFASClassifier()
        self._embedder = MS2Embedder()
        self._thread_pool = QThreadPool()
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("ML Analysis")
        title.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: #e94560;
            }
        """)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("ML 기반 PFAS 스코어링, MS2 유사도 검색, 딥러닝 학습용 데이터셋 생성")
        desc.setStyleSheet("color: #808080;")
        layout.addWidget(desc)
        
        # Main splitter
        splitter = QSplitter(Qt.Vertical)
        
        # Top panel - ML Scoring
        top_widget = self._create_scoring_section()
        splitter.addWidget(top_widget)
        
        # Bottom panel - Dataset Builder
        bottom_widget = self._create_dataset_section()
        splitter.addWidget(bottom_widget)
        
        splitter.setSizes([300, 200])
        layout.addWidget(splitter)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #404040;
                border-radius: 3px;
                text-align: center;
                background: #2d2d2d;
            }
            QProgressBar::chunk {
                background-color: #e94560;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Status
        self.status_label = QLabel("데이터를 로드하세요")
        self.status_label.setStyleSheet("color: #808080;")
        layout.addWidget(self.status_label)
    
    def _create_scoring_section(self) -> QWidget:
        """Create ML Scoring section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        group = QGroupBox("ML Scoring & Confidence Level")
        group_layout = QVBoxLayout(group)
        
        # Info
        info = QLabel("PFAS 확률 예측, 불확실성 추정, Schymanski 신뢰도 계산")
        info.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        group_layout.addWidget(info)
        
        # Controls
        controls = QHBoxLayout()
        
        self.run_ml_button = QPushButton("Run ML Scoring")
        self.run_ml_button.setEnabled(False)
        self.run_ml_button.clicked.connect(self._on_run_ml_scoring)
        self.run_ml_button.setStyleSheet("""
            QPushButton {
                background-color: #e94560;
                color: white;
                padding: 8px 20px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: #404040;
                color: #808080;
            }
            QPushButton:hover:enabled {
                background-color: #ff5577;
            }
        """)
        controls.addWidget(self.run_ml_button)
        
        controls.addStretch()
        
        self.export_button = QPushButton("Export Results")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        controls.addWidget(self.export_button)
        
        group_layout.addLayout(controls)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "ID", "m/z", "ML Score", "Uncertainty", "Confidence", "Rationale"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                color: #1a1a2e;
                gridline-color: #d0d5dd;
                border: 1px solid #d0d5dd;
                border-radius: 6px;
            }
            QTableWidget::item:selected {
                background-color: #e94560;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #e8eaed;
                color: #1a1a2e;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
        """)
        group_layout.addWidget(self.results_table)
        
        layout.addWidget(group)
        return widget
    
    def _create_dataset_section(self) -> QWidget:
        """Create Dataset Builder section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        group = QGroupBox("Dataset Builder (ML Training)")
        group_layout = QVBoxLayout(group)
        
        # Info
        info = QLabel("딥러닝 모델 학습을 위한 Parquet, MGF, NPZ 데이터셋 생성")
        info.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        group_layout.addWidget(info)
        
        # Configuration form
        form = QFormLayout()
        
        # Output directory
        dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit("./datasets/onfra_pfas_training")
        dir_layout.addWidget(self.output_dir_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_output_dir)
        dir_layout.addWidget(browse_btn)
        form.addRow("Output:", dir_layout)
        
        # EIC Length
        self.eic_length_spin = QSpinBox()
        self.eic_length_spin.setRange(64, 512)
        self.eic_length_spin.setValue(256)
        form.addRow("EIC Length:", self.eic_length_spin)
        
        # MS2 Bins
        self.ms2_bins_spin = QSpinBox()
        self.ms2_bins_spin.setRange(500, 4000)
        self.ms2_bins_spin.setValue(2000)
        form.addRow("MS2 Bins:", self.ms2_bins_spin)
        
        # Options
        options_layout = QHBoxLayout()
        self.include_mgf_check = QCheckBox("MGF")
        self.include_mgf_check.setChecked(True)
        options_layout.addWidget(self.include_mgf_check)
        
        self.include_binned_check = QCheckBox("Binned NPZ")
        self.include_binned_check.setChecked(True)
        options_layout.addWidget(self.include_binned_check)
        
        options_layout.addStretch()
        form.addRow("Include:", options_layout)
        
        group_layout.addLayout(form)
        
        # Build button
        button_layout = QHBoxLayout()
        
        self.build_button = QPushButton("Build Dataset")
        self.build_button.setEnabled(False)
        self.build_button.clicked.connect(self._on_build_dataset)
        self.build_button.setStyleSheet("""
            QPushButton {
                background-color: #4a90d9;
                color: white;
                padding: 8px 20px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: #404040;
                color: #808080;
            }
            QPushButton:hover:enabled {
                background-color: #5aa0e9;
            }
        """)
        button_layout.addWidget(self.build_button)
        
        button_layout.addStretch()
        group_layout.addLayout(button_layout)
        
        layout.addWidget(group)
        return widget
    
    def set_features(self, features_df: pd.DataFrame):
        """
        Set features from Prioritization tab.
        
        Args:
            features_df: Prioritized features DataFrame
        """
        self._features_df = features_df
        
        # Enable buttons
        self.run_ml_button.setEnabled(len(features_df) > 0)
        self.build_button.setEnabled(len(features_df) > 0)
        
        self.status_label.setText(f"{len(features_df)}개 피처 로드됨")
        self._log(f"Loaded {len(features_df)} features")
    
    def _on_run_ml_scoring(self):
        """Run ML scoring."""
        if self._features_df is None or len(self._features_df) == 0:
            return
        
        self.run_ml_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        worker = MLScoringWorker(self._features_df, self._classifier)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_ml_finished)
        worker.signals.error.connect(self._on_error)
        
        self._thread_pool.start(worker)
    
    def _on_progress(self, value: float, message: str):
        """Handle progress update."""
        self.progress_bar.setValue(int(value * 100))
        self.status_label.setText(message)
    
    def _on_ml_finished(self, result_df: pd.DataFrame):
        """Handle ML scoring completion."""
        self._result_df = result_df
        self.run_ml_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Update table
        self._update_results_table(result_df)
        
        self.status_label.setText(f"ML 스코어링 완료: {len(result_df)}개 피처")
        
        # Emit signal
        self.analysis_complete.emit(result_df)
    
    def _update_results_table(self, df: pd.DataFrame):
        """Update results table."""
        self.results_table.setRowCount(min(len(df), 500))
        
        for i, (_, row) in enumerate(df.head(500).iterrows()):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(int(row.get("feature_id", i)))))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{row.get('mz', 0):.4f}"))
            
            ml_score = row.get("ml_score", 0)
            score_item = QTableWidgetItem(f"{ml_score:.3f}")
            if ml_score >= 0.8:
                score_item.setForeground(Qt.green)
            elif ml_score >= 0.5:
                score_item.setForeground(Qt.yellow)
            self.results_table.setItem(i, 2, score_item)
            
            uncertainty = row.get("ml_uncertainty", 0)
            self.results_table.setItem(i, 3, QTableWidgetItem(f"±{uncertainty:.2f}"))
            
            confidence = row.get("confidence_level", 5)
            conf_item = QTableWidgetItem(f"Level {confidence}")
            if confidence <= 2:
                conf_item.setForeground(Qt.green)
            elif confidence <= 3:
                conf_item.setForeground(Qt.yellow)
            self.results_table.setItem(i, 4, conf_item)
            
            rationale = row.get("confidence_rationale", "")
            self.results_table.setItem(i, 5, QTableWidgetItem(str(rationale)[:50]))
    
    def _on_export(self):
        """Export results."""
        if self._result_df is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export ML Results",
            "ml_results.xlsx",
            "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)",
        )
        
        if file_path:
            try:
                if file_path.endswith(".csv"):
                    self._result_df.to_csv(file_path, index=False)
                else:
                    self._result_df.to_excel(file_path, index=False)
                self.status_label.setText(f"저장됨: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", str(e))
    
    def _browse_output_dir(self):
        """Browse for output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", str(Path.cwd())
        )
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def _on_build_dataset(self):
        """Build ML training dataset."""
        if self._features_df is None or len(self._features_df) == 0:
            return
        
        output_dir = Path(self.output_dir_edit.text())
        
        config_kwargs = {
            "eic_length": self.eic_length_spin.value(),
            "ms2_bins": self.ms2_bins_spin.value(),
            "include_mgf": self.include_mgf_check.isChecked(),
            "include_binned": self.include_binned_check.isChecked(),
        }
        
        self.build_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        worker = DatasetBuildWorker(self._features_df, output_dir, config_kwargs)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_dataset_finished)
        worker.signals.error.connect(self._on_error)
        
        self._thread_pool.start(worker)
    
    def _on_dataset_finished(self, output_path: Path):
        """Handle dataset build completion."""
        self.build_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.status_label.setText(f"데이터셋 생성 완료: {output_path}")
        
        QMessageBox.information(
            self,
            "Dataset Built",
            f"ML 훈련 데이터셋이 생성되었습니다:\n{output_path}\n\n"
            "- features.parquet (메타데이터)\n"
            "- eic/ (EIC 시계열)\n"
            "- ms2/ (MS2 스펙트럼)\n"
            "- labels.json (레이블)",
        )
    
    def _on_error(self, message: str):
        """Handle error."""
        self.run_ml_button.setEnabled(True)
        self.build_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.status_label.setText(f"오류: {message}")
        QMessageBox.warning(self, "Error", message)
    
    def _log(self, message: str):
        """Log message."""
        logger.info(message)
