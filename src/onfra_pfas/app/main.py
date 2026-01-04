"""
Main application entry point for ONFRA PFAS.

Launches splash screen and main window with tabs.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QStatusBar,
    QLabel,
    QMessageBox,
)

from .splash import SplashScreen
from .tabs.feature_tab import FeatureTab
from .tabs.prioritize_tab import PrioritizeTab
from .tabs.viz_tab import VizTab
from .tabs.ml_tab import MLTab

from ..core.config import PipelineConfig, get_default_config
from ..core.utils import resource_path
from ..core.logging import setup_root_logger
from ..core.backend import probe_gpu
from .. import __version__

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config

        self._setup_window()
        self._setup_ui()
        self._connect_signals()

    def _setup_window(self):
        """Setup window properties."""
        self.setWindowTitle(f"ONFRA PFAS v{__version__}")
        self.setMinimumSize(1200, 800)

        # Set window icon
        icon_path = resource_path("assets/onfra_logo.png")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Brighter modern theme stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
            }
            QWidget {
                background-color: #f0f2f5;
                color: #1a1a2e;
                font-size: 13px;
            }
            QTabWidget::pane {
                border: 1px solid #d0d5dd;
                background-color: #ffffff;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #e8eaed;
                color: #5f6368;
                padding: 12px 24px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                color: #1a1a2e;
                border-bottom: 3px solid #e94560;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #d8dade;
            }
            QGroupBox {
                border: 1px solid #d0d5dd;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: #ffffff;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: #e94560;
                font-size: 14px;
            }
            QLabel {
                color: #1a1a2e;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #ffffff;
                border: 1px solid #d0d5dd;
                border-radius: 6px;
                padding: 8px 12px;
                color: #1a1a2e;
                font-size: 13px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 2px solid #e94560;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #1a1a2e;
                selection-background-color: #e94560;
                selection-color: #ffffff;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #d0d5dd;
                border-radius: 6px;
                padding: 10px 18px;
                color: #1a1a2e;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #f8f9fa;
                border-color: #e94560;
                color: #e94560;
            }
            QPushButton:pressed {
                background-color: #e8eaed;
            }
            QProgressBar {
                background-color: #e8eaed;
                border: none;
                border-radius: 6px;
                text-align: center;
                color: #1a1a2e;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #e94560,
                    stop: 1 #ff6b8a
                );
                border-radius: 6px;
            }
            QScrollBar:vertical {
                background-color: #f0f2f5;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #c4c7cc;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #a8acb3;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            QStatusBar {
                background-color: #ffffff;
                color: #5f6368;
                border-top: 1px solid #d0d5dd;
            }
            QCheckBox {
                color: #1a1a2e;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid #d0d5dd;
                background-color: #ffffff;
            }
            QCheckBox::indicator:checked {
                background-color: #e94560;
                border-color: #e94560;
            }
        """)

    def _setup_ui(self):
        """Setup the UI."""
        # Central widget with tabs
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        # Tab widget
        self.tabs = QTabWidget()

        # Create tabs
        self.feature_tab = FeatureTab(self.config)
        self.prioritize_tab = PrioritizeTab(self.config)
        self.viz_tab = VizTab(self.config)
        self.ml_tab = MLTab(self.config)

        self.tabs.addTab(self.feature_tab, "🔬 Feature Finding")
        self.tabs.addTab(self.prioritize_tab, "🎯 Prioritization")
        self.tabs.addTab(self.viz_tab, "📊 Visualization")
        self.tabs.addTab(self.ml_tab, "🤖 ML Analysis")

        layout.addWidget(self.tabs)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # GPU status in status bar
        gpu_info = probe_gpu()
        if gpu_info.available:
            gpu_text = f"GPU: {gpu_info.device_name}"
        else:
            gpu_text = "GPU: Not available (CPU mode)"

        self.gpu_label = QLabel(gpu_text)
        self.status_bar.addPermanentWidget(self.gpu_label)

        self.status_bar.showMessage("Ready")

    def _connect_signals(self):
        """Connect signals between tabs."""
        # When features are found, pass to prioritization tab
        self.feature_tab.features_found.connect(self._on_features_found)

        # When prioritization completes, pass to viz tab and ml tab
        self.prioritize_tab.prioritization_complete.connect(self._on_prioritization_complete)
        self.prioritize_tab.prioritization_complete.connect(self._on_prioritization_for_ml)

    def _on_features_found(self, result):
        """Handle features found from feature finding."""
        from ..core.featurefinding import FeatureResult

        if isinstance(result, FeatureResult):
            self.prioritize_tab.set_features(result.features_df)
            self.status_bar.showMessage(f"Found {len(result.features_df)} features")

            # Switch to prioritization tab
            self.tabs.setCurrentIndex(1)

    def _on_prioritization_complete(self, result_df):
        """Handle prioritization completion."""
        import pandas as pd

        if isinstance(result_df, pd.DataFrame):
            self.viz_tab.set_data(result_df)
            self.status_bar.showMessage(f"Prioritized {len(result_df)} features")

            # Switch to visualization tab
            self.tabs.setCurrentIndex(2)

    def _on_prioritization_for_ml(self, result_df):
        """Handle prioritization completion for ML tab."""
        import pandas as pd

        if isinstance(result_df, pd.DataFrame):
            self.ml_tab.set_features(result_df)

    def closeEvent(self, event):
        """Handle window close."""
        # Could add confirmation dialog here
        event.accept()


def loading_generator():
    """Generator for splash screen loading progress."""
    import time

    yield 10, "Initializing logging..."
    time.sleep(0.8)
    setup_root_logger()

    yield 30, "Loading configuration..."
    time.sleep(0.8)
    # Config is loaded in main()

    yield 50, "Checking GPU availability..."
    time.sleep(0.8)
    gpu_info = probe_gpu()
    if gpu_info.available:
        yield 60, f"GPU found: {gpu_info.device_name}"
    else:
        yield 60, "GPU not available, using CPU"
    time.sleep(0.8)

    yield 80, "Loading UI components..."
    time.sleep(0.8)

    yield 100, "Ready!"
    time.sleep(1.0)


def main():
    """Main entry point."""
    # Create application
    app = QApplication(sys.argv)

    # Set application info
    app.setApplicationName("ONFRA PFAS")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("ONFRA")

    # Set font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Show splash screen
    splash = SplashScreen()
    splash.show()
    app.processEvents()

    # Load with progress
    for progress, message in loading_generator():
        splash.set_progress(progress, message)
        app.processEvents()

    # Load config
    config = get_default_config()

    # Create main window
    main_window = MainWindow(config)

    # Close splash and show main window
    splash.close()
    main_window.show()

    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
