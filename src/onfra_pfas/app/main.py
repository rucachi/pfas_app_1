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

        # Dark theme stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            QWidget {
                background-color: #1a1a2e;
                color: #d4d4d4;
            }
            QTabWidget::pane {
                border: 1px solid #404040;
                background-color: #16213e;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #2d2d44;
                color: #a0a0a0;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #16213e;
                color: #ffffff;
                border-bottom: 2px solid #e94560;
            }
            QTabBar::tab:hover {
                background-color: #3d3d5c;
            }
            QGroupBox {
                border: 1px solid #404040;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #e94560;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #2d2d44;
                border: 1px solid #404040;
                border-radius: 3px;
                padding: 5px;
                color: #d4d4d4;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border-color: #e94560;
            }
            QPushButton {
                background-color: #2d2d44;
                border: 1px solid #404040;
                border-radius: 5px;
                padding: 8px 15px;
                color: #d4d4d4;
            }
            QPushButton:hover {
                background-color: #3d3d5c;
                border-color: #e94560;
            }
            QPushButton:pressed {
                background-color: #4d4d6c;
            }
            QProgressBar {
                background-color: #2d2d44;
                border: none;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #e94560,
                    stop: 1 #ff6b6b
                );
                border-radius: 3px;
            }
            QScrollBar:vertical {
                background-color: #2d2d44;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #4d4d6c;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            QStatusBar {
                background-color: #0f3460;
                color: #a0a0a0;
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

        self.tabs.addTab(self.feature_tab, "🔬 Feature Finding")
        self.tabs.addTab(self.prioritize_tab, "🎯 Prioritization")
        self.tabs.addTab(self.viz_tab, "📊 Visualization")

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

        # When prioritization completes, pass to viz tab
        self.prioritize_tab.prioritization_complete.connect(self._on_prioritization_complete)

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
