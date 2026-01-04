"""
Splash screen for ONFRA PFAS application.

Displays logo, app name, developer info, and version.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QFont, QColor, QPainter, QLinearGradient
from PySide6.QtWidgets import (
    QSplashScreen,
    QApplication,
    QLabel,
    QVBoxLayout,
    QWidget,
    QProgressBar,
)

from ..core.utils import resource_path
from .. import __version__


class SplashScreen(QSplashScreen):
    """Splash screen with logo and loading progress."""

    def __init__(self):
        # Create a blank pixmap first
        super().__init__()

        self.setFixedSize(600, 400)

        # Setup UI
        self._setup_ui()

    def _setup_ui(self):
        """Setup the splash screen UI."""
        # Create central widget
        self.container = QWidget(self)
        self.container.setGeometry(0, 0, 600, 400)
        self.container.setStyleSheet("""
            QWidget {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #1a1a2e,
                    stop: 0.5 #16213e,
                    stop: 1 #0f3460
                );
            }
        """)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(15)

        # Organization name (instead of logo)
        self.org_label = QLabel("(재)국제도시물정보과학연구원")
        self.org_label.setAlignment(Qt.AlignCenter)
        self.org_label.setStyleSheet("""
            QLabel {
                color: #e94560;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.org_label)

        # App name
        self.title_label = QLabel("ONFRA PFAS")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 32px;
                font-weight: bold;
                letter-spacing: 2px;
            }
        """)
        layout.addWidget(self.title_label)

        # Slogan
        self.slogan_label = QLabel("Non-Target Screening Analysis Platform")
        self.slogan_label.setAlignment(Qt.AlignCenter)
        self.slogan_label.setStyleSheet("""
            QLabel {
                color: #a0a0a0;
                font-size: 14px;
                font-style: italic;
            }
        """)
        layout.addWidget(self.slogan_label)

        layout.addStretch()

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #2d2d44;
                border: none;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #e94560,
                    stop: 1 #ff6b6b
                );
                border-radius: 2px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Status message
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #808080;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.status_label)

        layout.addStretch()

        # Developer info
        self.dev_label = QLabel("개발 : 김태형 / 연락처 : 010-9411-7143")
        self.dev_label.setAlignment(Qt.AlignCenter)
        self.dev_label.setStyleSheet("""
            QLabel {
                color: #606060;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.dev_label)

        # Version
        self.version_label = QLabel(f"Version {__version__}")
        self.version_label.setAlignment(Qt.AlignCenter)
        self.version_label.setStyleSheet("""
            QLabel {
                color: #505050;
                font-size: 10px;
            }
        """)
        layout.addWidget(self.version_label)

    def set_progress(self, value: int, message: str = ""):
        """Update progress bar and status message."""
        self.progress_bar.setValue(value)
        if message:
            self.status_label.setText(message)
        QApplication.processEvents()

    def paintEvent(self, event):
        """Custom paint for rounded corners."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw rounded rectangle background
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor("#1a1a2e"))
        gradient.setColorAt(0.5, QColor("#16213e"))
        gradient.setColorAt(1, QColor("#0f3460"))

        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 15, 15)


def show_splash_and_load(app: QApplication, load_function) -> None:
    """
    Show splash screen while loading.

    Args:
        app: QApplication instance
        load_function: Function that yields (progress, message) tuples
    """
    splash = SplashScreen()
    splash.show()
    app.processEvents()

    try:
        for progress, message in load_function():
            splash.set_progress(progress, message)
    except Exception as e:
        splash.set_progress(100, f"Error: {e}")

    splash.close()
