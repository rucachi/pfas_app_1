"""
Visualization tab for ONFRA PFAS application.

Displays EIC, spectrum, and correlation visualizations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PySide6.QtCore import Qt, Signal
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
)
from PySide6.QtWebEngineWidgets import QWebEngineView

import pandas as pd

from ...core.config import PipelineConfig
from ...core.visualization import VizPayload

logger = logging.getLogger(__name__)


class VizTab(QWidget):
    """Visualization tab widget."""

    def __init__(self, config: PipelineConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._features_df: pd.DataFrame | None = None
        self._viz_payload: VizPayload | None = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Title
        title = QLabel("Visualization")
        title.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: #e94560;
            }
        """)
        layout.addWidget(title)

        # Controls
        controls_layout = QHBoxLayout()

        # Feature selector
        controls_layout.addWidget(QLabel("Feature:"))
        self.feature_combo = QComboBox()
        self.feature_combo.setMinimumWidth(200)
        self.feature_combo.currentIndexChanged.connect(self._on_feature_selected)
        controls_layout.addWidget(self.feature_combo)

        # View type
        controls_layout.addWidget(QLabel("View:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["EIC", "MS2 Spectrum", "Correlations", "Homologous Series"])
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        controls_layout.addWidget(self.view_combo)

        controls_layout.addStretch()

        # Refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self._refresh_view)
        controls_layout.addWidget(self.refresh_button)

        layout.addLayout(controls_layout)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left - Feature list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        list_group = QGroupBox("Features")
        list_layout = QVBoxLayout(list_group)

        self.feature_table = QTableWidget()
        self.feature_table.setColumnCount(4)
        self.feature_table.setHorizontalHeaderLabels(["ID", "m/z", "Score", "Evidence"])
        self.feature_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.feature_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.feature_table.setSelectionMode(QTableWidget.SingleSelection)
        self.feature_table.itemSelectionChanged.connect(self._on_table_selection)
        self.feature_table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                color: #1a1a2e;
                gridline-color: #d0d5dd;
                border: 1px solid #d0d5dd;
                border-radius: 6px;
            }
        """)
        list_layout.addWidget(self.feature_table)

        left_layout.addWidget(list_group)

        splitter.addWidget(left_panel)

        # Right - Visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        viz_group = QGroupBox("Plot")
        viz_layout = QVBoxLayout(viz_group)

        # Web view for Plotly charts
        self.web_view = QWebEngineView()
        self.web_view.setMinimumSize(400, 300)
        viz_layout.addWidget(self.web_view)

        right_layout.addWidget(viz_group)

        # Details
        details_group = QGroupBox("Details")
        details_layout = QVBoxLayout(details_group)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(150)
        self.details_text.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                color: #1a1a2e;
                font-family: Consolas, monospace;
                font-size: 12px;
                border: 1px solid #d0d5dd;
                border-radius: 6px;
            }
        """)
        details_layout.addWidget(self.details_text)

        right_layout.addWidget(details_group)

        splitter.addWidget(right_panel)

        # Set splitter sizes
        splitter.setSizes([300, 500])
        layout.addWidget(splitter)

        # Export buttons
        button_layout = QHBoxLayout()

        self.export_plot_button = QPushButton("Export Plot")
        self.export_plot_button.clicked.connect(self._export_plot)
        self.export_plot_button.setEnabled(False)
        button_layout.addWidget(self.export_plot_button)

        self.export_payload_button = QPushButton("Export Viz Payload")
        self.export_payload_button.clicked.connect(self._export_payload)
        self.export_payload_button.setEnabled(False)
        button_layout.addWidget(self.export_payload_button)

        button_layout.addStretch()

        layout.addLayout(button_layout)

        # Show placeholder
        self._show_placeholder()

    def _show_placeholder(self):
        """Show placeholder content."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {
                    background-color: #ffffff;
                    color: #5f6368;
                    font-family: 'Segoe UI', Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }
                .placeholder {
                    text-align: center;
                }
                h2 { color: #1a1a2e; }
            </style>
        </head>
        <body>
            <div class="placeholder">
                <h2>No Data Loaded</h2>
                <p>Run Feature Finding and Prioritization to visualize results.</p>
            </div>
        </body>
        </html>
        """
        self.web_view.setHtml(html)

    def set_data(self, features_df: pd.DataFrame, viz_payload: VizPayload | None = None):
        """
        Set visualization data.

        Args:
            features_df: Prioritized features DataFrame
            viz_payload: Optional pre-computed viz payload
        """
        self._features_df = features_df
        self._viz_payload = viz_payload

        # Update feature combo
        self.feature_combo.clear()
        if len(features_df) > 0:
            for _, row in features_df.head(100).iterrows():
                label = f"{row['feature_id']}: m/z {row['mz']:.4f}"
                if "pfas_score" in row:
                    label += f" (score: {row['pfas_score']:.1f})"
                self.feature_combo.addItem(label, row["feature_id"])

        # Update table
        self._update_feature_table(features_df)

        # Enable export
        self.export_plot_button.setEnabled(True)
        self.export_payload_button.setEnabled(viz_payload is not None)

        # Show first feature
        if len(features_df) > 0:
            self._on_feature_selected(0)

    def _update_feature_table(self, df: pd.DataFrame):
        """Update feature table."""
        self.feature_table.setRowCount(min(len(df), 500))

        for i, (_, row) in enumerate(df.head(500).iterrows()):
            self.feature_table.setItem(i, 0, QTableWidgetItem(str(int(row["feature_id"]))))
            self.feature_table.setItem(i, 1, QTableWidgetItem(f"{row['mz']:.4f}"))

            score = row.get("pfas_score", 0)
            self.feature_table.setItem(i, 2, QTableWidgetItem(f"{score:.1f}"))

            evidence = row.get("evidence_types", "")
            self.feature_table.setItem(i, 3, QTableWidgetItem(str(evidence)))

    def _on_feature_selected(self, index: int):
        """Handle feature selection from combo box."""
        if index < 0 or self._features_df is None:
            return

        feature_id = self.feature_combo.itemData(index)
        if feature_id is not None:
            self._show_feature(feature_id)

    def _on_table_selection(self):
        """Handle feature selection from table."""
        selected = self.feature_table.selectedItems()
        if not selected:
            return

        row = selected[0].row()
        feature_id_item = self.feature_table.item(row, 0)
        if feature_id_item:
            feature_id = int(feature_id_item.text())
            # Update combo box
            for i in range(self.feature_combo.count()):
                if self.feature_combo.itemData(i) == feature_id:
                    self.feature_combo.setCurrentIndex(i)
                    break

    def _on_view_changed(self, index: int):
        """Handle view type change."""
        self._refresh_view()

    def _show_feature(self, feature_id: int):
        """Show visualization for a feature."""
        if self._features_df is None:
            return

        feature_row = self._features_df[self._features_df["feature_id"] == feature_id]
        if len(feature_row) == 0:
            return

        feature = feature_row.iloc[0]
        view_type = self.view_combo.currentText()

        # Update details
        details = [
            f"Feature ID: {feature_id}",
            f"m/z: {feature['mz']:.6f}",
            f"RT: {feature['rt']:.2f} s ({feature['rt']/60:.2f} min)",
            f"Intensity: {feature['intensity']:.2e}",
        ]

        if "pfas_score" in feature:
            details.append(f"PFAS Score: {feature['pfas_score']:.2f}")

        if "evidence_details" in feature and feature["evidence_details"]:
            details.append(f"\nEvidence:\n{feature['evidence_details']}")

        self.details_text.setText("\n".join(details))

        # Show appropriate plot
        if view_type == "EIC":
            self._show_eic_plot(feature_id, feature)
        elif view_type == "MS2 Spectrum":
            self._show_spectrum_plot(feature_id)
        elif view_type == "Correlations":
            self._show_correlation_plot(feature_id)
        elif view_type == "Homologous Series":
            self._show_series_plot(feature_id)

    def _show_eic_plot(self, feature_id: int, feature):
        """Show EIC plot for a feature."""
        # Generate Plotly HTML
        mz = feature["mz"]
        rt = feature["rt"]

        # If we have EIC data in payload, use it
        eic_data = None
        if self._viz_payload:
            for eic in self._viz_payload.eics:
                if eic.feature_id == feature_id:
                    eic_data = eic
                    break

        if eic_data:
            # Use actual EIC data
            rt_values = list(eic_data.rt_values / 60)  # Convert to minutes
            int_values = list(eic_data.intensity_values)
        else:
            # Placeholder data
            import numpy as np
            rt_values = list(np.linspace(rt/60 - 2, rt/60 + 2, 100))
            int_values = list(np.exp(-((np.array(rt_values) - rt/60)**2) / 0.5) * 1e6)

        html = self._generate_plotly_html(
            "EIC Plot",
            [{"x": rt_values, "y": int_values, "type": "scatter", "name": f"m/z {mz:.4f}"}],
            "Retention Time (min)",
            "Intensity",
        )
        self.web_view.setHtml(html)

    def _show_spectrum_plot(self, feature_id: int):
        """Show MS2 spectrum plot."""
        spectrum_data = None
        if self._viz_payload:
            for spec in self._viz_payload.spectra:
                if spec.feature_id == feature_id:
                    spectrum_data = spec
                    break

        if spectrum_data:
            mz_values = list(spectrum_data.mz_values)
            int_values = list(spectrum_data.intensity_values)
        else:
            # Placeholder
            mz_values = [100, 150, 200, 250, 300]
            int_values = [50000, 100000, 30000, 80000, 20000]

        # Create bar chart for spectrum
        html = self._generate_plotly_html(
            "MS2 Spectrum",
            [{"x": mz_values, "y": int_values, "type": "bar", "name": "MS2"}],
            "m/z",
            "Intensity",
        )
        self.web_view.setHtml(html)

    def _show_correlation_plot(self, feature_id: int):
        """Show correlation network for a feature."""
        edges = []
        if self._viz_payload:
            for edge in self._viz_payload.correlations:
                if edge.source_feature_id == feature_id or edge.target_feature_id == feature_id:
                    edges.append(edge)

        # Display as table for now (network would need vis.js or similar)
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ background: #1e1e1e; color: #d4d4d4; font-family: Arial; padding: 20px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; border: 1px solid #404040; text-align: left; }}
                th {{ background: #2d2d2d; }}
            </style>
        </head>
        <body>
            <h3>Correlated Features (r ≥ {self.config.visualization.correlation_threshold})</h3>
            <table>
                <tr><th>Feature</th><th>Correlation</th></tr>
        """

        for edge in edges[:20]:  # Limit to 20
            other_id = edge.target_feature_id if edge.source_feature_id == feature_id else edge.source_feature_id
            html += f"<tr><td>Feature {other_id}</td><td>{edge.correlation:.3f}</td></tr>"

        if not edges:
            html += "<tr><td colspan='2'>No correlations found</td></tr>"

        html += """
            </table>
        </body>
        </html>
        """
        self.web_view.setHtml(html)

    def _show_series_plot(self, feature_id: int):
        """Show homologous series containing this feature."""
        series_info = []
        if self._viz_payload and self._viz_payload.homologous_series:
            for series in self._viz_payload.homologous_series:
                if feature_id in series.get("members", []):
                    series_info.append(series)

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ background: #1e1e1e; color: #d4d4d4; font-family: Arial; padding: 20px; }}
                .series {{ margin-bottom: 20px; padding: 15px; background: #2d2d2d; border-radius: 5px; }}
                h4 {{ margin: 0 0 10px 0; color: #e94560; }}
            </style>
        </head>
        <body>
            <h3>Homologous Series</h3>
        """

        if series_info:
            for series in series_info:
                html += f"""
                <div class="series">
                    <h4>{series.get('repeat_unit', 'Unknown')} Series</h4>
                    <p>Members: {len(series.get('members', []))}</p>
                    <p>KMD: {series.get('kmd_value', 0):.4f}</p>
                    <p>Member IDs: {', '.join(map(str, series.get('members', [])[:10]))}...</p>
                </div>
                """
        else:
            html += "<p>This feature is not part of any detected homologous series.</p>"

        html += "</body></html>"
        self.web_view.setHtml(html)

    def _generate_plotly_html(
        self,
        title: str,
        data: list[dict],
        xaxis_title: str,
        yaxis_title: str,
    ) -> str:
        """Generate Plotly HTML."""
        import json

        layout = {
            "title": {"text": title, "font": {"color": "#1a1a2e", "size": 16}},
            "paper_bgcolor": "#ffffff",
            "plot_bgcolor": "#f8f9fa",
            "xaxis": {
                "title": xaxis_title,
                "color": "#1a1a2e",
                "gridcolor": "#e8eaed",
            },
            "yaxis": {
                "title": yaxis_title,
                "color": "#1a1a2e",
                "gridcolor": "#e8eaed",
            },
            "font": {"color": "#1a1a2e"},
        }

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body style="margin:0; background:#ffffff;">
            <div id="plot" style="width:100%; height:100vh;"></div>
            <script>
                var data = {json.dumps(data)};
                var layout = {json.dumps(layout)};
                Plotly.newPlot('plot', data, layout, {{responsive: true}});
            </script>
        </body>
        </html>
        """
        return html

    def _refresh_view(self):
        """Refresh current view."""
        index = self.feature_combo.currentIndex()
        if index >= 0:
            self._on_feature_selected(index)

    def _export_plot(self):
        """Export current plot."""
        # Would need to use Plotly's export functionality
        QMessageBox.information(
            self,
            "Export Plot",
            "Right-click on the plot to save as PNG, or use the camera icon in the plot toolbar.",
        )

    def _export_payload(self):
        """Export viz payload to JSON."""
        if self._viz_payload is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Visualization Payload",
            "viz_payload.json",
            "JSON Files (*.json);;All Files (*)",
        )

        if file_path:
            try:
                self._viz_payload.to_json(file_path)
                QMessageBox.information(self, "Export", f"Payload saved to {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", str(e))
