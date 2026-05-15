"""
_style.py — Raycast-inspired dark theme for Florian napari plugins.

Provides:
  • STYLESHEET — apply to the root QWidget of each plugin
  • CollapsibleSection — a card that can be expanded/collapsed
  • make_separator — thin horizontal rule between sections
"""
from __future__ import annotations
from qtpy.QtCore import Qt, QPropertyAnimation, QEasingCurve
from qtpy.QtWidgets import (
    QFrame, QLabel, QPushButton, QSizePolicy,
    QToolButton, QVBoxLayout, QWidget,
)

# ─────────────────────────────────────────────────────────────────────────────
# Colour tokens
# ─────────────────────────────────────────────────────────────────────────────
BG_BASE      = "#111113"
BG_CARD      = "#1E1E21"
BG_INPUT     = "#28282C"
BG_HOVER     = "rgba(255,255,255,0.055)"
BORDER       = "#383840"
BORDER_FOCUS = "#7B61FF"
TEXT_PRIMARY = "#F0F0F5"
TEXT_SECOND  = "#8E8E9A"
TEXT_MUTED   = "#56565E"
ACCENT       = "#7B61FF"
ACCENT_LIGHT = "#9B82FF"
SUCCESS      = "#30D158"
WARNING      = "#FFD60A"
ERROR        = "#FF453A"

# ─────────────────────────────────────────────────────────────────────────────
# Master stylesheet
# ─────────────────────────────────────────────────────────────────────────────
STYLESHEET = f"""
/* ── Root ──────────────────────────────────────────────────────────── */
QWidget {{
    background-color: {BG_BASE};
    color: {TEXT_PRIMARY};
    font-size: 12px;
    font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
}}

/* ── Scroll area ────────────────────────────────────────────────────── */
QScrollArea  {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: transparent; width: 5px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 2px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: #555560; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

/* ── Group boxes → styled cards ─────────────────────────────────────── */
QGroupBox {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 20px;
    padding: 14px 12px 10px 12px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: 4px;
    color: {TEXT_MUTED};
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}}

/* ── Form labels ────────────────────────────────────────────────────── */
QLabel {{
    color: {TEXT_SECOND};
    background: transparent;
    font-size: 12px;
}}

/* ── Inputs ─────────────────────────────────────────────────────────── */
QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 5px 8px;
    color: {TEXT_PRIMARY};
    min-height: 24px;
    selection-background-color: {ACCENT};
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: #555560;
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {BORDER_FOCUS};
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 7px;
    selection-background-color: {ACCENT};
    color: {TEXT_PRIMARY};
    padding: 4px;
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: transparent; border: none; width: 16px;
}}

/* ── Checkboxes ─────────────────────────────────────────────────────── */
QCheckBox {{
    color: {TEXT_PRIMARY};
    spacing: 8px;
    font-size: 12px;
}}
QCheckBox::indicator {{
    width: 15px; height: 15px;
    border: 1.5px solid {BORDER};
    border-radius: 4px;
    background: {BG_INPUT};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QCheckBox::indicator:hover {{
    border-color: {ACCENT};
}}

/* ── Progress bar ───────────────────────────────────────────────────── */
QProgressBar {{
    background-color: {BG_INPUT};
    border: none;
    border-radius: 4px;
    max-height: 5px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {ACCENT}, stop:1 {ACCENT_LIGHT});
    border-radius: 4px;
}}

/* ── List widget ────────────────────────────────────────────────────── */
QListWidget {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 4px;
    color: {TEXT_PRIMARY};
    outline: none;
}}
QListWidget::item {{
    border-radius: 5px;
    padding: 5px 8px;
    color: {TEXT_PRIMARY};
}}
QListWidget::item:selected {{
    background-color: rgba(123, 97, 255, 0.22);
    color: {ACCENT_LIGHT};
}}
QListWidget::item:hover {{
    background-color: {BG_HOVER};
}}

/* ── Tooltips ───────────────────────────────────────────────────────── */
QToolTip {{
    background-color: #2A2A2F;
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 9px;
    font-size: 11px;
}}

/* ── Separator ──────────────────────────────────────────────────────── */
QFrame[frameShape="4"] {{   /* HLine */
    color: {BORDER};
    background: {BORDER};
    max-height: 1px;
    border: none;
}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Button helpers
# ─────────────────────────────────────────────────────────────────────────────

_BTN_PRIMARY = f"""
QPushButton {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {ACCENT}, stop:1 {ACCENT_LIGHT});
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.2px;
}}
QPushButton:hover {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #8B72FF, stop:1 #AC94FF);
}}
QPushButton:pressed {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #6A52E0, stop:1 #8A72E0);
}}
QPushButton:disabled {{
    background: #2C2C30;
    color: {TEXT_MUTED};
}}
"""

_BTN_SECONDARY = f"""
QPushButton {{
    background: transparent;
    color: {TEXT_SECOND};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 500;
}}
QPushButton:hover {{
    background: {BG_HOVER};
    color: {TEXT_PRIMARY};
    border-color: #555560;
}}
QPushButton:pressed {{
    background: rgba(255,255,255,0.04);
}}
QPushButton:disabled {{
    color: {TEXT_MUTED};
    border-color: #2A2A2E;
}}
"""

_BTN_GHOST = f"""
QPushButton {{
    background: transparent;
    color: {ACCENT_LIGHT};
    border: none;
    padding: 4px 8px;
    font-size: 11px;
    font-weight: 500;
    text-decoration: underline;
    text-decoration-color: transparent;
}}
QPushButton:hover {{
    color: #FFFFFF;
}}
"""

_TOGGLE_BTN = f"""
QToolButton {{
    background: transparent;
    color: {TEXT_MUTED};
    border: none;
    padding: 4px 0px;
    font-size: 11px;
    font-weight: 700;
    text-align: left;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}}
QToolButton:hover {{
    color: {TEXT_SECOND};
}}
QToolButton:checked {{
    color: {TEXT_SECOND};
}}
"""


def style_primary_btn(btn: QPushButton) -> None:
    btn.setStyleSheet(_BTN_PRIMARY)


def style_secondary_btn(btn: QPushButton) -> None:
    btn.setStyleSheet(_BTN_SECONDARY)


def style_ghost_btn(btn: QPushButton) -> None:
    btn.setStyleSheet(_BTN_GHOST)


# ─────────────────────────────────────────────────────────────────────────────
# Result / status label helpers
# ─────────────────────────────────────────────────────────────────────────────

_RESULT_BASE = f"""
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    font-weight: 600;
"""


def set_result_success(label: QLabel, text: str) -> None:
    label.setText(text)
    label.setStyleSheet(_RESULT_BASE + f"color: {SUCCESS};")
    label.setVisible(True)


def set_result_error(label: QLabel, text: str) -> None:
    label.setText(text)
    label.setStyleSheet(_RESULT_BASE + f"color: {ERROR};")
    label.setVisible(True)


def set_result_info(label: QLabel, text: str) -> None:
    label.setText(text)
    label.setStyleSheet(_RESULT_BASE + f"color: {TEXT_SECOND};")
    label.setVisible(True)


# ─────────────────────────────────────────────────────────────────────────────
# Separator
# ─────────────────────────────────────────────────────────────────────────────

def make_separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Plain)
    line.setFixedHeight(1)
    line.setStyleSheet(f"background: {BORDER}; border: none;")
    return line


# ─────────────────────────────────────────────────────────────────────────────
# Plugin header
# ─────────────────────────────────────────────────────────────────────────────

def make_plugin_header(icon: str, title: str, subtitle: str) -> QWidget:
    """Returns a styled header widget with icon, title, and one-liner."""
    w = QWidget()
    w.setStyleSheet(f"""
        QWidget {{ background: transparent; }}
        QLabel#title {{
            color: {TEXT_PRIMARY};
            font-size: 16px;
            font-weight: 700;
            background: transparent;
        }}
        QLabel#sub {{
            color: {TEXT_MUTED};
            font-size: 11px;
            background: transparent;
        }}
        QLabel#icon {{
            font-size: 24px;
            background: transparent;
        }}
    """)
    layout = QVBoxLayout(w)
    layout.setContentsMargins(4, 8, 4, 4)
    layout.setSpacing(2)

    icon_lbl = QLabel(icon)
    icon_lbl.setObjectName("icon")
    icon_lbl.setAlignment(Qt.AlignLeft)

    title_lbl = QLabel(title)
    title_lbl.setObjectName("title")

    sub_lbl = QLabel(subtitle)
    sub_lbl.setObjectName("sub")
    sub_lbl.setWordWrap(True)

    layout.addWidget(icon_lbl)
    layout.addWidget(title_lbl)
    layout.addWidget(sub_lbl)
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Collapsible section
# ─────────────────────────────────────────────────────────────────────────────

class CollapsibleSection(QWidget):
    """A card section that can be toggled open/closed.

    Usage::
        sec = CollapsibleSection("Advanced Options")
        sec.layout().addWidget(my_form)
        outer_layout.addWidget(sec)
    """

    def __init__(self, title: str, collapsed: bool = True, parent=None):
        super().__init__(parent)
        self._title = title.upper()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Toggle button
        self._toggle = QToolButton()
        self._toggle.setCheckable(True)
        self._toggle.setChecked(not collapsed)
        self._toggle.setStyleSheet(_TOGGLE_BTN)
        self._toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._toggle.setFixedHeight(28)
        self._update_label()
        outer.addWidget(self._toggle)

        # Content container
        self._content = QWidget()
        self._content.setStyleSheet(f"""
            QWidget {{
                background: {BG_CARD};
                border: 1px solid {BORDER};
                border-radius: 10px;
                padding: 4px;
            }}
        """)
        self._inner_layout = QVBoxLayout(self._content)
        self._inner_layout.setContentsMargins(10, 10, 10, 10)
        self._inner_layout.setSpacing(8)
        self._content.setVisible(not collapsed)
        outer.addWidget(self._content)

        self._toggle.toggled.connect(self._on_toggled)

    def _update_label(self):
        arrow = "▾" if self._toggle.isChecked() else "▸"
        self._toggle.setText(f"  {arrow}  {self._title}")

    def _on_toggled(self, checked: bool):
        self._content.setVisible(checked)
        self._update_label()

    def inner_layout(self):
        return self._inner_layout

    def addWidget(self, w: QWidget):  # noqa: N802 — Qt naming
        self._inner_layout.addWidget(w)

    def addLayout(self, lay):  # noqa: N802
        self._inner_layout.addLayout(lay)
