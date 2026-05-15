# florian-tools — PanOCT Plugin Suite

> **A modular napari plugin suite for retinal tumor volumetry using panretinal OCT (PanOCT).**

---

## Overview

`florian-tools` is a meta-package that installs and registers four focused napari plugins. Each plugin handles one stage of the volumetric measurement pipeline, so they can be used independently or chained together.

```
┌──────────────────────────────────────────────────────────────────┐
│                    Recommended Workflow                           │
│                                                                  │
│  [Optional]                                                       │
│  Crop Z-Slices            ──►  Remove edge artifacts             │
│           │                                                       │
│           ▼                                                       │
│  [Optional]                                                       │
│  PanOCT Curve Correction  ──►  Geometric pre-processing          │
│           │                                                       │
│           ▼                                                       │
│  Generate Tumor Mask (Linear)                                     │
│          or                    ──►  Tumor detection & volume      │
│  Generate Tumor Mask (Quadratic)                                  │
│           │                                                       │
│           ▼                                                       │
│  [Optional]                                                       │
│  Generate 3D Tumor Render  ──►  Surface mesh & 3D visualisation  │
└──────────────────────────────────────────────────────────────────┘
```

### The Four Plugins

| Plugin | Napari menu entry | When to use |
|---|---|---|
| `florian-z-slice-crop` | **Crop Z-Slices** | Remove fundus projection images or edge artifacts from 3D volumes |
| `florian-panOCT-curve-correction` | **PanOCT Curve Correction** | Optional pre-processing, if your system has fan-beam distortion |
| `florian-linear-tumor-vol` | **Generate Tumor Mask (Linear)** | Flat/elevated tumors; simpler, faster |
| `florian-quadratic-tumor-vol` | **Generate Tumor Mask (Quadratic)** | Dome-shaped tumors that follow the globe curvature |
| `florian-tumor-3d-render` | **Generate 3D Tumor Render** | After either detection plugin |

---

## Installation

```bash
conda activate napari-env
pip install -e ./florian-tools
```

This automatically installs the four sub-packages as editable dependencies.

---

## Shared UI System (`_style.py`)

Every florian plugin imports a `_style.py` module that provides a consistent **Raycast-inspired dark theme**. This file is copied verbatim into each plugin's source package so there are no cross-package imports.

### Design Tokens

| Token | Hex | Role |
|---|---|---|
| `BG_BASE` | `#111113` | Widget root background |
| `BG_CARD` | `#1E1E21` | Group box "card" surfaces |
| `BG_INPUT` | `#28282C` | Input fields, list widgets |
| `BORDER` | `#383840` | Card and input borders |
| `BORDER_FOCUS` | `#7B61FF` | Focused input ring |
| `TEXT_PRIMARY` | `#F0F0F5` | Body text, labels |
| `TEXT_SECOND` | `#8E8E9A` | Form row labels, secondary text |
| `TEXT_MUTED` | `#56565E` | Section titles, hints |
| `ACCENT` | `#7B61FF` | Primary buttons, checkboxes, focus |
| `ACCENT_LIGHT` | `#9B82FF` | Hover/gradient end for primary buttons |
| `SUCCESS` | `#30D158` | Result labels on success |
| `WARNING` | `#FFD60A` | Non-critical warnings |
| `ERROR` | `#FF453A` | Result labels on error |

### Component Library

#### `STYLESHEET`
A Qt Style Sheet string. Apply it to the root `QWidget` of any new widget:

```python
from florian_myplugin._style import STYLESHEET

class MyWidget(QWidget):
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.setStyleSheet(STYLESHEET)
```

#### `make_plugin_header(icon, title, subtitle) → QWidget`
Returns a header block with an emoji icon, bold title, and a one-line description.

```python
lay.addWidget(make_plugin_header("📐", "My Plugin", "One sentence description."))
```

#### `make_separator() → QFrame`
Returns a 1 px horizontal rule using the border colour.

#### `style_primary_btn(btn)` / `style_secondary_btn(btn)` / `style_ghost_btn(btn)`
Apply the primary (purple gradient), secondary (ghost outline), or ghost (text-only link) styles to a `QPushButton`.

- **Primary** — the main action button. One per widget.
- **Secondary** — auxiliary actions (e.g. "Recalculate", "Preview").
- **Ghost** — destructive or navigational links.

#### `CollapsibleSection(title, collapsed=True) → QWidget`
A card that can be toggled open/closed with a `▸ SECTION TITLE` toggle button.  
Use for **Advanced Options** that most users never need to change.

```python
adv = CollapsibleSection("Advanced Options", collapsed=True)
adv.addWidget(my_form_widget)
lay.addWidget(adv)
```

#### `set_result_success(label, text)` / `set_result_error(label, text)` / `set_result_info(label, text)`
Style a `QLabel` as a coloured result card (green / red / grey).

---

## UI Rules for New Plugins

Follow these rules when creating or modifying a florian plugin to maintain visual consistency.

### Layout
1. Root layout is `QVBoxLayout` with no margins, containing a `QScrollArea`.
2. Inner layout has `10 px` horizontal padding, `6 px` top, `16 px` bottom, `8 px` spacing.
3. First widget is always `make_plugin_header(...)`.
4. Second widget is always `make_separator()`.
5. `QGroupBox` sections act as "cards" — one logical concept per card.

### Buttons
| Rule | Reason |
|---|---|
| Exactly **one** primary button per widget | Visual hierarchy — only one dominant call to action |
| Primary button `setFixedHeight(44)` | Comfortable touch target |
| Secondary buttons `setFixedHeight(30–32)` | Visually subordinate |
| Progress bar immediately below the primary button | Spatial association with the running action |

### Progress Feedback
```python
# In __init__ / _build_ui():
self._progress = QProgressBar()
self._progress.setRange(0, 0)   # indeterminate by default
self._progress.setFixedHeight(5)
self._progress.setVisible(False)

# During run:
self._progress.setVisible(True)

# In _on_progress():
import re
m = re.search(r'(\d+)\s*%', msg)
if m:
    self._progress.setRange(0, 100)
    self._progress.setValue(int(m.group(1)))
else:
    self._progress.setRange(0, 0)  # bounce animation

# After completion:
self._progress.setVisible(False)
```

### Advanced Options
All parameters that are not needed for 90 % of cases must go inside a `CollapsibleSection("Advanced Options", collapsed=True)`. Only expose in the main body parameters the user must set on every run.

### Result Labels
```python
self._result_label = QLabel("")
self._result_label.setVisible(False)   # hidden until first run completes
self._result_label.setMinimumHeight(42)
```

Call `set_result_success / set_result_error` in `_on_finished / _on_error`, which also sets `setVisible(True)`.

### Tooltips
Every input widget **must** have a tooltip that includes:
1. What the parameter does
2. The default value
3. A heuristic for when to change it (if non-obvious)

---

## Adding a New Plugin

1. Create `florian-myplugin/` directory with standard pyproject.toml / napari.yaml structure.
2. Copy `_style.py` from any existing plugin into `src/florian_myplugin/_style.py`.
3. Build the widget following the layout rules above.
4. Add the plugin to `florian-tools/setup.cfg` `install_requires`.
5. Register the widget command + widget in `florian-tools/src/florian_tools/napari.yaml`.
6. Install: `pip install -e ./florian-myplugin`.
