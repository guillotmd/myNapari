"""
_widget.py — 3D tumor surface render widget (modern UI).
"""
from __future__ import annotations
import numpy as np
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_warning
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QScrollArea, QSpinBox, QVBoxLayout, QWidget, QFileDialog, QLineEdit
)
from florian_tumor_3d_render._style import (
    STYLESHEET, CollapsibleSection,
    make_plugin_header, make_separator,
    style_primary_btn, style_secondary_btn,
    set_result_success, set_result_error, set_result_info,
    BG_CARD, BORDER, SUCCESS,
)


class TumorRenderWidget(QWidget):
    """Generates a 3D surface mesh from a tumor mask Labels layer."""

    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self._last_mask_name: str | None = None
        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        self._connect_signals()

    @staticmethod
    def _form_row(form, label, widget, tip):
        lbl = QLabel(label); lbl.setToolTip(tip); widget.setToolTip(tip)
        form.addRow(lbl, widget)

    def _spin(self, lo, hi, val, decimals=0):
        if decimals:
            w = QDoubleSpinBox(); w.setDecimals(decimals); w.setValue(float(val))
        else:
            w = QSpinBox(); w.setValue(int(val))
        w.setRange(lo, hi)
        return w

    def _build_ui(self):
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(10, 6, 10, 16); lay.setSpacing(8)
        scroll.setWidget(inner); root.addWidget(scroll)

        lay.addWidget(make_plugin_header(
            "🫀", "3D Tumor Render",
            "Generates a 3D surface mesh from a tumor mask using marching cubes + optional Laplacian smoothing."
        ))
        lay.addWidget(make_separator())

        # ── Input ─────────────────────────────────────────────────────
        grp_in = QGroupBox("Input"); f_in = QFormLayout(grp_in); f_in.setSpacing(6)

        self._mask_combo = QComboBox()
        self._form_row(f_in, "Tumor mask (Labels):", self._mask_combo,
            "Select the tumor mask Labels layer produced by a detection plugin.\n"
            "e.g. 'MyScan_Tumor_Mask_Linear' or 'MyScan_Tumor_Mask_Quadratic'\n\n"
            "The tumor label ID is auto-detected when you change this selection.")

        # Auto-detect badge
        self._label_id_row = QHBoxLayout()
        self._tumor_label_id = self._spin(1, 255, 4)
        self._tumor_label_id.setToolTip(
            "Label value that identifies tumor voxels.\n"
            "Auto-detected from the mask on selection.\n"
            "Linear plugin default: 4.  Quadratic plugin default: 3.")
        self._autodetect_hint = QLabel("auto-detected")
        self._autodetect_hint.setStyleSheet(f"""
            color: {SUCCESS}; font-size: 10px; font-weight: 600;
            background: transparent; padding-left: 6px;
        """)
        self._label_id_row.addWidget(self._tumor_label_id)
        self._label_id_row.addWidget(self._autodetect_hint)
        self._label_id_row.addStretch()
        f_in.addRow(QLabel("Tumor label ID:"), self._label_id_row)

        self._vol_combo = QComboBox()
        self._form_row(f_in, "OCT volume (Image, opt.):", self._vol_combo,
            "Optional: the original OCT volume for context in the 3D view.")
            
        lay.addWidget(grp_in)

        # ── Physical Resolution ────────────────────────────────────────
        grp_res = QGroupBox("Physical Resolution (µm / px)")
        res_h = QHBoxLayout(grp_res); res_h.setSpacing(6)

        self._axial_res   = self._spin(0.1, 100, 3.87, 2)
        self._lateral_res = self._spin(0.1, 100, 11.5, 2)
        self._spacing     = self._spin(0.1, 500, 120.0, 1)
        self._axial_res.setToolTip("Axial (depth) pixel size in µm. Default: 3.87 µm")
        self._lateral_res.setToolTip("Lateral (within B-scan) pixel size in µm. Default: 11.5 µm")
        self._spacing.setToolTip("Inter-slice spacing in µm (between B-scans). Default: 120 µm")

        res_h.addWidget(QLabel("Axial:")); res_h.addWidget(self._axial_res)
        res_h.addWidget(QLabel("Lat:")); res_h.addWidget(self._lateral_res)
        res_h.addWidget(QLabel("Gap:")); res_h.addWidget(self._spacing)
        lay.addWidget(grp_res)

        # ── Mesh Options ───────────────────────────────────────────────
        grp_mesh = QGroupBox("Mesh Options"); f_mesh = QFormLayout(grp_mesh); f_mesh.setSpacing(6)

        self._mesh_smoothing = self._spin(0, 100, 10)
        self._form_row(f_mesh, "Laplacian smoothing:", self._mesh_smoothing,
            "Number of Laplacian smoothing passes applied to the surface mesh.\n"
            "0 = no smoothing.  10–20 gives a clean surface without distorting the shape.\n"
            "Requires trimesh.")
        lay.addWidget(grp_mesh)

        # ── Animation Options ──────────────────────────────────────────
        grp_anim = CollapsibleSection("Animation Options", collapsed=True)
        f_anim = QFormLayout(); f_anim.setSpacing(6)

        self._anim_type = QComboBox()
        self._anim_type.addItems(["360° Continuous", "See-Saw Oscillate"])
        self._form_row(f_anim, "Type:", self._anim_type, "Type of animation loop.")

        self._anim_axis = QLineEdit("0.0, 1.0, 0.0")
        self._form_row(f_anim, "Rotation Vector:", self._anim_axis, "Enter rotation axis as X, Y, Z (e.g. 0.0, 1.0, 0.0 for horizontal).")

        self._anim_sweep = self._spin(10, 360, 90)
        self._form_row(f_anim, "Sweep Angle (°):", self._anim_sweep, "Total rotation angle (used for See-Saw).")

        self._anim_duration = self._spin(1, 60, 4, decimals=1)
        self._form_row(f_anim, "Duration (s):", self._anim_duration, "Length of the animation in seconds.")

        self._anim_fps = self._spin(1, 60, 30)
        self._form_row(f_anim, "FPS:", self._anim_fps, "Frames per second.")

        grp_anim.addLayout(f_anim)
        lay.addWidget(grp_anim)

        # ── Buttons ───────────────────────────────────────────────────
        lay.addSpacing(4)
        self._run_btn = QPushButton("Generate 3D Render")
        self._run_btn.setFixedHeight(44); style_primary_btn(self._run_btn)
        self._run_btn.setToolTip(
            "Format the selected layers for 3D viewing, calculate volume, and switch to 3D mode.")
        lay.addWidget(self._run_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0); self._progress.setFixedHeight(5); self._progress.setVisible(False)
        lay.addWidget(self._progress)

        self._recalc_btn = QPushButton("Recalculate Volume from Mask")
        self._recalc_btn.setFixedHeight(32); style_secondary_btn(self._recalc_btn)
        self._recalc_btn.setToolTip(
            "Instantly recompute volume from the current mask data using the resolution settings above.")
        lay.addWidget(self._recalc_btn)

        anim_btn_lay = QHBoxLayout()
        anim_btn_lay.setSpacing(6)
        
        self._is_previewing = False
        self._preview_anim_btn = QPushButton("Preview Animation")
        self._preview_anim_btn.setFixedHeight(32); style_secondary_btn(self._preview_anim_btn)
        self._preview_anim_btn.setToolTip("Play the animation in the viewer without saving.")
        anim_btn_lay.addWidget(self._preview_anim_btn)

        self._anim_btn = QPushButton("Export Video")
        self._anim_btn.setFixedHeight(32); style_secondary_btn(self._anim_btn)
        self._anim_btn.setToolTip(
            "Captures a rotating turntable animation of the current 3D view.\n"
            "Exports to MP4 or GIF (takes a few seconds to render).")
        anim_btn_lay.addWidget(self._anim_btn)
        
        lay.addLayout(anim_btn_lay)

        self._result_label = QLabel("")
        self._result_label.setAlignment(Qt.AlignCenter); self._result_label.setWordWrap(True)
        self._result_label.setVisible(False); self._result_label.setMinimumHeight(42)
        lay.addWidget(self._result_label)
        lay.addStretch()

    # ──────────────────────────────────────────────────────────────────
    # Signals
    # ──────────────────────────────────────────────────────────────────

    def _connect_signals(self):
        self._run_btn.clicked.connect(self._on_run)
        self._recalc_btn.clicked.connect(self._on_recalculate)
        self._preview_anim_btn.clicked.connect(self._on_preview_animation)
        self._anim_btn.clicked.connect(self._on_export_animation)
        self._mask_combo.currentTextChanged.connect(self._on_mask_changed)
        self.viewer.layers.events.inserted.connect(self._refresh_combos)
        self.viewer.layers.events.removed.connect(self._refresh_combos)
        self._refresh_combos()

    def _refresh_combos(self, event=None):
        labels = ["<None>"] + [l.name for l in self.viewer.layers if isinstance(l, Labels)]
        images = [l.name for l in self.viewer.layers if isinstance(l, Image)]

        cur_m = self._mask_combo.currentText()
        self._mask_combo.blockSignals(True)
        self._mask_combo.clear(); self._mask_combo.addItems(labels)
        idx = self._mask_combo.findText(cur_m)
        if idx >= 0: self._mask_combo.setCurrentIndex(idx)
        self._mask_combo.blockSignals(False)
        if self._mask_combo.currentText() != cur_m:
            self._on_mask_changed(self._mask_combo.currentText())

        vol_names = ["<None>"] + images; cur_v = self._vol_combo.currentText()
        self._vol_combo.blockSignals(True)
        self._vol_combo.clear(); self._vol_combo.addItems(vol_names)
        idx = self._vol_combo.findText(cur_v)
        if idx >= 0: self._vol_combo.setCurrentIndex(idx)
        self._vol_combo.blockSignals(False)

    def _on_mask_changed(self, name: str):
        """Auto-detect tumor label ID when a new mask is selected."""
        layer = self._get_layer(name)
        if layer is None or not isinstance(layer, Labels): return
        unique = [int(v) for v in np.unique(layer.data) if v > 0]
        if not unique: return
        detected = max(unique)
        self._tumor_label_id.setValue(detected)
        self._autodetect_hint.setText(f"auto-detected: {detected}")

    def _get_layer(self, name):
        for l in self.viewer.layers:
            if l.name == name: return l
        return None

    def _set_running(self, running: bool):
        self._run_btn.setEnabled(not running)
        self._run_btn.setText("Processing..." if running else "Generate 3D Render")
        self._progress.setVisible(running)
        self._recalc_btn.setEnabled(not running)

    def _build_job(self, mask_layer) -> dict:
        vn = self._vol_combo.currentText()
        vl = None if vn == "<None>" else self._get_layer(vn)
        
        m_data = mask_layer.data
        v_data = vl.data if vl else None

        return dict(
            mask_data=m_data, mask_name=mask_layer.name,
            tumor_label_id=self._tumor_label_id.value(),
            axial_um=self._axial_res.value(),
            lateral_um=self._lateral_res.value(),
            spacing_um=self._spacing.value(),
            smooth_iters=self._mesh_smoothing.value(),
            vol_data=v_data,
            vol_name=vl.name if vl else None,
            vol_colormap=getattr(vl, 'colormap', 'gray') if vl else 'gray',
        )

    # ──────────────────────────────────────────────────────────────────
    # Run
    # ──────────────────────────────────────────────────────────────────

    def _on_run(self):
        mask_name = self._mask_combo.currentText()
        mask_layer = None if mask_name == "<None>" else self._get_layer(mask_name)
        
        vol_name = self._vol_combo.currentText()
        vol_layer = None if vol_name == "<None>" else self._get_layer(vol_name)

        if mask_layer is None and vol_layer is None:
            show_info("Please select a Tumor Mask OR an OCT Volume layer."); return

        # If only Volume is selected, skip marching cubes and just jump to 3D volume view
        if mask_layer is None and vol_layer is not None:
            ax_um = self._axial_res.value()
            lat_um = self._lateral_res.value()
            isl_um = self._spacing.value()
            maxr = max(ax_um, lat_um, isl_um)
            scale = (isl_um / maxr, ax_um / maxr, lat_um / maxr)
            
            v_data = vol_layer.data
                
            # Create a cropped 3D layer instead of modifying the original
            vn3 = f"{vol_layer.name}_3D"
            ex = self._get_layer(vn3)
            if ex: ex.data = v_data; ex.scale = scale
            else:
                ex = self.viewer.add_image(v_data, name=vn3, colormap=vol_layer.colormap, scale=scale)
            if hasattr(ex, 'depiction'): ex.depiction = 'volume'
            if hasattr(ex, 'rendering'): ex.rendering = 'attenuated_mip'
            
            # Hide all other layers to prevent floating 2D planes in 3D
            for layer in self.viewer.layers:
                if layer != ex:
                    layer.visible = False
            
            self.viewer.dims.ndisplay = 3
            self.viewer.reset_view()
            
            set_result_success(self._result_label, "3D Volume Render initialized")
            show_info("Switched to 3D view for OCT Volume.")
            return

        self._last_mask_name = mask_layer.name
        self._set_running(True); self._result_label.setVisible(False)
        worker = _run_render_thread(self._build_job(mask_layer))
        worker.yielded.connect(self._on_progress)
        worker.returned.connect(self._on_finished)
        worker.errored.connect(self._on_error)
        worker.start()

    def _on_progress(self, msg: str):
        self._progress.setRange(0, 0)  # marching cubes is not divisible into %

    def _on_finished(self, result):
        verts, faces, vals, scale_tuple, volume_mm3, mask_name, vol_data, vol_name, vol_colormap = result
        self._set_running(False)

        # Temporarily drop to 2D mode to avoid vispy crashes when adding invisible layers
        if self.viewer.dims.ndisplay == 3:
            self.viewer.dims.ndisplay = 2

        # Hide all layers to prevent floating 2D planes in 3D
        for layer in self.viewer.layers:
            layer.visible = False

        if vol_data is not None:
            vn3 = f"{vol_name}_3D"; ex = self._get_layer(vn3)
            if ex: ex.data = vol_data; ex.scale = scale_tuple
            else: 
                self.viewer.add_image(vol_data, name=vn3,
                      scale=scale_tuple, colormap=vol_colormap, visible=False)
                ex = self._get_layer(vn3)
            
            if ex:
                if hasattr(ex, 'depiction'): ex.depiction = 'volume'
                if hasattr(ex, 'rendering'): ex.rendering = 'attenuated_mip'

        if verts is not None and len(verts) > 0:
            sn = f"{mask_name}_3D_Surface"; ex_s = self._get_layer(sn)
            if ex_s: ex_s.data = (verts, faces, vals); ex_s.scale = scale_tuple
            else: self.viewer.add_surface((verts, faces, vals),
                      name=sn, colormap="turbo", scale=scale_tuple)
            self.viewer.dims.ndisplay = 3; self.viewer.reset_view()
            set_result_success(self._result_label, f"Volume: {volume_mm3:.4f} mm³")
            show_info(f"3D render complete — Volume: {volume_mm3:.4f} mm³")
        else:
            show_warning("No surface found — check the tumor label ID.")
            set_result_error(self._result_label, "No surface found — wrong label ID?")

    def _on_error(self, exc):
        self._set_running(False)
        set_result_error(self._result_label, f"Error: {exc}")
        show_warning(f"3D render failed: {exc}"); raise exc

    # ──────────────────────────────────────────────────────────────────
    # Recalculate
    # ──────────────────────────────────────────────────────────────────

    def _on_recalculate(self):
        mask_name = self._mask_combo.currentText()
        mask_layer = None if mask_name == "<None>" else self._get_layer(mask_name)
        if mask_layer is None:
            show_info("Please select a Tumor Mask to calculate volume."); return
            
        m_data = mask_layer.data
        
        label_id    = self._tumor_label_id.value()
        ax_um       = self._axial_res.value()
        lat_um      = self._lateral_res.value()
        isl_um      = self._spacing.value()
        n_voxels    = int(np.sum(m_data == label_id))
        vol_mm3     = n_voxels * (ax_um * 1e-3) * (lat_um * 1e-3) * (isl_um * 1e-3)
        set_result_success(self._result_label,
            f"Volume: {vol_mm3:.4f} mm³  (recalculated)")
        show_info(f"Volume recalculated: {vol_mm3:.4f} mm³")

    # ──────────────────────────────────────────────────────────────────
    # Preview Animation
    # ──────────────────────────────────────────────────────────────────

    def _on_preview_animation(self):
        if self.viewer.dims.ndisplay != 3:
            show_warning("Please generate a 3D render and switch to 3D mode first.")
            return

        if getattr(self, '_is_previewing', False):
            self._is_previewing = False
            return

        self._is_previewing = True
        self._set_running(True)
        self._preview_anim_btn.setEnabled(True)
        self._preview_anim_btn.setText("Stop Preview")
        
        try:
            import time, math
            from qtpy.QtWidgets import QApplication

            anim_type = self._anim_type.currentText()
            try:
                axis_vec = [float(x.strip()) for x in self._anim_axis.text().split(',')]
                if len(axis_vec) != 3: raise ValueError
            except:
                show_warning("Rotation Vector must be 3 comma-separated numbers (e.g. 0, 1, 0).")
                return
            
            sweep     = self._anim_sweep.value()
            duration  = self._anim_duration.value()
            fps       = self._anim_fps.value()
            frames    = int(fps * duration)

            orig_angles = tuple(self.viewer.camera.angles)
            
            for i in range(frames):
                if not getattr(self, '_is_previewing', False):
                    break
                
                progress = i / max(1, frames - 1)
                
                if anim_type == "See-Saw Oscillate":
                    offset = math.sin(progress * 2 * math.pi) * (sweep / 2.0)
                else:
                    offset = progress * 360.0
                
                new_angles = [orig_angles[j] + offset * axis_vec[j] for j in range(3)]
                self.viewer.camera.angles = tuple(new_angles)
                
                QApplication.processEvents()
                time.sleep(1.0 / fps)
            
            self.viewer.camera.angles = orig_angles
            
        except Exception as e:
            show_warning(f"Preview failed: {e}")
        finally:
            self._is_previewing = False
            self._set_running(False)
            self._preview_anim_btn.setText("Preview Animation")

    # ──────────────────────────────────────────────────────────────────
    # Export Animation
    # ──────────────────────────────────────────────────────────────────

    def _on_export_animation(self):
        if self.viewer.dims.ndisplay != 3:
            show_warning("Please generate a 3D render and switch to 3D mode first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Turntable Animation", "", "MP4 Video (*.mp4);;GIF Animation (*.gif)"
        )
        if not path: return

        try:
            import imageio
        except ImportError:
            show_warning("Please install imageio to save animations (conda install imageio)")
            return

        self._set_running(True)
        self._result_label.setVisible(False)
        self._run_btn.setText("Rendering video...")
        
        try:
            anim_type = self._anim_type.currentText()
            try:
                axis_vec = [float(x.strip()) for x in self._anim_axis.text().split(',')]
                if len(axis_vec) != 3: raise ValueError
            except:
                show_warning("Rotation Vector must be 3 comma-separated numbers (e.g. 0, 1, 0).")
                return

            sweep     = self._anim_sweep.value()
            duration  = self._anim_duration.value()
            fps       = self._anim_fps.value()
            frames    = int(fps * duration)

            orig_angles = tuple(self.viewer.camera.angles)
            
            kwargs = {}
            if path.endswith('.mp4'):
                kwargs = {'quality': 8, 'macro_block_size': None}
                
            with imageio.get_writer(path, fps=fps, **kwargs) as writer:
                import math
                from qtpy.QtWidgets import QApplication

                for i in range(frames):
                    progress = i / max(1, frames - 1)
                    
                    if anim_type == "See-Saw Oscillate":
                        offset = math.sin(progress * 2 * math.pi) * (sweep / 2.0)
                    else:
                        offset = progress * 360.0
                    
                    new_angles = [orig_angles[j] + offset * axis_vec[j] for j in range(3)]
                    self.viewer.camera.angles = tuple(new_angles)
                    
                    QApplication.processEvents()
                    img = self.viewer.screenshot(flash=False)
                    writer.append_data(img)
            
            self.viewer.camera.angles = orig_angles
            set_result_success(self._result_label, f"✓ Animation saved to {path.split('/')[-1]}")
            show_info(f"Animation saved to {path}")
            
        except Exception as e:
            show_warning(f"Failed to save animation: {e}")
            set_result_error(self._result_label, f"Error: {e}")
        finally:
            self._set_running(False)


# ──────────────────────────────────────────────────────────────────────────────
# Background thread
# ──────────────────────────────────────────────────────────────────────────────

@thread_worker
def _run_render_thread(job: dict):
    import queue, threading
    pq = queue.Queue(); result_holder = [None, None]

    def _run():
        try:
            md, mn = job['mask_data'], job['mask_name']
            lid     = job['tumor_label_id']
            ax, lat, isl = job['axial_um'], job['lateral_um'], job['spacing_um']
            smooth  = job['smooth_iters']
            vd, vn, vc = job['vol_data'], job['vol_name'], job['vol_colormap']

            n_vox = int(np.sum(md == lid))
            vol   = n_vox * (ax * 1e-3) * (lat * 1e-3) * (isl * 1e-3)
            maxr  = max(ax, lat, isl)
            scale = (isl / maxr, ax / maxr, lat / maxr)

            pq.put("⏳ Running marching cubes…")
            from skimage.measure import marching_cubes
            bv = (md == lid)
            if not bv.any():
                result_holder[0] = (None, None, None, scale, vol, mn, vd, vn, vc); return

            # Pad with False to ensure the mesh closes fully on the array boundaries
            bv_padded = np.pad(bv, pad_width=1, mode='constant', constant_values=False)
            verts, faces, _, _ = marching_cubes(bv_padded, level=0.5, spacing=(1, 1, 1))
            verts -= 1.0  # correct for the +1 offset introduced by padding

            if smooth > 0 and len(verts) > 0:
                pq.put(f"⏳ Smoothing mesh ({smooth} iterations)…")
                import trimesh
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                trimesh.smoothing.filter_laplacian(mesh, iterations=smooth)
                verts, faces = mesh.vertices, mesh.faces

            vals = np.ones(len(verts), dtype=np.float32)
            result_holder[0] = (verts, faces, vals, scale, vol, mn, vd, vn, vc)
        except Exception as e:
            result_holder[1] = e

    t = threading.Thread(target=_run); t.start()
    while t.is_alive():
        try: yield pq.get_nowait()
        except queue.Empty:
            import time; time.sleep(0.05)
    if result_holder[1]: raise result_holder[1]
    return result_holder[0]
