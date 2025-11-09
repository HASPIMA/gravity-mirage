from __future__ import annotations
from pathlib import Path
from typing import List
from flask import Flask, request, redirect, url_for, send_from_directory, send_file
from markupsafe import Markup
from werkzeug.utils import secure_filename
from PIL import Image
from .physics import SchwarzschildBlackHole
from .ray_tracer import GravitationalRayTracer
import os
import io
import numpy as np

# Simple Flask app to upload and serve images from local disk.
UPLOAD_FOLDER = Path.cwd() / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)


def list_uploaded_images() -> List[str]:
    """Return list of filenames in the upload folder (sorted)."""
    return sorted([f.name for f in UPLOAD_FOLDER.iterdir() if f.is_file()])


def render_lensing_image(filename: str, mass: float = 10.0, scale_Rs: float = 100.0, out_width: int = 512, method: str = 'weak') -> bytes:
    """Render a preview image with a black hole in the middle using a simple weak-field deflection.

    - filename: name of uploaded file in UPLOAD_FOLDER
    - mass: black hole mass in solar masses
    - scale_Rs: how many Schwarzschild radii correspond to the image's max radius
    - out_width: output image width in pixels (maintains aspect ratio)
    Returns PNG bytes.
    """
    src_path = UPLOAD_FOLDER / filename
    if not src_path.exists():
        raise FileNotFoundError(filename)

    # Load source image and resize to desired width for faster processing
    src = Image.open(src_path).convert("RGB")
    w0, h0 = src.size
    aspect = h0 / w0
    out_w = int(out_width)
    out_h = max(1, int(out_w * aspect))
    src_small = src.resize((out_w, out_h), Image.Resampling.BILINEAR)
    src_arr = np.array(src_small)

    # Setup black hole
    bh = SchwarzschildBlackHole(mass=mass)

    # Coordinate grids
    cx = out_w / 2.0
    cy = out_h / 2.0
    ys, xs = np.mgrid[0:out_h, 0:out_w]
    dx = xs - cx
    dy = ys - cy
    r = np.sqrt(dx ** 2 + dy ** 2)  # pixels
    max_r = np.max(r)
    max_r = max(max_r, 1.0)

    # Map pixel radius to physical impact parameter b (meters)
    # Image radius max_r corresponds to (scale_Rs * Rs)
    Rs = bh.schwarzschild_radius
    meters_per_pixel = (scale_Rs * Rs) / max_r
    b = r * meters_per_pixel

    # Compute deflection angle alpha(b)
    if method == 'weak':
        with np.errstate(divide='ignore', invalid='ignore'):
            alpha = np.vectorize(bh.deflection_angle_weak_field)(b)
    else:
        # Geodesic tracing: compute deflection for radial bins (coarse) using numeric geodesics
        tracer = GravitationalRayTracer(bh)
        # Use up to 128 radial bins for performance
        bins = min(128, max(8, int(max_r)))
        radii = np.linspace(0, max_r, bins)
        alpha_bins = np.zeros_like(radii)

        # set observer large radius
        r0 = max(1e4 * Rs, 1e6)

        for i, rb in enumerate(radii):
            b_phys = rb * meters_per_pixel
            # initial velocities: dr0 negative (incoming), dtheta0=0, dphi0 = b / r0^2
            dr0 = -1.0
            dtheta0 = 0.0
            dphi0 = b_phys / (r0 ** 2 + 1e-30)
            try:
                sol = tracer.trace_photon_geodesic((r0, np.pi / 2.0, 0.0), (dr0, dtheta0, dphi0), lambda_max=1e3)
                y = getattr(sol, 'y', None)
                if y is not None and y.shape[1] > 0:
                    phi_final = y[3, -1]
                    # deflection α ≈ φ_final - π (incoming from +inf to +inf)
                    alpha_bins[i] = float(np.abs(phi_final) - np.pi)
                else:
                    alpha_bins[i] = 0.0
            except Exception:
                alpha_bins[i] = 0.0

        # Interpolate alpha for all pixel radii
        alpha = np.interp(r.flatten(), radii, alpha_bins).reshape(r.shape)

    # For captured rays (alpha==inf), we'll set a mask
    captured = ~np.isfinite(alpha)

    # Original angle and new (source) angle: source_angle = observed_angle + alpha
    theta = np.arctan2(dy, dx)
    theta_src = theta + alpha

    # Source positions (keep same radius)
    src_x = cx + r * np.cos(theta_src)
    src_y = cy + r * np.sin(theta_src)

    # Nearest-neighbor sampling
    src_xi = np.clip(np.rint(src_x).astype(int), 0, out_w - 1)
    src_yi = np.clip(np.rint(src_y).astype(int), 0, out_h - 1)

    result = np.empty_like(src_arr)
    result[:, :, 0] = src_arr[src_yi, src_xi, 0]
    result[:, :, 1] = src_arr[src_yi, src_xi, 1]
    result[:, :, 2] = src_arr[src_yi, src_xi, 2]

    # Draw black disk for captured region (r < Rs_pixels)
    Rs_pixels = Rs / meters_per_pixel
    mask_disk = (r <= Rs_pixels) | captured
    result[mask_disk] = 0

    out_img = Image.fromarray(result)
    bio = io.BytesIO()
    out_img.save(bio, format='PNG')
    return bio.getvalue()


@app.route('/uploads/<path:filename>')
def uploaded_file(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route('/', methods=['GET'])
def index():
    images = list_uploaded_images()

    # Build the thumbnail items (delete button + thumbnail that sets preview when clicked)
    items = []
    for img in images:
        url = url_for('uploaded_file', filename=img)
        delete_url = url_for('delete')
        delete_form = (
            f'<form method="post" action="{delete_url}" '
            f'onsubmit="return confirm(\'Delete {img}?\');" '
            f'style="display:inline-block;margin:0;padding:0;">\n'
            f'  <input type="hidden" name="filename" value="{img}" />\n'
            f'  <button type="submit" '
            f'style="position:absolute; top:4px; right:4px; background:#d9534f; color:white; border:none; width:24px; height:24px; border-radius:12px; cursor:pointer;">✖</button>\n'
            f'</form>'
        )
        items.append(
            f'<li style="list-style:none; display:inline-block; margin:8px; position:relative; width:160px; vertical-align:top;">'
            f'{delete_form}'
            f'<a href="{url}" style="display:block; text-decoration:none; color:inherit;">{img}</a>'
            f'<img src="{url}" style="height:120px; display:block; margin-top:4px; border-radius:4px;" onclick="setPreview(\'{img}\')" />'
            f'</li>'
        )

    items_html = Markup(''.join(items))
    options_html = ''.join([f'<option value="{i}">{i}</option>' for i in images])

    pre = """
        <!doctype html>
        <html>
            <head>
                <meta charset="utf-8" />
                <title>Gravity Mirage — Upload & Preview</title>
                        <style>
                            :root{ --bg:#0f1720; --card:#0b1320; --muted:#9aa6b2; --accent:#6ee7b7; --panel: #0b1220 }
                            body{ margin:0; font-family:Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; background:linear-gradient(180deg,#071025,#0b1220); color:#e6eef6; display:flex; align-items:center; justify-content:center; min-height:100vh; }
                            /* Center the app and give a comfortable max width for 16:9 displays */
                                /* wider center column for large 16:9 (1920px) screens */
                                .app{ display:grid; grid-template-columns:1280px 1fr; gap:28px; padding:28px; box-sizing:border-box; max-width:1800px; margin:0; align-items:start; }
                                .sidebar{ background:rgba(255,255,255,0.02); border-radius:10px; padding:18px; overflow:auto; box-shadow: 0 6px 18px rgba(2,6,23,0.6); min-width:360px; max-width:1280px; }
                            /* Main column becomes a vertical flex so preview can expand nicely */
                            .main{ padding:14px; border-radius:10px; overflow:auto; display:flex; flex-direction:column; gap:12px; }
            h1{ margin:0 0 12px 0; font-size:18px; text-align:center; }
                    .uploader{ display:flex; gap:8px; align-items:center; }
                    .uploader input[type=file]{ color:var(--muted); }
                      .uploads-list{ margin-top:12px; display:flex; flex-wrap:wrap; gap:12px; justify-content:center; }
                      .thumb{ width:160px; background:var(--card); padding:10px; border-radius:8px; text-align:center; cursor:pointer; margin:0 auto; }
                      .thumb img{ width:100%; height:96px; object-fit:cover; border-radius:6px; display:block; }
                    .controls{ margin-top:14px; background:rgba(255,255,255,0.015); padding:10px; border-radius:8px; text-align:center; }
                    label{ font-size:13px; color:var(--muted); display:block; margin-bottom:6px; }
                    .range-wrap{ display:flex; align-items:center; gap:10px; justify-content:center; }
                    input[type=range]{ flex:1; }
                    .value{ width:64px; text-align:right; color:var(--accent); font-weight:600 }
                      .preview-card{ background:rgba(255,255,255,0.02); border-radius:10px; padding:12px; flex:1 1 auto; box-shadow: 0 6px 30px rgba(2,6,23,0.6); display:flex; flex-direction:column; }
                      /* previewContainer fills available space and keeps aspect nicely */
                      #previewContainer{ width:100%; flex:1 1 auto; display:flex; align-items:center; justify-content:center; background:#02050a; border-radius:8px; overflow:hidden; }
                      #previewImg{ max-width:100%; max-height:100%; display:block; margin:0 auto; }
                    .small{ font-size:12px; color:var(--muted); }
                    /* center the Black hole preview heading in the sidebar */
                    .sidebar h2{ text-align:center; margin-top:12px; }
                    .topbar{ display:flex; justify-content:center; align-items:center; gap:12px; margin-bottom:12px; }
                    /* center form elements in sidebar */
                    .sidebar form{ margin:0 auto; }
                    .controls select, .controls input[type=range]{ margin:0 auto; }
                    /* Unified larger Upload-like button for chooser + upload (apply globally) */
                    .control-button { display:inline-flex; align-items:center; justify-content:center; gap:6px; padding:10px 14px; min-height:36px; line-height:1; font-size:14px; font-weight:600; border-radius:8px; background:#2b3137; color:inherit; border:1px solid rgba(255,255,255,0.06); cursor:pointer; box-shadow: 0 2px 6px rgba(2,6,23,0.45); transition: background 120ms, border-color 120ms, transform 80ms; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex:0 0 auto; }
                    .control-button:hover { background:#353b42; border-color:rgba(255,255,255,0.09); transform:translateY(-1px); }
                    .control-button:active { transform:translateY(0); }
                    .control-button:focus { outline:2px solid rgba(110,231,183,0.12); outline-offset:2px; }
                    .control-button[disabled], .control-button:disabled { opacity:0.55; cursor:not-allowed; transform:none; }
                </style>
            </head>
            <body>
                <div class="app">
                    <aside class="sidebar">
                        <h1>Uploads</h1>
                        <div class="uploader">
                            <form id="uploadForm" method="post" action="/upload" enctype="multipart/form-data" style="display:flex; gap:8px; align-items:center; width:100%; flex-wrap:nowrap;">
                                <!-- Hidden native file input; label acts as the visible chooser button -->
                                <input id="fileInput" type="file" name="file" accept="image/*" style="display:none" />
                                <button type="button" id="chooseBtn" class="control-button" onclick="document.getElementById('fileInput').click();">Choose file</button>
                                <div id="fileName" class="small" style="flex:1; text-align:left; margin-left:6px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">No file chosen</div>
                                <input id="uploadSubmit" class="control-button" type="submit" value="Upload" style="margin-left:8px;" />
                            </form>
                        </div>
                        <div class="uploads-list" id="uploadsList">
                            <ul style="padding:0; margin:0; display:flex; flex-wrap:wrap; gap:8px;">
        """

    mid = """
                            </ul>
                        </div>

                        <hr />
                        <h2>Black hole preview</h2>
                        <div class="controls">
                            <div style="display:inline-block; margin-right:12px; vertical-align:middle;">
                                    <label for="imageSelect" style="display:block; margin-bottom:4px;">Image</label>
                                    <select id="imageSelect" style="vertical-align:middle; padding:6px; border-radius:6px;">
    """

    mid2 = """
                    </select>
                    </div>
                    <div style="display:inline-block; margin-left:12px; vertical-align:middle;">
                        <label for="massSlider" style="display:block; margin-bottom:4px;">Mass (M_sun)</label>
                        <input id="massSlider" type="range" min="1" max="1000000" step="1" value="10" style="vertical-align:middle; width:240px;" />
                        <div class="value" id="massValue" style="display:inline-block; width:70px; text-align:right;">10</div>
                    </div>
                    <div style="display:inline-block; margin-left:12px; vertical-align:middle;">
                        <label for="scaleSlider">Scale (Rs across image radius):</label>
                        <input id="scaleSlider" type="range" min="1" max="20" step="1" value="5" style="vertical-align:middle; width:240px;" />
                        <span id="scaleValue" style="display:inline-block; width:70px; text-align:right;">100</span>
                    </div>
                    <div style="display:inline-block; margin-left:12px; vertical-align:middle;">
                        <label for="methodSelect" style="display:block; margin-bottom:4px;">Method:</label>
                        <select id="methodSelect" style="vertical-align:middle; padding:6px; border-radius:6px;">
                            <option value="weak">Weak-field (fast)</option>
                            <option value="geodesic">Geodesic (slower, more accurate)</option>
                        </select>
                    </div>
                    <!-- render button removed: sliders and selection auto-trigger preview -->
                </div>
                <div style="margin-top:12px;">
                    <img id="previewImg" src="" style="max-width:100%; border:1px solid #ccc; background:#000;" />
                </div>

                <script>
                    function buildPreviewUrl(name){
                        var mass = document.getElementById('massSlider').value || 10;
                        var scale = document.getElementById('scaleSlider').value || 100;
                        var width = 512;
                        var method = document.getElementById('methodSelect') ? document.getElementById('methodSelect').value : 'weak';
                        return '/preview/' + encodeURIComponent(name)
                                + '?mass=' + encodeURIComponent(mass)
                                + '&scale=' + encodeURIComponent(scale)
                                + '&width=' + encodeURIComponent(width)
                                + '&method=' + encodeURIComponent(method);
                    }
                    function setPreview(name){
                        document.getElementById('imageSelect').value = name;
                        document.getElementById('previewImg').src = buildPreviewUrl(name);
                    }
                    document.addEventListener('DOMContentLoaded', function(){
                        // When the selection changes, auto-render the selected image
                        var imageSelect = document.getElementById('imageSelect');
                        if(imageSelect){
                            imageSelect.addEventListener('change', function(){ var name = imageSelect.value; if(name) setPreview(name); });
                        }
                        // Wire sliders to display current integer value
                        var massSlider = document.getElementById('massSlider');
                        var massValue = document.getElementById('massValue');
                        if(massSlider && massValue){
                            massValue.textContent = massSlider.value;
                            massSlider.addEventListener('input', function(){ massValue.textContent = massSlider.value; scheduleRender(); });
                        }
                        var scaleSlider = document.getElementById('scaleSlider');
                        var scaleValue = document.getElementById('scaleValue');
                        if(scaleSlider && scaleValue){
                            scaleValue.textContent = scaleSlider.value;
                            scaleSlider.addEventListener('input', function(){ scaleValue.textContent = scaleSlider.value; scheduleRender(); });
                        }

                        // Debounced auto-render when sliders change to avoid spamming render requests
                        var _debounceTimer = null;
                        function scheduleRender(){
                            if(_debounceTimer) clearTimeout(_debounceTimer);
                            var sel = document.getElementById('imageSelect');
                            var name = sel ? sel.value : null;
                            _debounceTimer = setTimeout(function(){ if(name) setPreview(name); }, 300);
                        }

                        // File chooser preview: keep Upload button next to filename and update text dynamically
                        try{
                            var fileInput = document.getElementById('fileInput');
                            var fileName = document.getElementById('fileName');
                            var uploadSubmit = document.getElementById('uploadSubmit');
                            var chooseBtn = document.getElementById('chooseBtn');
                            var defaultText = (navigator.language && navigator.language.startsWith && navigator.language.startsWith('es')) ? 'Ningún archivo seleccionado' : 'No file chosen';
                            if(fileName) fileName.textContent = defaultText;
                            if(fileInput){
                                fileInput.addEventListener('change', function(){
                                    if(fileInput.files && fileInput.files.length>0){
                                        fileName.textContent = fileInput.files[0].name;
                                        if(uploadSubmit) uploadSubmit.disabled = false;
                                    }else{
                                        fileName.textContent = defaultText;
                                        if(uploadSubmit) uploadSubmit.disabled = false;
                                    }
                                });
                            }
                            // Clicking the visible choose button triggers the hidden input (label[for] does this automatically)
                        }catch(e){ /* ignore if elements missing */ }

                        // initialize preview with first image if present
                        var sel = document.getElementById('imageSelect');
                        if(sel && sel.options.length>0){
                            var first = sel.options[0].value;
                            document.getElementById('previewImg').src = buildPreviewUrl(first);
                        }
                    });
                </script>
        </html>
        """

    html = pre + str(items_html) + mid + options_html + mid2
    return html


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    # Save uploaded files under a neutral name so original names are not preserved.
    # Naming pattern: image<N><ext> where N is an incrementing integer starting at 1
    orig = secure_filename(str(file.filename))
    ext = Path(orig).suffix.lower()
    if not ext:
        # fallback if no extension provided
        ext = '.png'

    # find existing files starting with 'image' and extract numeric suffixes
    existing = [p.name for p in Path(app.config['UPLOAD_FOLDER']).iterdir() if p.is_file()]
    max_n = 0
    for name in existing:
        stem = Path(name).stem  # e.g., 'image12' or 'image'
        if stem.startswith('image'):
            num_part = stem[5:]
            try:
                n = int(num_part) if num_part != '' else 0
                if n > max_n:
                    max_n = n
            except Exception:
                continue

    new_n = max_n + 1
    new_name = f"image{new_n}{ext}"
    dest = Path(app.config['UPLOAD_FOLDER']) / new_name
    file.save(str(dest))
    return redirect(url_for('index'))


@app.route('/preview/<path:filename>')
def preview(filename: str):
    # Query params: mass (float), scale (float), width (int)
    try:
        mass = float(request.args.get('mass', 10.0))
    except Exception:
        mass = 10.0
    try:
        scale = float(request.args.get('scale', 100.0))
    except Exception:
        scale = 100.0
    try:
        width = int(request.args.get('width', 512))
    except Exception:
        width = 512

    try:
        png = render_lensing_image(filename, mass=mass, scale_Rs=scale, out_width=width)
    except FileNotFoundError:
        return redirect(url_for('index'))

    return send_file(io.BytesIO(png), mimetype='image/png')



@app.route('/delete', methods=['POST'])
def delete():
    filename = request.form.get('filename')
    if not filename:
        return redirect(url_for('index'))
    if filename not in list_uploaded_images():
        return redirect(url_for('index'))
    target = UPLOAD_FOLDER / filename
    try:
        target.unlink()
    except FileNotFoundError:
        pass
    return redirect(url_for('index'))


def run(port: int | None = None):
    env_port = os.getenv('PORT')
    if env_port:
        port = int(env_port)
    if port is None:
        port = 8000
    app.run(host='0.0.0.0', port=port, debug=True)


if __name__ == '__main__':
    run()

