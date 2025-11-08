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
        <title>Gravity Mirage — Upload Images</title>
      </head>
      <body>
        <h1>Upload images to be distorted by the black hole</h1>
        <form method="post" action="/upload" enctype="multipart/form-data">
          <input type="file" name="file" accept="image/*" />
          <input type="submit" value="Upload" />
        </form>
        <h2>Uploaded images</h2>
        <div>
          <ul style="padding:0; margin:0;">
    """

    mid = """
          </ul>
        </div>

        <hr />
        <h2>Black hole preview</h2>
        <div>
          <label for="imageSelect">Image:</label>
          <select id="imageSelect">
    """

    mid2 = """
          </select>
          <label for="massInput">Mass (M_sun):</label>
          <input id="massInput" type="number" value="10" step="1" style="width:80px;" />
                    <label for="scaleInput">Scale (Rs across image radius):</label>
                    <input id="scaleInput" type="number" value="100" step="10" style="width:80px;" />
                    <label for="methodSelect">Method:</label>
                    <select id="methodSelect">
                        <option value="weak">Weak-field (fast)</option>
                        <option value="geodesic">Geodesic (slower, more accurate)</option>
                    </select>
          <button id="renderBtn">Render preview</button>
        </div>
        <div style="margin-top:12px;">
          <img id="previewImg" src="" style="max-width:100%; border:1px solid #ccc; background:#000;" />
        </div>

        <script>
          function buildPreviewUrl(name){
            var mass = document.getElementById('massInput').value || 10;
            var scale = document.getElementById('scaleInput').value || 100;
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
            var btn = document.getElementById('renderBtn');
            if(btn){
              btn.addEventListener('click', function(){
                var sel = document.getElementById('imageSelect');
                var name = sel.value;
                if(name) setPreview(name);
              });
            }
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

    filename = secure_filename(str(file.filename))
    dest = Path(app.config['UPLOAD_FOLDER']) / filename
    # If file with same name exists, append a short suffix
    if dest.exists():
        base = dest.stem
        suf = dest.suffix
        i = 1
        while dest.exists():
            dest = Path(app.config['UPLOAD_FOLDER']) / f"{base}_{i}{suf}"
            i += 1

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

