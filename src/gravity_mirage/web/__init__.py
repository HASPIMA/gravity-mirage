from __future__ import annotations

import io
import queue as _queue
import threading
import uuid
from importlib.metadata import version
from os import getenv
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    StreamingResponse,
)
from jinja2 import DictLoader, Environment, select_autoescape
from PIL import Image

from gravity_mirage.physics import SchwarzschildBlackHole
from gravity_mirage.ray_tracer import GravitationalRayTracer
from gravity_mirage.utils.files import (
    allocate_export_path,
    allocate_image_path,
    list_exported_images,
    list_uploaded_images,
    resolve_export_file,
    resolve_uploaded_file,
    sanitize_extension,
)
from gravity_mirage.web.constants import (
    ALLOWED_METHODS,
    CHUNK_SIZE,
    EXPORT_FOLDER,
    INDEX_TEMPLATE,
    PREVIEW_WIDTH,
    UPLOAD_FOLDER,
)

app = FastAPI(
    title="Gravity Mirage Web",
    version=version("gravity_mirage"),
)

# Simple in-memory job queue for GIF exports. This is intentionally lightweight
# and suitable for development. Jobs are stored in `JOBS` and processed by a
# background worker thread that writes the generated GIF to `exports/`.
JOB_QUEUE: _queue.Queue = _queue.Queue()
JOBS: dict[str, dict[str, Any]] = {}


def _gif_worker() -> None:
    while True:
        job = JOB_QUEUE.get()
        if job is None:
            break
        job_id = job["id"]
        try:
            JOBS[job_id]["status"] = "processing"
            JOBS[job_id]["progress"] = 0

            path = Path(job["path"]).resolve()
            with Image.open(path) as src_image:
                src = src_image.convert("RGB")
                w0, h0 = src.size
                aspect = h0 / max(w0, 1)
                out_w = max(1, int(job.get("width", PREVIEW_WIDTH)))
                out_h = max(1, int(out_w * aspect))
                src_small = src.resize((out_w, out_h), Image.Resampling.BILINEAR)
                src_arr0 = np.array(src_small)

            frames = int(job.get("frames", 24))
            frames_list: list[Image.Image] = []
            for i in range(frames):
                # update a coarse progress indicator
                JOBS[job_id]["progress"] = (i // frames) * 100
                shift = round(i * (out_w / frames))
                rolled = np.roll(src_arr0, -shift, axis=1)
                result_arr = _compute_lensed_array_from_src_arr(
                    rolled,
                    mass=job.get("mass", 10.0),
                    scale_Rs=job.get("scale", 100.0),
                    method=job.get("method", "weak"),
                )
                frames_list.append(Image.fromarray(result_arr))

            # Allocate the next sequential export filename (image1.gif, image2.gif, ...)
            out_file = allocate_export_path(".gif")

            # Save with 20 frames per second (50 ms per frame)
            frames_list[0].save(
                out_file,
                format="GIF",
                save_all=True,
                append_images=frames_list[1:],
                loop=0,
                duration=50,
                optimize=False,
            )
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["result"] = str(out_file.name)
            JOBS[job_id]["progress"] = 100
        except (OSError, ValueError, RuntimeError) as exc:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(exc)
        finally:
            JOB_QUEUE.task_done()


# Start worker thread
_worker_thread = threading.Thread(target=_gif_worker, daemon=True)
_worker_thread.start()


template_env = Environment(
    loader=DictLoader({"index.html": INDEX_TEMPLATE}),
    autoescape=select_autoescape(["html", "xml"]),
)
index_template = template_env.get_template("index.html")


def render_lensing_image(
    src_path: Path,
    mass: float = 10.0,
    scale_Rs: float = 100.0,
    out_width: int = PREVIEW_WIDTH,
    method: str = "weak",
) -> bytes:
    """Render a PNG preview that visualizes gravitational lensing."""
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    with Image.open(src_path) as src_image:
        src = src_image.convert("RGB")
        w0, h0 = src.size
        aspect = h0 / max(w0, 1)
        out_w = max(1, int(out_width))
        out_h = max(1, int(out_w * aspect))
        src_small = src.resize((out_w, out_h), Image.Resampling.BILINEAR)
        src_arr = np.array(src_small)

    bh = SchwarzschildBlackHole(mass=mass)
    cx = out_w / 2.0
    cy = out_h / 2.0
    ys, xs = np.mgrid[0:out_h, 0:out_w]
    dx = xs - cx
    dy = ys - cy
    r = np.sqrt(dx**2 + dy**2)
    max_r = max(np.max(r), 1.0)

    Rs = bh.schwarzschild_radius
    meters_per_pixel = (scale_Rs * Rs) / max_r
    b = r * meters_per_pixel

    if method == "weak":
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = np.vectorize(bh.deflection_angle_weak_field)(b)
    else:
        tracer = GravitationalRayTracer(bh)
        bins = min(128, max(8, int(max_r)))
        radii = np.linspace(0, max_r, bins)
        alpha_bins = np.zeros_like(radii)
        r0 = max(1e4 * Rs, 1e6)

        for i, rb in enumerate(radii):
            b_phys = rb * meters_per_pixel
            dr0 = -1.0
            dtheta0 = 0.0
            dphi0 = b_phys / (r0**2 + 1e-30)
            try:
                # Allow integration long enough for the photon to escape back
                # to large radius. The tracer now supports stopping at the
                # escape event, so a large lambda_max is acceptable.
                sol = tracer.trace_photon_geodesic(
                    (r0, np.pi / 2.0, 0.0),
                    (dr0, dtheta0, dphi0),
                    lambda_max=max(1e3, float(r0) * 2.0),
                )
                y = getattr(sol, "y", None)
                if y is not None and y.shape[1] > 0:
                    phi_final = y[3, -1]
                    alpha_bins[i] = float(abs(phi_final) - np.pi)
                else:
                    alpha_bins[i] = 0.0
            except Exception:
                alpha_bins[i] = 0.0

        alpha = np.interp(r.flatten(), radii, alpha_bins).reshape(r.shape)

    captured = ~np.isfinite(alpha)
    theta = np.arctan2(dy, dx)
    theta_src = theta + alpha
    src_x = cx + r * np.cos(theta_src)
    src_y = cy + r * np.sin(theta_src)
    src_xi = np.clip(np.rint(src_x).astype(int), 0, out_w - 1)
    src_yi = np.clip(np.rint(src_y).astype(int), 0, out_h - 1)

    result = np.empty_like(src_arr)
    result[:, :, 0] = src_arr[src_yi, src_xi, 0]
    result[:, :, 1] = src_arr[src_yi, src_xi, 1]
    result[:, :, 2] = src_arr[src_yi, src_xi, 2]

    Rs_pixels = Rs / meters_per_pixel
    mask_disk = (r <= Rs_pixels) | captured
    result[mask_disk] = 0

    out_img = Image.fromarray(result)
    bio = io.BytesIO()
    out_img.save(bio, format="PNG")
    bio.seek(0)
    return bio.getvalue()


def _compute_lensed_array_from_src_arr(
    src_arr: np.ndarray,
    mass: float = 10.0,
    scale_Rs: float = 100.0,
    method: str = "weak",
) -> np.ndarray:
    """
    Compute a lensed RGB image array from an already-resized source array.

    src_arr: HxWx3 uint8 RGB array
    returns: HxWx3 uint8 RGB array with lensing applied
    """
    out_h, out_w = src_arr.shape[0], src_arr.shape[1]

    bh = SchwarzschildBlackHole(mass=mass)
    cx = out_w / 2.0
    cy = out_h / 2.0
    ys, xs = np.mgrid[0:out_h, 0:out_w]
    dx = xs - cx
    dy = ys - cy
    r = np.sqrt(dx**2 + dy**2)
    max_r = max(np.max(r), 1.0)

    Rs = bh.schwarzschild_radius
    meters_per_pixel = (scale_Rs * Rs) / max_r
    b = r * meters_per_pixel

    if method == "weak":
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = np.vectorize(bh.deflection_angle_weak_field)(b)
    else:
        tracer = GravitationalRayTracer(bh)
        bins = min(128, max(8, int(max_r)))
        radii = np.linspace(0, max_r, bins)
        alpha_bins = np.zeros_like(radii)
        r0 = max(1e4 * Rs, 1e6)

        for i, rb in enumerate(radii):
            b_phys = rb * meters_per_pixel
            dr0 = -1.0
            dtheta0 = 0.0
            dphi0 = b_phys / (r0**2 + 1e-30)
            try:
                sol = tracer.trace_photon_geodesic(
                    (r0, np.pi / 2.0, 0.0),
                    (dr0, dtheta0, dphi0),
                    lambda_max=max(1e3, float(r0) * 2.0),
                )
                y = getattr(sol, "y", None)
                if y is not None and y.shape[1] > 0:
                    phi_final = y[3, -1]
                    alpha_bins[i] = float(abs(phi_final) - np.pi)
                else:
                    alpha_bins[i] = 0.0
            except (ValueError, RuntimeError):
                alpha_bins[i] = 0.0

        alpha = np.interp(r.flatten(), radii, alpha_bins).reshape(r.shape)

    captured = ~np.isfinite(alpha)
    theta = np.arctan2(dy, dx)
    theta_src = theta + alpha
    src_x = cx + r * np.cos(theta_src)
    src_y = cy + r * np.sin(theta_src)
    src_xi = np.clip(np.rint(src_x).astype(int), 0, out_w - 1)
    src_yi = np.clip(np.rint(src_y).astype(int), 0, out_h - 1)

    result = np.empty_like(src_arr)
    result[:, :, 0] = src_arr[src_yi, src_xi, 0]
    result[:, :, 1] = src_arr[src_yi, src_xi, 1]
    result[:, :, 2] = src_arr[src_yi, src_xi, 2]

    Rs_pixels = Rs / meters_per_pixel
    mask_disk = (r <= Rs_pixels) | captured
    result[mask_disk] = 0

    return result


@app.get("/export_gif/{filename}")
async def export_gif(
    filename: str,
    mass: Annotated[float, Query(gt=0.0)] = 10.0,
    scale: Annotated[float, Query(gt=0.0)] = 100.0,
    width: Annotated[int, Query(gt=0)] = PREVIEW_WIDTH,
    method: Annotated[str, Query()] = "weak",
    frames: Annotated[int, Query(ge=2, le=200)] = 24,
) -> StreamingResponse:
    """
    Generate an animated GIF that scrolls the image right-to-left.

    The scrolling is implemented by rolling the resized source image horizontally
    across the requested number of frames and applying the lensing renderer
    to each frame.
    """
    clean_method = method.lower()
    if clean_method not in ALLOWED_METHODS:
        raise HTTPException(status_code=400, detail="Unsupported render method")

    path = resolve_uploaded_file(filename)

    def _build_gif_bytes():
        with Image.open(path) as src_image:
            src = src_image.convert("RGB")
            w0, h0 = src.size
            aspect = h0 / max(w0, 1)
            out_w = max(1, int(width))
            out_h = max(1, int(out_w * aspect))
            src_small = src.resize((out_w, out_h), Image.Resampling.BILINEAR)
            src_arr0 = np.array(src_small)

        frames_list = []
        # for each frame, roll the image left by a fraction of the width
        for i in range(frames):
            shift = round(i * (out_w / frames))
            rolled = np.roll(src_arr0, -shift, axis=1)
            result_arr = _compute_lensed_array_from_src_arr(
                rolled,
                mass=mass,
                scale_Rs=scale,
                method=clean_method,
            )
            frames_list.append(Image.fromarray(result_arr))

        bio = io.BytesIO()
        # Save as animated GIF
        frames_list[0].save(
            bio,
            format="GIF",
            save_all=True,
            append_images=frames_list[1:],
            loop=0,
            # Use 20 frames per second => 50 ms per frame
            duration=50,
            optimize=False,
        )
        bio.seek(0)
        return bio.getvalue()

    try:
        gif_bytes = await run_in_threadpool(_build_gif_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Image not found") from exc

    headers = {
        "Content-Disposition": f'attachment; filename="{Path(filename).stem}-scroll.gif"',
    }
    return StreamingResponse(
        io.BytesIO(gif_bytes),
        media_type="image/gif",
        headers=headers,
    )


@app.post("/export_gif_async/{filename}")
async def export_gif_async(
    filename: str,
    mass: Annotated[float, Query(gt=0.0)] = 10.0,
    scale: Annotated[float, Query(gt=0.0)] = 100.0,
    width: Annotated[int, Query(gt=0)] = PREVIEW_WIDTH,
    method: Annotated[str, Query()] = "weak",
    frames: Annotated[int, Query(ge=2, le=200)] = 24,
) -> dict:
    """Queue a GIF export job and return a job id for polling."""
    clean_method = method.lower()
    if clean_method not in ALLOWED_METHODS:
        raise HTTPException(status_code=400, detail="Unsupported render method")

    path = resolve_uploaded_file(filename)

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "path": str(path),
        "mass": float(mass),
        "scale": float(scale),
        "width": int(width),
        "method": clean_method,
        "frames": int(frames),
    }
    JOB_QUEUE.put(JOBS[job_id])
    return {"job_id": job_id, "status": "queued"}


@app.get("/export_gif_status/{job_id}")
async def export_gif_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "progress": job.get("progress", 0),
        "error": job.get("error"),
    }


@app.get("/export_gif_result/{job_id}")
async def export_gif_result(job_id: str) -> FileResponse:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "done":
        raise HTTPException(status_code=409, detail="Job not ready")
    result_name = job.get("result")
    if not result_name:
        raise HTTPException(status_code=500, detail="Result missing")
    result_path = EXPORT_FOLDER / result_name
    return FileResponse(result_path, media_type="image/gif", filename=result_name)


@app.get("/exports_list")
async def exports_list() -> dict:
    """Return a JSON listing of files in the exports folder."""
    return {"exports": list_exported_images()}


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Render the landing page with the upload + preview UI."""
    images = list_uploaded_images()
    exports = list_exported_images()

    # Prefer a gif placed in ./img/ (repo asset) for the full-page background;
    # fall back to an uploaded copy in the uploads/ folder if present.
    img_dir = Path.cwd() / "img"
    img_file = img_dir / "nasa-black-hole-visualization.gif"
    if img_file.exists():
        background_image_url = f"/img/{img_file.name}"
    else:
        bg_file = UPLOAD_FOLDER / "nasa-black-hole-visualization.gif"
        background_image_url = f"/uploads/{bg_file.name}" if bg_file.exists() else ""

    html = index_template.render(
        images=images,
        exports=exports,
        first_image=images[0] if images else "",
        preview_width=PREVIEW_WIDTH,
        background_image_url=background_image_url,
    )
    return HTMLResponse(html)


@app.post("/upload")
async def upload(file: Annotated[UploadFile, File()]) -> RedirectResponse:
    """Persist an uploaded file and redirect back to the UI."""
    if file is None or not file.filename:
        return RedirectResponse("/", status_code=303)

    ext = sanitize_extension(Path(file.filename).suffix)
    dest = allocate_image_path(ext)

    with dest.open("wb") as buffer:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            buffer.write(chunk)
    await file.close()
    return RedirectResponse("/", status_code=303)


@app.get("/uploads/{filename:path}")
async def uploaded_file(filename: str) -> FileResponse:
    """Serve original uploaded assets."""
    path = resolve_uploaded_file(filename)
    return FileResponse(path)


@app.get("/img/{filename:path}")
async def img_file(filename: str) -> FileResponse:
    """
    Serve files from the repository's ./img/ directory (for repo assets).

    This lets the template reference `/img/nasa-black-hole-visualization.gif`
    without requiring the user to re-upload the asset into uploads/.
    """
    # Ensure we don't allow path traversal outside the img directory.
    img_base = (Path.cwd() / "img").resolve()
    target = (img_base / Path(filename).name).resolve()
    try:
        target.relative_to(img_base)
    except ValueError as e:
        raise HTTPException(status_code=404, detail="File not found") from e
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)


@app.get("/exports/{filename:path}")
async def export_file(filename: str) -> FileResponse:
    """Serve files from the repository's `exports/` directory (generated GIFs)."""
    # Ensure we don't allow path traversal outside the exports directory.
    export_base = EXPORT_FOLDER.resolve()
    target = (export_base / Path(filename).name).resolve()
    try:
        target.relative_to(export_base)
    except ValueError as e:
        raise HTTPException(status_code=404, detail="File not found") from e
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target, media_type="image/gif")


@app.post("/delete")
async def delete(filename: Annotated[str, Form()]) -> RedirectResponse:
    """Remove an uploaded asset."""
    try:
        path = resolve_uploaded_file(filename)
    except HTTPException:
        return RedirectResponse("/", status_code=303)
    path.unlink(missing_ok=True)
    return RedirectResponse("/", status_code=303)


@app.post("/delete_export")
async def delete_export(filename: Annotated[str, Form()]) -> RedirectResponse:
    """Remove an exported GIF."""
    try:
        path = resolve_export_file(filename)
    except HTTPException:
        return RedirectResponse("/", status_code=303)
    path.unlink(missing_ok=True)
    return RedirectResponse("/", status_code=303)


@app.get("/preview/{filename}", response_class=StreamingResponse)
async def preview(
    filename: str,
    mass: Annotated[float, Query(gt=0.0)] = 10.0,
    scale: Annotated[float, Query(gt=0.0)] = 100.0,
    width: Annotated[int, Query(gt=0)] = PREVIEW_WIDTH,
    method: Annotated[str, Query()] = "weak",
) -> StreamingResponse:
    """Generate and stream a PNG preview for the requested file."""
    clean_method = method.lower()
    if clean_method not in ALLOWED_METHODS:
        raise HTTPException(status_code=400, detail="Unsupported render method")

    render_width = int(max(64, min(width, 2048)))
    path = resolve_uploaded_file(filename)

    try:
        png = await run_in_threadpool(
            render_lensing_image,
            path,
            float(mass),
            float(scale),
            render_width,
            clean_method,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Image not found") from exc

    return StreamingResponse(io.BytesIO(png), media_type="image/png")


def run(
    *,
    port: int | None = None,
    host: str | None = None,
    reload: bool = False,
) -> None:
    """
    Start the web server for the Gravity Mirage API.

    This function serves as the entry point for running the web application using uvicorn.
    It handles port and host configuration, with support for environment variable overrides.

    Args:
        port: The port number to run the server on (keyword-only).
            Defaults to the PORT provided (if specified), environment variable if set, otherwise 2025.
        host: The host address to bind the server to (keyword-only).
            Defaults to '127.0.0.1' if not specified.
        reload: Enable auto-reload when code changes are detected (keyword-only).
            Defaults to False.

    Returns:
        None

    Example:
        >>> run()  # Runs on 127.0.0.1:2025
        >>> run(port=8000, host='0.0.0.0')  # Runs on 0.0.0.0:8000
        >>> run(reload=True)  # Runs with auto-reload enabled

    """
    env_port = getenv("PORT")
    if env_port and not port:
        port = int(env_port)
    if port is None:
        port = 2025

    if not host:
        host = "127.0.0.1"

    import uvicorn

    uvicorn.run("gravity_mirage.web:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run()
