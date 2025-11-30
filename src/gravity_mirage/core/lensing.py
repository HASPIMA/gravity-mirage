import io
from pathlib import Path

import numpy as np
from PIL import Image

from gravity_mirage.core.physics import SchwarzschildBlackHole
from gravity_mirage.core.ray_tracer import GravitationalRayTracer
from gravity_mirage.web.constants import PREVIEW_WIDTH


def compute_lensed_array_from_src_arr(  # noqa: PLR0915
    src_arr: np.ndarray,
    mass: float = 10.0,
    scale_rs: float = 100.0,
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

    rs = bh.schwarzschild_radius
    meters_per_pixel = (scale_rs * rs) / max_r
    b = r * meters_per_pixel

    if method == "weak":
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = np.vectorize(bh.deflection_angle_weak_field)(b)
    else:
        tracer = GravitationalRayTracer(bh)
        bins = min(128, max(8, int(max_r)))
        radii = np.linspace(0, max_r, bins)
        alpha_bins = np.zeros_like(radii)
        r0 = max(1e4 * rs, 1e6)

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

    rs_pixels = rs / meters_per_pixel
    mask_disk = (r <= rs_pixels) | captured
    result[mask_disk] = 0

    return result


def render_lensing_image(  # noqa: PLR0915
    src_path: Path,
    mass: float = 10.0,
    scale_rs: float = 100.0,
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

    rs = bh.schwarzschild_radius
    meters_per_pixel = (scale_rs * rs) / max_r
    b = r * meters_per_pixel

    if method == "weak":
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = np.vectorize(bh.deflection_angle_weak_field)(b)
    else:
        tracer = GravitationalRayTracer(bh)
        bins = min(128, max(8, int(max_r)))
        radii = np.linspace(0, max_r, bins)
        alpha_bins = np.zeros_like(radii)
        r0 = max(1e4 * rs, 1e6)

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

    rs_pixels = rs / meters_per_pixel
    mask_disk = (r <= rs_pixels) | captured
    result[mask_disk] = 0

    out_img = Image.fromarray(result)
    bio = io.BytesIO()
    out_img.save(bio, format="PNG")
    bio.seek(0)
    return bio.getvalue()
