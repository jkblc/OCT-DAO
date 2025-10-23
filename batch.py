"""
Stream an OCT spectral cube to a multipage TIFF while showing a live preview.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile                              # pip install tifffile
from scipy.fft import fft, fftshift
from scipy.interpolate import PchipInterpolator
import scipy.io
from scipy.signal import butter, filtfilt, windows

from DAO import dls_dao_refocus

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
NUM_A             : int            = 2048
NUM_K             : int            = 2048
RAW_FILE          : Path           = Path("RawBuffer5.raw")
LAMBDA_CAL_FILE   : Path           = Path("Lin_loc.mat")
OUTPUT_TIFF       : Path           = Path("OCT_volume.tiff")   # keep .tiff!

C2A, C3A          = -50.0, 0.0
GAUSS_STD_ALPHA   : float          = 1.6
BUTTER_N, BUTTER_CUTOFF = 3, 0.05
LOG_RANGE         : Tuple[float, float] = (1, 4)

# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities (unchanged core logic)
# ──────────────────────────────────────────────────────────────────────────────
def load_wavelength(path: str | Path, column: int = 1) -> np.ndarray:
    mat = scipy.io.loadmat(path, squeeze_me=True,
                           struct_as_record=False, simplify_cells=True)
    def _v(a: np.ndarray) -> np.ndarray:
        return np.asarray(a, np.float64).ravel() * 1e-9

    for key in ("b", "y"):
        if key in mat and isinstance(mat[key], np.ndarray) and mat[key].size > 100:
            return _v(mat[key])

    if "L" in mat:
        L = mat["L"]
        if isinstance(L, dict):
            cols = list(L.keys())
            if len(cols) > column:
                return _v(L[cols[column]])
        arr = np.asarray(L)
        if arr.dtype.names and len(arr.dtype.names) > column:
            return _v(arr[arr.dtype.names[column]])
        if arr.ndim == 2 and arr.shape[1] > column:
            return _v(arr[:, column])

    for val in mat.values():
        arr = np.asarray(val)
        if arr.ndim == 2 and arr.shape[1] > column and arr.size > 100:
            return _v(arr[:, column])

    raise ValueError("Wavelength vector not found in Lin_loc.mat")

def gaussian_window(m: int, alpha: float) -> np.ndarray:
    std = (m - 1) / (2.0 * alpha)
    return windows.gaussian(m, std, sym=False)

def dispersion_kernel(n: int, c2: float, c3: float, fc: float = 0.0) -> np.ndarray:
    fdd = np.linspace(0.0, 1.0, n, dtype=np.float64)
    return np.exp(-1j * (c2 * (fdd - fc) ** 2 + c3 * (fdd - fc) ** 3))

def linear_freq_grid(wl: np.ndarray) -> np.ndarray:
    c0 = 3.0e8
    f = c0 / wl
    if f[0] > f[-1]:
        f = f[::-1]
    return np.linspace(f[0], f[-1], len(f))

def read_cube(path: str | Path, k: int, a: int) -> Tuple[np.ndarray, int]:
    vec = np.fromfile(path, np.uint16)
    per = k * a
    if vec.size % per:
        raise ValueError("Raw size not multiple of K×A")
    n = vec.size // per
    return vec.reshape((k, a, n), order="F"), n

# ──────────────────────────────────────────────────────────────────────────────
# Frame processing
# ──────────────────────────────────────────────────────────────────────────────
def process(fringe: np.ndarray, wl: np.ndarray) -> np.ndarray:
    """
    Reconstruct one B-scan magnitude (log-scaled) with consistent math.
    Fixes: FFT sizing/cropping tied to spectral length; DC along k-axis;
    dispersion/window length tied to resampled grid; correct op order.
    """
    # 1) Build k-linear grid (length M)
    flin = linear_freq_grid(wl)                   # shape (M,)
    c0 = 3.0e8
    f_raw = c0 / wl
    if f_raw[0] > f_raw[-1]:
        f_raw = f_raw[::-1]
        fringe = fringe[::-1, :]                  # keep (k, a) aligned

    # 2) DC removal per spectrum (along k)
    # Do NOT average across A-scans and filter that; subtract per-k mean directly.
    fr = fringe - np.mean(fringe, axis=1, keepdims=True)

    # 3) Resample to linear k with PCHIP (vectorized over A-scans)
    fr = PchipInterpolator(f_raw, fr, axis=0)(flin)  # shape (M, A)

    # 4) Cast up to complex128 (phase-sensitive steps ahead)
    fr = fr.astype(np.complex128, copy=False)

    # 5) Match kernel/window lengths to resampled grid (M), not NUM_K blindly
    M = fr.shape[0]
    fr *= dispersion_kernel(M, C2A, C3A)[:, None]
    fr *= gaussian_window(M, GAUSS_STD_ALPHA)[:, None]

    # 6) FFT with *consistent* zero-padding and center/crop using M
    depth_fft = fft(fr, 2 * M, axis=0, workers=-1)   # size 2M along depth
    depth_fft = fftshift(depth_fft, axes=0)
    half = M                                        # crop positive depths
    mag = np.abs(depth_fft[half:, :])

    # 7) Log compression
    return np.log10(np.maximum(mag, 1e-12)).astype(np.float32)


def apply_dao_to_volume(volume_complex: np.ndarray,
                        patch_size: int = 256,
                        depth_index: int | None = None,
                        visualize: bool = True):
    """
    Apply digital adaptive optics (DAO) to a 3-D OCT complex volume.

    Parameters
    ----------
    volume_complex : ndarray
        Complex-valued OCT volume, shape (depth, x, y).
    patch_size : int
        Size of square patch (in pixels) for DAO estimation.
    depth_index : int or None
        Depth index to select for en-face DAO patch.
        If None, use the layer with maximum total intensity.
    visualize : bool
        If True, show before/after PSF comparison.

    Returns
    -------
    volume_corrected : ndarray
        Phase-corrected complex volume, same shape as input.
    dao_result : dict
        Full dictionary from `dls_dao_refocus`, including
        WFE map, Zernike coefficients, and refocused PSF.
    """

    if volume_complex.ndim != 3:
        raise ValueError("volume_complex must have shape (depth, x, y)")

    depth, nx, ny = volume_complex.shape

    # 1. Choose depth slice
    if depth_index is None:
        intensity_profile = np.sum(np.abs(volume_complex)**2, axis=(1, 2))
        depth_index = int(np.argmax(intensity_profile))
    enface_complex = volume_complex[depth_index]

    # 2. Extract centered square patch
    cx, cy = nx // 2, ny // 2
    half = patch_size // 2
    psf_patch = enface_complex[cx - half:cx + half, cy - half:cy + half]

    if psf_patch.shape[0] < patch_size or psf_patch.shape[1] < patch_size:
        raise ValueError("Patch size too large for the input volume dimensions.")

    # 3. Run DAO
    dao_result = dls_dao_refocus(psf_patch, pupil_radius_pix=patch_size // 2)
    wfe = dao_result["WFE_rad"]

    # 4. Apply phase correction to full volume
    # Extend WFE map to match (x,y) dimensions by tiling
    corr_map = np.exp(-1j * wfe)
    if wfe.shape != (nx, ny):
        corr_map = np.pad(corr_map,
                          (((nx - wfe.shape[0]) // 2, (nx - wfe.shape[0] + 1) // 2),
                           ((ny - wfe.shape[1]) // 2, (ny - wfe.shape[1] + 1) // 2)),
                          mode='edge')
        corr_map = corr_map[:nx, :ny]

    volume_corrected = volume_complex * corr_map

    # 5. Optional visualization
    if visualize:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title(f"Original en-face (z={depth_index})")
        plt.imshow(np.abs(enface_complex), cmap="gray")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title("DAO-corrected patch")
        plt.imshow(np.abs(dao_result["psf_refocused"]), cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return volume_corrected, dao_result

# ──────────────────────────────────────────────────────────────────────────────
# Main volume loop + live preview
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    wl, (cube, n) = load_wavelength(LAMBDA_CAL_FILE), read_cube(RAW_FILE, NUM_K, NUM_A)
    print(f"Processing {n} frames …")

    # Pre-allocate in half precision (≈3 GB instead of 6 GB)
    volume_complex = np.memmap("volume_complex.dat",
                               dtype=np.complex64, mode="w+",
                               shape=(941, NUM_A, n))

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(np.zeros((941, NUM_A), np.float32),
                   cmap="gray", vmin=LOG_RANGE[0], vmax=LOG_RANGE[1],
                   origin="upper", aspect="auto")
    ax.set_xlabel("A-scan index")
    ax.set_ylabel("Depth (pixels)")
    fig.canvas.manager.set_window_title("OCT + DAO stream")

    # ─── stream frames ───────────────────────────────────────────────
    for idx in range(n):
        flin = linear_freq_grid(wl)
        f = 3.0e8 / wl
        dc = filtfilt(*butter(BUTTER_N, BUTTER_CUTOFF, btype="low"),
                      cube[:, :, idx].mean(1), axis=0)
        fr = cube[:, :, idx] - dc[:, None]
        if f[0] > f[-1]:
            f, fr = f[::-1], fr[::-1, :]
        fr = PchipInterpolator(f, fr, axis=0)(flin)
        fr = fr.astype(np.complex64, copy=False)
        fr *= dispersion_kernel(NUM_K, C2A, C3A).astype(np.complex64)[:, None]
        fr *= gaussian_window(NUM_K, GAUSS_STD_ALPHA)[:, None]
        bscan = fftshift(fft(fr, 2 * 941, axis=0), 0)[941:, :].astype(np.complex64)
        volume_complex[:, :, idx] = bscan  # save complex slice on disk

        mag = np.log10(np.maximum(np.abs(bscan), 1e-12)).astype(np.float32)
        tifffile.imwrite(OUTPUT_TIFF, mag, append=True,
                         photometric="minisblack", metadata=None)

        im.set_data(mag)
        ax.set_title(f"Frame {idx+1}/{n}")
        fig.canvas.draw_idle(); fig.canvas.flush_events()

    plt.ioff()
    print("Streaming done — running DAO correction …")

    # ─── DAO phase correction on disk-mapped volume ─────────────────────────
    volume_complex.flush()                      # ensure data written
    volume_mm = np.memmap("volume_complex.dat", dtype=np.complex64,
                          mode="r", shape=(941, NUM_A, n))

    volume_corrected, dao_result = apply_dao_to_volume(volume_mm,
                                                       patch_size=256,
                                                       depth_index=None,
                                                       visualize=True)

    # write corrected version
    OUTPUT_DAO = OUTPUT_TIFF.with_stem(OUTPUT_TIFF.stem + "_DAO")
    for idx in range(n):
        mag_corr = np.log10(np.maximum(np.abs(volume_corrected[:, :, idx]), 1e-12)).astype(np.float32)
        tifffile.imwrite(OUTPUT_DAO, mag_corr, append=True,
                         photometric="minisblack", metadata=None)

    print(f"Uncorrected → {OUTPUT_TIFF.resolve()}")
    print(f"DAO-corrected → {OUTPUT_DAO.resolve()}")


if __name__ == "__main__":
    main()
