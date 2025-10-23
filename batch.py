"""
Stream an OCT spectral cube to a multipage TIFF while showing a live preview.
Exactly the same reconstruction maths as before—only the writer changed.
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
    flin = linear_freq_grid(wl)
    f = 3.0e8 / wl

    dc = filtfilt(*butter(BUTTER_N, BUTTER_CUTOFF, btype="low"),
                  fringe.mean(1), axis=0)
    fr = fringe - dc[:, None]
    if f[0] > f[-1]:
        f, fr = f[::-1], fr[::-1, :]

    fr = PchipInterpolator(f, fr, axis=0)(flin)
    fr = fr.astype(np.complex128, copy=False)
    fr *= dispersion_kernel(NUM_K, C2A, C3A)[:, None]
    fr *= gaussian_window(NUM_K, GAUSS_STD_ALPHA)[:, None]

    mag = fftshift(np.abs(fft(fr, 2 * 941, axis=0, workers=-1)), 0)[941:, :]
    return np.log10(np.maximum(mag, 1e-12)).astype(np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# Main volume loop + live preview
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    wl   = load_wavelength(LAMBDA_CAL_FILE)
    cube, n = read_cube(RAW_FILE, NUM_K, NUM_A)
    print(f"Processing {n} frames …")

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(np.zeros((941, NUM_A), np.float32), cmap="gray",
                   vmin=LOG_RANGE[0], vmax=LOG_RANGE[1],
                   origin="upper", aspect="auto")
    ax.set_xlabel("A-scan index")
    ax.set_ylabel("Depth (pixels)")
    fig.canvas.manager.set_window_title("OCT volume processing")

    # High-level helper—one call per slice, append=True streams into one file
    for idx in range(n):
        slice_img = process(cube[:, :, idx], wl)

        tifffile.imwrite(
            OUTPUT_TIFF,
            slice_img,
            photometric="minisblack",
            append=True,
            metadata=None,          # avoids ImageJ metadata bloat
        )

        im.set_data(slice_img)
        ax.set_title(f"Frame {idx + 1} / {n}")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    plt.ioff()
    print(f"Done. Stack saved to: {OUTPUT_TIFF.resolve()}")

if __name__ == "__main__":
    main()
