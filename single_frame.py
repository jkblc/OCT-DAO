"""Single-frame OCT B-scan preview script.

Reads a raw spectral cube, applies background removal, k-linearisation,
dispersion compensation and windowing, then displays the log-magnitude
depth profile of one B-scan.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.fft import fft, fftshift
from scipy.interpolate import PchipInterpolator
import scipy.optimize as opt  # kept to preserve original import set
from scipy.signal import butter, filtfilt, windows

# -----------------------------------------------------------------------------
# Acquisition parameters
# -----------------------------------------------------------------------------
NUM_A: int = 2048                    # A-scans per B-scan (columns)
NUM_K: int = 2048                    # Camera pixels / spectral samples (rows)
RAW_FILE: Path = Path("RawBuffer5.raw")
LAMBDA_CAL_FILE: Path = Path("Lin_loc.mat")

FRAME_IDX: int = 0                   # B-scan index to display (0-based)

# Dispersion compensation coefficients
C2A: float = -50.0                   # Quadratic term
C3A: float = 0.0                     # Cubic term

GAUSS_STD_ALPHA: float = 1.6         # Spectral shaping parameter

# Background smoothing
BUTTER_N: int = 3
BUTTER_CUTOFF: float = 0.05

LOG_RANGE: Tuple[float, float] = (1, 4)      # Log-scale display range

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def load_wavelength(cal_path: str | Path, column: int = 1) -> np.ndarray:
    """Extract a wavelength vector (in meters) from *Lin_loc.mat*."""
    mat = scipy.io.loadmat(cal_path, squeeze_me=True,
                           struct_as_record=False, simplify_cells=True)

    def _as_vector(array: np.ndarray) -> np.ndarray:
        return np.asarray(array, dtype=np.float64).ravel() * 1e-9

    for key in ("b", "y"):
        if key in mat and isinstance(mat[key], np.ndarray):
            vec = mat[key]
            if vec.size > 100:
                return _as_vector(vec)

    if "L" in mat:
        L = mat["L"]
        if isinstance(L, dict):
            cols = list(L.keys())
            if len(cols) > column:
                return _as_vector(L[cols[column]])
        arr = np.asarray(L)
        if arr.dtype.names and len(arr.dtype.names) > column:
            return _as_vector(arr[arr.dtype.names[column]])
        if arr.ndim == 2 and arr.shape[1] > column:
            return _as_vector(arr[:, column])

    for val in mat.values():
        arr = np.asarray(val)
        if arr.ndim == 2 and arr.shape[1] > column and arr.size > 100:
            return _as_vector(arr[:, column])

    raise ValueError("Unable to locate wavelength vector in Lin_loc.mat")


def gaussian_window(m: int, alpha: float) -> np.ndarray:
    """Return a non-symmetric Gaussian window of length *m*."""
    std = (m - 1) / (2.0 * alpha)
    return windows.gaussian(m, std, sym=False)


def dispersion_kernel(num_samples: int, c2: float, c3: float,
                      fc: float = 0.0) -> np.ndarray:
    """Generate a quadratic-plus-cubic phase kernel for dispersion compensation."""
    fdd = np.linspace(0.0, 1.0, num_samples, dtype=np.float64)
    phase = -1j * (c2 * (fdd - fc) ** 2 + c3 * (fdd - fc) ** 3)
    return np.exp(phase)


def build_linear_freq_grids(wavelength: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return frequency and k-space axes resampled onto uniform grids."""
    c0 = 3.0e8
    freq = c0 / wavelength
    kvec = 2.0 * np.pi / wavelength

    if freq[0] > freq[-1]:           # ensure increasing frequency order
        freq, kvec = freq[::-1], kvec[::-1]

    freq_lin = np.linspace(freq[0], freq[-1], len(freq))
    kvec_lin = np.linspace(kvec[0], kvec[-1], len(kvec))
    return freq_lin, kvec_lin


def read_raw_cube(path: str | Path, num_k: int, num_a: int
                  ) -> Tuple[np.ndarray, int]:
    """Read a *.raw* cube into an array with shape (K, A, frames)."""
    data = np.fromfile(path, dtype=np.uint16)
    pixels_per_frame = num_k * num_a
    if data.size % pixels_per_frame:
        raise ValueError("Raw file size is not an integer multiple of K×A")
    n_frames = data.size // pixels_per_frame
    cube = data.reshape((num_k, num_a, n_frames), order="F")
    return cube, n_frames

# -----------------------------------------------------------------------------
# Single-frame processing
# -----------------------------------------------------------------------------
def process_single_frame(fringe: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    """Return log-magnitude depth spectrum of one B-scan after preprocessing."""
    freq_lin, _ = build_linear_freq_grids(wavelength)
    freq = 3.0e8 / wavelength

    # Background estimation and removal
    dc_raw = fringe.mean(axis=1)
    b_b, a_b = butter(BUTTER_N, BUTTER_CUTOFF, btype="low")
    dc_vec = filtfilt(b_b, a_b, dc_raw, axis=0)
    frame = fringe - dc_vec[:, None]

    if freq[0] > freq[-1]:
        freq, frame = freq[::-1], frame[::-1, :]

    # k-linearisation
    frame_lin = PchipInterpolator(freq, frame, axis=0)(freq_lin)

    # Dispersion compensation
    frame_lin = frame_lin.astype(np.complex128, copy=False)
    frame_lin *= dispersion_kernel(NUM_K, C2A, C3A)[:, None]

    # Complex background subtraction
    # Estimate the stationary parasitic signal (complex mean along A-scan axis)
    bg_complex = frame_lin.mean(axis=1, keepdims=True)
    # Remove it from every column
    frame_lin -= bg_complex

    # Spectral window
    frame_lin *= gaussian_window(NUM_K, GAUSS_STD_ALPHA)[:, None]

    # FFT to depth domain (zero-padding = 2×941)
    spec = fft(frame_lin, 2 * 941, axis=0, workers=-1)
    spec = fftshift(np.abs(spec), axes=0)[941:, :]        # positive depths only
    return np.log10(np.maximum(spec, 1e-12))

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main() -> None:
    wavelength = load_wavelength(LAMBDA_CAL_FILE)
    cube, n_frames = read_raw_cube(RAW_FILE, NUM_K, NUM_A)

    if FRAME_IDX >= n_frames:
        raise IndexError(f"Dataset contains {n_frames} frames, "
                         f"but FRAME_IDX={FRAME_IDX} was requested.")

    print(f"Loaded frame {FRAME_IDX} of {n_frames} from {RAW_FILE}")
    log_img = process_single_frame(cube[:, :, FRAME_IDX], wavelength)
    
    freq = 3e8 / wavelength
    freq_lin, _ = build_linear_freq_grids(wavelength)
    plt.figure()
    plt.plot(freq, label="Original Frequency")
    plt.plot(freq_lin, label="Linearized Frequency")
    plt.title("Frequency Resampling")
    plt.xlabel("Index")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    kernel = dispersion_kernel(NUM_K, C2A, C3A)
    plt.figure()
    plt.plot(np.angle(kernel))
    plt.title("Dispersion Kernel Phase Profile")
    plt.xlabel("Pixel Index")
    plt.ylabel("Phase (radians)")
    plt.tight_layout()
    plt.show()


    plt.figure()
    plt.plot(cube[:, 1000, FRAME_IDX])
    plt.title("Raw Interferogram (A-scan 1000)")
    plt.xlabel("Spectral Pixel Index")
    plt.ylabel("Intensity (a.u.)")
    plt.tight_layout()
    plt.show()

    plt.figure("OCT single-frame preview", figsize=(8, 5))
    plt.imshow(log_img, cmap="gray", vmin=LOG_RANGE[0], vmax=LOG_RANGE[1],
               aspect="auto", origin="upper")
    plt.title(f"B-scan {FRAME_IDX} | C2A={C2A}  C3A={C3A}")
    plt.xlabel("A-scan index")
    plt.ylabel("Depth (pixels)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
