import numpy as np
import math

# -----------------------------
# Utilities
# -----------------------------
def make_pupil_mask(N, radius_pix):
    yy, xx = np.indices((N, N)) - (N-1)/2
    rr = np.sqrt(xx**2 + yy**2)
    return (rr <= radius_pix).astype(float)

def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft2c(X):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X)))

def phase_diff(a, b):
    # returns wrapped phase of a*conj(b)
    return np.angle(a * np.conj(b))

# -----------------------------
# Zernike basis (unit disk) and gradients
# -----------------------------
def zernike_nm(n, m, rho, theta):
    # n>=0, |m|<=n, (n-m) even. OSA/Fringe indexing can be mapped as needed.
    m_abs = abs(m)
    R = np.zeros_like(rho)
    for k in range((n - m_abs)//2 + 1):
        c = ((-1)**k * math.factorial(n - k) /
             (math.factorial(k) *
              math.factorial((n + m_abs)//2 - k) *
              math.factorial((n - m_abs)//2 - k)))
        R += c * rho**(n - 2*k)
    if m == 0:
        Z = R
    elif m > 0:
        Z = R * np.cos(m_abs * theta)
    else:
        Z = R * np.sin(m_abs * theta)
    return Z

def zernike_set_with_grads(mask, orders):
    """
    Build Zernike basis Z_i and their Cartesian gradients dZ/dx, dZ/dy
    on a unit disk. Map pixels -> unit disk by x,y in [-1,1].
    """
    N = mask.shape[0]
    yy, xx = np.indices((N, N))
    x = (xx - (N-1)/2) / ((N-1)/2)
    y = (yy - (N-1)/2) / ((N-1)/2)
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)

    # Avoid outside-disk contributions
    inside = (rho <= 1.0) & (mask > 0.5)

    Z_list, dZdx_list, dZdy_list = [], [], []

    # numerical gradient via central differences on unit grid
    # safe for masked interior
    def num_grad(F):
        # gradients in unit-disk coordinates (x,y)
        # map pixel to unit scale spacing:
        # one pixel step equals delta = 2/(N-1) in x and y
        delta = 2.0 / (N - 1)
        Fx = (np.roll(F, -1, axis=1) - np.roll(F, 1, axis=1)) / (2*delta)
        Fy = (np.roll(F, -1, axis=0) - np.roll(F, 1, axis=0)) / (2*delta)
        # zero outside disk to avoid artifacts
        Fx[~inside] = 0.0
        Fy[~inside] = 0.0
        return Fx, Fy

    for (n, m) in orders:
        Z = zernike_nm(n, m, rho, theta)
        Z[~inside] = 0.0
        dZdx, dZdy = num_grad(Z)
        Z_list.append(Z)
        dZdx_list.append(dZdx)
        dZdy_list.append(dZdy)

    # Stack as [num_modes, N, N]
    Z = np.stack(Z_list, axis=0)
    dZdx = np.stack(dZdx_list, axis=0)
    dZdy = np.stack(dZdy_list, axis=0)
    return Z, dZdx, dZdy, inside

# Standard OSA low-order set through 4th radial order (exclude piston/tilt if desired)
def osa_orders_thru_4():
    return [
        # (n,m)
        (0,0),             # piston
        (1,-1),(1,1),      # tilt x,y
        (2,-2),(2,0),(2,2),# astig x/y, defocus, astig @45
        (3,-3),(3,-1),(3,1),(3,3),  # coma/trefoil
        (4,-4),(4,-2),(4,0),(4,2),(4,4) # primary spherical etc.
    ]

# -----------------------------
# Main pipeline
# -----------------------------
def dls_dao_refocus(psf_enface_complex, pupil_radius_pix, modes=None):
    """
    psf_enface_complex : complex2D at selected retinal depth (A * exp(i*phi))
    pupil_radius_pix   : radius of pupil in pixels within pupil plane
    modes              : list of (n,m); default OSA through 4th order
    Returns: dict with fields including WFE map, Zernike coeffs, corrected PSF, etc.
    """
    N = psf_enface_complex.shape[0]
    assert psf_enface_complex.shape[0] == psf_enface_complex.shape[1], "use square grid"

    # 1) Pupil-plane complex field via 2D FFT of en-face PSF
    E_pupil = fft2c(psf_enface_complex)

    # Optional apodization before/after FFT can help with ringing.
    # Normalize
    E_pupil /= np.max(np.abs(E_pupil)) + 1e-12

    # 2) Pupil mask in pixel domain
    mask = make_pupil_mask(N, pupil_radius_pix)

    # 3) Digital lateral shearing: 1-pixel shifts in x and y
    Exp = np.roll(E_pupil, -1, axis=1)
    Eyp = np.roll(E_pupil, -1, axis=0)

    dphidx_wrapped = phase_diff(Exp, E_pupil)
    dphidy_wrapped = phase_diff(Eyp, E_pupil)

    # Zero outside pupil
    dphidx_wrapped *= mask
    dphidy_wrapped *= mask

    # 4) Build Zernike gradient design matrices and solve least squares:
    if modes is None:
        modes = osa_orders_thru_4()

    Z, dZdx, dZdy, inside = zernike_set_with_grads(mask, modes)

    # Flatten inside-pupil samples
    sx = dphidx_wrapped[inside].ravel()
    sy = dphidy_wrapped[inside].ravel()

    # Build A matrix for [sx; sy] ~ [dZdx; dZdy] * c
    # A has shape [2*M, K], c has shape [K], where M=#pixels inside, K=#modes
    K = Z.shape[0]
    A = np.zeros((2*inside.sum(), K), dtype=np.float64)
    for k in range(K):
        A[:inside.sum(), k] = dZdx[k][inside]
        A[inside.sum():, k] = dZdy[k][inside]

    b = np.concatenate([sx, sy], axis=0)

    # Least-squares solve with mild Tikhonov to stabilize against noise
    lam = 1e-3
    ATA = A.T @ A + lam * np.eye(K)
    ATb = A.T @ b
    coeffs = np.linalg.solve(ATA, ATb)   # Zernike coefficients in radians at the pupil plane

    # 5) Reconstruct WFE (phase map) from Zernike basis (integrated solution)
    WFE = np.zeros((N, N), dtype=np.float64)
    for k in range(K):
        WFE += coeffs[k] * Z[k]

    # Mask outside pupil
    WFE *= mask

    # Optional: drop piston and tilt (first three OSA terms)
    # Set coeffs[0:3]=0 and recompute WFE if needed.

    # 6) Phase-conjugate in pupil plane and invert to refocus PSF
    E_corr = E_pupil * np.exp(-1j * WFE)
    psf_refocused = ifft2c(E_corr)

    out = {
        "E_pupil": E_pupil,
        "mask": mask,
        "dphidx": dphidx_wrapped,
        "dphidy": dphidy_wrapped,
        "zernike_modes": modes,
        "zernike_coeffs_rad": coeffs,
        "WFE_rad": WFE,
        "psf_refocused": psf_refocused
    }
    return out

# -----------------------------
# Power-vector conversion (Thibos) from low-order Zernike coeffs
# Requires mapping OSA indices -> (n,m) positions used above.
# -----------------------------
def low_order_power_vectors(coeffs, modes, pupil_radius_m):
    """
    Convert Zernike coefficients (in radians) to dioptric power vectors M, J0, J45.
    Uses Appendix A relations with R_max = pupil radius (meters) and phase->OPD scaling.
    Need wavelength to convert phase radians to OPD (meters): OPD = (lambda/2π)*phase.
    """
    # Pull needed terms (defocus Z_2^0, astig Z_2^{±2}) in our modes list
    # Find indices:
    idx_defocus = modes.index((2,0))
    idx_astig0 = modes.index((2,2))
    idx_astig45 = modes.index((2,-2))

    # Choose wavelength (meters) used by the OCT system (e.g., 1032 nm)
    lam = 1032e-9

    # Convert phase-radians coefficients -> OPD-meters coefficients
    c20 = coeffs[idx_defocus] * lam / (2*np.pi)
    c22 = coeffs[idx_astig0] * lam / (2*np.pi)
    c2m2 = coeffs[idx_astig45] * lam / (2*np.pi)

    Rmax = pupil_radius_m

    # Appendix A (paper): M, J0, J45 from Zernike coeffs and Rmax
    # Signs follow the convention in the paper; verify against your basis normalization.
    M   = (-c20) / (4*np.sqrt(3) * Rmax**2)
    J0  = (-c22) / (2*np.sqrt(6) * Rmax**2)
    J45 = (-c2m2) / (2*np.sqrt(6) * Rmax**2)

    # Cylinder/Axis and Sphere (negative cyl convention)
    C = -2*np.sqrt(J0**2 + J45**2)
    S = M - C/2
    alpha_rad = 0.5 * np.arctan2(J45, J0)
    alpha_deg = np.degrees(alpha_rad) % 180.0

    return dict(M=M, J0=J0, J45=J45, Sphere=S, Cylinder=C, Axis_deg=alpha_deg)
