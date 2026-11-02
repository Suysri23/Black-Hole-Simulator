"""
Shared physics for the Schwarzschild black-hole simulator.

Everything the renderer needs that is *not* per-pixel lives here, so that
main.py (GPU) and validate_physics.py (tests) use the same equations:

  * Schwarzschild geometry and the isotropic <-> areal coordinate maps
  * Novikov-Thorne (Page & Thorne 1974) thin-disk flux profile
  * Physical disk temperature for a given mass / Eddington ratio
  * Blackbody -> linear sRGB colour table (CIE 1931, Wyman et al. 2013 fit)
  * A NumPy port of the shader's ray integrator (used only for testing)
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

# ============================================================
# CONSTANTS (SI)
# ============================================================

G = 6.67430e-11
C = 299792458.0
M_SUN = 1.98847e30
SIGMA_SB = 5.670374419e-8
H_PLANCK = 6.62607015e-34
K_B = 1.380649e-23
M_PROTON = 1.67262192e-27
SIGMA_THOMSON = 6.6524587321e-29

# Radiative efficiency of a thin disk ending at the Schwarzschild ISCO
ISCO_EFFICIENCY = 1.0 - np.sqrt(8.0 / 9.0)


# ============================================================
# GEOMETRY
# ============================================================

def schwarzschild_radius(mass_solar):
    return 2.0 * G * mass_solar * M_SUN / C**2


def gravitational_radius(mass_solar):
    return G * mass_solar * M_SUN / C**2


def critical_impact_parameter(rs=1.0):
    return 1.5 * np.sqrt(3.0) * rs


def isotropic_to_areal(rho, rs=1.0):
    """r = rho (1 + rs/(4 rho))^2"""
    rho = np.asarray(rho, dtype=float)
    return rho * (1.0 + 0.25 * rs / rho) ** 2


def areal_to_isotropic(r, rs=1.0):
    """Inverse of isotropic_to_areal (exterior branch, r >= rs)."""
    r = np.asarray(r, dtype=float)
    return 0.5 * ((r - 0.5 * rs) + np.sqrt(r * (r - rs)))


def optical_index(rho, rs=1.0):
    """
    Effective refractive index of Schwarzschild space in isotropic
    coordinates, n = (1 + a/rho)^3 / (1 - a/rho), a = rs/4.
    Light rays are the rays of this medium; n*rho*sin(psi) = b is conserved.
    """
    a = 0.25 * rs
    rho = np.asarray(rho, dtype=float)
    return (1.0 + a / rho) ** 3 / (1.0 - a / rho)


# ============================================================
# EMITTER REDSHIFT
# ============================================================

def redshift_factor(r_areal, l_over_e, rs=1.0):
    """
    g = nu_observed / nu_emitted for a photon emitted by gas on a circular
    geodesic orbit at areal radius r (valid for r >= 3 rs, the ISCO).

        g = 1 / ( u^t (1 - Omega * L/E) )
        u^t   = 1 / sqrt(1 - 3M/r)       (M = rs/2)
        Omega = sqrt(M / r^3)

    l_over_e is the photon's angular momentum about the disk axis divided by
    its energy (a length, conserved along the ray). It contains the
    Doppler shift; u^t contains transverse Doppler + gravitational shift.
    """
    m = 0.5 * rs
    ut = 1.0 / np.sqrt(1.0 - 3.0 * m / r_areal)
    omega = np.sqrt(m / r_areal**3)
    return 1.0 / (ut * (1.0 - omega * l_over_e))


# ============================================================
# NOVIKOV-THORNE THIN DISK (SCHWARZSCHILD)
# ============================================================

_NT_CACHE = {}


def novikov_thorne_profile(r_max=1.0e6, n=40000):
    """
    Dimensionless Page-Thorne flux for a Schwarzschild black hole.

    Returns (r, flux, energy) with r in units of M (G=c=1), and

        flux = F * 4 pi r_g^2 / (Mdot c^2)  =  f(r) / r

    where F is the radiated flux per disk face in the gas rest frame.

        f(r) = -Omega,r (E - Omega L)^-2  Int_{6M}^{r} (E - Omega L) L,r dr

    For Schwarzschild: E - Omega L = sqrt(1 - 3/r), so the integrand has a
    closed form and the zero-torque boundary at the ISCO (r = 6M) is exact.
    """
    key = (r_max, n)
    if key in _NT_CACHE:
        return _NT_CACHE[key]

    r = np.geomspace(6.0, r_max, n)
    x = 1.0 - 3.0 / r
    energy = (1.0 - 2.0 / r) / np.sqrt(x)
    dL_dr = 0.5 / np.sqrt(r) * (1.0 - 6.0 / r) / x**1.5
    integrand = np.sqrt(x) * dL_dr
    cumulative = cumulative_trapezoid(integrand, r, initial=0.0)
    f = 1.5 * r**-2.5 / x * cumulative
    result = (r, f / r, energy, f)
    _NT_CACHE[key] = result
    return result


def disk_temperature_lut(n=512, r_out_rs=15.0):
    """
    Relative emitted temperature tau(r) = (F/F_max)^(1/4) sampled
    logarithmically in areal radius from 3 rs (ISCO) to r_out_rs.
    Returns float32 array of length n.
    """
    r_m, flux, _, _ = novikov_thorne_profile()
    r_rs = np.geomspace(3.0, r_out_rs, n)
    f_here = np.interp(2.0 * r_rs, r_m, flux)
    return ((f_here / flux.max()) ** 0.25).astype(np.float32)


def eddington_luminosity(mass_solar):
    """Watts."""
    return 4.0 * np.pi * G * mass_solar * M_SUN * M_PROTON * C / SIGMA_THOMSON


def accretion_rate(mass_solar, eddington_ratio):
    """Mass accretion rate in kg/s for L = eddington_ratio * L_Edd."""
    luminosity = eddington_ratio * eddington_luminosity(mass_solar)
    return luminosity / (ISCO_EFFICIENCY * C**2)


def disk_peak_temperature(mass_solar, eddington_ratio):
    """
    Maximum effective temperature (kelvin) of the thin disk.
    T^4 = F_max / sigma,  F = Mdot c^2 / (4 pi r_g^2) * f(r)/r
    Scales as (Eddington ratio)^(1/4) * M^(-1/4).
    """
    _, flux, _, _ = novikov_thorne_profile()
    r_g = gravitational_radius(mass_solar)
    mdot_c2 = accretion_rate(mass_solar, eddington_ratio) * C**2
    f_max = mdot_c2 / (4.0 * np.pi * r_g**2) * flux.max()
    return (f_max / SIGMA_SB) ** 0.25


def isco_orbital_frequency(mass_solar):
    """Orbital frequency (Hz) at r = 6 GM/c^2, as seen from infinity."""
    return C**3 / (2.0 * np.pi * G * mass_solar * M_SUN * 6.0**1.5)


# ============================================================
# BLACKBODY COLOUR
# ============================================================

def _cie_lobe(wl, mu, s1, s2):
    s = np.where(wl < mu, s1, s2)
    return np.exp(-0.5 * ((wl - mu) / s) ** 2)


def _cie_xyz(wl_nm):
    """Multi-lobe Gaussian fit of the CIE 1931 colour matching functions."""
    x = (1.056 * _cie_lobe(wl_nm, 599.8, 37.9, 31.0)
         + 0.362 * _cie_lobe(wl_nm, 442.0, 16.0, 26.7)
         - 0.065 * _cie_lobe(wl_nm, 501.1, 20.4, 26.2))
    y = (0.821 * _cie_lobe(wl_nm, 568.8, 46.9, 40.5)
         + 0.286 * _cie_lobe(wl_nm, 530.9, 16.3, 31.1))
    z = (1.217 * _cie_lobe(wl_nm, 437.0, 11.8, 36.0)
         + 0.681 * _cie_lobe(wl_nm, 459.0, 26.0, 13.8))
    return x, y, z


_XYZ_TO_SRGB = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570],
])


def blackbody_xyz(temperature):
    """CIE XYZ of the visible-band spectral radiance of a blackbody."""
    t = np.atleast_1d(np.asarray(temperature, dtype=float))[:, None]
    wl_nm = np.linspace(380.0, 780.0, 161)
    wl = wl_nm * 1e-9
    planck = (2.0 * H_PLANCK * C**2 / wl**5
              / np.expm1(H_PLANCK * C / (wl * K_B * t)))
    x, y, z = _cie_xyz(wl_nm)
    d_wl = wl_nm[1] - wl_nm[0]
    return np.stack([(planck * x).sum(1), (planck * y).sum(1),
                     (planck * z).sum(1)], axis=1) * d_wl


def blackbody_rgb_lut(n=1024, t_min=500.0, t_max=60000.0, t_ref=6500.0):
    """
    Linear-sRGB visible radiance of a blackbody on a log-T grid, scaled so a
    6500 K blackbody has luminance Y = 1. Returns (temperatures, rgb[n,3]).

    Because I_nu / nu^3 is invariant along a ray, a blackbody seen with
    redshift g is exactly a blackbody at temperature g*T. The renderer
    therefore only needs LUT(g * T_emitted) -- colour *and* brightness shift
    come out together.
    """
    temps = np.geomspace(t_min, t_max, n)
    xyz = blackbody_xyz(temps) / blackbody_xyz(t_ref)[0, 1]
    rgb = np.clip(xyz @ _XYZ_TO_SRGB.T, 0.0, None)
    return temps, rgb.astype(np.float32)


# ============================================================
# CPU REFERENCE OF THE SHADER'S RAY INTEGRATOR (for testing)
# ============================================================

def _normalize(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _acceleration(p, d, rs):
    a = 0.25 * rs
    r = np.maximum(np.linalg.norm(p, axis=1), 0.2601 * rs)
    deriv = -(a / r) * (1.0 / (r - a) + 3.0 / (r + a))
    grad = (deriv / r)[:, None] * p
    return grad - np.sum(grad * d, axis=1)[:, None] * d


def _rk4(p, d, h, rs):
    k1p, k1d = d, _acceleration(p, d, rs)
    p2, d2 = p + 0.5 * h * k1p, _normalize(d + 0.5 * h * k1d)
    k2p, k2d = d2, _acceleration(p2, d2, rs)
    p3, d3 = p + 0.5 * h * k2p, _normalize(d + 0.5 * h * k2d)
    k3p, k3d = d3, _acceleration(p3, d3, rs)
    p4, d4 = p + h * k3p, _normalize(d + h * k3d)
    k4p, k4d = d4, _acceleration(p4, d4, rs)
    p_new = p + h * (k1p + 2 * k2p + 2 * k3p + k4p) / 6.0
    d_new = _normalize(d + h * (k1d + 2 * k2d + 2 * k3d + k4d) / 6.0)
    return p_new, d_new


CAPTURED, ESCAPED, UNRESOLVED = 1, 2, 0


def trace_isotropic_rays(p0, d0, rs=1.0, escape_radius=25.0,
                         base_step=0.045, max_steps=200000):
    """
    Vectorised NumPy port of the fragment shader's integration loop
    (same equations, same step-size rules, no disk). Rays are advanced
    together; finished rays get step 0.

    Returns (status, position, direction); status is CAPTURED, ESCAPED or
    UNRESOLVED.
    """
    p = np.array(p0, dtype=float)
    d = _normalize(np.array(d0, dtype=float))
    status = np.zeros(len(p), dtype=int)
    capture = 0.252 * rs

    for _ in range(max_steps):
        r = np.linalg.norm(p, axis=1)
        active = status == 0
        status[active & (r <= capture)] = CAPTURED
        status[active & (r > escape_radius * rs)
               & (np.sum(p * d, axis=1) > 0.0)] = ESCAPED
        active = status == 0
        if not active.any():
            break

        near = np.clip((r - capture) / max(5.0 * rs - capture, 1e-3 * rs),
                       0.12, 1.0)
        far = np.clip(r / (6.0 * rs), 1.0, 4.0)
        scale = np.where(r < 5.0 * rs, near, far)
        h = (base_step * rs * scale * active)[:, None]
        p, d = _rk4(p, d, h, rs)

    return status, p, d


def launch_parallel_rays(impact_parameters, rs=1.0, start_radius=300.0):
    """Rays travelling along +x from x = -start_radius with exact b = n rho sin(psi)."""
    b = np.asarray(impact_parameters, dtype=float) * rs
    y0 = b.copy()
    for _ in range(4):
        rho0 = np.hypot(start_radius * rs, y0)
        y0 = b / optical_index(rho0, rs)
    p0 = np.stack([-start_radius * rs * np.ones_like(b), y0,
                   np.zeros_like(b)], axis=1)
    d0 = np.tile([1.0, 0.0, 0.0], (len(b), 1))
    return p0, d0
