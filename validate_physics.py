import numpy as np
from scipy.integrate import solve_ivp, trapezoid

import blackhole_physics as bh


# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

G = 6.67430e-11
C = 299792458.0
M_SUN = 1.98847e30


# ============================================================
# SCHWARZSCHILD GEOMETRY
# ============================================================

def schwarzschild_radius(mass_solar):
    """Return the Schwarzschild radius in metres."""
    if mass_solar <= 0:
        raise ValueError("Black hole mass must be positive.")

    mass_kg = mass_solar * M_SUN
    return 2.0 * G * mass_kg / C**2


def critical_impact_parameter(rs):
    """Critical impact parameter for a Schwarzschild black hole."""
    if rs <= 0:
        raise ValueError("Schwarzschild radius must be positive.")

    return 3.0 * np.sqrt(3.0) * rs / 2.0


# ============================================================
# SCHWARZSCHILD NULL GEODESIC
# ============================================================

def geodesic(phi, state, rs):
    """
    Null-geodesic equation in areal Schwarzschild coordinates.

    u'' + u = (3/2) rs u^2
    u = 1/r
    """
    u, v = state
    return (v, -u + 1.5 * rs * u**2)


# ============================================================
# PHOTON TRAJECTORY CLASSIFICATION
# ============================================================

def trace_photon(
    impact_parameter,
    rs,
    *,
    rtol=1e-10,
    atol=1e-12,
    max_phi=1.0e4,
    max_step=0.01,
):
    """
    Classify a photon as escaped, captured, or critical/unresolved.

    The photon starts at a large finite radius R = 10000 b. The initial
    derivative is obtained from the first integral of the null-geodesic
    equation. A ray that reaches the areal horizon r = rs is classified
    as captured. A ray whose radial inverse reaches a turning point while
    increasing u is classified as scattered/escaped.

    Returns a dictionary. Deflection is in radians for escaped rays.
    """
    b = float(impact_parameter)
    rs = float(rs)

    if b <= 0 or rs <= 0:
        raise ValueError("Impact parameter and Schwarzschild radius must be positive.")

    b_critical = critical_impact_parameter(rs)

    # At the exact separatrix, the ray asymptotically approaches the
    # photon sphere. Floating-point ODE integration can spuriously report
    # a finite turning point, so explicitly classify a tiny tolerance.
    if np.isclose(b, b_critical, rtol=1e-11, atol=0.0):
        return {
            "status": "critical",
            "deflection": None,
            "phi_turn": None,
            "final_u": 2.0 / (3.0 * rs),
            "solver_message": "Analytically identified critical photon orbit.",
        }

    R = 10000.0 * b
    u0 = 1.0 / R

    radial_term = 1.0 / b**2 - u0**2 + rs * u0**3
    if radial_term <= 0:
        raise ValueError("Invalid initial conditions: radial velocity is not real.")

    v0 = np.sqrt(radial_term)

    def turning_point(phi, state):
        return state[1]

    turning_point.terminal = True
    turning_point.direction = -1

    def horizon(phi, state):
        return state[0] - 1.0 / rs

    horizon.terminal = True
    horizon.direction = 1

    solution = solve_ivp(
        lambda phi, y: geodesic(phi, y, rs),
        (0.0, max_phi),
        (u0, v0),
        events=(turning_point, horizon),
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )

    if len(solution.t_events[0]) > 0:
        phi_turn = float(solution.t_events[0][0])
        phi_flat = np.arccos(np.clip(b / R, -1.0, 1.0))
        alpha = 2.0 * (phi_turn - phi_flat)

        return {
            "status": "escaped",
            "deflection": float(alpha),
            "phi_turn": phi_turn,
            "final_u": float(solution.y[0, -1]),
            "solver_message": solution.message,
        }

    if len(solution.t_events[1]) > 0:
        return {
            "status": "captured",
            "deflection": None,
            "phi_turn": None,
            "final_u": float(solution.y[0, -1]),
            "solver_message": solution.message,
        }

    return {
        "status": "near_critical_or_unresolved",
        "deflection": None,
        "phi_turn": None,
        "final_u": float(solution.y[0, -1]),
        "solver_message": solution.message,
    }


def calculate_deflection(b, rs, **kwargs):
    """
    Backwards-compatible helper for weak/ordinary scattering tests.
    Raises RuntimeError if the ray is not classified as escaped.
    """
    result = trace_photon(b, rs, **kwargs)

    if result["status"] != "escaped":
        raise RuntimeError(
            f"Deflection is undefined for a ray classified as {result['status']}."
        )

    return result["deflection"]


# ============================================================
# ANALYTICAL DEFLECTION APPROXIMATIONS
# ============================================================

def first_order_deflection(b, rs):
    x = rs / b
    return 2.0 * x


def second_order_deflection(b, rs):
    x = rs / b
    return 2.0 * x + (15.0 * np.pi / 16.0) * x**2


def third_order_deflection(b, rs):
    x = rs / b
    return (
        2.0 * x
        + (15.0 * np.pi / 16.0) * x**2
        + (16.0 / 3.0) * x**3
    )


# ============================================================
# VALIDATION HELPERS
# ============================================================

passed = 0
failed = 0


def check(name, condition):
    global passed, failed

    if condition:
        passed += 1
        print(f"[PASS] {name}")
    else:
        failed += 1
        print(f"[FAIL] {name}")


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():
    global passed, failed
    passed = 0
    failed = 0

    mass = 10.0
    rs = schwarzschild_radius(mass)
    b_critical = critical_impact_parameter(rs)

    print("\nBLACK HOLE PHYSICS VALIDATION")
    print("=" * 64)
    print(f"Mass: {mass:.2f} solar masses")
    print(f"Schwarzschild radius: {rs / 1000:.6f} km")
    print(f"Critical impact parameter: {b_critical / 1000:.6f} km")

    # --------------------------------------------------------
    # 1. THEORETICAL REFERENCE VALUES
    # --------------------------------------------------------

    photon_sphere = 1.5 * rs
    isco = 3.0 * rs

    check("Schwarzschild radius is positive", rs > 0)
    check("Photon sphere is at 1.5 rs", np.isclose(photon_sphere / rs, 1.5))
    check("ISCO is at 3 rs", np.isclose(isco / rs, 3.0))
    check(
        "Critical impact parameter is 3sqrt(3)rs/2",
        np.isclose(b_critical / rs, 3.0 * np.sqrt(3.0) / 2.0),
    )

    # --------------------------------------------------------
    # 2. PHOTON SPHERE EQUILIBRIUM
    # --------------------------------------------------------

    u_photon = 1.0 / photon_sphere
    equilibrium = -u_photon + 1.5 * rs * u_photon**2

    print("\nPHOTON SPHERE")
    print("-" * 64)
    print(f"Geodesic acceleration: {equilibrium:.12e}")
    check("Photon sphere equilibrium", abs(equilibrium) < 1e-12)

    # --------------------------------------------------------
    # 3. WEAK-FIELD GRAVITATIONAL LENSING
    # --------------------------------------------------------

    print("\nWEAK-FIELD GRAVITATIONAL LENSING")
    print("-" * 64)

    b = 10.0 * rs
    numerical = calculate_deflection(b, rs)
    first = first_order_deflection(b, rs)
    second = second_order_deflection(b, rs)
    third = third_order_deflection(b, rs)

    def relative_error(reference, approximation):
        return abs(reference - approximation) / abs(reference) * 100.0

    error_first = relative_error(numerical, first)
    error_second = relative_error(numerical, second)
    error_third = relative_error(numerical, third)

    print(f"Impact parameter: {b / rs:.2f} rs")
    print(f"Numerical deflection: {np.degrees(numerical):.6f} degrees")
    print(
        f"First-order: {np.degrees(first):.6f} degrees "
        f"(error {error_first:.4f}%)"
    )
    print(
        f"Second-order: {np.degrees(second):.6f} degrees "
        f"(error {error_second:.4f}%)"
    )
    print(
        f"Third-order: {np.degrees(third):.6f} degrees "
        f"(error {error_third:.4f}%)"
    )

    check("Numerical photon deflection is positive", numerical > 0)
    check("Second-order improves over first-order", error_second < error_first)
    check("Third-order improves over second-order", error_third < error_second)
    check("Third-order approximation error below 1%", error_third < 1.0)

    # --------------------------------------------------------
    # 4. STRONG-FIELD PHOTON CLASSIFICATION
    # --------------------------------------------------------

    print("\nSTRONG-FIELD PHOTON CLASSIFICATION")
    print("-" * 64)

    captured = trace_photon(2.4 * rs, rs, max_phi=200.0)
    critical = trace_photon(b_critical, rs, max_phi=200.0)
    scattered = trace_photon(3.0 * rs, rs, max_phi=200.0)
    near_critical = trace_photon(1.001 * b_critical, rs, max_phi=2000.0)

    print(f"b = 2.4 rs: {captured['status']}")
    print(f"b = bc: {critical['status']}")
    print(f"b = 3.0 rs: {scattered['status']}")
    print(f"b = 1.001 bc: {near_critical['status']}")

    check("Subcritical ray is captured", captured["status"] == "captured")
    check("Supercritical ray escapes", scattered["status"] == "escaped")
    check("Critical ray is classified as critical", critical["status"] == "critical")
    check(
        "Near-critical supercritical ray escapes",
        near_critical["status"] == "escaped",
    )

    if scattered["status"] == "escaped":
        print(
            "Deflection at b = 3 rs: "
            f"{np.degrees(scattered['deflection']):.6f} degrees"
        )

    # --------------------------------------------------------
    # 5. ISOTROPIC COORDINATE REFERENCES
    # --------------------------------------------------------

    print("\nISOTROPIC COORDINATE REFERENCES")
    print("-" * 64)

    horizon_iso = rs / 4.0
    photon_iso = (2.0 + np.sqrt(3.0)) * rs / 4.0
    isco_iso = (5.0 + np.sqrt(24.0)) * rs / 4.0

    print(f"Event horizon: {horizon_iso / rs:.9f} rs")
    print(f"Photon sphere: {photon_iso / rs:.9f} rs")
    print(f"ISCO: {isco_iso / rs:.9f} rs")

    check("Isotropic event horizon", np.isclose(horizon_iso / rs, 0.25))
    check(
        "Isotropic photon sphere",
        np.isclose(photon_iso / rs, 0.9330127019, atol=1e-8),
    )
    check(
        "Isotropic ISCO",
        np.isclose(isco_iso / rs, 2.4747448714, atol=1e-8),
    )

    # --------------------------------------------------------
    # 6. SHADER RAY EQUATIONS vs AREAL-COORDINATE GEODESICS
    # --------------------------------------------------------
    # The renderer integrates rays of the isotropic optical medium
    # n(rho). Here the NumPy port of that exact integrator is checked
    # against the independent areal-coordinate geodesic solver above.

    print("\nSHADER RAY INTEGRATOR (ISOTROPIC OPTICAL MEDIUM)")
    print("-" * 64)

    rho_test = np.array([0.9, 1.7, 5.0, 40.0])
    r_areal = bh.isotropic_to_areal(rho_test)
    check(
        "Isotropic <-> areal maps are inverses",
        np.allclose(bh.areal_to_isotropic(r_areal), rho_test),
    )
    check(
        "Bouguer invariant: n*rho = r / sqrt(1 - rs/r)",
        np.allclose(
            bh.optical_index(rho_test) * rho_test,
            r_areal / np.sqrt(1.0 - 1.0 / r_areal),
        ),
    )

    p0, d0 = bh.launch_parallel_rays([10.0])
    status, _, d_end = bh.trace_isotropic_rays(p0, d0, escape_radius=300.0)
    shader_deflection = float(np.arccos(np.clip(d_end[0, 0], -1.0, 1.0)))
    rel = abs(shader_deflection - numerical) / numerical * 100.0
    print(f"b = 10 rs: shader integrator {np.degrees(shader_deflection):.5f} deg, "
          f"areal solver {np.degrees(numerical):.5f} deg ({rel:.3f}% apart)")
    check("Shader deflection matches areal geodesic within 1%", rel < 1.0)

    factors = np.array([0.97, 0.99, 1.01, 1.03])
    p0, d0 = bh.launch_parallel_rays(factors * 1.5 * np.sqrt(3.0))
    status, _, _ = bh.trace_isotropic_rays(p0, d0, escape_radius=300.0)
    print("capture at 0.97/0.99/1.01/1.03 b_c:", list(status))
    check(
        "Shader capture threshold sits at b_c (within 1%)",
        list(status) == [bh.CAPTURED, bh.CAPTURED, bh.ESCAPED, bh.ESCAPED],
    )

    # --------------------------------------------------------
    # 7. EMITTER REDSHIFT
    # --------------------------------------------------------

    print("\nEMITTER REDSHIFT g")
    print("-" * 64)

    r_test = np.array([3.0, 6.0, 20.0])
    g_face_on = bh.redshift_factor(r_test, 0.0)
    expected = np.sqrt(1.0 - 1.5 / r_test)
    print(f"face-on g at 3/6/20 rs: {np.round(g_face_on, 4)}")
    check("Face-on g = sqrt(1 - 3M/r) (grav. + transverse Doppler)",
          np.allclose(g_face_on, expected))
    check("g -> 1 far from the hole",
          abs(bh.redshift_factor(1e6, 5.0) - 1.0) < 1e-3)

    approaching = bh.redshift_factor(6.0, +5.0)
    receding = bh.redshift_factor(6.0, -5.0)
    print(f"r = 6 rs, |L/E| = 5 rs: g = {approaching:.4f} / {receding:.4f}")
    check("Orbital motion toward observer blueshifts, away redshifts",
          approaching > bh.redshift_factor(6.0, 0.0) > receding)

    # --------------------------------------------------------
    # 8. NOVIKOV-THORNE ACCRETION DISK
    # --------------------------------------------------------

    print("\nNOVIKOV-THORNE THIN DISK")
    print("-" * 64)

    r_m, flux, energy, f_func = bh.novikov_thorne_profile()
    efficiency = trapezoid(energy * f_func, r_m)
    print(f"Radiated efficiency: {efficiency:.6f}  "
          f"(1 - sqrt(8/9) = {bh.ISCO_EFFICIENCY:.6f})")
    check("Integrated luminosity gives efficiency 1 - sqrt(8/9) (0.1%)",
          abs(efficiency / bh.ISCO_EFFICIENCY - 1.0) < 1e-3)
    check("Zero-torque boundary: flux vanishes at the ISCO", flux[0] == 0.0)
    peak_r = r_m[np.argmax(flux)]
    print(f"Flux peaks at r = {peak_r:.3f} M")
    check("Flux peak lies between 9 M and 10 M", 9.0 < peak_r < 10.0)
    # The leading term converges only as r^-1/2 (0.4% low at r = 1e6 M).
    check("Far field follows Shakura-Sunyaev F = 3 Mdot / (8 pi r^3) (1%)",
          abs(flux[-1] * r_m[-1] ** 3 / 1.5 - 1.0) < 1e-2)

    t_phys = bh.disk_peak_temperature(10.0, 0.1)
    scaling = bh.disk_peak_temperature(20.0, 0.4) / t_phys
    print(f"10 Msun at 0.1 L_Edd: Tmax = {t_phys / 1e6:.2f} MK")
    check("Peak temperature is in the soft X-ray range (1-20 MK)",
          1e6 < t_phys < 2e7)
    check("Tmax scales as (Mdot/M^2)^(1/4) ~ (ratio/M)^(1/4)",
          np.isclose(scaling, (0.4 / 0.1 * 10.0 / 20.0) ** 0.25, rtol=1e-6))

    # --------------------------------------------------------
    # 9. BLACKBODY COLOUR TABLE
    # --------------------------------------------------------

    print("\nBLACKBODY COLOUR")
    print("-" * 64)

    temps, rgb = bh.blackbody_rgb_lut()
    at = lambda t: rgb[np.searchsorted(temps, t)]
    check("1500 K is red-dominated", at(1500)[0] > at(1500)[2] * 20)
    check("6500 K is near white (all channels within 10%)",
          np.ptp(at(6500)) / at(6500).mean() < 0.10)
    check("15000 K is blue-dominated", at(15000)[2] > at(15000)[0])
    check("Visible radiance increases monotonically with T",
          np.all(np.diff(rgb.sum(axis=1)) >= 0.0))

    # --------------------------------------------------------
    # 10. FINAL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 64)
    print(f"VALIDATION SUMMARY: {passed} passed, {failed} failed")
    print("=" * 64)

    if failed:
        raise SystemExit(1)

    print("ALL VALIDATION TESTS PASSED")


if __name__ == "__main__":
    main()
