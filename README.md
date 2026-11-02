# Schwarzschild Black Hole Simulator

A real-time, GPU ray-traced visualisation of a non-rotating (Schwarzschild) black hole with a thin accretion disk. Every pixel traces a photon backwards from the camera through curved spacetime, so gravitational lensing, the photon ring and relativistic Doppler effects emerge from the physics rather than being faked.

The renderer is backed by a validation suite that checks the shader's equations against independent solutions of the Schwarzschild geodesic equation.

## Features

- **Gravitational lensing** by integrating light rays with fourth-order Runge–Kutta on the GPU
- **Photon sphere, shadow and higher-order images** of the disk, including the far side bent over the top of the hole
- **Novikov–Thorne accretion disk** with the correct zero-torque inner edge at the ISCO
- **Full relativistic frequency shift** per pixel (orbital Doppler, transverse Doppler and gravitational redshift)
- **Blackbody colour** from CIE 1931 colour-matching functions, so colour and brightness shift together correctly
- **Keplerian differential rotation** of the disk's turbulent surface structure
- **HDR pipeline** with multi-level bloom and ACES tone mapping
- **Procedural sky** with coloured stars and a Milky Way band, all lensed by the black hole
- **Live HUD** showing physical quantities for the chosen mass and accretion rate
- **Validation suite** with 33 automated checks

## Requirements

- Python 3.9 or newer
- A GPU and driver supporting OpenGL 3.3 core
- Python packages: `numpy`, `scipy`, `pygame`, `moderngl`

```bash
pip install numpy scipy pygame moderngl
```

## Usage

```bash
python main.py
```

Run the physics validation (no GPU needed):

```bash
python validate_physics.py
```

## Controls

| Input | Action |
|---|---|
| Left mouse drag | Orbit the camera |
| Mouse wheel | Zoom |
| `[` / `]` | Decrease / increase black hole mass (5–20 M☉) |
| `-` / `=` | Decrease / increase accretion rate (0.01–1 L_Edd) |
| `,` / `.` | Decrease / increase exposure |
| `G` | Toggle relativistic redshift (compare with and without) |
| `Q` | Cycle render quality (0.5×, 0.75×, 1×, 1.5×) |
| `H` | Toggle the HUD |
| `Space` | Pause the disk animation |
| `R` | Reset |
| `Esc` | Quit |

If the frame rate is low, press `Q` to lower the render scale. Each pixel runs up to 1600 integration steps, so this is demanding on integrated GPUs.

## How it works

### Ray tracing

Rays are integrated in isotropic coordinates, where Schwarzschild space behaves like a medium with refractive index

$$n(\rho) = \frac{(1 + a/\rho)^3}{1 - a/\rho}, \qquad a = \frac{r_s}{4}$$

Light then follows

$$\frac{d\hat{d}}{ds} = \nabla \ln n - \hat{d}\,(\hat{d}\cdot\nabla \ln n)$$

The event horizon sits at $\rho = r_s/4$, the photon sphere at $\rho \approx 0.933\,r_s$ and the ISCO at $\rho \approx 2.475\,r_s$ (areal radius $3\,r_s$). Step sizes shrink near the hole and grow in the weak field.

### Accretion disk

The disk is geometrically thin and optically thick, with emission from the Page–Thorne (1974) flux profile for a Schwarzschild hole. The profile is computed on the CPU in `blackhole_physics.py` and uploaded as a lookup texture. The peak temperature scales physically as $(\dot{M}/M^2)^{1/4}$.

### Redshift

For gas on circular orbits the frequency ratio is

$$g = \frac{1}{u^t\,(1 - \Omega\,L/E)}, \qquad u^t = \frac{1}{\sqrt{1 - 3M/r}}, \qquad \Omega = \sqrt{M/r^3}$$

where $L/E$ is the photon's conserved angular momentum about the disk axis, taken from the camera ray via Bouguer's invariant $b = n\rho\sin\psi$. Since $I_\nu/\nu^3$ is conserved along a ray, a blackbody seen with shift $g$ is exactly a blackbody at temperature $gT$. The renderer therefore looks up its colour at $gT$ directly.

## Validation

`validate_physics.py` runs 33 checks, including:

- The shader's isotropic ray integrator (a NumPy port of the GLSL loop) agrees with an independent areal-coordinate geodesic solver, to about 0.05% on the deflection of a ray at $b = 10\,r_s$
- The captured/escaped boundary sits at the critical impact parameter $b_c = \tfrac{3\sqrt{3}}{2} r_s$
- Weak-field deflection matches the post-Newtonian series
- The disk's integrated luminosity gives efficiency $1 - \sqrt{8/9} \approx 0.0572$
- Flux vanishes at the ISCO, peaks near $9.55\,M$ and follows the Shakura–Sunyaev $r^{-3}$ law far away
- Face-on redshift equals $\sqrt{1 - 3M/r}$; approaching gas is blueshifted and receding gas redshifted
- Isotropic and areal coordinate maps are consistent, and the blackbody table has the right colour ordering

## Project structure

```
main.py                 Renderer: shaders, render passes, HUD and controls
blackhole_physics.py    Shared physics: geometry, disk model, redshift, colour tables
validate_physics.py     Automated validation of the equations used by the renderer
```

## Known limitations

- **Colours are false.** A real 10 M☉ disk peaks around 4 million K, which is soft X-ray and invisible. The simulator shows the same temperature structure mapped into visible light, with the reference disk peaking at 5,500 K. Mass and accretion rate scale this with the true $T \propto (\dot{M}/M^2)^{1/4}$ law, and the HUD states the real value.
- **Time is slowed.** The disk is animated roughly 200× slower than reality. The HUD shows the exact factor.
- **Schwarzschild only.** There is no spin, so no frame dragging and no Kerr shadow shape.
- **Idealised disk.** It is thin, opaque and exactly equatorial, with no scattering, atmosphere model, disk thickness or jets.
- **Static observer.** The camera does not move relativistically, so there is no aberration of the background sky.

## Ideas for future work

- Kerr (spinning) black holes
- Geometrically thick or tilted disks
- Moving-observer aberration
- Spectral colour hardening and non-blackbody emission

## References

- Page, D. N. & Thorne, K. S. (1974). *Disk-Accretion onto a Black Hole. Time-Averaged Structure of Accretion Disk.* ApJ 191, 499
- Novikov, I. D. & Thorne, K. S. (1973). *Astrophysics of Black Holes.*
- Luminet, J.-P. (1979). *Image of a spherical black hole with thin accretion disk.* A&A 75, 228
- Wyman, C., Sloan, P.-P. & Shirley, P. (2013). *Simple Analytic Approximations to the CIE XYZ Color Matching Functions.* JCGT 2(2)
