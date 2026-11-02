import math

import numpy as np
import pygame
import moderngl

import blackhole_physics as bh


# ============================================================
# CONFIGURATION
# ============================================================

WIDTH = 1000
HEIGHT = 650

INITIAL_MASS = 10.0  # Solar masses
REFERENCE_MASS = 10.0
MIN_MASS = 5.0
MAX_MASS = 20.0

INITIAL_EDDINGTON = 0.1
MIN_EDDINGTON = 0.01
MAX_EDDINGTON = 1.0

CAMERA_DISTANCE = 22.0
MIN_CAMERA_DISTANCE = 5.0
MAX_CAMERA_DISTANCE = 48.0
CAMERA_YAW = 0.0
CAMERA_PITCH = 0.24

# Coordinates are isotropic, in units of the Schwarzschild radius of a
# 10-solar-mass reference black hole.

DISK_OUTER_ISO = 12.0          # outer disk edge (isotropic, in rs)
DISK_INNER_ISO = (5.0 + math.sqrt(24.0)) / 4.0 * 1.0  # ISCO, r = 3 rs areal

# Disk material orbits ~200x slower than reality so it can be watched.
TIME_SCALE = 5.0

# A real 10 Msun disk peaks near 4e6 K (soft X-rays), which is invisible.
# The renderer shows what the *same temperature structure* would look like
# in visible light: the reference disk peaks at this temperature, and other
# masses / accretion rates scale from it with the true T ~ (Mdot/M^2)^(1/4).
DISPLAY_PEAK_TEMP_REFERENCE = 5500.0  # K

QUALITY_SCALES = [0.5, 0.75, 1.0, 1.5]
QUALITY_DEFAULT = 2

LUT_T_MIN = 500.0
LUT_T_MAX = 60000.0
DISK_OUTER_AREAL_RS = float(
    bh.isotropic_to_areal(DISK_OUTER_ISO, 1.0)
)


# ============================================================
# GPU SHADERS
# ============================================================

FULLSCREEN_VERTEX = """
#version 330

out vec2 v_uv;

void main() {
    vec2 positions[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );

    vec2 p = positions[gl_VertexID];
    v_uv = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
"""


SCENE_FRAGMENT = """
#version 330

uniform vec2 u_resolution;
uniform float u_time;
uniform float u_rs;

uniform float u_yaw;
uniform float u_pitch;
uniform float u_distance;

uniform float u_peakTemp;     // displayed peak disk temperature [K]
uniform float u_shift;        // 1 = relativistic redshift on, 0 = off

uniform sampler2D u_fluxLut;  // relative emitted temperature vs radius
uniform sampler2D u_colorLut; // blackbody visible radiance (linear sRGB)

out vec4 fragColor;

const float PI = 3.14159265359;

const float LUT_T_MIN = %%TMIN%%;
const float LUT_T_MAX = %%TMAX%%;
const float DISK_INNER_ISO = %%RIN%%;
const float DISK_OUTER_ISO = %%ROUT%%;
const float DISK_OUTER_AREAL = %%ROUTA%%;   // in rs
const float PATTERN_CYCLE = 24.0;


// ------------------------------------------------------------
// NOISE
// ------------------------------------------------------------

float hash31(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float vnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float n000 = hash31(i);
    float n100 = hash31(i + vec3(1.0, 0.0, 0.0));
    float n010 = hash31(i + vec3(0.0, 1.0, 0.0));
    float n110 = hash31(i + vec3(1.0, 1.0, 0.0));
    float n001 = hash31(i + vec3(0.0, 0.0, 1.0));
    float n101 = hash31(i + vec3(1.0, 0.0, 1.0));
    float n011 = hash31(i + vec3(0.0, 1.0, 1.0));
    float n111 = hash31(i + vec3(1.0, 1.0, 1.0));

    return mix(
        mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
        mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y),
        f.z
    );
}

float fbm(vec3 p, int octaves) {
    float sum = 0.0;
    float amp = 0.5;
    for (int i = 0; i < octaves; i++) {
        sum += amp * vnoise(p);
        p = p * 2.03 + 11.7;
        amp *= 0.5;
    }
    return sum;
}


// ------------------------------------------------------------
// BACKGROUND SKY: stars and a faint Milky Way band
// ------------------------------------------------------------

vec3 skyColor(vec3 direction) {

    vec3 d = normalize(direction);

    vec3 galacticNormal = vec3(0.303, 0.808, 0.505);
    float lat = dot(d, galacticNormal);
    float band = exp(-lat * lat * 18.0);

    float dust = fbm(d * 4.0, 4);
    float glow = band * (0.25 + 0.75 * dust);

    vec3 sky = vec3(0.0015, 0.002, 0.005)
             + vec3(0.050, 0.042, 0.038) * glow * glow
             + vec3(0.012, 0.016, 0.030) * band;

    vec3 cells = floor(d * 500.0);
    float star = hash31(cells);

    if (star > 0.9985 - 0.0045 * band) {
        float brightness = pow(hash31(cells + 17.0), 2.0);
        float tint = hash31(cells + 31.0);
        vec3 starColor = mix(
            vec3(1.0, 0.72, 0.5),
            vec3(0.7, 0.82, 1.0),
            tint
        );
        sky += starColor * brightness * 3.0;
    }

    return sky;
}


// ------------------------------------------------------------
// SCHWARZSCHILD OPTICAL GEOMETRY
// ------------------------------------------------------------

// Isotropic-coordinate optical index:
//
// n(rho) = (1 + a/rho)^3 / (1 - a/rho),   a = rs/4
//
// Light rays obey:
//
// d(direction)/ds = grad(ln n) - direction * dot(direction, grad(ln n))

vec3 rayAcceleration(vec3 p, vec3 direction) {

    float a = 0.25 * u_rs;

    // Prevent singular numerical evaluation at the horizon.
    float r = max(length(p), 0.2601 * u_rs);

    // Radial derivative of ln(n)
    float derivative =
        -(a / r) * (1.0 / (r - a) + 3.0 / (r + a));

    vec3 grad = derivative * p / r;

    return grad - dot(grad, direction) * direction;
}


// ------------------------------------------------------------
// FOURTH ORDER RUNGE-KUTTA RAY INTEGRATION
// ------------------------------------------------------------

void integrateRay(inout vec3 p, inout vec3 d, float h) {

    vec3 k1p = d;
    vec3 k1d = rayAcceleration(p, d);

    vec3 p2 = p + 0.5 * h * k1p;
    vec3 d2 = normalize(d + 0.5 * h * k1d);

    vec3 k2p = d2;
    vec3 k2d = rayAcceleration(p2, d2);

    vec3 p3 = p + 0.5 * h * k2p;
    vec3 d3 = normalize(d + 0.5 * h * k2d);

    vec3 k3p = d3;
    vec3 k3d = rayAcceleration(p3, d3);

    vec3 p4 = p + h * k3p;
    vec3 d4 = normalize(d + h * k3d);

    vec3 k4p = d4;
    vec3 k4d = rayAcceleration(p4, d4);

    p += h * (k1p + 2.0 * k2p + 2.0 * k3p + k4p) / 6.0;
    d = normalize(d + h * (k1d + 2.0 * k2d + 2.0 * k3d + k4d) / 6.0);
}


// ------------------------------------------------------------
// ACCRETION DISK
// ------------------------------------------------------------

// Turbulent surface structure, advected by Keplerian differential rotation.
// Two layers are cross-faded so the shear never winds up indefinitely.

float diskTurbulence(vec3 hit, float omega) {

    float result = 0.0;
    float phase = u_time / PATTERN_CYCLE;

    for (int k = 0; k < 2; k++) {

        float tau = fract(phase + 0.5 * float(k));
        float weight = 1.0 - abs(2.0 * tau - 1.0);

        float angle = omega * tau * PATTERN_CYCLE;
        float cs = cos(angle);
        float sn = sin(angle);

        vec2 q = vec2(
            cs * hit.x - sn * hit.z,
            sn * hit.x + cs * hit.z
        ) / u_rs;

        float n = fbm(vec3(q * 1.0, 0.5), 4);

        result += weight * n;
    }

    return result;
}


// lzOverE: photon angular momentum about the disk axis divided by its
// energy (conserved along the ray). Disk gas orbits with angular momentum
// along +y, i.e. with velocity direction (z, 0, -x).

vec3 diskColor(vec3 hit, vec3 rayDirection, float lzOverE) {

    float rho = length(hit.xz);

    float a = 0.25 * u_rs;
    float arealR = rho * (1.0 + a / rho) * (1.0 + a / rho);
    float rr = max(arealR / u_rs, 3.0);   // areal radius in units of rs

    // Emitted temperature from the Novikov-Thorne flux profile.

    float x = log(rr / 3.0) / log(DISK_OUTER_AREAL / 3.0);
    float tau = texture(u_fluxLut, vec2(clamp(x, 0.0, 1.0), 0.5)).r;

    // Circular-orbit angular velocity, Omega = sqrt(M / r^3), M = rs/2.

    float omega = sqrt(0.5 / (rr * rr * rr)) / u_rs;

    float turbulence = diskTurbulence(hit, omega);
    float tEmit = u_peakTemp * tau * mix(0.86, 1.14, clamp(turbulence * 1.6, 0.0, 1.0));

    // Frequency shift g = nu_obs / nu_emit:
    //   g = 1 / ( u^t (1 - Omega L/E) ),   u^t = 1/sqrt(1 - 1.5 rs/r)

    float g = 1.0;

    if (u_shift > 0.5) {
        float ut = inversesqrt(max(1.0 - 1.5 / rr, 1e-4));
        g = 1.0 / (ut * (1.0 - omega * lzOverE));
        g = clamp(g, 0.1, 4.0);
    }

    // I_nu / nu^3 is invariant, so a blackbody seen with shift g is a
    // blackbody at g * T: look up colour and brightness together.

    float tObs = g * tEmit;

    float lt = (log(max(tObs, LUT_T_MIN)) - log(LUT_T_MIN)) /
               (log(LUT_T_MAX) - log(LUT_T_MIN));

    vec3 radiance = texture(u_colorLut, vec2(clamp(lt, 0.0, 1.0), 0.5)).rgb;

    // Electron-scattering limb darkening (flux-normalised).

    float mu = abs(rayDirection.y);
    float limb = (1.0 + 2.06 * mu) / 2.373;

    return radiance * limb;
}


// ------------------------------------------------------------
// MAIN RENDERING FUNCTION
// ------------------------------------------------------------

void main() {

    vec2 uv = (
        2.0 * gl_FragCoord.xy - u_resolution
    ) / u_resolution.y;

    vec3 camera = vec3(
        sin(u_yaw) * cos(u_pitch),
        sin(u_pitch),
        cos(u_yaw) * cos(u_pitch)
    ) * u_distance;

    vec3 forward = normalize(-camera);
    vec3 worldUp = vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(forward, worldUp));
    vec3 up = normalize(cross(right, forward));

    float fov = 0.70;

    vec3 direction = normalize(
        forward + fov * uv.x * right + fov * uv.y * up
    );

    // Conserved photon angular momentum per unit energy, from the camera
    // ray. Bouguer's formula: b = n(rho) * rho * sin(psi).
    // The sign flip converts the backwards-traced ray to the real photon.

    float rc = length(camera);
    float aCam = 0.25 * u_rs;
    float nCam = pow(1.0 + aCam / rc, 3.0) / (1.0 - aCam / rc);

    vec3 jv = cross(camera / rc, direction);
    float sinPsi = length(jv);

    float impact = nCam * rc * sinPsi;
    float lzOverE = -impact * jv.y / max(sinPsi, 1e-6);

    vec3 p = camera;
    vec3 d = direction;

    vec3 color = vec3(0.0);
    bool finished = false;

    float stepSize = 0.045 * u_rs;

    // Slightly outside the isotropic event horizon (rho = rs/4).
    float captureRadius = 0.252 * u_rs;

    for (int i = 0; i < 1600; i++) {

        float radius = length(p);

        if (radius <= captureRadius) {
            color = vec3(0.0);
            finished = true;
            break;
        }

        if (radius > 25.0 * u_rs && dot(p, d) > 0.0) {
            color = skyColor(d);
            finished = true;
            break;
        }

        vec3 previousPosition = p;

        // Small steps in the strong field, larger steps far away where
        // the rays are almost straight.
        float localStep = stepSize;

        if (radius < 5.0 * u_rs) {
            float refinement = clamp(
                (radius - captureRadius) /
                max(5.0 * u_rs - captureRadius, 0.001 * u_rs),
                0.12,
                1.0
            );
            localStep *= refinement;
        } else {
            localStep *= clamp(radius / (6.0 * u_rs), 1.0, 4.0);
        }

        integrateRay(p, d, localStep);

        // Disk-plane crossing (the disk is geometrically thin and opaque).
        if (previousPosition.y * p.y < 0.0) {

            float t = previousPosition.y /
                (previousPosition.y - p.y);

            vec3 hit = mix(previousPosition, p, t);

            float diskRadius = length(hit.xz);

            if (
                diskRadius >= DISK_INNER_ISO * u_rs &&
                diskRadius <= DISK_OUTER_ISO * u_rs
            ) {
                color = diskColor(hit, d, lzOverE);
                finished = true;
                break;
            }
        }
    }

    if (!finished) {
        color = vec3(0.0);
    }

    // Linear HDR output: tone mapping and bloom happen in later passes.
    fragColor = vec4(color, 1.0);
}
"""


# Downsample (with optional bright-pass for the first bloom level).
DOWNSAMPLE_FRAGMENT = """
#version 330

uniform sampler2D u_src;
uniform vec2 u_srcTexel;
uniform float u_prefilter;

in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec2 o = 0.5 * u_srcTexel;

    vec3 c = 0.25 * (
        texture(u_src, v_uv + vec2(-o.x, -o.y)).rgb +
        texture(u_src, v_uv + vec2( o.x, -o.y)).rgb +
        texture(u_src, v_uv + vec2(-o.x,  o.y)).rgb +
        texture(u_src, v_uv + vec2( o.x,  o.y)).rgb
    );

    if (u_prefilter > 0.5) {
        float br = max(c.r, max(c.g, c.b));
        c *= max(br - 0.8, 0.0) / max(br, 1e-4);
    }

    fragColor = vec4(c, 1.0);
}
"""


BLUR_FRAGMENT = """
#version 330

uniform sampler2D u_src;
uniform vec2 u_direction;   // texel step along the blur axis

in vec2 v_uv;
out vec4 fragColor;

void main() {
    const float w[5] = float[](
        0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216
    );

    vec3 c = texture(u_src, v_uv).rgb * w[0];

    for (int i = 1; i < 5; i++) {
        vec2 off = u_direction * float(i);
        c += texture(u_src, v_uv + off).rgb * w[i];
        c += texture(u_src, v_uv - off).rgb * w[i];
    }

    fragColor = vec4(c, 1.0);
}
"""


COMPOSITE_FRAGMENT = """
#version 330

uniform sampler2D u_hdr;
uniform sampler2D u_bloom0;
uniform sampler2D u_bloom1;
uniform sampler2D u_bloom2;
uniform float u_exposure;
uniform float u_bloomStrength;

in vec2 v_uv;
out vec4 fragColor;

vec3 aces(vec3 x) {
    return clamp(
        (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
        0.0, 1.0
    );
}

void main() {
    vec3 color = texture(u_hdr, v_uv).rgb;

    vec3 bloom =
        texture(u_bloom0, v_uv).rgb * 0.50 +
        texture(u_bloom1, v_uv).rgb * 0.35 +
        texture(u_bloom2, v_uv).rgb * 0.25;

    color += bloom * u_bloomStrength;
    color *= u_exposure;

    color = aces(color);
    color = pow(color, vec3(1.0 / 2.2));

    fragColor = vec4(color, 1.0);
}
"""


HUD_VERTEX = """
#version 330

uniform vec4 u_rect;      // x, y, w, h in pixels (origin bottom-left)
uniform vec2 u_screen;

out vec2 v_uv;

void main() {
    vec2 quad[4] = vec2[](
        vec2(0.0, 0.0), vec2(1.0, 0.0),
        vec2(0.0, 1.0), vec2(1.0, 1.0)
    );

    vec2 q = quad[gl_VertexID];
    v_uv = q;

    vec2 pixel = u_rect.xy + q * u_rect.zw;
    vec2 ndc = pixel / u_screen * 2.0 - 1.0;

    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""


HUD_FRAGMENT = """
#version 330

uniform sampler2D u_tex;

in vec2 v_uv;
out vec4 fragColor;

void main() {
    fragColor = texture(u_tex, v_uv);
}
"""


def fill_template(source):
    return (
        source
        .replace("%%TMIN%%", f"{LUT_T_MIN:.1f}")
        .replace("%%TMAX%%", f"{LUT_T_MAX:.1f}")
        .replace("%%RIN%%", f"{DISK_INNER_ISO:.9f}")
        .replace("%%ROUT%%", f"{DISK_OUTER_ISO:.1f}")
        .replace("%%ROUTA%%", f"{DISK_OUTER_AREAL_RS:.9f}")
    )


# ============================================================
# INITIALIZE PYGAME AND OPENGL
# ============================================================

pygame.init()

pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
pygame.display.gl_set_attribute(
    pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
)

pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption("Schwarzschild Black Hole Simulator")

ctx = moderngl.create_context(require=330)
ctx.disable(moderngl.DEPTH_TEST)
ctx.disable(moderngl.CULL_FACE)

print("GPU:", ctx.info["GL_RENDERER"])
print("OpenGL:", ctx.info["GL_VERSION"])


# ============================================================
# PROGRAMS
# ============================================================

programs = {
    "scene": ctx.program(
        vertex_shader=FULLSCREEN_VERTEX,
        fragment_shader=fill_template(SCENE_FRAGMENT),
    ),
    "down": ctx.program(
        vertex_shader=FULLSCREEN_VERTEX, fragment_shader=DOWNSAMPLE_FRAGMENT
    ),
    "blur": ctx.program(
        vertex_shader=FULLSCREEN_VERTEX, fragment_shader=BLUR_FRAGMENT
    ),
    "composite": ctx.program(
        vertex_shader=FULLSCREEN_VERTEX, fragment_shader=COMPOSITE_FRAGMENT
    ),
    "hud": ctx.program(
        vertex_shader=HUD_VERTEX, fragment_shader=HUD_FRAGMENT
    ),
}

vaos = {name: ctx.vertex_array(prog, []) for name, prog in programs.items()}


def set_uniform(program, name, value):
    if name in program:
        program[name].value = value


# ============================================================
# LOOKUP TABLES (computed once from the physics module)
# ============================================================

# Emitted temperature profile of a Novikov-Thorne disk.
flux_lut = bh.disk_temperature_lut(n=512, r_out_rs=DISK_OUTER_AREAL_RS)
flux_texture = ctx.texture(
    (len(flux_lut), 1), 1, flux_lut.tobytes(), dtype="f4"
)

# Blackbody visible radiance versus temperature.
lut_temps, lut_rgb = bh.blackbody_rgb_lut(
    n=1024, t_min=LUT_T_MIN, t_max=LUT_T_MAX
)
color_texture = ctx.texture(
    (len(lut_temps), 1), 3, lut_rgb.tobytes(), dtype="f4"
)

for texture in (flux_texture, color_texture):
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    texture.repeat_x = False
    texture.repeat_y = False


# ============================================================
# RENDER TARGETS
# ============================================================

class Target:
    """Half-float colour texture plus framebuffer."""

    def __init__(self, size):
        self.size = size
        self.texture = ctx.texture(size, 4, dtype="f2")
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.texture.repeat_x = False
        self.texture.repeat_y = False
        self.fbo = ctx.framebuffer(color_attachments=[self.texture])

    def release(self):
        self.fbo.release()
        self.texture.release()


hdr_target = None
bloom_levels = []   # list of (target_a, target_b)


def build_targets(scale):
    global hdr_target, bloom_levels

    if hdr_target is not None:
        hdr_target.release()
        for a, b in bloom_levels:
            a.release()
            b.release()

    hdr_target = Target((int(WIDTH * scale), int(HEIGHT * scale)))

    bloom_levels = []
    for level in range(3):
        size = (max(WIDTH >> (level + 1), 8), max(HEIGHT >> (level + 1), 8))
        bloom_levels.append((Target(size), Target(size)))


def run_pass(name, target, inputs, **uniforms):
    program = programs[name]

    if target is None:
        ctx.screen.use()
        ctx.viewport = (0, 0, WIDTH, HEIGHT)
    else:
        target.fbo.use()
        ctx.viewport = (0, 0, target.size[0], target.size[1])

    for unit, (uniform_name, texture) in enumerate(inputs):
        texture.use(unit)
        set_uniform(program, uniform_name, unit)

    for key, value in uniforms.items():
        set_uniform(program, key, value)

    vaos[name].render(moderngl.TRIANGLES, vertices=3)


# ============================================================
# HUD
# ============================================================

hud_font = pygame.font.SysFont(
    "dejavusansmono,consolas,menlo,couriernew,monospace", 14
)
hud_texture = None
hud_size = (0, 0)
get_bytes = getattr(pygame.image, "tobytes", None) or pygame.image.tostring


def update_hud(lines):
    global hud_texture, hud_size

    line_height = hud_font.get_linesize()
    rendered = [
        hud_font.render(text, True, color) for text, color in lines
    ]
    width = max(s.get_width() for s in rendered) + 24
    height = line_height * len(rendered) + 18

    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(
        surface, (4, 8, 16, 165), surface.get_rect(), border_radius=8
    )

    for i, line in enumerate(rendered):
        surface.blit(line, (12, 9 + i * line_height))

    data = get_bytes(surface, "RGBA", True)

    if hud_texture is not None:
        hud_texture.release()

    hud_texture = ctx.texture((width, height), 4, data)
    hud_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    hud_size = (width, height)


def draw_hud():
    if hud_texture is None:
        return

    ctx.screen.use()
    ctx.viewport = (0, 0, WIDTH, HEIGHT)
    ctx.enable(moderngl.BLEND)

    program = programs["hud"]
    hud_texture.use(0)
    set_uniform(program, "u_tex", 0)
    set_uniform(program, "u_screen", (float(WIDTH), float(HEIGHT)))
    set_uniform(
        program,
        "u_rect",
        (12.0, float(HEIGHT - hud_size[1] - 12), float(hud_size[0]),
         float(hud_size[1])),
    )

    vaos["hud"].render(moderngl.TRIANGLE_STRIP, vertices=4)
    ctx.disable(moderngl.BLEND)


WHITE = (225, 232, 245)
DIM = (140, 152, 172)
ACCENT = (255, 190, 110)


def hud_lines(mass, eddington, peak_display, fps, shift_on, quality, exposure,
              distance, pitch_angle, paused):
    rs_scene = mass / REFERENCE_MASS
    rs_km = bh.schwarzschild_radius(mass) / 1000.0

    t_phys = bh.disk_peak_temperature(mass, eddington)
    mdot_msun_yr = (
        bh.accretion_rate(mass, eddington) * 3.15576e7 / bh.M_SUN
    )

    r_obs_rs = float(bh.isotropic_to_areal(distance, rs_scene)) / rs_scene
    time_dilation = math.sqrt(max(1.0 - 1.0 / r_obs_rs, 0.0))
    inclination = 90.0 - math.degrees(pitch_angle)

    f_phys = bh.isco_orbital_frequency(mass)
    f_shown = (
        math.sqrt(0.5 / 27.0) / rs_scene * TIME_SCALE / (2.0 * math.pi)
    )
    slowdown = f_phys / f_shown

    return [
        (f"SCHWARZSCHILD BLACK HOLE   {mass:.0f} M_sun", ACCENT),
        (f"rs = {rs_km:.1f} km    photon sphere 1.5 rs    ISCO 3 rs", WHITE),
        (f"critical impact parameter {1.5 * math.sqrt(3.0):.3f} rs", WHITE),
        ("", WHITE),
        (f"accretion {eddington:.2f} L_Edd   Mdot {mdot_msun_yr:.1e} M_sun/yr",
         WHITE),
        (f"disk Tmax {t_phys / 1e6:.2f} MK (X-ray, Novikov-Thorne)", WHITE),
        (f"shown in false visible colour, peak {peak_display:.0f} K", DIM),
        (f"ISCO orbit {f_phys:.0f} Hz, slowed {slowdown:.0f}x", WHITE),
        ("", WHITE),
        (f"observer r = {r_obs_rs:.1f} rs   inclination {inclination:.0f} deg",
         WHITE),
        (f"clock rate at observer {time_dilation:.3f} (static)", WHITE),
        (f"redshift g: {'ON' if shift_on else 'OFF'}   "
         f"quality {quality:.2f}x   exposure {exposure:.2f}", WHITE),
        (f"{fps:5.1f} FPS {'  PAUSED' if paused else ''}", DIM),
        ("", WHITE),
        ("drag orbit | wheel zoom | [ ] mass | - = accretion", DIM),
        (", . exposure | G redshift | Q quality | H hud", DIM),
        ("space pause | R reset | esc quit", DIM),
    ]


# ============================================================
# SIMULATION VARIABLES
# ============================================================

mass = INITIAL_MASS
eddington = INITIAL_EDDINGTON

camera_distance = CAMERA_DISTANCE
yaw = CAMERA_YAW
pitch = CAMERA_PITCH

exposure = 0.8
shift_on = True
show_hud = True
quality_index = QUALITY_DEFAULT

simulation_time = 0.0
paused = False

clock = pygame.time.Clock()
running = True
dragging = False
previous_mouse = None

hud_timer = 0.0
T_REFERENCE_PHYSICAL = bh.disk_peak_temperature(
    REFERENCE_MASS, INITIAL_EDDINGTON
)

build_targets(QUALITY_SCALES[quality_index])


# ============================================================
# MAIN LOOP
# ============================================================

while running:

    dt = min(clock.tick(60) / 1000.0, 0.05)

    if not paused:
        simulation_time += dt

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                dragging = True
                previous_mouse = event.pos

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                dragging = False
                previous_mouse = None

        elif event.type == pygame.MOUSEMOTION:
            if dragging and previous_mouse:
                dx = event.pos[0] - previous_mouse[0]
                dy = event.pos[1] - previous_mouse[1]

                yaw += dx * 0.005
                pitch -= dy * 0.005
                pitch = max(-1.35, min(1.35, pitch))

                previous_mouse = event.pos

        elif event.type == pygame.MOUSEWHEEL:
            camera_distance *= 0.92 ** event.y
            camera_distance = max(
                MIN_CAMERA_DISTANCE, min(MAX_CAMERA_DISTANCE, camera_distance)
            )

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:
                running = False

            elif event.key == pygame.K_SPACE:
                paused = not paused

            elif event.key == pygame.K_r:
                mass = INITIAL_MASS
                eddington = INITIAL_EDDINGTON
                camera_distance = CAMERA_DISTANCE
                yaw = CAMERA_YAW
                pitch = CAMERA_PITCH
                simulation_time = 0.0
                exposure = 0.8

            elif event.key == pygame.K_LEFTBRACKET:
                mass = max(MIN_MASS, mass - 1.0)

            elif event.key == pygame.K_RIGHTBRACKET:
                mass = min(MAX_MASS, mass + 1.0)

            elif event.key == pygame.K_MINUS:
                eddington = max(MIN_EDDINGTON, eddington / 1.25)

            elif event.key == pygame.K_EQUALS:
                eddington = min(MAX_EDDINGTON, eddington * 1.25)

            elif event.key == pygame.K_COMMA:
                exposure = max(0.1, exposure / 1.15)

            elif event.key == pygame.K_PERIOD:
                exposure = min(10.0, exposure * 1.15)

            elif event.key == pygame.K_g:
                shift_on = not shift_on

            elif event.key == pygame.K_h:
                show_hud = not show_hud

            elif event.key == pygame.K_q:
                quality_index = (quality_index + 1) % len(QUALITY_SCALES)
                build_targets(QUALITY_SCALES[quality_index])

    # Schwarzschild radius relative to the reference mass.

    rs = mass / REFERENCE_MASS

    # Keep the camera outside 2.5 rs even for the heaviest hole.
    camera_distance = max(camera_distance, 2.5 * rs)

    # Displayed disk temperature follows the true scaling of the
    # Novikov-Thorne peak temperature with mass and accretion rate.
    peak_display = DISPLAY_PEAK_TEMP_REFERENCE * (
        bh.disk_peak_temperature(mass, eddington) / T_REFERENCE_PHYSICAL
    )

    # ---- Pass 1: ray-traced HDR scene ----

    hdr_w, hdr_h = hdr_target.size

    run_pass(
        "scene",
        hdr_target,
        [("u_fluxLut", flux_texture), ("u_colorLut", color_texture)],
        u_resolution=(float(hdr_w), float(hdr_h)),
        u_time=simulation_time * TIME_SCALE,
        u_rs=rs,
        u_yaw=yaw,
        u_pitch=pitch,
        u_distance=camera_distance,
        u_peakTemp=peak_display,
        u_shift=1.0 if shift_on else 0.0,
    )

    # ---- Pass 2: bloom chain ----

    source = hdr_target
    for level, (blur_a, blur_b) in enumerate(bloom_levels):
        run_pass(
            "down",
            blur_a,
            [("u_src", source.texture)],
            u_srcTexel=(1.0 / source.size[0], 1.0 / source.size[1]),
            u_prefilter=1.0 if level == 0 else 0.0,
        )
        run_pass(
            "blur",
            blur_b,
            [("u_src", blur_a.texture)],
            u_direction=(1.0 / blur_a.size[0], 0.0),
        )
        run_pass(
            "blur",
            blur_a,
            [("u_src", blur_b.texture)],
            u_direction=(0.0, 1.0 / blur_a.size[1]),
        )
        source = blur_a

    # ---- Pass 3: composite, tone map, gamma ----

    run_pass(
        "composite",
        None,
        [
            ("u_hdr", hdr_target.texture),
            ("u_bloom0", bloom_levels[0][0].texture),
            ("u_bloom1", bloom_levels[1][0].texture),
            ("u_bloom2", bloom_levels[2][0].texture),
        ],
        u_exposure=exposure,
        u_bloomStrength=0.55,
    )

    # ---- HUD ----

    hud_timer -= dt
    if show_hud and (hud_timer <= 0.0 or hud_texture is None):
        update_hud(
            hud_lines(
                mass, eddington, peak_display, clock.get_fps(), shift_on,
                QUALITY_SCALES[quality_index], exposure, camera_distance,
                pitch, paused,
            )
        )
        hud_timer = 0.25

    if show_hud:
        draw_hud()

    pygame.display.flip()


# ============================================================
# CLEANUP
# ============================================================

for vao in vaos.values():
    vao.release()
for program in programs.values():
    program.release()

pygame.quit()
