"""Build the RoboCup Adult-size soccer field MJCF scene for MuJoCo.

The scene is generated at runtime from the base robot MJCF
(``resources/T1/T1_locomotion.xml``) so the robot model stays in a single
source of truth. We replace the checkerboard ground with a grass-like field
plane and inject:

* Size-5 ball (radius 0.11 m, mass 0.43 kg) with rolling friction
* Two goals at x = ±FIELD_LENGTH/2 (posts + crossbar, physical capsules)
* Field line markings (visual-only sites/geoms, no collision)

Field spec (RoboCup Humanoid League Adult-size, paper arXiv:2511.03996):
    field 14 x 9 m, goal width 2.6 m, goal height 1.8 m (H1 spec ~1.8 m).
"""

import os
import re

# ---------------------------------------------------------------------------
# Field constants (metres) — single source of truth for both sims eventually.
# ---------------------------------------------------------------------------
FIELD_LENGTH = 14.0
FIELD_WIDTH = 9.0
GOAL_WIDTH = 2.6
GOAL_HEIGHT = 1.8
GOAL_DEPTH = 0.6
GOAL_POST_RADIUS = 0.06
BORDER_STRIP = 1.0          # extra walkable margin around the field
LINE_WIDTH = 0.05

BALL_RADIUS = 0.11          # Size 5
BALL_MASS = 0.43

# Plane half-sizes (field + margin)
_PLANE_HX = FIELD_LENGTH / 2 + BORDER_STRIP
_PLANE_HY = FIELD_WIDTH / 2 + BORDER_STRIP


def _line_geom(name: str, x: float, y: float, hx: float, hy: float) -> str:
    """A thin white box slightly above the plane (visual only, no collision)."""
    return (
        f'        <geom name="{name}" type="box" pos="{x} {y} 0.001" '
        f'size="{hx} {hy} 0.001" rgba="1 1 1 1" '
        f'contype="0" conaffinity="0"/>\n'
    )


def _goal_xml(side: str, x_sign: float) -> str:
    """Physical goal: two posts + crossbar at x = x_sign * FIELD_LENGTH/2."""
    x = x_sign * FIELD_LENGTH / 2
    hw = GOAL_WIDTH / 2
    r = GOAL_POST_RADIUS
    h = GOAL_HEIGHT
    return f"""
        <body name="goal_{side}" pos="{x} 0 0">
            <geom name="goal_{side}_post_l" type="capsule"
                  fromto="0 {hw} 0  0 {hw} {h}" size="{r}" rgba="1 1 1 1"/>
            <geom name="goal_{side}_post_r" type="capsule"
                  fromto="0 {-hw} 0  0 {-hw} {h}" size="{r}" rgba="1 1 1 1"/>
            <geom name="goal_{side}_crossbar" type="capsule"
                  fromto="0 {-hw} {h}  0 {hw} {h}" size="{r}" rgba="1 1 1 1"/>
        </body>
"""


def _field_lines_xml() -> str:
    hx = FIELD_LENGTH / 2
    hy = FIELD_WIDTH / 2
    lw = LINE_WIDTH / 2
    parts = [
        _line_geom("line_side_n", 0, hy, hx, lw),
        _line_geom("line_side_s", 0, -hy, hx, lw),
        _line_geom("line_goal_e", hx, 0, lw, hy),
        _line_geom("line_goal_w", -hx, 0, lw, hy),
        _line_geom("line_center", 0, 0, lw, hy),
    ]
    return "".join(parts)


def _ball_xml(pos=(1.0, 0.0, None)) -> str:
    z = BALL_RADIUS if pos[2] is None else pos[2]
    # friction: sliding, torsional, rolling. Rolling friction makes the ball
    # slow down and stop like on short grass. solref tuned for ~60% bounce.
    return f"""
        <body name="ball" pos="{pos[0]} {pos[1]} {z}">
            <freejoint name="ball_freejoint"/>
            <inertial pos="0 0 0" mass="{BALL_MASS}"
                      diaginertia="{2/3*BALL_MASS*BALL_RADIUS**2:.6f} {2/3*BALL_MASS*BALL_RADIUS**2:.6f} {2/3*BALL_MASS*BALL_RADIUS**2:.6f}"/>
            <geom name="ball_geom" type="sphere" size="{BALL_RADIUS}"
                  rgba="1 0.3 0.1 1" friction="0.7 0.005 0.005"
                  solref="0.005 0.02" condim="6"/>
        </body>
"""


FIELD_MATERIAL = """
        <texture name="texfield" type="2d" builtin="checker" rgb1="0.15 0.45 0.15" rgb2="0.18 0.52 0.18" width="512" height="512"/>
        <material name="matfield" reflectance="0.05" texture="texfield" texrepeat="8 8" texuniform="true"/>
"""

FIELD_GEOM = (
    f'        <geom name="field" type="plane" pos="0 0 0" '
    f'size="{_PLANE_HX} {_PLANE_HY} 0.1" material="matfield" '
    f'friction="0.8 0.005 0.005" condim="3"/>\n'
)


def build_scene_xml(base_xml_path: str, ball_pos=(1.0, 0.0, None)) -> str:
    """Return the soccer-field MJCF as a string, derived from the robot MJCF."""
    with open(base_xml_path, "r", encoding="utf-8") as f:
        xml = f.read()

    # 1. Field material into <asset>.
    xml = xml.replace("</asset>", FIELD_MATERIAL + "    </asset>", 1)

    # 2. Replace the checkerboard ground with the field plane.
    xml, n = re.subn(
        r'<geom name="ground"[^/]*/>',
        FIELD_GEOM.strip(),
        xml,
        count=1,
    )
    if n != 1:
        raise ValueError(f"Expected exactly one ground geom in {base_xml_path}")

    # 3. Inject ball, goals and lines just before </worldbody>.
    injection = (
        _ball_xml(ball_pos)
        + _goal_xml("east", +1.0)
        + _goal_xml("west", -1.0)
        + _field_lines_xml()
    )
    xml = xml.replace("</worldbody>", injection + "    </worldbody>", 1)
    return xml


def write_scene(base_xml_path: str, out_path: str = None, ball_pos=(1.0, 0.0, None)) -> str:
    """Write the generated scene next to the base MJCF (meshdir stays valid)."""
    if out_path is None:
        out_path = os.path.join(os.path.dirname(base_xml_path), "_soccer_field_runtime.xml")
    xml = build_scene_xml(base_xml_path, ball_pos)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)
    return out_path


# ---------------------------------------------------------------------------
# Field geometry helpers (used by episode logic and tests)
# ---------------------------------------------------------------------------

def is_goal(ball_xy, attacking_east: bool = True) -> bool:
    """Ball fully crossed the goal line between the posts?"""
    x, y = float(ball_xy[0]), float(ball_xy[1])
    if abs(y) > GOAL_WIDTH / 2:
        return False
    if attacking_east:
        return x > FIELD_LENGTH / 2 + BALL_RADIUS
    return x < -(FIELD_LENGTH / 2 + BALL_RADIUS)


def is_out_of_bounds(ball_xy) -> bool:
    """Ball outside the field (excluding goal mouths)."""
    x, y = float(ball_xy[0]), float(ball_xy[1])
    if abs(y) > FIELD_WIDTH / 2 + BALL_RADIUS:
        return True
    if abs(x) > FIELD_LENGTH / 2 + BALL_RADIUS and abs(y) > GOAL_WIDTH / 2:
        return True
    # deep behind the goal line but it wasn't a goal (wide shots handled above)
    if abs(x) > FIELD_LENGTH / 2 + GOAL_DEPTH + BALL_RADIUS:
        return True
    return False
