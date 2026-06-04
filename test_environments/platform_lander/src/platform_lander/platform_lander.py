"""Standalone moving-platform booster landing environment.

The physics and API are adapted from Gymnasium's LunarLander v3 under the MIT
license, then modified to model a vertical booster landing on a floating
left-right moving platform.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from platform_lander.core import DependencyNotInstalled, Env, EzPickle
from platform_lander.spaces import Box, Discrete

try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
    )
except ImportError as exc:  # pragma: no cover - import-time dependency check
    raise DependencyNotInstalled(
        "Box2D is not installed. Install it with `pip install swig box2d-py`."
    ) from exc

if TYPE_CHECKING:
    import pygame


FPS = 50
SCALE = 30.0

VIEWPORT_W = 600
VIEWPORT_H = 400

BOTTOM_ENGINE_POWER = 18.0
TOP_ENGINE_POWER = 1.25
INITIAL_RANDOM = 700.0

BOOSTER_HALF_WIDTH = 8
BOOSTER_TOP = 52
BOOSTER_BOTTOM = 56
BOOSTER_START_CLEARANCE = 0.06
TOP_ENGINE_Y = 43
TOP_ENGINE_AWAY = 9
BOTTOM_ENGINE_Y = 54

PLATFORM_WIDTH = 118
PLATFORM_HEIGHT = 14
PLATFORM_SPEED = 1.15
MAX_JET_FIRES = 200
LANDING_ANGLE = math.radians(8.0)
LANDING_VX = 0.45
LANDING_VY = 0.65
LANDING_ANGULAR_V = 0.55


class ContactDetector(contactListener):
    """Track booster contacts with the platform and ocean."""

    def __init__(self, env: "PlatformLander") -> None:
        contactListener.__init__(self)
        self.env = env

    @staticmethod
    def _data(contact) -> tuple[object, object]:
        return contact.fixtureA.userData, contact.fixtureB.userData

    def BeginContact(self, contact) -> None:
        a, b = self._data(contact)
        labels = {a, b}
        booster_labels = {"booster_body", "left_foot", "right_foot"}

        if "ocean" in labels and labels.intersection(booster_labels):
            self.env.ocean_contact = True
            return

        if "platform" not in labels:
            return

        if "booster_body" in labels:
            self.env.body_platform_contact = True
            self.env.platform_contact = True
        if "left_foot" in labels:
            self.env.left_foot_contact = True
            self.env.platform_contact = True
        if "right_foot" in labels:
            self.env.right_foot_contact = True
            self.env.platform_contact = True

    def EndContact(self, contact) -> None:
        a, b = self._data(contact)
        labels = {a, b}
        if "platform" not in labels:
            return
        if "left_foot" in labels:
            self.env.left_foot_contact = False
        if "right_foot" in labels:
            self.env.right_foot_contact = False
        if "booster_body" in labels:
            self.env.body_platform_contact = False
        self.env.platform_contact = (
            self.env.left_foot_contact
            or self.env.right_foot_contact
            or self.env.body_platform_contact
        )


class PlatformLander(Env, EzPickle):
    """Land a reusable booster vertically on a moving ocean platform.

    Actions are ``Discrete(4)`` by default:

    - 0: do nothing
    - 1: fire the upper-left attitude jet
    - 2: fire the bottom engine
    - 3: fire the upper-right attitude jet

    With ``continuous=True``, actions are ``Box(-1, 1, shape=(2,))`` where the
    first value controls the bottom engine and the second controls the top jets.

    Observations contain eleven float values:
    relative x/y to the platform landing point, x/y velocity, booster angle,
    angular velocity, left/right foot contact flags, platform x, and platform
    velocity, and the fraction of jet fires remaining. Wind is applied from
    ``wind_direction`` with force ``wind_power``. If ``variable_wind=True`` the
    force varies over time using LunarLander v3's deterministic wind pattern.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        render_mode: str | None = None,
        continuous: bool = False,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        wind_direction: float | tuple[float, float] = 0.0,
        turbulence_power: float = 1.5,
        variable_wind: bool = True,
        platform_speed: float = PLATFORM_SPEED,
        max_jet_fires: int = MAX_JET_FIRES,
    ) -> None:
        EzPickle.__init__(
            self,
            render_mode,
            continuous,
            gravity,
            enable_wind,
            wind_power,
            wind_direction,
            turbulence_power,
            variable_wind,
            platform_speed,
            max_jet_fires,
        )
        if not -12.0 < gravity < 0.0:
            raise ValueError(f"gravity must be between -12 and 0, got {gravity}")

        self.gravity = gravity
        self.continuous = continuous
        self.enable_wind = enable_wind
        self.wind_power = float(wind_power)
        self.wind_direction = wind_direction
        self.turbulence_power = float(turbulence_power)
        self.variable_wind = variable_wind
        self.platform_speed = float(platform_speed)
        self.max_jet_fires = int(max_jet_fires)
        self.render_mode = render_mode

        self.screen: pygame.Surface | None = None
        self.clock = None
        self.isopen = True

        self.world = Box2D.b2World(gravity=(0, gravity))
        self.booster: Box2D.b2Body | None = None
        self.platform: Box2D.b2Body | None = None
        self.ocean: Box2D.b2Body | None = None
        self.bottom_flame_power = 0.0
        self.top_flame_power = 0.0
        self.top_flame_direction = 0
        self.jet_fires_used = 0
        self.prev_shaping = None

        low = np.array(
            [-2.5, -2.5, -10.0, -10.0, -2 * math.pi, -10.0, 0.0, 0.0, -1.0, -2.0, 0.0],
            dtype=np.float32,
        )
        high = np.array(
            [2.5, 2.5, 10.0, 10.0, 2 * math.pi, 10.0, 1.0, 1.0, 1.0, 2.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = Box(low, high, dtype=np.float32)
        self.action_space = (
            Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            if continuous
            else Discrete(4)
        )

    def _destroy(self) -> None:
        self.world.contactListener = None
        for body_name in ("booster", "platform", "ocean"):
            body = getattr(self, body_name)
            if body is not None:
                self.world.DestroyBody(body)
                setattr(self, body_name, None)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._destroy()

        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref

        self.ocean_contact = False
        self.platform_contact = False
        self.body_platform_contact = False
        self.left_foot_contact = False
        self.right_foot_contact = False
        self.failure_reason: str | None = None
        self.prev_shaping = None
        self.bottom_flame_power = 0.0
        self.top_flame_power = 0.0
        self.top_flame_direction = 0
        self.jet_fires_used = 0

        w = VIEWPORT_W / SCALE
        h = VIEWPORT_H / SCALE
        self.platform_y = h / 4
        self.platform_half_width = PLATFORM_WIDTH / SCALE / 2
        self.platform_half_height = PLATFORM_HEIGHT / SCALE / 2
        self.platform_min_x = self.platform_half_width + 0.25
        self.platform_max_x = w - self.platform_half_width - 0.25
        self.platform_direction = int(self.np_random.choice([-1, 1]))

        platform_x = float(self.np_random.uniform(self.platform_min_x, self.platform_max_x))
        self.platform = self.world.CreateKinematicBody(position=(platform_x, self.platform_y))
        platform_fixture = self.platform.CreateFixture(
            fixtureDef(
                shape=polygonShape(box=(self.platform_half_width, self.platform_half_height)),
                density=0.0,
                friction=0.9,
                restitution=0.0,
                categoryBits=0x0001,
            )
        )
        platform_fixture.userData = "platform"
        self.platform.color1 = (38, 42, 48)
        self.platform.color2 = (235, 235, 235)

        self.ocean_y = self.platform_y - self.platform_half_height * 0.75
        self.ocean = self.world.CreateStaticBody()
        ocean_fixture = self.ocean.CreateFixture(
            fixtureDef(
                shape=edgeShape(vertices=[(0, self.ocean_y), (w, self.ocean_y)]),
                isSensor=True,
                categoryBits=0x0001,
            )
        )
        ocean_fixture.userData = "ocean"

        initial_x = w / 2
        initial_y = h + BOOSTER_BOTTOM / SCALE + BOOSTER_START_CLEARANCE
        self.booster = self.world.CreateDynamicBody(position=(initial_x, initial_y), angle=0.0)
        body_fixture = self.booster.CreateFixture(
            fixtureDef(
                shape=polygonShape(
                    vertices=[
                        (-BOOSTER_HALF_WIDTH / SCALE, -48 / SCALE),
                        (-BOOSTER_HALF_WIDTH / SCALE, 43 / SCALE),
                        (-5 / SCALE, BOOSTER_TOP / SCALE),
                        (5 / SCALE, BOOSTER_TOP / SCALE),
                        (BOOSTER_HALF_WIDTH / SCALE, 43 / SCALE),
                        (BOOSTER_HALF_WIDTH / SCALE, -48 / SCALE),
                    ]
                ),
                density=4.5,
                friction=0.25,
                categoryBits=0x0010,
                maskBits=0x0001,
                restitution=0.0,
            )
        )
        body_fixture.userData = "booster_body"

        left_foot_fixture = self.booster.CreateFixture(
            fixtureDef(
                shape=polygonShape(box=(8 / SCALE, 2 / SCALE, (-7 / SCALE, -54 / SCALE), 0.0)),
                density=0.7,
                friction=1.0,
                categoryBits=0x0010,
                maskBits=0x0001,
                restitution=0.0,
            )
        )
        left_foot_fixture.userData = "left_foot"

        right_foot_fixture = self.booster.CreateFixture(
            fixtureDef(
                shape=polygonShape(box=(8 / SCALE, 2 / SCALE, (7 / SCALE, -54 / SCALE), 0.0)),
                density=0.7,
                friction=1.0,
                categoryBits=0x0010,
                maskBits=0x0001,
                restitution=0.0,
            )
        )
        right_foot_fixture.userData = "right_foot"

        self.booster.color1 = (230, 232, 235)
        self.booster.color2 = (40, 44, 52)

        self.booster.ApplyForceToCenter(
            (
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            ),
            True,
        )

        if self.enable_wind:
            self.wind_idx = int(self.np_random.integers(-9999, 9999))
            self.torque_idx = int(self.np_random.integers(-9999, 9999))

        self.drawlist = [self.platform, self.booster]

        if self.render_mode == "human":
            self.render()
        return self.step(np.array([0.0, 0.0], dtype=np.float32) if self.continuous else 0)[0], {}

    def _wind_unit(self) -> tuple[float, float]:
        if isinstance(self.wind_direction, tuple):
            x, y = self.wind_direction
            norm = math.hypot(x, y)
            return (0.0, 0.0) if norm == 0 else (x / norm, y / norm)
        return math.cos(float(self.wind_direction)), math.sin(float(self.wind_direction))

    def _wind_scale(self) -> float:
        if not self.variable_wind:
            return 1.0
        scale = math.tanh(
            math.sin(0.02 * self.wind_idx)
            + math.sin(math.pi * 0.01 * self.wind_idx)
        )
        self.wind_idx += 1
        return scale

    def set_wind(
        self,
        *,
        power: float | None = None,
        direction: float | tuple[float, float] | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Adjust wind during an episode."""

        if power is not None:
            self.wind_power = float(power)
        if direction is not None:
            self.wind_direction = direction
        if enabled is not None:
            self.enable_wind = bool(enabled)

    def _update_platform(self) -> None:
        assert self.platform is not None
        x = self.platform.position.x
        if x <= self.platform_min_x + 1e-6 and self.platform_direction < 0:
            self.platform_direction = 1
        elif x >= self.platform_max_x - 1e-6 and self.platform_direction > 0:
            self.platform_direction = -1
        self.platform.linearVelocity = (self.platform_speed * self.platform_direction, 0.0)

    def _clamp_platform(self) -> None:
        assert self.platform is not None
        x = min(max(self.platform.position.x, self.platform_min_x), self.platform_max_x)
        if x != self.platform.position.x:
            if x <= self.platform_min_x + 1e-6:
                self.platform_direction = 1
            elif x >= self.platform_max_x - 1e-6:
                self.platform_direction = -1
            self.platform.position = (x, self.platform.position.y)
            self.platform.linearVelocity = (self.platform_speed * self.platform_direction, 0.0)

    def _apply_wind(self) -> None:
        assert self.booster is not None
        if not self.enable_wind or self.platform_contact:
            return
        wx, wy = self._wind_unit()
        wind_mag = self.wind_power * self._wind_scale()
        self.booster.ApplyForceToCenter((wx * wind_mag, wy * wind_mag), True)

        torque_mag = math.tanh(
            math.sin(0.02 * self.torque_idx)
            + math.sin(math.pi * 0.01 * self.torque_idx)
        ) * self.turbulence_power
        self.torque_idx += 1
        self.booster.ApplyTorque(torque_mag, True)

    def _apply_engines(self, action) -> tuple[float, float]:
        assert self.booster is not None

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float64)
        else:
            if not self.action_space.contains(action):
                raise AssertionError(f"{action!r} ({type(action)}) is not a valid action")

        tip = (math.sin(self.booster.angle), math.cos(self.booster.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, 1.0) / SCALE for _ in range(2)]

        bottom_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action == 2):
            if self.jet_fires_used < self.max_jet_fires:
                bottom_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5 if self.continuous else 1.0
                self.jet_fires_used += 1
                ox = -tip[0] * (BOTTOM_ENGINE_Y / SCALE + dispersion[0]) + side[0] * dispersion[1]
                oy = -tip[1] * (BOTTOM_ENGINE_Y / SCALE + dispersion[0]) + side[1] * dispersion[1]
                impulse_pos = (self.booster.position[0] + ox, self.booster.position[1] + oy)
                impulse = (tip[0] * BOTTOM_ENGINE_POWER * bottom_power, tip[1] * BOTTOM_ENGINE_POWER * bottom_power)
                self.booster.ApplyLinearImpulse(impulse, impulse_pos, True)

        top_power = 0.0
        top_direction = 0
        if self.continuous and abs(action[1]) > 0.5:
            top_direction = -1 if action[1] < 0 else 1
            top_power = float(np.clip(abs(action[1]), 0.5, 1.0))
        elif not self.continuous and action in (1, 3):
            top_direction = -1 if action == 1 else 1
            top_power = 1.0

        if top_direction and self.jet_fires_used < self.max_jet_fires:
            # top_direction < 0 means upper-left jet, > 0 means upper-right jet.
            self.jet_fires_used += 1
            local_side = TOP_ENGINE_AWAY / SCALE * (-top_direction)
            impulse_sign = top_direction
            ox = tip[0] * (TOP_ENGINE_Y / SCALE + dispersion[0]) + side[0] * local_side
            oy = tip[1] * (TOP_ENGINE_Y / SCALE + dispersion[0]) + side[1] * local_side
            impulse_pos = (self.booster.position[0] + ox, self.booster.position[1] + oy)
            impulse = (
                side[0] * impulse_sign * TOP_ENGINE_POWER * top_power,
                side[1] * impulse_sign * TOP_ENGINE_POWER * top_power,
            )
            self.booster.ApplyLinearImpulse(impulse, impulse_pos, True)
        elif top_direction:
            top_power = 0.0

        self.bottom_flame_power = max(bottom_power, self.bottom_flame_power * 0.68)
        if top_direction:
            self.top_flame_direction = top_direction
        self.top_flame_power = max(top_power, self.top_flame_power * 0.68)

        return float(bottom_power), float(top_power)

    def _get_state(self) -> np.ndarray:
        assert self.booster is not None
        assert self.platform is not None

        pos = self.booster.position
        vel = self.booster.linearVelocity
        platform_pos = self.platform.position
        platform_vel = self.platform.linearVelocity
        target_y = self.platform_y + self.platform_half_height + BOOSTER_BOTTOM / SCALE
        half_w = VIEWPORT_W / SCALE / 2
        half_h = VIEWPORT_H / SCALE / 2

        state = np.array(
            [
                (pos.x - platform_pos.x) / half_w,
                (pos.y - target_y) / half_h,
                (vel.x - platform_vel.x) * half_w / FPS,
                vel.y * half_h / FPS,
                self._normalized_angle(self.booster.angle),
                20.0 * self.booster.angularVelocity / FPS,
                1.0 if self.left_foot_contact else 0.0,
                1.0 if self.right_foot_contact else 0.0,
                (platform_pos.x - half_w) / half_w,
                platform_vel.x,
                max(0.0, (self.max_jet_fires - self.jet_fires_used) / self.max_jet_fires),
            ],
            dtype=np.float32,
        )
        return state

    @staticmethod
    def _normalized_angle(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _is_standing_landing(self, state: np.ndarray) -> bool:
        assert self.booster is not None
        return bool(
            self.left_foot_contact
            and self.right_foot_contact
            and abs(state[0]) < (self.platform_half_width - 0.15) / (VIEWPORT_W / SCALE / 2)
            and abs(state[2]) < LANDING_VX
            and abs(state[3]) < LANDING_VY
            and abs(state[4]) < LANDING_ANGLE
            and abs(self.booster.angularVelocity) < LANDING_ANGULAR_V
        )

    def _terminal_status(self, state: np.ndarray) -> tuple[bool, bool, str | None]:
        assert self.booster is not None
        if self.ocean_contact or self.booster.position.y < self.ocean_y - 0.5:
            return True, False, "ocean"
        if abs(state[0]) > 2.0:
            return True, False, "out_of_bounds"
        if self.body_platform_contact:
            return True, False, "booster_body_hit_platform"
        if self.platform_contact and abs(state[4]) >= LANDING_ANGLE:
            return True, False, "non_vertical_platform_contact"
        if self._is_standing_landing(state) and (not self.booster.awake or abs(state[3]) < 0.08):
            return True, True, None
        if not self.booster.awake:
            return True, False, "settled_not_vertical"
        return False, False, None

    def _settle_successful_landing(self) -> None:
        assert self.booster is not None
        assert self.platform is not None
        target_y = self.platform_y + self.platform_half_height + BOOSTER_BOTTOM / SCALE
        platform_x = self.platform.position.x
        half_width = self.platform_half_width - 0.15
        x = float(np.clip(self.booster.position.x, platform_x - half_width, platform_x + half_width))

        self.booster.position = (x, target_y)
        self.booster.angle = 0.0
        self.booster.linearVelocity = self.platform.linearVelocity
        self.booster.angularVelocity = 0.0
        self.booster.awake = False
        self.left_foot_contact = True
        self.right_foot_contact = True
        self.body_platform_contact = False
        self.platform_contact = True

    def step(self, action):
        if self.booster is None:
            raise AssertionError("You forgot to call reset()")

        self._update_platform()
        self._apply_wind()
        bottom_power, top_power = self._apply_engines(action)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self._clamp_platform()

        state = self._get_state()
        shaping = (
            -100 * math.sqrt(state[0] * state[0] + state[1] * state[1])
            -100 * math.sqrt(state[2] * state[2] + state[3] * state[3])
            -120 * abs(state[4])
            + 12 * state[6]
            + 12 * state[7]
        )
        reward = 0.0 if self.prev_shaping is None else float(shaping - self.prev_shaping)
        self.prev_shaping = shaping
        reward -= bottom_power * 0.30
        reward -= top_power * 0.03

        terminated, success, failure_reason = self._terminal_status(state)
        self.failure_reason = failure_reason
        if terminated:
            reward = 100.0 if success else -100.0
            if success:
                self._settle_successful_landing()
                state = self._get_state()

        if self.render_mode == "human":
            self.render()

        info = {
            "success": success,
            "failure_reason": failure_reason,
            "platform_x": float(self.platform.position.x) if self.platform else None,
            "wind_power": self.wind_power if self.enable_wind else 0.0,
            "wind_direction": self.wind_direction,
            "jet_fires_used": self.jet_fires_used,
            "jet_fires_remaining": max(0, self.max_jet_fires - self.jet_fires_used),
        }
        return state, reward, terminated, False, info

    def render(self):
        if self.render_mode is None:
            return None

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as exc:  # pragma: no cover - optional rendering dependency
            raise DependencyNotInstalled("pygame is not installed. Install it with `pip install pygame`.") from exc

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        surf.fill((186, 220, 235))
        w = VIEWPORT_W / SCALE
        h = VIEWPORT_H / SCALE

        ocean_top = (self.ocean_y if hasattr(self, "ocean_y") else h / 4) * SCALE
        ocean_rect = pygame.Rect(0, 0, VIEWPORT_W, ocean_top)
        pygame.draw.rect(surf, (31, 91, 133), ocean_rect)
        for i in range(0, VIEWPORT_W, 48):
            pygame.draw.arc(surf, (62, 132, 176), (i, ocean_top - 8, 56, 18), 0, math.pi, 2)

        self._draw_flames(surf, pygame, gfxdraw)

        for obj in self.drawlist:
            for fixture in obj.fixtures:
                if fixture.userData == "ocean":
                    continue
                trans = fixture.body.transform
                if type(fixture.shape) is circleShape:
                    center = trans * fixture.shape.pos * SCALE
                    radius = max(2, int(fixture.shape.radius * SCALE))
                    pygame.draw.circle(surf, obj.color1, center, radius)
                    pygame.draw.circle(surf, obj.color2, center, max(1, radius - 1))
                    continue

                path = [trans * vertex * SCALE for vertex in fixture.shape.vertices]
                color = obj.color1
                outline = obj.color2
                if fixture.userData == "platform":
                    color = (45, 48, 54)
                    outline = (235, 235, 235)
                elif fixture.userData in {"left_foot", "right_foot"}:
                    color = (42, 46, 54)
                    outline = (245, 245, 245)
                pygame.draw.polygon(surf, color, path)
                gfxdraw.aapolygon(surf, path, color)
                pygame.draw.aalines(surf, outline, True, path)

        self._draw_booster_details(surf, pygame)
        surf = pygame.transform.flip(surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
            return None
        if self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2))
        return None

    def _draw_flames(self, surf, pygame, gfxdraw) -> None:
        if self.booster is None:
            return

        layer = pygame.Surface((VIEWPORT_W, VIEWPORT_H), pygame.SRCALPHA)

        if self.bottom_flame_power > 0.03:
            nozzle, direction = self._local_flame_anchor(
                (0.0, -BOTTOM_ENGINE_Y / SCALE),
                (0.0, -(BOTTOM_ENGINE_Y + 1.0) / SCALE),
            )
            self._draw_flame_cone(
                layer,
                pygame,
                gfxdraw,
                nozzle=nozzle,
                direction=direction,
                power=self.bottom_flame_power,
                length_px=34,
                base_width_px=7,
                end_width_px=20,
            )

        if self.top_flame_power > 0.03 and self.top_flame_direction:
            local_x = TOP_ENGINE_AWAY / SCALE * (-self.top_flame_direction)
            nozzle, direction = self._local_flame_anchor(
                (local_x, TOP_ENGINE_Y / SCALE),
                (
                    local_x - self.top_flame_direction / SCALE,
                    TOP_ENGINE_Y / SCALE,
                ),
            )
            self._draw_flame_cone(
                layer,
                pygame,
                gfxdraw,
                nozzle=nozzle,
                direction=direction,
                power=self.top_flame_power,
                length_px=18,
                base_width_px=4,
                end_width_px=10,
            )

        surf.blit(layer, (0, 0))

    def _local_flame_anchor(
        self,
        nozzle_local: tuple[float, float],
        direction_local: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        assert self.booster is not None
        nozzle_world = self.booster.transform * nozzle_local
        direction_world = self.booster.transform * direction_local
        nozzle = (nozzle_world[0] * SCALE, nozzle_world[1] * SCALE)
        direction = (
            direction_world[0] * SCALE - nozzle[0],
            direction_world[1] * SCALE - nozzle[1],
        )
        return nozzle, direction

    @staticmethod
    def _draw_flame_cone(
        layer,
        pygame,
        gfxdraw,
        *,
        nozzle: tuple[float, float],
        direction: tuple[float, float],
        power: float,
        length_px: float,
        base_width_px: float,
        end_width_px: float,
    ) -> None:
        direction_vec = np.array(direction, dtype=np.float64)
        norm = np.linalg.norm(direction_vec)
        if norm == 0:
            return

        direction_vec /= norm
        perp = np.array([-direction_vec[1], direction_vec[0]])
        power = float(np.clip(power, 0.0, 1.0))
        start = np.array(nozzle, dtype=np.float64)
        end = start + direction_vec * length_px * (0.45 + 0.55 * power)

        def poly(base_width: float, end_width: float, length_scale: float) -> list[tuple[int, int]]:
            scaled_end = start + (end - start) * length_scale
            return [
                tuple(np.round(start + perp * base_width / 2).astype(int)),
                tuple(np.round(start - perp * base_width / 2).astype(int)),
                tuple(np.round(scaled_end - perp * end_width / 2).astype(int)),
                tuple(np.round(scaled_end + perp * end_width / 2).astype(int)),
            ]

        alpha = int(210 * power)
        outer = poly(base_width_px, end_width_px, 1.0)
        middle = poly(base_width_px * 0.65, end_width_px * 0.58, 0.78)
        inner = poly(base_width_px * 0.32, end_width_px * 0.24, 0.48)

        pygame.draw.polygon(layer, (69, 39, 24, int(120 * power)), outer)
        gfxdraw.aapolygon(layer, outer, (69, 39, 24, int(120 * power)))
        pygame.draw.polygon(layer, (255, 112, 28, alpha), middle)
        gfxdraw.aapolygon(layer, middle, (255, 112, 28, alpha))
        pygame.draw.polygon(layer, (255, 235, 132, int(245 * power)), inner)
        gfxdraw.aapolygon(layer, inner, (255, 235, 132, int(245 * power)))

    def _draw_booster_details(self, surf, pygame) -> None:
        if self.booster is None:
            return
        trans = self.booster.transform
        stripe_points = [
            (-6 / SCALE, 20 / SCALE),
            (6 / SCALE, 20 / SCALE),
            (6 / SCALE, 28 / SCALE),
            (-6 / SCALE, 28 / SCALE),
        ]
        stripe = [trans * point * SCALE for point in stripe_points]
        pygame.draw.polygon(surf, (35, 40, 48), stripe)

        logo_center = trans * (0, 35 / SCALE) * SCALE
        font_rect = pygame.Rect(0, 0, 18, 7)
        font_rect.center = logo_center
        pygame.draw.rect(surf, (35, 40, 48), font_rect, border_radius=1)

    def close(self) -> None:
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def heuristic(env: PlatformLander, state: np.ndarray):
    """A simple controller for quick smoke tests and demos."""

    angle_target = state[0] * 0.55 + state[2] * 0.9
    angle_target = float(np.clip(angle_target, -0.35, 0.35))
    hover_target = 0.45 * abs(state[0])

    angle_todo = (angle_target - state[4]) * 0.55 - state[5] * 0.85
    hover_todo = (hover_target - state[1]) * 0.55 - state[3] * 0.55

    if state[6] or state[7]:
        angle_todo = -state[4] * 0.9 - state[5] * 0.5
        hover_todo = -state[3] * 0.5

    if env.unwrapped.continuous:
        action = np.array([hover_todo * 18 - 1, angle_todo * 18], dtype=np.float32)
        return np.clip(action, -1, +1)

    action = 0
    if hover_todo > abs(angle_todo) and hover_todo > 0.04:
        action = 2
    elif angle_todo < -0.04:
        action = 1
    elif angle_todo > 0.04:
        action = 3
    return action


def demo_heuristic_lander(seed: int | None = None, render: bool = False) -> float:
    env = PlatformLander(render_mode="human" if render else None)
    total_reward = 0.0
    state, _ = env.reset(seed=seed)
    for _ in range(1000):
        action = heuristic(env, state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    env.close()
    return total_reward


if __name__ == "__main__":
    demo_heuristic_lander(render=True)
