import subprocess
import sys
import os
from pathlib import Path

import pytest

pytest.importorskip("Box2D")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))


def test_platform_lander_imports_without_gymnasium():
    code = (
        "import sys; "
        "from platform_lander import PlatformLander; "
        "assert 'gymnasium' not in sys.modules; "
        "env = PlatformLander(); "
        "env.reset(seed=0); "
        "env.close(); "
        "print('ok')"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.strip() == "ok"


@pytest.mark.parametrize("continuous", [False, True])
def test_platform_lander_reset_and_step(continuous):
    from platform_lander import PlatformLander

    env = PlatformLander(continuous=continuous, enable_wind=True, wind_direction=(1, 0.25))
    obs, info = env.reset(seed=123)
    assert obs.shape == (11,)
    assert info == {}

    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == (11,)
        assert isinstance(reward, float)
        assert truncated is False
        assert "success" in info
        if terminated:
            break
    env.close()


def test_platform_never_leaves_screen_and_reverses():
    from platform_lander import PlatformLander

    env = PlatformLander(platform_speed=20.0)
    env.reset(seed=2)
    xs = []
    directions = set()
    for _ in range(120):
        env.step(0)
        xs.append(env.platform.position.x)
        directions.add(env.platform_direction)

    assert min(xs) >= env.platform_min_x - 1e-6
    assert max(xs) <= env.platform_max_x + 1e-6
    assert directions == {-1, 1}
    env.close()


def test_booster_starts_above_visible_sky():
    from platform_lander import PlatformLander
    from platform_lander.platform_lander import SCALE, VIEWPORT_H

    env = PlatformLander()
    env.reset(seed=3)

    assert env.booster.position.y > VIEWPORT_H / SCALE
    env.close()


def test_jets_push_and_rotate_booster():
    from platform_lander import PlatformLander

    env = PlatformLander()
    env.reset(seed=1)
    initial_vy = env.booster.linearVelocity.y
    env.step(2)
    assert env.booster.linearVelocity.y > initial_vy
    env.close()

    left_env = PlatformLander()
    right_env = PlatformLander()
    left_env.reset(seed=1)
    right_env.reset(seed=1)
    for _ in range(5):
        left_env.step(1)
        right_env.step(3)
    assert left_env.booster.angularVelocity < 0
    assert right_env.booster.angularVelocity > 0
    left_env.close()
    right_env.close()


def test_jet_fire_budget_caps_engine_impulses():
    from platform_lander import PlatformLander

    env = PlatformLander(max_jet_fires=1)
    env.reset(seed=1)

    initial_vy = env.booster.linearVelocity.y
    env.step(2)
    after_first_fire_vy = env.booster.linearVelocity.y
    env.step(2)
    after_exhausted_fire_vy = env.booster.linearVelocity.y

    assert env.jet_fires_used == 1
    assert after_first_fire_vy > initial_vy
    assert after_exhausted_fire_vy < after_first_fire_vy
    assert env._get_state()[-1] == 0.0
    env.close()


def test_successful_landing_settles_booster_upright():
    from platform_lander import PlatformLander
    from platform_lander.platform_lander import BOOSTER_BOTTOM, SCALE

    env = PlatformLander()
    env.reset(seed=4)
    env.booster.position = (env.platform.position.x, env.platform_y + env.platform_half_height + BOOSTER_BOTTOM / SCALE)
    env.booster.angle = 0.04
    env.booster.linearVelocity = env.platform.linearVelocity
    env.booster.angularVelocity = 0.02
    env.left_foot_contact = True
    env.right_foot_contact = True
    env.platform_contact = True

    state, reward, terminated, truncated, info = env.step(0)

    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert reward == 100.0
    assert env.booster.angle == 0.0
    assert env.booster.angularVelocity == 0.0
    assert env.left_foot_contact is True
    assert env.right_foot_contact is True
    env.close()
