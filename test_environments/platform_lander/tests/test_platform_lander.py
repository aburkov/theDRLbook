import math
import subprocess
import sys
import os
from pathlib import Path

import numpy as np
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


def test_default_wind_power_is_exported_single_source():
    from platform_lander import DEFAULT_WIND_POWER, PlatformLander

    env = PlatformLander()
    assert env.wind_power == DEFAULT_WIND_POWER
    env.close()


def test_default_jet_fire_budget_is_50():
    from platform_lander import PlatformLander

    env = PlatformLander()
    assert env.max_jet_fires == 50
    env.close()


def test_leg_mass_increased_without_increasing_total_booster_mass():
    from platform_lander import PlatformLander
    from platform_lander.platform_lander import BOOSTER_BODY_DENSITY, LANDING_LEG_DENSITY

    env = PlatformLander()
    env.reset(seed=0)
    fixture_by_label = {fixture.userData: fixture for fixture in env.booster.fixtures}

    assert fixture_by_label["booster_body"].density == pytest.approx(BOOSTER_BODY_DENSITY)
    assert fixture_by_label["left_foot"].density == pytest.approx(LANDING_LEG_DENSITY)
    assert fixture_by_label["right_foot"].density == pytest.approx(LANDING_LEG_DENSITY)
    assert env.booster.mass == pytest.approx(7.9677, abs=1e-3)
    env.close()


def test_max_jet_fire_budget_must_be_positive():
    from platform_lander import PlatformLander

    with pytest.raises(ValueError, match="max_jet_fires"):
        PlatformLander(max_jet_fires=0)


def test_default_rewards_and_shaping_factor_are_unchanged():
    from platform_lander import (
        DEFAULT_FAILURE_REWARD,
        DEFAULT_SHAPING_FACTOR,
        DEFAULT_SUCCESS_REWARD,
        PlatformLander,
    )

    env = PlatformLander()
    assert env.success_reward == DEFAULT_SUCCESS_REWARD == 100.0
    assert env.failure_reward == DEFAULT_FAILURE_REWARD == -100.0
    assert env.shaping_factor == DEFAULT_SHAPING_FACTOR == 1.0
    env.close()


def test_custom_terminal_rewards_are_used():
    from platform_lander import PlatformLander
    from platform_lander.platform_lander import BOOSTER_BOTTOM, SCALE

    success_env = PlatformLander(success_reward=500.0)
    success_env.reset(seed=4)
    success_env.booster.position = (
        success_env.platform.position.x,
        success_env.platform_y + success_env.platform_half_height + BOOSTER_BOTTOM / SCALE,
    )
    success_env.booster.angle = 0.04
    success_env.booster.linearVelocity = success_env.platform.linearVelocity
    success_env.booster.angularVelocity = 0.02
    success_env.booster.awake = False
    success_env.left_foot_contact = True
    success_env.right_foot_contact = True
    success_env.platform_contact = True

    _state, reward, terminated, _truncated, info = success_env.step(0)
    assert terminated is True
    assert info["success"] is True
    assert reward == 500.0
    success_env.close()

    failure_env = PlatformLander(failure_reward=-500.0)
    failure_env.reset(seed=4)
    failure_env.ocean_contact = True
    _state, reward, terminated, _truncated, info = failure_env.step(0)
    assert terminated is True
    assert info["success"] is False
    assert reward == -500.0
    failure_env.close()


def test_shaping_factor_scales_dense_reward():
    from platform_lander import PlatformLander

    default_env = PlatformLander()
    zero_shaping_env = PlatformLander(shaping_factor=0.0)
    default_env.reset(seed=8)
    zero_shaping_env.reset(seed=8)

    default_env.step(0)
    zero_shaping_env.step(0)
    _obs, default_reward, _terminated, _truncated, _info = default_env.step(0)
    _obs, zero_reward, _terminated, _truncated, _info = zero_shaping_env.step(0)

    assert default_reward != 0.0
    assert zero_reward == 0.0
    default_env.close()
    zero_shaping_env.close()


def test_dense_shaping_rewards_vertical_position_only_when_bottom_is_over_platform():
    from platform_lander import PlatformLander

    env = PlatformLander()
    far_high_state = np.zeros(11, dtype=np.float32)
    far_high_state[0] = 0.4
    far_high_state[1] = 1.5
    far_high_state[2] = 0.2
    far_high_state[3] = -0.5
    far_high_state[4] = 0.1

    far_lower_state = far_high_state.copy()
    far_lower_state[1] = 0.1

    over_high_state = far_high_state.copy()
    over_high_state[0] = 0.0
    over_lower_state = over_high_state.copy()
    over_lower_state[1] = 0.1

    tilted_center_over_high_state = over_high_state.copy()
    tilted_center_over_high_state[0] = 0.18
    tilted_center_over_high_state[4] = -math.pi / 2
    tilted_center_over_lower_state = tilted_center_over_high_state.copy()
    tilted_center_over_lower_state[1] = 0.1

    assert env._dense_shaping(far_lower_state) == pytest.approx(env._dense_shaping(far_high_state))
    assert env._dense_shaping(over_lower_state) > env._dense_shaping(over_high_state)
    assert env._dense_shaping(tilted_center_over_lower_state) == pytest.approx(
        env._dense_shaping(tilted_center_over_high_state)
    )
    env.close()


def test_dense_shaping_penalizes_angular_velocity():
    from platform_lander import PlatformLander

    env = PlatformLander()
    still_state = np.zeros(11, dtype=np.float32)
    spinning_state = still_state.copy()
    spinning_state[5] = 0.5

    assert env._dense_shaping(still_state) > env._dense_shaping(spinning_state)
    env.close()


def test_dense_shaping_penalizes_horizontal_and_vertical_speed_separately():
    from platform_lander import PlatformLander

    env = PlatformLander()
    still_state = np.zeros(11, dtype=np.float32)
    horizontal_state = still_state.copy()
    vertical_state = still_state.copy()
    horizontal_state[2] = 0.5
    vertical_state[3] = -0.5

    assert env._dense_shaping(still_state) > env._dense_shaping(horizontal_state)
    assert env._dense_shaping(still_state) > env._dense_shaping(vertical_state)
    assert env._dense_shaping(horizontal_state) == pytest.approx(env._dense_shaping(vertical_state))
    env.close()


def test_negative_shaping_factor_is_rejected():
    from platform_lander import PlatformLander

    with pytest.raises(ValueError):
        PlatformLander(shaping_factor=-0.1)


def test_zero_power_wind_matches_no_wind():
    from platform_lander import PlatformLander

    no_wind = PlatformLander(enable_wind=False)
    zero_wind = PlatformLander(enable_wind=True, wind_power=0.0)

    no_wind_obs, _ = no_wind.reset(seed=7)
    zero_wind_obs, _ = zero_wind.reset(seed=7)
    assert np.allclose(no_wind_obs, zero_wind_obs)

    for _ in range(20):
        no_wind_obs, _, _, _, no_wind_info = no_wind.step(0)
        zero_wind_obs, _, _, _, zero_wind_info = zero_wind.step(0)

    assert np.allclose(no_wind_obs, zero_wind_obs)
    assert zero_wind_info["wind_power"] == 0.0
    no_wind.close()
    zero_wind.close()


def test_wind_only_changes_horizontal_velocity():
    from platform_lander import PlatformLander

    no_wind = PlatformLander(enable_wind=False)
    wind = PlatformLander(enable_wind=True, wind_power=5.0, wind_direction=(1.0, 0.5), variable_wind=False)

    no_wind.reset(seed=11)
    wind.reset(seed=11)
    no_wind.step(0)
    wind.step(0)

    assert wind.booster.linearVelocity.x > no_wind.booster.linearVelocity.x
    assert wind.booster.linearVelocity.y == pytest.approx(no_wind.booster.linearVelocity.y)
    assert wind.booster.angularVelocity == pytest.approx(no_wind.booster.angularVelocity)
    no_wind.close()
    wind.close()


def test_variable_wind_changes_lateral_direction_and_magnitude():
    from platform_lander import PlatformLander

    wind = PlatformLander(enable_wind=True, wind_power=5.0, wind_direction=(1.0, 0.0))
    wind.reset(seed=13)
    wind.wind_idx = 0

    scales = [wind._wind_scale() for _ in range(600)]
    assert any(scale > 0.0 for scale in scales)
    assert any(scale < 0.0 for scale in scales)
    assert len({round(abs(scale), 3) for scale in scales}) > 10
    assert wind._wind_unit() == (1.0, 0.0)
    wind.close()


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


def test_booster_starts_over_platform_with_random_tilt_and_zero_velocity():
    from platform_lander import PlatformLander
    from platform_lander.platform_lander import BOOSTER_START_MAX_ANGLE

    env = PlatformLander()
    env.reset(seed=3)

    assert env.booster.position.x == pytest.approx(env.platform.position.x)
    assert 0.0 <= abs(env.booster.angle) <= BOOSTER_START_MAX_ANGLE
    assert env.booster.linearVelocity.x == pytest.approx(0.0)
    assert env.booster.linearVelocity.y == pytest.approx(0.0)
    assert env.booster.angularVelocity == pytest.approx(0.0)
    env.close()


def test_booster_start_tilt_uses_both_directions():
    from platform_lander import PlatformLander

    env = PlatformLander()
    signs = set()
    for seed in range(20):
        env.reset(seed=seed)
        if abs(env.booster.angle) > 1e-9:
            signs.add(1 if env.booster.angle > 0 else -1)

    assert signs == {-1, 1}
    env.close()


def test_training_start_randomizes_platform_and_tilt():
    from platform_lander import PlatformLander
    from platform_lander.platform_lander import BOOSTER_BOTTOM, BOOSTER_START_CLEARANCE, BOOSTER_START_MAX_ANGLE, SCALE
    from vanilla_reinforce import randomize_start

    env = PlatformLander()
    env.reset(seed=3)
    observation = randomize_start(env, np.random.default_rng(4))
    expected_y = 400 / 30.0 + BOOSTER_BOTTOM / SCALE + BOOSTER_START_CLEARANCE

    assert env.booster.position.x == pytest.approx(env.platform.position.x)
    assert env.booster.position.y == pytest.approx(expected_y)
    assert 0.0 <= abs(env.booster.angle) <= BOOSTER_START_MAX_ANGLE
    assert env.booster.linearVelocity.x == pytest.approx(0.0)
    assert env.booster.linearVelocity.y == pytest.approx(0.0)
    assert env.booster.angularVelocity == pytest.approx(0.0)
    assert observation[0] == pytest.approx(0.0)
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


def test_side_jet_nozzle_side_matches_action_name():
    from platform_lander import PlatformLander

    left_env = PlatformLander()
    right_env = PlatformLander()
    left_env.reset(seed=1)
    right_env.reset(seed=1)
    left_env.step(1)
    right_env.step(3)

    assert left_env.top_flame_direction < 0
    assert right_env.top_flame_direction > 0
    assert -left_env.top_flame_direction == right_env.top_flame_direction
    left_env.close()
    right_env.close()


def test_training_rollout_timeout_is_failure():
    from platform_lander import PlatformLander
    from vanilla_reinforce import Policy, run_episode_data

    rng = np.random.default_rng(0)
    env = PlatformLander()
    policy = Policy(
        obs_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.n),
    )

    _observations, _log_probs, rewards, episode_return, steps, info = run_episode_data(
        env,
        policy,
        rng,
        gamma=1.0,
        max_steps=1,
        train=False,
    )

    assert steps == 1
    assert rewards[-1] == -100.0
    assert episode_return == -100.0
    assert info["success"] is False
    assert info["failure_reason"] == "timeout"
    env.close()


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
    env.booster.awake = False
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


def test_stable_platform_contact_succeeds_without_box2d_sleep():
    from platform_lander import PlatformLander
    from platform_lander.platform_lander import BOOSTER_BOTTOM, SCALE, STABLE_LANDING_STEPS

    env = PlatformLander()
    env.reset(seed=4)
    env.booster.position = (env.platform.position.x, env.platform_y + env.platform_half_height + BOOSTER_BOTTOM / SCALE)
    env.booster.angle = 0.04
    env.booster.linearVelocity = env.platform.linearVelocity
    env.booster.angularVelocity = 0.02
    env.booster.awake = True
    env.left_foot_contact = True
    env.right_foot_contact = True
    env.platform_contact = True

    for _ in range(STABLE_LANDING_STEPS):
        state = env._get_state()
        env._update_landing_stability(state)

    terminated, success, failure_reason = env._terminal_status(state)

    assert terminated is True
    assert success is True
    assert failure_reason is None
    env.close()


def test_missing_second_leg_anchors_when_visually_touching_platform():
    from platform_lander import PlatformLander
    from platform_lander.platform_lander import BOOSTER_BOTTOM, SCALE

    env = PlatformLander()
    env.reset(seed=4)
    env.booster.position = (
        env.platform.position.x - 0.8,
        env.platform_y + env.platform_half_height + BOOSTER_BOTTOM / SCALE + 0.02,
    )
    env.booster.angle = 0.0
    env.booster.linearVelocity = env.platform.linearVelocity
    env.booster.angularVelocity = 0.0
    env.right_foot_contact = True
    env.platform_contact = True

    env.anchored_feet.add("right_foot")
    env._queue_feet_visually_touching_platform()
    env._process_pending_leg_anchors()

    assert tuple(sorted(env.anchored_feet)) == ("left_foot", "right_foot")
    assert env.left_foot_contact is True
    assert env.right_foot_contact is True
    env.close()


def test_one_anchored_leg_allows_side_jet_rotation():
    from platform_lander import PlatformLander
    from platform_lander.platform_lander import BOOSTER_BOTTOM, SCALE

    env = PlatformLander()
    env.reset(seed=4)
    env.booster.position = (
        env.platform.position.x,
        env.platform_y + env.platform_half_height + BOOSTER_BOTTOM / SCALE,
    )
    env.booster.angle = 0.0
    env.booster.linearVelocity = env.platform.linearVelocity
    env.booster.angularVelocity = 0.0
    env.left_foot_contact = True
    env.platform_contact = True
    env._queue_leg_anchor("left_foot")
    env._process_pending_leg_anchors()

    initial_anchor_count = len(env.leg_anchor_joints)
    joint = env.leg_anchor_joints["left_foot"]
    initial_angular_velocity = float(env.booster.angularVelocity)
    env._apply_engines(3)

    assert initial_anchor_count == 1
    assert joint.collideConnected is True
    assert tuple(sorted(env.anchored_feet)) == ("left_foot",)
    assert abs(float(env.booster.angularVelocity) - initial_angular_velocity) > 0.0
    env.close()


def test_leg_anchor_lifts_pierced_foot_to_platform_surface():
    from platform_lander import PlatformLander
    from platform_lander.platform_lander import BOOSTER_BOTTOM, SCALE

    env = PlatformLander()
    env.reset(seed=4)
    env.booster.position = (
        env.platform.position.x,
        env.platform_y + env.platform_half_height + BOOSTER_BOTTOM / SCALE - 0.4,
    )
    env.booster.angle = 0.0
    env.left_foot_contact = True
    env.platform_contact = True
    env._queue_leg_anchor("left_foot")
    env._process_pending_leg_anchors()

    _foot_x, foot_y = env._foot_anchor_world("left_foot")

    assert foot_y >= env._platform_top_y() - 1e-6
    assert env.anchored_feet == {"left_foot"}
    env.close()


def test_two_anchored_legs_trigger_success():
    from platform_lander import PlatformLander
    from platform_lander.platform_lander import BOOSTER_BOTTOM, SCALE

    env = PlatformLander()
    env.reset(seed=4)
    env.booster.position = (
        env.platform.position.x,
        env.platform_y + env.platform_half_height + BOOSTER_BOTTOM / SCALE,
    )
    env.booster.angle = 0.0
    env.booster.linearVelocity = env.platform.linearVelocity
    env.booster.angularVelocity = 0.0
    env.left_foot_contact = True
    env.right_foot_contact = True
    env.platform_contact = True
    env._queue_leg_anchor("left_foot")
    env._queue_leg_anchor("right_foot")
    env._process_pending_leg_anchors()

    _state, reward, terminated, _truncated, info = env.step(0)

    assert terminated is True
    assert info["success"] is True
    assert reward == 100.0
    env.close()


def test_fast_platform_impact_does_not_fail_before_settling():
    from platform_lander import PlatformLander
    from platform_lander.platform_lander import BOOSTER_BOTTOM, SCALE

    env = PlatformLander()
    env.reset(seed=4)
    env.booster.position = (env.platform.position.x, env.platform_y + env.platform_half_height + BOOSTER_BOTTOM / SCALE)
    env.booster.angle = 0.0
    env.booster.linearVelocity = env.platform.linearVelocity
    env.booster.angularVelocity = 0.0
    env.left_foot_contact = True
    env.right_foot_contact = True
    env.platform_contact = True
    env.platform_impact_velocity = (0.0, -12.0, 0.0)

    state = env._get_state()
    terminated, success, failure_reason = env._terminal_status(state)

    assert terminated is False
    assert success is False
    assert failure_reason is None
    env.close()


def test_crash_effect_starts_on_failure():
    from platform_lander import PlatformLander
    from platform_lander.platform_lander import CRASH_EFFECT_FRAMES

    env = PlatformLander()
    env.reset(seed=4)
    env.ocean_contact = True

    _state, _reward, terminated, _truncated, info = env.step(0)

    assert terminated is True
    assert info["success"] is False
    assert env.crash_effect_position is not None
    assert env.crash_effect_frame == 0
    assert env.crash_effect_frames == CRASH_EFFECT_FRAMES
    env.close()
