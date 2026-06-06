# Platform Lander

A standalone reusable-booster landing environment based on Gymnasium LunarLander v3 physics, but without importing Gymnasium. The task is to land a SpaceX-style booster upright on a moving floating platform. Missing the platform and falling into the ocean, or contacting the platform in a non-vertical position, terminates the episode as failure.

## Install

After the package has been published to PyPI:

```bash
pip install platform_lander
```

Before the PyPI release is available, install the same package directly from
the book repository subdirectory:

```bash
pip install "platform_lander @ git+https://github.com/aburkov/theDRLbook.git#subdirectory=test_environments/platform_lander"
```

For local development from this folder:

```bash
pip install -e .
```

## Google Colab

Use the same install command in the first notebook cell. Colab usually needs `swig` before Box2D builds:

```python
!apt-get -qq install swig
!pip install -q platform_lander
```

Then import normally:

```python
from platform_lander import PlatformLander

env = PlatformLander(render_mode="rgb_array", enable_wind=True, wind_power=5.0)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(2)
frame = env.render()
```

Display a rendered frame in Colab:

```python
import matplotlib.pyplot as plt

plt.imshow(frame)
plt.axis("off")
plt.show()
```

## Local Script

To watch the booster in a local Pygame window, install the package in editable
mode and run the demo:

```bash
pip install -e .
python examples/demo.py
```

The test file is headless, so running `pytest` or `python tests/test_platform_lander.py`
will not open an animation window.

To train a discrete policy with the textbook single-trajectory REINFORCE
algorithm and then show three animated runs:

```bash
pip install -e ".[train]"
python vanilla_reinforce.py
```

The repository also includes incremental REINFORCE variants:

```bash
python rtg_reinforce.py                                  # vanilla + per-timestep reward-to-go
python average_reinforcement_baseline_reinforce.py       # reward-to-go + running scalar RTG baseline
python value_function_baseline_reinforce.py              # reward-to-go + learned value-function baseline
python batch_reinforce.py                                # vanilla + trajectory batches
python full_reinforce.py                                 # batches + reward-to-go + selectable scalar baseline
```

Each training script writes a log, per-episode CSV data, and a checkpoint under
`runs/` by default, for example `runs/full_reinforce.log`,
`runs/full_reinforce.csv`, and `runs/full_reinforce.pt`. Override those paths
with `--log-file`, `--csv-file`, and `--model-file`.

To load the hardcoded `runs/full_reinforce.pt` checkpoint and watch several
animated policy rollouts:

```bash
python watch_trained_policy.py
```

To generate one side-by-side results graph per variant from the saved CSV
files:

```bash
python plot_reinforce_results.py
```

For a quick smoke test without opening the animation window:

```bash
python vanilla_reinforce.py --episodes 3 --max-steps 20 --no-animation
```

Training scripts also expose reward-scale controls for experiments:

```bash
python gamma_dropped_rtg_reinforce.py --success-reward 500 --failure-reward -500 --shaping-factor 0.5
```

```python
from platform_lander import PlatformLander

env = PlatformLander(enable_wind=True, wind_direction=(1, 0.2), wind_power=5.0)
obs, info = env.reset(seed=0)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(info)
        break

env.close()
```

## API Notes

- `PlatformLander(continuous=False)` uses `Discrete(4)` actions.
- Actions: `0` no-op, `1` upper-left attitude jet, `2` bottom engine, `3` upper-right attitude jet.
- `continuous=True` uses a two-value `Box(-1, 1, shape=(2,))` action.
- Wind is controlled with `enable_wind`, `wind_power`, `wind_direction`, and `set_wind(...)`.
- The platform moves horizontally at `1.15 / 3.0` world units per second by default.
- Episodes start with the platform at a random x/direction and the booster directly above it, tilted like `\` by 20 degrees with zero initial velocity.
- Terminal rewards default to `success_reward=100.0` and `failure_reward=-100.0`.
- Dense shaping is multiplied by `shaping_factor`, which defaults to `1.0`.
- Dense shaping rewards lateral alignment, low relative horizontal speed, low vertical speed, vertical attitude, low angular velocity, and foot contact. It also rewards vertical closeness to the platform, but only while the booster's bottom is horizontally above the platform.
- The booster has 50 available jet fires by default. After they are exhausted,
  engine commands have no effect and the booster continues ballistically.
- The observation includes the fraction of jet fires remaining.
- The package provides local `Box` and `Discrete` spaces and does not import Gymnasium.

## Publishing

Build the package from this directory:

```bash
python -m build
```

Upload the generated `dist/platform_lander-*.tar.gz` and
`dist/platform_lander-*.whl` files to PyPI with a PyPI account that owns the
`platform_lander` project name:

```bash
python -m twine upload dist/*
```
