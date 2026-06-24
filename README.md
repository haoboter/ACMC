# ACMC

Automatic Cross-scale Modular Coordination (ACMC) is the codebase for the
CrossBot study, "Miniature cross-scale magnetic robots learn to coordinate".
CrossBot combines a millimeter-scale magnetic millicore with nanoscale magnetic
nanounits and learns to coordinate millicore locomotion, nanounit locomotion,
and assembly/disassembly under one shared rotating magnetic field.

The repository follows the workflow used in the paper:

1. Learn elementary behavior models, also called digital twins, from active or
   offline experiments.
2. Use the learned behavior models inside a Gym-style simulation environment.
3. Train a reinforcement-learning policy in simulation.
4. Deploy the trained policy on the physical magnetic-actuation and camera
   platform.

## Repository layout

| Path | Purpose |
| --- | --- |
| `Elementar_behavior_modeling/` | Active-learning scripts, datasets, and trained behavior models for millicore motion, nanounit motion, and release/reconfiguration. |
| `Independent cross-scale coordination/` | Simulation environment and SAC-based RL training code. The environment loads the three behavior-model weights from `SL_model/AL_all/`. |
| `Exp/` | Real-world deployment code, hardware interfaces, YOLO-based perception, and saved RL checkpoints. |
| `LICENSE` | MIT license. |

Vendor/device code is bundled in the repository for reproducibility:

- Sensory 826 DAQ wrappers and SDK files are under `S826.py` and `sdk_826_*`
  folders.
- Daheng Imaging camera Python wrappers are under `gxipy/`.

## Environment

Use a Python environment with PyTorch, OpenCV, Gym, SciPy, pandas, and
Ultralytics. A CUDA-enabled PyTorch build is optional but useful for training.

Example setup:

```bash
conda create -n acmc python=3.9
conda activate acmc

pip install numpy scipy matplotlib pandas openpyxl opencv-python gym ultralytics
pip install torch torchvision
```

If you need a specific CUDA or CPU PyTorch build, use the official PyTorch
installation selector and keep the rest of the packages above unchanged.

The real-world and active-learning scripts also require the physical platform:

- Sensory 826 DAQ board and installed driver.
- Daheng GX industrial camera and working camera runtime.
- Magnetic coil setup driven through the S826 board.
- A YOLO model file named `best.pt` in the working directory for scripts that
  perform millicore detection.

Simulation-only reproduction does not require the DAQ board or camera.

## Reproduction workflow

Run commands from the repository root unless a step says otherwise.

### 1. Behavior models

Pretrained behavior models are already included:

```text
Elementar_behavior_modeling/behavior_models/milli_velocity.pth
Elementar_behavior_modeling/behavior_models/nano_velocity.pth
Elementar_behavior_modeling/behavior_models/release_grayscale.pth
```

The three behavior models correspond to the elementary digital twins described in
the paper:

- millicore locomotion velocity,
- nanounit locomotion velocity,
- nanounit release/reconfiguration response measured from grayscale.

To retrain a behavior model from the included Excel data, edit
`EXPERIMENT_TYPE` in
`Elementar_behavior_modeling/code/offline/train_offline.py`.
Valid values are:

```python
EXPERIMENT_TYPE = "milli"
EXPERIMENT_TYPE = "nano"
EXPERIMENT_TYPE = "reconfiguration"
```

Then run:

```bash
cd Elementar_behavior_modeling/code/offline
python train_offline.py
```

The script expects these columns in the selected Excel sheet:

- Inputs: `Flux`, `Frequency`, `Pitch`, `Direction`
- Output for `milli` and `nano`: `Velocity`
- Output for `reconfiguration`: `Final_Gray`

It saves outputs under the corresponding data folder:

```text
Elementar_behavior_modeling/data/*/train_output/<model_name>/
```

### 2. Active-learning data collection

The active-learning scripts reproduce the autonomous experimental sampling loop
from the paper, but they require the physical DAQ/camera/coil setup.

```bash
cd Elementar_behavior_modeling/code/milli
python active_learning_millicore_motion.py

cd ../nano
python active_learning_nanounits_motion.py

cd ../reconfig
python active_learning_release_rate.py
```

These scripts read and write local `.npy`, `.pth`, and `.xlsx` files in their
current working directories. Run each script from its own folder so relative
paths resolve as expected.

### 3. RL training in simulation

The RL environment is implemented in:

```text
Independent cross-scale coordination/sim_env.py
```

Its observation is an 8D vector:

```text
[target_millicore, target_nano, target_release,
 current_millicore, current_nano, current_release,
 theta_millicore, theta_nano]
```

Its action is a normalized 4D rotating-field vector:

```text
[flux_density, frequency, pitch_angle, direction_angle]
```

The normalized action is mapped to physical ranges used in the paper:

- flux density: `0-20 mT`
- frequency: `0-40 Hz`
- pitch angle: `0-180 deg`
- direction angle: `0-360 deg`

Train the policy:

```bash
cd "Independent cross-scale coordination"
python main_RL.py
```

Key output files include:

```text
actor_last
critic_1_last
critic_2_last
value_last
target_value_last
score_plot.png
task_completion_rates.png
current_episode.npy
```

To deploy a trained policy in the real-world code, copy the five `*_last`
checkpoint files into:

```text
Exp/agent/
```

This repository already includes an `Exp/agent/` checkpoint set.

### 4. Real-world deployment

Run these scripts only on the experimental machine with the camera, DAQ board,
coil driver, and `best.pt` available.

Policy-controlled experiment:

```bash
cd Exp
python main_exp.py
```

Human-in-the-loop baseline/control:

```bash
cd Exp
python main_exp_human.py
```

At launch, the real-world code initializes the camera, asks for or loads selected
pipe-centerline points, activates the coil, and starts segment-level execution.
Each run creates a timestamped folder under:

```text
Exp/output/run_<timestamp>/
Exp/output/run_human_<timestamp>/
```

Typical outputs are:

```text
experiment_config.json
loaded_agent_models/
obs_action_history.npz
segment_results.xlsx
all_episodes_actions.xlsx
episode_* videos and info files
```

Use `Ctrl+C` to interrupt; the scripts attempt to save completed segment data and
stop the coil in the shutdown path.

## Paper-to-code correspondence

| Paper/SI concept | Code location |
| --- | --- |
| Active experimental sampling for behavior models | `Elementar_behavior_modeling/code/{milli,nano,reconfig}/active_learning_*.py` |
| Offline supervised behavior-model training | `Elementar_behavior_modeling/code/offline/train_offline.py` |
| Millicore and nanounit velocity networks | `Elementar_behavior_modeling/code/offline/behavior_networks.py`, `Independent cross-scale coordination/utils/util.py` |
| Release/grayscale prediction network | `Elementar_behavior_modeling/code/offline/behavior_networks.py`, `Independent cross-scale coordination/utils/util.py` |
| RL simulation environment | `Independent cross-scale coordination/sim_env.py` |
| SAC-style agent and replay buffer | `Independent cross-scale coordination/utils/rl_agent.py`, `Independent cross-scale coordination/utils/buffer.py` |
| Real-world state observation and coordinate normalization | `Exp/real_world_env.py`, `Exp/devices.py`, `Exp/utils.py` |
| Physical deployment with learned policy | `Exp/main_exp.py` |
| Human-control deployment | `Exp/main_exp_human.py` |

## License

This repository is released under the MIT License.
