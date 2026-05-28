# ACMC
**Automatic Cross-scale Modular Coordination Frameword** — a research codebase for coordinating millimeter-scale modular robots (millicore), nanoscale nanounits, and reconfiguration/release behaviors under a shared rotating magnetic field.
The pipeline has three stages: learn elementary **behavior models** (digital twins), train **cross-scale coordination** in simulation, then run **real-world experiments** on hardware.
## Repository layout
| Directory | Role |
|-----------|------|
| `Elementar_behavior_modeling/` | Active learning and offline training of motion / release-rate predictors from coil parameters and vision. |
| `Independent cross-scale coordination/` | Gym-style simulated pipe-navigation environment driven by the digital twins; SAC-style RL training (`main_RL.py`). |
| `Exp/` | Deployment of a trained policy on the physical setup (camera, coil, optional human-in-the-loop variant). |
Bundled vendor SDKs (not installed via pip): **Sensory 826** DAQ (`S826.py`, `sdk_826_*`) and **Daheng Imaging** camera API (`gxipy/`).
## Typical workflow
1. **Behavior modeling** — Run active-learning scripts under `Elementar_behavior_modeling/code/{milli,nano,reconfig}/`, or offline training via `code/offline/train_offline.py` when labeled Excel data is available.
2. **Simulated RL** — Place trained twin weights under `Independent cross-scale coordination/SL_model/AL_all/` (`milli_velocity.pth`, `nano_velocity.pth`, `release_grayscale.pth`), then train with `main_RL.py`.
3. **Hardware experiments** — Copy RL checkpoints into `Exp/agent/` and run `main_exp.py` (or `main_exp_human.py`). Real-world perception uses a YOLO weights file (`best.pt`) in the `Exp/` working directory.
> **Note:** Pretrained weights, training datasets (`.xlsx`), and `SL_model/` checkpoints are not shipped with this repository and must be produced or supplied locally.
## Dependencies
### Python (install via pip)
Core stack used across modules:
- `torch`
- `numpy`, `scipy`
- `opencv-python`
- `gym` (classic OpenAI Gym API)
- `matplotlib`
Additional packages by module:
- **Behavior modeling / experiments:** `pandas`, `openpyxl`
- **Real-world experiments (`Exp/`):** `ultralytics` (YOLO), `pandas`
A CUDA-capable PyTorch build is recommended when GPU acceleration is available.
### Hardware & drivers
- **Magnetic field:** Sensory 826 DAQ board (drivers/SDK included under `sdk_826_win_*` / `sdk_826_linux_*`).
- **Imaging:** Daheng GX industrial camera (controlled through bundled `gxipy`).
- **Detection (real-world runs):** Custom YOLO model weights (`best.pt`).
Install the S826 drivers for your OS before running any script that imports `S826`.
## License
MIT — see [LICENSE](LICENSE).
