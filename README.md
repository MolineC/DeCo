# DeCo

Source Code of "Learning to Handle Constrained Routing Problems From a Decoupling Perspective"

  ## Quick Start

  This project implements DeCo-based models for TSPDL and TSPTW. The following minimal steps get you running quickly.

  Requirements:

  - Python 3.8+
  - PyTorch, numpy, matplotlib, pytz

  Install basic dependencies:

  ```bash
  pip install torch numpy matplotlib pytz
  ```

  Run training (example for TSPDL):

  ```bash
  cd DeCo/TSPDL
  # edit train.py to set data/model/config if needed
  python train.py
  ```

  Run testing (example for TSPDL):

  ```bash
  cd DeCo/TSPDL
  # set model_load_path and model_load_epoch inside test.py
  python test.py
  ```

  Notes:

  - Ensure `env_params` and `trainer_params` in `train.py` (or `tester_params` in `test.py`) point to existing data and pretrained model paths.

  That's it — edit the config in `train.py`/`test.py`, then run the appropriate script.

