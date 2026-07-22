# Code Overview

This repository contains a concise implementation of DPSA.

DPSA samples a fixed number of DINOv3-guided operator paths per iteration. Some paths start from OPS-style neighborhood points, while the rest start from the current adversarial image. Gradients from the main view and sampled paths are combined by Gradient Distribution Synthesis, which balances stable consensus directions and residual diversity.

The main implementation is in `transferattack/dpsa.py`; the image operator pool is in `transferattack/operator_lib.py`; `main.py` provides generation and evaluation commands.
