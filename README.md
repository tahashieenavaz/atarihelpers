# Atari Helpers

Lightweight utilities for Atari reinforcement learning: launch environments fast, capture episodic video if you want, and preprocess frames so they are ready for your agent. All signal, no cargo cult. 🎯

- 🎮 `make_environment`: spin up Gymnasium Atari envs with optional video recording
- 🖼️ `process_state`: grayscale + resize frames for downstream stacks
- 🧰 Zero-fluff dependency set (Gymnasium, ALE-Py, NumPy, OpenCV)

## Installation

```bash
pip install atarihelpers
```

## Quickstart

```python
from atarihelpers import make_environment, process_state

env = make_environment(
    "ALE/Pong-v5",
    record=True,       # 🎥 save videos to ./videos
    record_every=25,   # capture every 25th episode
)

state, _ = env.reset()
processed = process_state(
    state,
    image_size=84,     # target square size
    grayscale=True,    # convert to single channel
    resize=True,       # keep original resolution if False
)
```

Note: inputs should be NumPy arrays shaped `(H, W, C)` in BGR order (OpenCV style). Returns the processed NumPy array ready for stacking or feeding to your model.
