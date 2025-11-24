# Atari Helpers

A tiny toolkit for tidying up Atari observations before they hit your RL agent. Keep installs light, configs simple, and pre-processing consistent.

- 🚀 Quick preprocessing for Atari frames (grayscale + resize)
- 🧰 Single helper focused on DRL needs; no extra baggage
- ✅ Tested on Python 3.9+ with NumPy + OpenCV

## Installation

```bash
pip install atarihelpers
```

## Usage

```python
import gymnasium as gym
from atarihelpers import process_state

env = gym.make("ALE/Pong-v5")
state, _ = env.reset()

processed = process_state(
    state,
    image_size=84,   # target square size
    grayscale=True,  # convert to single channel
    resize=True,     # skip if you want original resolution
)
```

📝 Input should be a NumPy array shaped `(H, W, C)` in BGR order (OpenCV style). The helper returns the processed NumPy array ready for stacking or feeding to your model.
