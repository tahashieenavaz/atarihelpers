import numpy
import cv2
import gymnasium
import ale_py


def make_episode_trigger(record_every: int):
    def episode_trigger(episode):
        return episode == 0 or episode % record_every == 0

    return episode_trigger


def make_environment(
    name: str,
    path: str = "./videos",
    record: bool = False,
    record_every: int = 50,
    **kwargs,
):
    gymnasium.register_envs(ale_py)
    env = gymnasium.make(name, render_mode="rgb_array", **kwargs)
    if record:
        env = gymnasium.wrappers.RecordVideo(
            env, path, make_episode_trigger(record_every)
        )
    return env


# TODO: we can add curated crops for every single environment
def process_state(
    state: numpy.array,
    image_size: int = 84,
    grayscale: bool = True,
    resize: bool = True,
) -> numpy.array:
    if grayscale:
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)

    if resize:
        state = cv2.resize(state, (image_size, image_size))

    return state
