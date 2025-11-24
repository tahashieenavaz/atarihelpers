import numpy
import cv2


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
