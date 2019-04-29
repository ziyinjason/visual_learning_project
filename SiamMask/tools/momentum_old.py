import scipy
import numpy as np


def momentum(history_speed, current_speed, direction_threshold,
             magnitude_threshold, lr):
    '''
        Update the object speed using momentum
        Input:
            history_speed: np.array(x,y)
            current_speed: np.array(x,y)
        Output:
            new history speed: np.array(x,y)
    '''
    direction_distance = scipy.spatial.distacne.cosine(history_speed,
                                                       current_speed)
    magnitude_distance = np.linalg.norm(history_speed - current_speed)
    if direction_distance > direction_threshold or magnitude_distance > magnitude_threshold:
        current_speed = history_speed
    else:
        current_speed = history_speed * lr + current_speed * (1 - lr)

    return current_speed