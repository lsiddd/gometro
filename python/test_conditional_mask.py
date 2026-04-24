import numpy as np

from constants import (
    ACTION_DIMS,
    MASK_SIZE,
    COND_LINE_OFFSET,
    COND_STATION_OFFSET,
    COND_OPTION_OFFSET,
    MAX_LINE_SLOTS,
    MAX_STATION_SLOTS,
    NUM_OPTIONS,
)
from env import _actions_valid_for_masks


def _empty_mask() -> np.ndarray:
    return np.zeros((1, MASK_SIZE), dtype=bool)


def test_actions_valid_for_conditional_mask_accepts_valid_tuple():
    mask = _empty_mask()
    action = np.array([[7, 2, 0, 0]], dtype=np.int64)
    act, line, station, option = action[0]
    mask[0, act] = True
    mask[0, COND_LINE_OFFSET + act * MAX_LINE_SLOTS + line] = True
    mask[0, COND_STATION_OFFSET + ((act * MAX_LINE_SLOTS + line) * MAX_STATION_SLOTS) + station] = True
    mask[0, COND_OPTION_OFFSET + act * NUM_OPTIONS + option] = True

    assert _actions_valid_for_masks(action, mask).tolist() == [True]


def test_actions_valid_for_conditional_mask_rejects_wrong_parameter():
    mask = _empty_mask()
    action = np.array([[7, 2, 0, 0]], dtype=np.int64)
    act, line, station, option = action[0]
    mask[0, act] = True
    mask[0, COND_LINE_OFFSET + act * MAX_LINE_SLOTS + 1] = True
    mask[0, COND_STATION_OFFSET + ((act * MAX_LINE_SLOTS + line) * MAX_STATION_SLOTS) + station] = True
    mask[0, COND_OPTION_OFFSET + act * NUM_OPTIONS + option] = True

    assert _actions_valid_for_masks(action, mask).tolist() == [False]


def test_actions_valid_for_conditional_mask_rejects_wrong_shape_bounds():
    mask = np.ones((1, MASK_SIZE), dtype=bool)
    bad_option = np.array([[0, 0, 0, ACTION_DIMS[-1]]], dtype=np.int64)

    assert _actions_valid_for_masks(bad_option, mask).tolist() == [False]
