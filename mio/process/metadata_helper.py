"""
Helper functions for processing metadata in recordings.
"""

from typing import List, Optional

import numpy as np


def frame_timestamp_match_ratio(timestamp_lists: List[List[int]]) -> float:
    """
    Compute the fraction of timestamps that are same within each ReconstructedBufferList.
    The denominator is the maximum number of timestamps across the provided lists.

    Parameters:
        timestamp_lists: List[List[int]]
            A list of lists of timestamps to compare.

    Returns:
        float: The fraction of timestamps that are same within each ReconstructedBufferList.

    """
    # Edge cases
    if timestamp_lists is None or len(timestamp_lists) == 0:
        return 0.0

    base_timestamp_list = timestamp_lists[0]
    match_count = 0
    for timestamp in base_timestamp_list:
        if all(timestamp in timestamp_list for timestamp_list in timestamp_lists):
            match_count += 1

    maximum_timestamp_length = max(len(timestamp_list) for timestamp_list in timestamp_lists)
    return match_count / maximum_timestamp_length


def make_combined_list(list_of_lists: List[List]) -> List:
    """
    Combine multiple lists into a single list with unique elements, preserving order.

    Parameters:
        list_of_lists: List[List]
            A list of lists to combine.

    Returns:
        List: A combined list with unique elements.
    """
    combined_list = []
    for lst in list_of_lists:
        for item in lst:
            if item not in combined_list:
                combined_list.append(item)
    return combined_list


def linearity_mse(sequence: List[int], index: int, slope: Optional[float] = None) -> float:
    """
    Compute how well `sequence[index]` fits a global line.
    If slope is provided, only the intercept is fitted.
    Otherwise, slope and intercept are fitted from all other points.

    Parameters
    ----------
    sequence : list of int
        The numeric sequence.
    index : int
        Index to evaluate.
    slope : float, optional
        Known slope. If None, slope is estimated.

    Returns
    -------
    float
        Squared residual of the index under the fitted line,
        or -1.0 if not enough data.
    """
    n = len(sequence)
    if n < 3 or not (0 <= index < n):
        return -1.0

    x_all = np.arange(n, dtype=float)
    y_all = np.asarray(sequence, dtype=float)

    # Exclude the target index for fitting
    mask = np.ones(n, dtype=bool)
    mask[index] = False
    x = x_all[mask]
    y = y_all[mask]

    if slope is not None:
        m = slope
        c = np.mean(y - m * x)
    else:
        m, c = np.polyfit(x, y, 1)  # slope, intercept

    expected = c + m * index
    residual = y_all[index] - expected
    return residual**2
