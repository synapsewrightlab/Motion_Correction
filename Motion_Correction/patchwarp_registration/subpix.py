import numpy as np

FLOAT_EPSILON = 1.192092896e-07  # smallest float32 step where 1.0+eps != 1.0
FLOAT_VALUE_CHANGE = FLOAT_EPSILON * 10.0


def _fit_parabola(p1, p2, p3):
    """
    Fit y = A*x^2 + B*x + C through 3 points and return the vertex (xv, yv).

    p1, p2, p3: (x, y) tuples. p2 is expected to be the center/peak point.
    Direct port of SubPixFitParabola.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if denom == 0:
        # Degenerate (shouldn't happen since x's come from distinct pixel
        # locations) -- fall back to the center point, same as the C++
        # default-before-computation behavior.
        return x2, y2

    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    b = ((x3**2) * (y1 - y2) + (x2**2) * (y3 - y1) + (x1**2) * (y2 - y3)) / denom
    c = (
        x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3
    ) / denom

    xv = -b / (2 * a)
    yv = c - (b * b) / (4 * a)
    return xv, yv


def min_max_loc_subpix(heatmap: np.ndarray, loc_in, method: int = 0):
    """
    Refine an integer peak location on `heatmap` to sub-pixel precision.

    Parameters
    ----------
    heatmap : np.ndarray, 2D float32
        The correlation surface (e.g. output of cv2.matchTemplate).
    loc_in : (x, y) tuple/list of ints
        Integer peak location, e.g. from cv2.minMaxLoc's maxLoc.
    method : int
        -1 disables refinement and returns loc_in unchanged (as floats).
        Any other value uses the parabola fit (matches the original, which
        only branches on Method == -1; all other values use the same math).

    Returns
    -------
    subpix_loc : (x, y) tuple of floats
    error_val : float
        0.0 if the fit succeeded cleanly, 1.0 if a scan ran out of room or
        a fitted vertex had to be clipped back into range (mirrors the
        original's error/clipping behavior).
    """
    loc_x, loc_y = int(loc_in[0]), int(loc_in[1])

    # default result in case of early return, mirrors the C++ pre-set
    subpix_loc = (float(loc_x), float(loc_y))

    if method == -1:
        return subpix_loc, 0.0

    rows, cols = heatmap.shape[0], heatmap.shape[1]

    min_scan = 1
    max_scan_w = max(min_scan, cols >> 4)
    max_scan_h = max(min_scan, rows >> 4)

    scan_min_x = max(0, loc_x - max_scan_w)
    scan_min_y = max(0, loc_y - max_scan_h)
    scan_max_x = min(cols - 1, loc_x + max_scan_w)
    scan_max_y = min(rows - 1, loc_y + max_scan_h)

    src_start = float(heatmap[loc_y, loc_x])

    center_x, center_y = loc_x, loc_y

    # scan right
    x = center_x
    while True:
        x += 1
        if x > scan_max_x:
            return subpix_loc, 1.0  # ran out of room
        if abs(float(heatmap[center_y, x]) - src_start) > FLOAT_VALUE_CHANGE:
            break
    scan_right = (x, center_y)

    # scan left
    x = center_x
    while True:
        x -= 1
        if x < scan_min_x:
            return subpix_loc, 1.0
        if abs(float(heatmap[center_y, x]) - src_start) > FLOAT_VALUE_CHANGE:
            break
    scan_left = (x, center_y)

    # scan up (+y direction, matching the original's convention)
    y = center_y
    while True:
        y += 1
        if y > scan_max_y:
            return subpix_loc, 1.0
        if abs(float(heatmap[y, center_x]) - src_start) > FLOAT_VALUE_CHANGE:
            break
    scan_up = (center_x, y)

    # scan down (-y direction)
    y = center_y
    while True:
        y -= 1
        if y < scan_min_y:
            return subpix_loc, 1.0
        if abs(float(heatmap[y, center_x]) - src_start) > FLOAT_VALUE_CHANGE:
            break
    scan_down = (center_x, y)

    error_val = 0.0

    # X axis fit: points are (pixel_x, heatmap_value) at left/center/right
    a_pt = (scan_left[0], float(heatmap[scan_left[1], scan_left[0]]))
    b_pt = (center_x, float(heatmap[center_y, center_x]))
    c_pt = (scan_right[0], float(heatmap[scan_right[1], scan_right[0]]))
    result_x, _ = _fit_parabola(a_pt, b_pt, c_pt)

    if result_x < scan_left[0]:
        result_x = float(scan_left[0])
        error_val = 1.0
    if result_x > scan_right[0]:
        result_x = float(scan_right[0])
        error_val = 1.0

    # Y axis fit: x/y swapped in the fit (parabola math always solves along
    # its "x" axis), points are (pixel_y, heatmap_value) at down/center/up
    a_pt = (scan_down[1], float(heatmap[scan_down[1], scan_down[0]]))
    b_pt = (center_y, float(heatmap[center_y, center_x]))
    c_pt = (scan_up[1], float(heatmap[scan_up[1], scan_up[0]]))
    result_y, _ = _fit_parabola(a_pt, b_pt, c_pt)

    if result_y < scan_down[1]:
        result_y = float(scan_down[1])
        error_val = 1.0
    if result_y > scan_up[1]:
        result_y = float(scan_up[1])
        error_val = 1.0

    subpix_loc = (result_x, result_y)
    return subpix_loc, error_val
