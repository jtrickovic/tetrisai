from numba import njit

from .. import config

HEIGHT_KNEE = config.HEIGHT_KNEE


@njit(cache=True, nogil=True)
def get_board_metrics_numba(grid):
    holes = 0
    agg_height = 0
    max_height = 0
    bumpiness = 0
    col_transitions = 0
    prev_height = -1
    other_wells = 0

    for x in range(10):
        col_height = 0
        block_found = False
        prev_filled = False
        depth_counter = 0
        for y in range(20):
            filled = grid[y, x] != 0
            if filled:
                if not block_found:
                    col_height = 20 - y
                    block_found = True
            else:
                if block_found:
                    holes += 1
            if x < 9 and y > 0 and prev_filled != filled:
                col_transitions += 1
            prev_filled = filled
            if x < 9:
                if not filled:
                    left_filled = (x == 0) or (grid[y, x - 1] != 0)
                    right_filled = grid[y, x + 1] != 0
                    if left_filled and right_filled:
                        depth_counter += 1
                        other_wells += depth_counter
                    else:
                        depth_counter = 0
                else:
                    depth_counter = 0
        if x < 9 and not prev_filled:
            col_transitions += 1
        agg_height += col_height
        if col_height > max_height:
            max_height = col_height
        if prev_height != -1 and x < 9:
            bumpiness += abs(col_height - prev_height)
        prev_height = col_height

    row_transitions = 0
    for y in range(20):
        prev = True
        for x in range(9):
            filled = grid[y, x] != 0
            if filled != prev:
                row_transitions += 1
            prev = filled

    col9_depth = 0
    for y in range(20):
        if grid[y, 9] == 0 and grid[y, 8] != 0:
            col9_depth += 1
    if col9_depth > 8:
        col9_depth = 8

    height_excess = max(0, max_height - HEIGHT_KNEE)

    deep_well = max(0, col9_depth - 4)

    return (agg_height, holes, bumpiness, row_transitions, col_transitions,
            col9_depth, other_wells, height_excess, deep_well)


@njit(cache=True, nogil=True, inline='always')
def _line_points(n, penalty):
    if n >= 4:
        return 8
    if n == 3:
        return 5 - penalty
    if n == 2:
        return 3 - penalty
    if n == 1:
        return 1 - penalty
    return 0


@njit(cache=True, nogil=True, inline='always')
def _state_score(grid, lp, landing_height, well_foul, w):
    m = get_board_metrics_numba(grid)
    return (w[0] * lp +
            w[1] * m[0] +
            w[2] * m[1] +
            w[3] * m[2] +
            w[4] * m[3] +
            w[5] * m[4] +
            w[6] * m[5] +
            w[7] * m[6] +
            w[8] * landing_height +
            w[9] * well_foul +
            w[10] * m[7] +
            w[11] * m[8])
