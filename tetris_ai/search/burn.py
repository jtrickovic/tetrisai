from numba import njit

from ..tetris_base.tables import ALL_SHAPE_OFFSETS


@njit(cache=True, nogil=True)
def burn_and_clear_numba(grid, piece_id, lx, ly, lrot):
    new_grid = grid.copy()
    color_id = piece_id + 1

    offsets = ALL_SHAPE_OFFSETS[piece_id, lrot]
    landing_sum = 0
    for j in range(4):
        by = ly + offsets[j, 1]
        bx = lx + offsets[j, 0]
        if by >= 0:
            new_grid[by, bx] = color_id
            landing_sum += 20 - by
    landing_height = landing_sum / 4.0

    lines_cleared = 0
    write_y = 19
    for read_y in range(19, -1, -1):
        is_full = True
        for x in range(10):
            if new_grid[read_y, x] == 0:
                is_full = False
                break
        if is_full:
            lines_cleared += 1
        else:
            if write_y != read_y:
                for x in range(10):
                    new_grid[write_y, x] = new_grid[read_y, x]
            write_y -= 1

    while write_y >= 0:
        for x in range(10):
            new_grid[write_y, x] = 0
        write_y -= 1

    well_foul = 0
    for y in range(20):
        c = new_grid[y, 9]
        if c != 0 and c != 1:
            well_foul += 1

    return new_grid, lines_cleared, landing_height, well_foul
