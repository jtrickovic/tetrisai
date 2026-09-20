import numpy as np
from numba import njit

from ..tetris_base.tables import ALL_SHAPE_OFFSETS, KICKS_JLSTZ_NUMBA, KICKS_I_NUMBA


@njit(cache=True, nogil=True, inline='always')
def fast_check_collision_numba(grid, offsets, x, y):
    for i in range(4):
        bx = x + offsets[i, 0]
        by = y + offsets[i, 1]
        if bx < 0 or bx >= 10 or by >= 20:
            return True
        if by >= 0 and grid[by, bx] != 0:
            return True
    return False


@njit(cache=True, nogil=True)
def get_reachable_locks_numba(grid, piece_id, start_x, start_y, start_rot):
    Y_OFFSET = 2

    queue = np.zeros((2000, 3), dtype=np.int32)
    head = 0
    tail = 0

    visited = np.zeros((10, 26, 4), dtype=np.bool_)
    locked_visited = np.zeros((10, 26, 4), dtype=np.bool_)

    valid_locks = np.zeros((2000, 3), dtype=np.int32)
    lock_count = 0

    start_offsets = ALL_SHAPE_OFFSETS[piece_id, start_rot]
    if fast_check_collision_numba(grid, start_offsets, start_x, start_y):
        return np.zeros((0, 3), dtype=np.int32)

    queue[tail, 0] = start_x
    queue[tail, 1] = start_y
    queue[tail, 2] = start_rot
    tail += 1
    visited[start_x, start_y + Y_OFFSET, start_rot] = True

    while head < tail:
        cx = queue[head, 0]
        cy = queue[head, 1]
        crot = queue[head, 2]
        head += 1
        offsets = ALL_SHAPE_OFFSETS[piece_id, crot]

        ny = cy + 1
        if not fast_check_collision_numba(grid, offsets, cx, ny):
            if not visited[cx, ny + Y_OFFSET, crot]:
                visited[cx, ny + Y_OFFSET, crot] = True
                queue[tail, 0] = cx
                queue[tail, 1] = ny
                queue[tail, 2] = crot
                tail += 1
        else:
            all_inside = True
            for j in range(4):
                if cy + offsets[j, 1] < 0:
                    all_inside = False
                    break
            if all_inside and not locked_visited[cx, cy + Y_OFFSET, crot]:
                locked_visited[cx, cy + Y_OFFSET, crot] = True
                valid_locks[lock_count, 0] = cx
                valid_locks[lock_count, 1] = cy
                valid_locks[lock_count, 2] = crot
                lock_count += 1

        nx = cx - 1
        if nx >= 0 and not visited[nx, cy + Y_OFFSET, crot]:
            if not fast_check_collision_numba(grid, offsets, nx, cy):
                visited[nx, cy + Y_OFFSET, crot] = True
                queue[tail, 0] = nx
                queue[tail, 1] = cy
                queue[tail, 2] = crot
                tail += 1

        nx = cx + 1
        if nx < 10 and not visited[nx, cy + Y_OFFSET, crot]:
            if not fast_check_collision_numba(grid, offsets, nx, cy):
                visited[nx, cy + Y_OFFSET, crot] = True
                queue[tail, 0] = nx
                queue[tail, 1] = cy
                queue[tail, 2] = crot
                tail += 1

        if piece_id != 3:
            for d_rot in (1, 3):
                new_rot = (crot + d_rot) % 4
                kicks = KICKS_I_NUMBA[crot, new_rot] if piece_id == 0 else KICKS_JLSTZ_NUMBA[crot, new_rot]
                rot_offsets = ALL_SHAPE_OFFSETS[piece_id, new_rot]
                for k in range(5):
                    rx = cx + kicks[k, 0]
                    ry = cy + kicks[k, 1]
                    if 0 <= rx < 10 and -2 <= ry < 22:
                        if not visited[rx, ry + Y_OFFSET, new_rot]:
                            if not fast_check_collision_numba(grid, rot_offsets, rx, ry):
                                visited[rx, ry + Y_OFFSET, new_rot] = True
                                queue[tail, 0] = rx
                                queue[tail, 1] = ry
                                queue[tail, 2] = new_rot
                                tail += 1
                                break
                        else:
                            break

    return valid_locks[:lock_count]
