import numpy as np

from .shapes import SHAPES_SRS, KICK_TABLE_JLSTZ, KICK_TABLE_I

SHAPE_OFFSETS = {}
for name, rots in SHAPES_SRS.items():
    SHAPE_OFFSETS[name] = []
    for rot_grid in rots:
        offsets = []
        for row in range(rot_grid.shape[0]):
            for col in range(rot_grid.shape[1]):
                if rot_grid[row, col] != 0:
                    offsets.append((col, row))
        SHAPE_OFFSETS[name].append(offsets)

PIECE_TO_ID = {'I': 0, 'J': 1, 'L': 2, 'O': 3, 'S': 4, 'T': 5, 'Z': 6}
ID_TO_PIECE = {v: k for k, v in PIECE_TO_ID.items()}

ALL_SHAPE_OFFSETS = np.zeros((7, 4, 4, 2), dtype=np.int32)
for name, rots in SHAPE_OFFSETS.items():
    arr = np.zeros((4, 4, 2), dtype=np.int32)
    for r in range(4):
        for i, (dx, dy) in enumerate(rots[r]):
            arr[r, i, 0] = dx
            arr[r, i, 1] = dy
    ALL_SHAPE_OFFSETS[PIECE_TO_ID[name]] = arr

KICKS_JLSTZ_NUMBA = np.zeros((4, 4, 5, 2), dtype=np.int32)
KICKS_I_NUMBA = np.zeros((4, 4, 5, 2), dtype=np.int32)
for (r_from, r_to), offsets in KICK_TABLE_JLSTZ.items():
    for i, (dx, dy) in enumerate(offsets):
        KICKS_JLSTZ_NUMBA[r_from, r_to, i, 0] = dx
        KICKS_JLSTZ_NUMBA[r_from, r_to, i, 1] = dy
for (r_from, r_to), offsets in KICK_TABLE_I.items():
    for i, (dx, dy) in enumerate(offsets):
        KICKS_I_NUMBA[r_from, r_to, i, 0] = dx
        KICKS_I_NUMBA[r_from, r_to, i, 1] = dy
