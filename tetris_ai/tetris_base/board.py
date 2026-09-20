import random

import numpy as np

from .shapes import SHAPES


class Piece:
    def __init__(self, shape_name):
        self.name = shape_name
        self.shape = SHAPES[shape_name].copy()
        self.x = 3
        self.y = 0
        self.rotation_index = 0


class Board:
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.int8)
        self.current_piece = None
        self.lines_cleared = 0

    def is_valid_position(self, piece):
        for row in range(piece.shape.shape[0]):
            for col in range(piece.shape.shape[1]):
                if piece.shape[row, col] != 0:
                    bx = piece.x + col
                    by = piece.y + row
                    if bx < 0 or bx >= self.width or by >= self.height:
                        return False
                    if by >= 0 and self.grid[by, bx] != 0:
                        return False
        return True


class I_Piece(Piece): __init__ = lambda self: super().__init__('I')
class J_Piece(Piece): __init__ = lambda self: super().__init__('J')
class L_Piece(Piece): __init__ = lambda self: super().__init__('L')
class O_Piece(Piece): __init__ = lambda self: super().__init__('O')
class S_Piece(Piece): __init__ = lambda self: super().__init__('S')
class T_Piece(Piece): __init__ = lambda self: super().__init__('T')
class Z_Piece(Piece): __init__ = lambda self: super().__init__('Z')

PIECE_CLASSES = [I_Piece, J_Piece, L_Piece, O_Piece, S_Piece, T_Piece, Z_Piece]


class Bag:
    def __init__(self, rng=None):
        self.rng = rng if rng is not None else random
        self.pieces = []
        self._fill()

    def _fill(self):
        self.pieces = [cls() for cls in PIECE_CLASSES]
        self.rng.shuffle(self.pieces)

    def draw(self):
        if not self.pieces:
            self._fill()
        return self.pieces.pop()


class Next:
    def __init__(self, bag, preview_count=1):
        self.bag = bag
        self.preview_count = preview_count
        self.queue = []
        self._replenish()

    def _replenish(self):
        while len(self.queue) < self.preview_count:
            self.queue.append(self.bag.draw())

    def pop_next(self):
        piece = self.queue.pop(0)
        self._replenish()
        return piece


class Hold:
    def __init__(self):
        self.piece = None
        self.can_swap = True

    def swap(self, current_piece):
        if not self.can_swap:
            return current_piece
        self.can_swap = False
        if self.piece is None:
            self.piece = current_piece
            return None
        temp = self.piece
        self.piece = current_piece
        temp.x, temp.y = 3, 0
        return temp

    def reset_lock(self):
        self.can_swap = True
