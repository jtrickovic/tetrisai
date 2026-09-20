import os

ENGINE_MODE = 'hold_aware'
CROSSOVER_MODE = 'blx'
BLX_ALPHA = 0.5

HEIGHT_KNEE = 15
LOOKAHEAD_PLY = 4
BEAM_WIDTH = 5
NON_TETRIS_PENALTY = 1.5
CHAIN_BONUS = 4.0

SEED_BASE = 10_000
TIER1_SEEDS = 1
EVAL_MAX_PIECES = 400_000
STAGNATION_LIMIT = 4
IMPROVE_THRESHOLD = 1.02
DISPLAY_MAX_PIECES = 2_000
POP_SIZE = 24
GENERATIONS = 40
GA_SEED = 42
EVAL_WORKERS = None

TAG = 'v33'

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_PKG_ROOT, 'out')


def out_path(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)
