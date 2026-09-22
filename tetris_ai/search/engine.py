import random
import threading

import numpy as np

from .. import config
from ..tetris_base.board import Board, Bag, Next, Hold
from ..tetris_base.tables import PIECE_TO_ID
from .bfs import get_reachable_locks_numba
from .beam import evaluate_beam_numba, evaluate_beam_hold_numba

GUIDELINE_LINE_POINTS = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
BAR = {'r20': -1.0}
BAR_LOCK = threading.Lock()


def play_game(genome, max_pieces=1000, seed=0):
    if config.ENGINE_MODE == 'hold_aware':
        return _play_holdaware(genome, max_pieces=max_pieces, seed=seed)
    return _play_holdblind(genome, max_pieces=max_pieces, seed=seed)


def guideline_score(lines_cleared, combo, level):
    mult = level + 1
    base = GUIDELINE_LINE_POINTS.get(lines_cleared, 0)
    combo_bonus = 50 * combo if combo > 0 else 0
    return (base + combo_bonus) * mult


def _play_holdaware(genome, max_pieces=1000, seed=0):
    board = Board(10, 20)
    bag = Bag(random.Random(seed))
    next_queue = Next(bag, preview_count=config.LOOKAHEAD_PLY)
    hold_sys = Hold()

    board.current_piece = next_queue.pop_next()
    pieces_placed = 0

    score_total = 0
    combo = -1
    max_combo = 0
    tetris_count = 0
    tets20 = None
    b2b_count = 0
    b2b_active = False
    double_tetris_count = 0
    last_clear_lines = 0
    i_hold_turns = 0

    while pieces_placed < max_pieces:
        p1_id = PIECE_TO_ID[board.current_piece.name]
        upcoming = [PIECE_TO_ID[next_queue.queue[k].name]
                    for k in range(config.LOOKAHEAD_PLY)]
        ph = PIECE_TO_ID[hold_sys.piece.name] if hold_sys.piece is not None else -1

        best_score_curr = -float('inf')
        best_grid_curr = None
        best_lines_curr = 0
        locks_curr = get_reachable_locks_numba(
            board.grid, p1_id,
            board.current_piece.x, board.current_piece.y, board.current_piece.rotation_index)
        if len(locks_curr) > 0:
            curr_pieces = np.array([p1_id] + upcoming[:config.LOOKAHEAD_PLY - 1], dtype=np.int32)
            best_score_curr, best_grid_curr, best_lines_curr, *_ = evaluate_beam_hold_numba(
                board.grid, locks_curr, curr_pieces, genome.weights, config.NON_TETRIS_PENALTY,
                last_clear_lines, ph,
                config.BEAM_WIDTH, config.CHAIN_BONUS, config.HEIGHT_KNEE)

        best_score_hold = -float('inf')
        best_grid_hold = None
        best_lines_hold = 0
        if hold_sys.can_swap:
            if hold_sys.piece is not None:
                h_id = PIECE_TO_ID[hold_sys.piece.name]
                hold_pieces = np.array([h_id] + upcoming[:config.LOOKAHEAD_PLY - 1], dtype=np.int32)
            else:
                h_id = upcoming[0]
                hold_pieces = np.array(upcoming[:config.LOOKAHEAD_PLY], dtype=np.int32)
            locks_hold = get_reachable_locks_numba(board.grid, h_id, 3, 0, 0)
            if len(locks_hold) > 0:
                best_score_hold, best_grid_hold, best_lines_hold, *_ = evaluate_beam_hold_numba(
                    board.grid, locks_hold, hold_pieces, genome.weights, config.NON_TETRIS_PENALTY,
                    last_clear_lines, p1_id,
                    config.BEAM_WIDTH, config.CHAIN_BONUS, config.HEIGHT_KNEE)

        if best_score_curr > best_score_hold and best_score_curr != -float('inf'):
            best_grid = best_grid_curr
            best_lines_cleared = best_lines_curr
            used_hold = False
        elif best_score_hold != -float('inf'):
            best_grid = best_grid_hold
            best_lines_cleared = best_lines_hold
            used_hold = True
        else:
            break

        if used_hold:
            swapped_in = hold_sys.swap(board.current_piece)
            if swapped_in is None:
                next_queue.pop_next()

        board.grid = best_grid
        if best_lines_cleared > 0:
            board.lines_cleared += best_lines_cleared

        level = board.lines_cleared // 10
        combo = combo + 1 if best_lines_cleared > 0 else -1
        if combo > max_combo:
            max_combo = combo
        if best_lines_cleared == 4:
            tetris_count += 1
            if last_clear_lines == 4:
                double_tetris_count += 1
            if b2b_active:
                b2b_count += 1
            b2b_active = True
        elif best_lines_cleared > 0:
            b2b_active = False
        last_clear_lines = best_lines_cleared
        if hold_sys.piece is not None and hold_sys.piece.name == 'I':
            i_hold_turns += 1
        score_total += guideline_score(best_lines_cleared, combo, level)

        pieces_placed += 1
        if tets20 is None and pieces_placed >= 20_000:
            tets20 = tetris_count
            with BAR_LOCK:
                dominated = BAR['r20'] > 100.0 * tetris_count / 20_000.0
            if dominated:
                break
        board.current_piece = next_queue.pop_next()
        hold_sys.reset_lock()

        if not board.is_valid_position(board.current_piece):
            break

    survived = pieces_placed >= max_pieces
    if survived and tets20 is not None:
        with BAR_LOCK:
            BAR['r20'] = max(BAR['r20'], 100.0 * tets20 / 20_000.0)
    return {
        'survived': survived,
        'pieces': pieces_placed,
        'lines': board.lines_cleared,
        'score': score_total,
        'tetrises': tetris_count,
        'b2b': b2b_count,
        'double_tetris': double_tetris_count,
        'i_hold_turns': i_hold_turns,
        'max_combo': max_combo,
    }


def _play_holdblind(genome, max_pieces=1000, seed=0):
    board = Board(10, 20)
    bag = Bag(random.Random(seed))
    next_queue = Next(bag, preview_count=config.LOOKAHEAD_PLY)
    hold_sys = Hold()

    board.current_piece = next_queue.pop_next()
    pieces_placed = 0

    score_total = 0
    combo = -1
    max_combo = 0
    tetris_count = 0
    tets20 = None
    b2b_count = 0
    b2b_active = False
    double_tetris_count = 0
    last_clear_lines = 0
    i_hold_turns = 0

    while pieces_placed < max_pieces:
        p1_id = PIECE_TO_ID[board.current_piece.name]
        upcoming = [PIECE_TO_ID[next_queue.queue[k].name]
                    for k in range(config.LOOKAHEAD_PLY)]

        best_score_curr = -float('inf')
        best_grid_curr = None
        best_lines_curr = 0
        locks_curr = get_reachable_locks_numba(
            board.grid, p1_id,
            board.current_piece.x, board.current_piece.y, board.current_piece.rotation_index)
        if len(locks_curr) > 0:
            curr_pieces = np.array([p1_id] + upcoming[:config.LOOKAHEAD_PLY - 1], dtype=np.int32)
            best_score_curr, best_grid_curr, best_lines_curr, *_ = evaluate_beam_numba(
                board.grid, locks_curr, curr_pieces, genome.weights, config.NON_TETRIS_PENALTY,
                last_clear_lines,
                config.BEAM_WIDTH, config.CHAIN_BONUS, config.HEIGHT_KNEE)

        best_score_hold = -float('inf')
        best_grid_hold = None
        best_lines_hold = 0
        if hold_sys.can_swap:
            if hold_sys.piece is not None:
                h_id = PIECE_TO_ID[hold_sys.piece.name]
                hold_pieces = np.array([h_id] + upcoming[:config.LOOKAHEAD_PLY - 1], dtype=np.int32)
            else:
                h_id = upcoming[0]
                hold_pieces = np.array(upcoming[:config.LOOKAHEAD_PLY], dtype=np.int32)
            locks_hold = get_reachable_locks_numba(board.grid, h_id, 3, 0, 0)
            if len(locks_hold) > 0:
                best_score_hold, best_grid_hold, best_lines_hold, *_ = evaluate_beam_numba(
                    board.grid, locks_hold, hold_pieces, genome.weights, config.NON_TETRIS_PENALTY,
                    last_clear_lines,
                    config.BEAM_WIDTH, config.CHAIN_BONUS, config.HEIGHT_KNEE)

        if best_score_curr > best_score_hold and best_score_curr != -float('inf'):
            best_grid = best_grid_curr
            best_lines_cleared = best_lines_curr
            used_hold = False
        elif best_score_hold != -float('inf'):
            best_grid = best_grid_hold
            best_lines_cleared = best_lines_hold
            used_hold = True
        else:
            break

        if used_hold:
            swapped_in = hold_sys.swap(board.current_piece)
            if swapped_in is None:
                next_queue.pop_next()

        board.grid = best_grid
        if best_lines_cleared > 0:
            board.lines_cleared += best_lines_cleared

        level = board.lines_cleared // 10
        combo = combo + 1 if best_lines_cleared > 0 else -1
        if combo > max_combo:
            max_combo = combo
        if best_lines_cleared == 4:
            tetris_count += 1
            if last_clear_lines == 4:
                double_tetris_count += 1
            if b2b_active:
                b2b_count += 1
            b2b_active = True
        elif best_lines_cleared > 0:
            b2b_active = False
        last_clear_lines = best_lines_cleared
        if hold_sys.piece is not None and hold_sys.piece.name == 'I':
            i_hold_turns += 1
        score_total += guideline_score(best_lines_cleared, combo, level)

        pieces_placed += 1
        if tets20 is None and pieces_placed >= 20_000:
            tets20 = tetris_count
            with BAR_LOCK:
                dominated = BAR['r20'] > 100.0 * tetris_count / 20_000.0
            if dominated:
                break
        board.current_piece = next_queue.pop_next()
        hold_sys.reset_lock()

        if not board.is_valid_position(board.current_piece):
            break

    survived = pieces_placed >= max_pieces
    if survived and tets20 is not None:
        with BAR_LOCK:
            BAR['r20'] = max(BAR['r20'], 100.0 * tets20 / 20_000.0)
    return {
        'survived': survived,
        'pieces': pieces_placed,
        'lines': board.lines_cleared,
        'score': score_total,
        'tetrises': tetris_count,
        'b2b': b2b_count,
        'double_tetris': double_tetris_count,
        'i_hold_turns': i_hold_turns,
        'max_combo': max_combo,
    }
