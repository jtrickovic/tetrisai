import numpy as np
from numba import njit

from .metrics import _state_score, _line_points
from .bfs import get_reachable_locks_numba
from .burn import burn_and_clear_numba


@njit(cache=True, nogil=True)
def _beam_keep_top(cand_score, cand_n, B,
                   cand_grid, cand_meta, cand_acc, cand_prev,
                   beam_grid, beam_meta, beam_acc, beam_prev):
    order = np.argsort(cand_score[:cand_n])
    take = B if B < cand_n else cand_n
    for k in range(take):
        idx = order[cand_n - 1 - k]
        beam_grid[k] = cand_grid[idx]
        beam_meta[k, 0] = cand_meta[idx, 0]
        beam_meta[k, 1] = cand_meta[idx, 1]
        beam_meta[k, 2] = cand_meta[idx, 2]
        beam_meta[k, 3] = cand_meta[idx, 3]
        beam_acc[k] = cand_acc[idx]
        beam_prev[k] = cand_prev[idx]
    return take


@njit(cache=True, nogil=True)
def evaluate_beam_numba(grid, p1_locks, piece_ids, weights, penalty,
                        incoming_prev_clear, beam_width, chain_bonus, knee):
    B = beam_width
    MAX_LOCKS = 64
    MAX_CAND = B * MAX_LOCKS
    w = weights

    beam_grid = np.zeros((B, 20, 10), dtype=np.int8)
    beam_meta = np.zeros((B, 4), dtype=np.int32)
    beam_acc = np.zeros(B, dtype=np.float64)
    beam_prev = np.zeros(B, dtype=np.int32)

    cand_grid = np.zeros((MAX_CAND, 20, 10), dtype=np.int8)
    cand_meta = np.zeros((MAX_CAND, 4), dtype=np.int32)
    cand_acc = np.zeros(MAX_CAND, dtype=np.float64)
    cand_score = np.zeros(MAX_CAND, dtype=np.float64)
    cand_prev = np.zeros(MAX_CAND, dtype=np.int32)

    best_score = -1e9
    best_lines = 0
    best_lx = np.int32(0)
    best_ly = np.int32(0)
    best_lrot = np.int32(0)

    cand_n = 0
    p1_id = piece_ids[0]
    for i in range(min(len(p1_locks), MAX_LOCKS)):
        lx = p1_locks[i, 0]
        ly = p1_locks[i, 1]
        lrot = p1_locks[i, 2]
        g, lines, lh, wf = burn_and_clear_numba(grid, p1_id, lx, ly, lrot)
        lp = _line_points(lines, penalty)
        if lines == 4 and incoming_prev_clear == 4:
            lp += chain_bonus
        sc = _state_score(g, lp, lh, wf, w, knee)
        c = cand_n
        cand_grid[c] = g
        cand_meta[c, 0] = lines
        cand_meta[c, 1] = lx
        cand_meta[c, 2] = ly
        cand_meta[c, 3] = lrot
        cand_acc[c] = lp
        cand_prev[c] = lines
        cand_score[c] = sc
        cand_n += 1
        if sc > best_score:
            best_score = sc
            best_lines = lines
            best_lx = lx
            best_ly = ly
            best_lrot = lrot

    if cand_n == 0:
        return best_score, np.zeros((20, 10), dtype=np.int8), best_lines, best_lx, best_ly, best_lrot

    beam_n = _beam_keep_top(cand_score, cand_n, B,
                            cand_grid, cand_meta, cand_acc, cand_prev,
                            beam_grid, beam_meta, beam_acc, beam_prev)

    for ply in range(1, len(piece_ids)):
        pid = piece_ids[ply]
        cand_n = 0
        for s in range(beam_n):
            locks = get_reachable_locks_numba(beam_grid[s], pid, 3, 0, 0)
            for j in range(min(len(locks), MAX_LOCKS)):
                lx = locks[j, 0]
                ly = locks[j, 1]
                lrot = locks[j, 2]
                g, lines, lh, wf = burn_and_clear_numba(beam_grid[s], pid, lx, ly, lrot)
                lp = beam_acc[s] + _line_points(lines, penalty)
                if lines == 4 and beam_prev[s] == 4:
                    lp += chain_bonus
                sc = _state_score(g, lp, lh, wf, w, knee)
                c = cand_n
                cand_grid[c] = g
                cand_meta[c, 0] = beam_meta[s, 0]
                cand_meta[c, 1] = beam_meta[s, 1]
                cand_meta[c, 2] = beam_meta[s, 2]
                cand_meta[c, 3] = beam_meta[s, 3]
                cand_acc[c] = lp
                cand_prev[c] = lines
                cand_score[c] = sc
                cand_n += 1
                if sc > best_score:
                    best_score = sc
                    best_lines = beam_meta[s, 0]
                    best_lx = beam_meta[s, 1]
                    best_ly = beam_meta[s, 2]
                    best_lrot = beam_meta[s, 3]
        if cand_n == 0:
            break
        beam_n = _beam_keep_top(cand_score, cand_n, B,
                                cand_grid, cand_meta, cand_acc, cand_prev,
                                beam_grid, beam_meta, beam_acc, beam_prev)

    best_grid, _, _, _ = burn_and_clear_numba(grid, piece_ids[0], best_lx, best_ly, best_lrot)
    return best_score, best_grid, best_lines, best_lx, best_ly, best_lrot


@njit(cache=True, nogil=True)
def _beam_keep_top_hold(cand_score, cand_n, B,
                        cand_grid, cand_meta, cand_acc, cand_prev,
                        beam_grid, beam_meta, beam_acc, beam_prev):
    order = np.argsort(cand_score[:cand_n])
    take = B if B < cand_n else cand_n
    for k in range(take):
        idx = order[cand_n - 1 - k]
        beam_grid[k] = cand_grid[idx]
        for q in range(5):
            beam_meta[k, q] = cand_meta[idx, q]
        beam_acc[k] = cand_acc[idx]
        beam_prev[k] = cand_prev[idx]
    return take


@njit(cache=True, nogil=True)
def evaluate_beam_hold_numba(grid, p1_locks, piece_ids, weights, penalty,
                             incoming_prev_clear, post_hold,
                             beam_width, chain_bonus, knee):
    B = beam_width
    MAX_LOCKS = 64
    MAX_CAND = 2 * B * MAX_LOCKS
    w = weights

    beam_grid = np.zeros((B, 20, 10), dtype=np.int8)
    beam_meta = np.zeros((B, 5), dtype=np.int32)
    beam_acc = np.zeros(B, dtype=np.float64)
    beam_prev = np.zeros(B, dtype=np.int32)

    cand_grid = np.zeros((MAX_CAND, 20, 10), dtype=np.int8)
    cand_meta = np.zeros((MAX_CAND, 5), dtype=np.int32)
    cand_acc = np.zeros(MAX_CAND, dtype=np.float64)
    cand_score = np.zeros(MAX_CAND, dtype=np.float64)
    cand_prev = np.zeros(MAX_CAND, dtype=np.int32)

    best_score = -1e9
    best_lines = 0
    best_lx = np.int32(0)
    best_ly = np.int32(0)
    best_lrot = np.int32(0)

    cand_n = 0
    p1_id = piece_ids[0]
    for i in range(min(len(p1_locks), MAX_LOCKS)):
        lx = p1_locks[i, 0]
        ly = p1_locks[i, 1]
        lrot = p1_locks[i, 2]
        g, lines, lh, wf = burn_and_clear_numba(grid, p1_id, lx, ly, lrot)
        lp = _line_points(lines, penalty)
        if lines == 4 and incoming_prev_clear == 4:
            lp += chain_bonus
        sc = _state_score(g, lp, lh, wf, w, knee)
        c = cand_n
        cand_grid[c] = g
        cand_meta[c, 0] = lines
        cand_meta[c, 1] = lx
        cand_meta[c, 2] = ly
        cand_meta[c, 3] = lrot
        cand_meta[c, 4] = post_hold
        cand_acc[c] = lp
        cand_prev[c] = lines
        cand_score[c] = sc
        cand_n += 1
        if sc > best_score:
            best_score = sc
            best_lines = lines
            best_lx = lx
            best_ly = ly
            best_lrot = lrot

    if cand_n == 0:
        return best_score, np.zeros((20, 10), dtype=np.int8), best_lines, best_lx, best_ly, best_lrot

    beam_n = _beam_keep_top_hold(cand_score, cand_n, B,
                                 cand_grid, cand_meta, cand_acc, cand_prev,
                                 beam_grid, beam_meta, beam_acc, beam_prev)

    for ply in range(1, len(piece_ids)):
        pid = piece_ids[ply]
        cand_n = 0
        for s in range(beam_n):
            hold_s = beam_meta[s, 4]

            locks = get_reachable_locks_numba(beam_grid[s], pid, 3, 0, 0)
            for j in range(min(len(locks), MAX_LOCKS)):
                lx = locks[j, 0]
                ly = locks[j, 1]
                lrot = locks[j, 2]
                g, lines, lh, wf = burn_and_clear_numba(beam_grid[s], pid, lx, ly, lrot)
                lp = beam_acc[s] + _line_points(lines, penalty)
                if lines == 4 and beam_prev[s] == 4:
                    lp += chain_bonus
                sc = _state_score(g, lp, lh, wf, w, knee)
                c = cand_n
                cand_grid[c] = g
                cand_meta[c, 0] = beam_meta[s, 0]
                cand_meta[c, 1] = beam_meta[s, 1]
                cand_meta[c, 2] = beam_meta[s, 2]
                cand_meta[c, 3] = beam_meta[s, 3]
                cand_meta[c, 4] = hold_s
                cand_acc[c] = lp
                cand_prev[c] = lines
                cand_score[c] = sc
                cand_n += 1
                if sc > best_score:
                    best_score = sc
                    best_lines = beam_meta[s, 0]
                    best_lx = beam_meta[s, 1]
                    best_ly = beam_meta[s, 2]
                    best_lrot = beam_meta[s, 3]

            if hold_s >= 0 and hold_s != pid:
                locks2 = get_reachable_locks_numba(beam_grid[s], hold_s, 3, 0, 0)
                for j in range(min(len(locks2), MAX_LOCKS)):
                    lx = locks2[j, 0]
                    ly = locks2[j, 1]
                    lrot = locks2[j, 2]
                    g, lines, lh, wf = burn_and_clear_numba(beam_grid[s], hold_s, lx, ly, lrot)
                    lp = beam_acc[s] + _line_points(lines, penalty)
                    if lines == 4 and beam_prev[s] == 4:
                        lp += chain_bonus
                    sc = _state_score(g, lp, lh, wf, w, knee)
                    c = cand_n
                    cand_grid[c] = g
                    cand_meta[c, 0] = beam_meta[s, 0]
                    cand_meta[c, 1] = beam_meta[s, 1]
                    cand_meta[c, 2] = beam_meta[s, 2]
                    cand_meta[c, 3] = beam_meta[s, 3]
                    cand_meta[c, 4] = pid
                    cand_acc[c] = lp
                    cand_prev[c] = lines
                    cand_score[c] = sc
                    cand_n += 1
                    if sc > best_score:
                        best_score = sc
                        best_lines = beam_meta[s, 0]
                        best_lx = beam_meta[s, 1]
                        best_ly = beam_meta[s, 2]
                        best_lrot = beam_meta[s, 3]
        if cand_n == 0:
            break
        beam_n = _beam_keep_top_hold(cand_score, cand_n, B,
                                     cand_grid, cand_meta, cand_acc, cand_prev,
                                     beam_grid, beam_meta, beam_acc, beam_prev)

    best_grid, _, _, _ = burn_and_clear_numba(grid, piece_ids[0], best_lx, best_ly, best_lrot)
    return best_score, best_grid, best_lines, best_lx, best_ly, best_lrot
