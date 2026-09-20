from collections import deque

import numpy as np

from . import config
from .ga.genome import Genome
from .search import engine as _eng
from .search.beam import evaluate_beam_hold_numba, evaluate_beam_numba
from .search.bfs import get_reachable_locks_numba
from .tetris_base.board import Board, Bag, Next, Hold
from .tetris_base.tables import (ALL_SHAPE_OFFSETS, ID_TO_PIECE, KICK_TABLE_I,
                                 KICK_TABLE_JLSTZ, PIECE_TO_ID, SHAPE_OFFSETS)


DEFAULT_CHAMPION = [2.647786, 0.148468, -10.239377, 2.900157, -4.703681,
                    -0.115185, 1.030894, -4.747972, -2.276079, -9.759769,
                    -16.621865, -1.402996]


def stress_test_champion(weights, num_games=5, survival_goal=100000):
    """Play `num_games` with `weights`; 100k pieces is essentially infinite."""
    genome = Genome(weights)
    print("--- STRESS TESTING CHAMPION ---")
    print(f"Weights: {weights}")
    print(f"Target: Survive {survival_goal:,} pieces per game\n")

    for i in range(num_games):
        with _eng.BAR_LOCK:
            _eng.BAR['r20'] = -1.0
        print(f"Starting Game {i + 1}...")
        stats = _eng.play_game(genome, max_pieces=survival_goal)
        if stats['pieces'] >= survival_goal:
            print(f"  Game {i + 1} CLEARED! Hit the {survival_goal:,}-piece limit "
                  f"({stats['lines']:,} lines, {stats['tetrises']} Tetrises, "
                  f"score {stats['score']:,}).")
        else:
            print(f"  Game {i + 1} FAILED after {stats['pieces']:,} pieces "
                  f"({stats['lines']:,} lines, {stats['tetrises']} Tetrises, "
                  f"score {stats['score']:,}).")


def burn_piece_only(grid, piece_id, lx, ly, lrot):
    new_grid = grid.copy()
    color_id = piece_id + 1
    offsets = ALL_SHAPE_OFFSETS[piece_id, lrot]
    for j in range(offsets.shape[0]):
        bx = lx + offsets[j, 0]
        by = ly + offsets[j, 1]
        if 0 <= bx < 10 and 0 <= by < 20:
            new_grid[by, bx] = color_id
    return new_grid


def count_full_rows(grid):
    return int(np.sum(np.all(grid > 0, axis=1)))


def clear_full_rows(grid):
    full = np.all(grid > 0, axis=1)
    n = int(np.sum(full))
    if n == 0:
        return grid.copy()
    kept = grid[~full]
    empty = np.zeros((n, grid.shape[1]), dtype=grid.dtype)
    return np.vstack((empty, kept))


def fast_check_collision(grid, shape_name, x, y, rot):
    for dx, dy in SHAPE_OFFSETS[shape_name][rot]:
        bx, by = x + dx, y + dy
        if bx < 0 or bx >= 10 or by >= 20:
            return True
        if by >= 0 and grid[by, bx] != 0:
            return True
    return False


def _fully_visible(shape_name, x, y, rot):
    for dx, dy in SHAPE_OFFSETS[shape_name][rot]:
        cy = y + dy
        if cy < 0 or cy >= 20:
            return False
    return True


def find_piece_path(grid, piece_id, start_x, start_y, start_rot,
                    target_lx, target_ly, target_lrot):
    """BFS from spawn to the chosen lock, returning the ordered list of
    (x, y, rot) positions (spawn first, target last). Mirrors the move set of
    get_reachable_locks_numba (down / left / right / SRS rotations)."""
    shape_name = ID_TO_PIECE[piece_id]
    start = (start_x, start_y, start_rot)
    target = (target_lx, target_ly, target_lrot)
    if start == target:
        return [start]

    queue = deque([start])
    parent = {start: None}

    def neighbors(cx, cy, crot):
        """Legal successor states of (cx, cy, crot)."""
        out = []
        for dx, dy in ((0, 1), (-1, 0), (1, 0)):
            nx, ny = cx + dx, cy + dy
            if not fast_check_collision(grid, shape_name, nx, ny, crot):
                out.append((nx, ny, crot))
        if shape_name != 'O':
            kick_table = KICK_TABLE_I if shape_name == 'I' else KICK_TABLE_JLSTZ
            for d_rot in (1, -1):
                new_rot = (crot + d_rot) % 4
                for dx, dy in kick_table.get((crot, new_rot), [(0, 0)]):
                    rx, ry = cx + dx, cy + dy
                    if 0 <= rx < 10 and -2 <= ry < 22:
                        if not fast_check_collision(grid, shape_name, rx, ry, new_rot):
                            if _fully_visible(shape_name, rx, ry, new_rot):
                                out.append((rx, ry, new_rot))
                                break
        return out

    while queue:
        cur = queue.popleft()
        for nxt in neighbors(*cur):
            if nxt not in parent:
                parent[nxt] = cur
                if nxt == target:
                    path = [nxt]
                    p = cur
                    while p is not None:
                        path.append(p)
                        p = parent[p]
                    path.reverse()
                    return path
                queue.append(nxt)

    return [start, target]


def play_game_visual(genome):
    """Like play_game, but yields every intermediate board state so the GUI can
    animate the AI's reasoning. Each placement yields up to three frames:

        ('move',   ...) -> the piece travels from spawn to its lock spot
                           (a dim ghost shows the destination).
        ('locked', ...) -> the piece is burned in, lines NOT yet cleared.
        ('cleared',...) -> the cleared rows are removed (or 'settled' if none).

    Yield format:
        (grid, pieces, lines, phase, lines_clearing, piece_id,
         lx, ly, lrot, moving, preview_ids, hold_id, used_hold)
        moving = (x, y, rot) of the in-flight piece during 'move', else None.
    """
    board = Board(10, 20)
    bag = Bag()
    next_queue = Next(bag, preview_count=max(config.LOOKAHEAD_PLY, 2))
    hold_sys = Hold()
    board.current_piece = next_queue.pop_next()
    pieces_placed = 0
    last_clear = 0
    if not hasattr(genome, 'weights'):
        genome = Genome(genome)

    while True:
        p1_id = PIECE_TO_ID[board.current_piece.name]
        upcoming = [PIECE_TO_ID[next_queue.queue[k].name]
                    for k in range(config.LOOKAHEAD_PLY)]
        ph = (PIECE_TO_ID[hold_sys.piece.name]
              if hold_sys.piece is not None else -1)

        locks_curr = get_reachable_locks_numba(
            board.grid, p1_id,
            board.current_piece.x, board.current_piece.y, board.current_piece.rotation_index)
        best_score_curr = -float('inf')
        best_lock_curr = None
        if len(locks_curr) > 0:
            curr_pieces = np.array([p1_id] + upcoming[:config.LOOKAHEAD_PLY - 1], dtype=np.int32)
            if config.ENGINE_MODE == 'hold_aware':
                res = evaluate_beam_hold_numba(board.grid, locks_curr, curr_pieces, genome.weights, config.NON_TETRIS_PENALTY, last_clear, ph)
            else:
                res = evaluate_beam_numba(board.grid, locks_curr, curr_pieces, genome.weights, config.NON_TETRIS_PENALTY, last_clear)
            best_score_curr = res[0]
            best_lock_curr = (res[3], res[4], res[5])

        best_score_hold = -float('inf')
        best_lock_hold = None
        hold_played_id = None
        if hold_sys.can_swap:
            if hold_sys.piece is not None:
                hold_played_id = PIECE_TO_ID[hold_sys.piece.name]
                hold_pieces = np.array([hold_played_id] + upcoming[:config.LOOKAHEAD_PLY - 1], dtype=np.int32)
            else:
                hold_played_id = upcoming[0]
                hold_pieces = np.array(upcoming[:config.LOOKAHEAD_PLY], dtype=np.int32)
            locks_hold = get_reachable_locks_numba(board.grid, hold_played_id, 3, 0, 0)
            if len(locks_hold) > 0:
                if config.ENGINE_MODE == 'hold_aware':
                    res = evaluate_beam_hold_numba(board.grid, locks_hold, hold_pieces, genome.weights, config.NON_TETRIS_PENALTY, last_clear, p1_id)
                else:
                    res = evaluate_beam_numba(board.grid, locks_hold, hold_pieces, genome.weights, config.NON_TETRIS_PENALTY, last_clear)
                best_score_hold = res[0]
                best_lock_hold = (res[3], res[4], res[5])

        if best_score_curr > best_score_hold and best_score_curr != -float('inf'):
            played_id = p1_id
            lock = best_lock_curr
            used_hold = False
        elif best_score_hold != -float('inf'):
            played_id = hold_played_id
            lock = best_lock_hold
            used_hold = True
        else:
            break

        if used_hold:
            swapped_in = hold_sys.swap(board.current_piece)
            if swapped_in is None:
                next_queue.pop_next()

        preview_ids = (PIECE_TO_ID[next_queue.queue[0].name],
                       PIECE_TO_ID[next_queue.queue[1].name])
        hold_id = (PIECE_TO_ID[hold_sys.piece.name]
                   if hold_sys.piece is not None else -1)

        pieces_placed += 1
        lx, ly, lrot = lock

        path = find_piece_path(board.grid, played_id, 3, 0, 0, lx, ly, lrot)
        for (px, py, prot) in path:
            yield (board.grid, pieces_placed, board.lines_cleared,
                   'move', 0, played_id, lx, ly, lrot, (px, py, prot),
                   preview_ids, hold_id, used_hold)

        locked_grid = burn_piece_only(board.grid, played_id, lx, ly, lrot)
        lines_about_to_clear = count_full_rows(locked_grid)
        last_clear = lines_about_to_clear
        yield (locked_grid, pieces_placed, board.lines_cleared,
               'locked', lines_about_to_clear, played_id, lx, ly, lrot, None,
               preview_ids, hold_id, used_hold)

        if lines_about_to_clear > 0:
            board.grid = clear_full_rows(locked_grid)
            board.lines_cleared += lines_about_to_clear
            yield (board.grid, pieces_placed, board.lines_cleared,
                   'cleared', lines_about_to_clear, played_id, lx, ly, lrot, None,
                   preview_ids, hold_id, used_hold)
        else:
            board.grid = locked_grid
            yield (board.grid, pieces_placed, board.lines_cleared,
                   'settled', 0, played_id, lx, ly, lrot, None,
                   preview_ids, hold_id, used_hold)

        board.current_piece = next_queue.pop_next()
        hold_sys.reset_lock()
        if not board.is_valid_position(board.current_piece):
            break


def watch_champion(weights=None, lock_ms=450, clear_ms=250, move_ms=45):
    """Visualise the AI playing, animating every move.

    For each piece you see it travel from spawn to its slot (a dim ghost shows
    the destination), lock into place, then the lines clear with a
    SINGLE/DOUBLE/TRIPLE/TETRIS banner and a running combo counter.

    Controls (while running): UP/DOWN speed, SPACE pause, ESC quit.

    weights : optional 12-weight genome. Defaults to the certified vfinalc
              champion (DEFAULT_CHAMPION above).
    """
    if weights is None:
        weights = list(DEFAULT_CHAMPION)
    genome = Genome(weights)

    import pygame

    pygame.init()
    BLOCK_SIZE = 30
    TOP_MARGIN = BLOCK_SIZE
    WIDTH, HEIGHT = 10 * BLOCK_SIZE, 20 * BLOCK_SIZE + TOP_MARGIN
    SIDEBAR_WIDTH = 200

    screen = pygame.display.set_mode((WIDTH + SIDEBAR_WIDTH, HEIGHT))
    pygame.display.set_caption("Tetris AI Champion (Tetris / Combo)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)
    big_font = pygame.font.SysFont("Arial", 30, bold=True)

    COLORS = {
        0: (30, 30, 30), 1: (0, 255, 255), 2: (0, 0, 255), 3: (255, 165, 0),
        4: (255, 255, 0), 5: (0, 255, 0), 6: (128, 0, 128), 7: (255, 0, 0),
    }
    CLEAR_NAMES = {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE", 4: "TETRIS!"}

    game_gen = play_game_visual(genome)
    running = True
    paused = False
    speed_mult = 1.0
    combo = -1
    timer = 0.0
    state = None

    def advance():
        """Pull the next frame from the generator; update combo. Returns False
        when the game ends."""
        nonlocal state, combo
        try:
            s = next(game_gen)
        except StopIteration:
            return False
        state = s
        if s[3] in ('locked', 'settled'):
            if s[4] > 0:
                combo += 1
            else:
                combo = -1
        return True

    if not advance():
        running = False

    while running:
        dt = clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    speed_mult = min(speed_mult * 1.5, 64.0)
                elif event.key == pygame.K_DOWN:
                    speed_mult = max(speed_mult / 1.5, 0.05)

        if not paused and state is not None:
            timer += dt * speed_mult
            phase = state[3]
            if phase == 'move':
                hold_ms = move_ms
            elif phase in ('locked', 'settled'):
                hold_ms = lock_ms
            else:
                hold_ms = clear_ms
            if timer >= hold_ms:
                timer = 0.0
                if not advance():
                    print("Game Over!")
                    state = None

        screen.fill((0, 0, 0))

        if state is not None:
            (grid, pieces, lines, phase, n_clear, pid,
             lx, ly, lrot, moving, preview_ids, hold_id, used_hold) = state
        else:
            grid = np.zeros((20, 10), dtype=np.int8)
            pieces = lines = 0
            phase = 'over'
            n_clear = pid = 0
            lx = ly = lrot = 0
            moving = None
            preview_ids = (-1, -1)
            hold_id = -1
            used_hold = False

        for y in range(20):
            for x in range(10):
                color_id = int(grid[y, x])
                rect = (x * BLOCK_SIZE, y * BLOCK_SIZE + TOP_MARGIN,
                        BLOCK_SIZE - 1, BLOCK_SIZE - 1)
                pygame.draw.rect(screen, COLORS[color_id], rect)

        def draw_cells(x, y, rot, pid_, fill=None, outline=None, width=2):
            """Draw (or outline) a piece at (x, y, rot) using its offset list."""
            offsets = ALL_SHAPE_OFFSETS[pid_][rot]
            for j in range(offsets.shape[0]):
                cx = x + offsets[j, 0]
                cy = y + offsets[j, 1]
                if 0 <= cx < 10 and 0 <= cy < 20:
                    r = (cx * BLOCK_SIZE, cy * BLOCK_SIZE + TOP_MARGIN,
                         BLOCK_SIZE - 1, BLOCK_SIZE - 1)
                    if outline is not None:
                        pygame.draw.rect(screen, outline, r, width)
                    if fill is not None:
                        pygame.draw.rect(screen, fill, r)

        def draw_mini(pid_, ox, oy, size=15):
            """Draw piece pid_ (spawn rotation) small, in the sidebar."""
            if pid_ is None or pid_ < 0:
                return
            offs = ALL_SHAPE_OFFSETS[pid_][0]
            for j in range(offs.shape[0]):
                dx = offs[j, 0]
                dy = offs[j, 1]
                pygame.draw.rect(screen, COLORS[pid_ + 1],
                                 (ox + dx * size, oy + dy * size, size - 1, size - 1))

        if phase == 'move' and pid is not None:
            draw_cells(lx, ly, lrot, pid, outline=(90, 90, 90), width=2)
            if moving is not None:
                mx, my, mrot = moving
                draw_cells(mx, my, mrot, pid, fill=COLORS[pid + 1])

        if phase in ('locked', 'settled') and pid is not None:
            draw_cells(lx, ly, lrot, pid, outline=(255, 255, 255), width=2)

        if phase == 'cleared' and n_clear > 0:
            label = CLEAR_NAMES.get(n_clear, f"x{n_clear}")
            color = (0, 255, 255) if n_clear == 4 else (255, 255, 255)
            screen.blit(big_font.render(label, True, color), (10, HEIGHT - 40))
        if combo > 0 and phase in ('locked', 'settled'):
            screen.blit(big_font.render(f"COMBO x{combo}", True, (255, 200, 0)),
                        (10, HEIGHT - 80))

        sx = WIDTH + 12
        stats = [
            f"Pieces:  {pieces}",
            f"Lines:   {lines}",
            f"Combo:   {combo if combo > 0 else 0}",
            f"Speed:   {speed_mult:.1f}x",
            f"Phase:   {phase}",
            "",
            "UP/DOWN: speed",
            "SPACE:   pause",
            "ESC:     quit",
        ]
        for i, text in enumerate(stats):
            screen.blit(font.render(text, True, (255, 255, 255)), (sx, 20 + i * 26))

        py = 300
        screen.blit(font.render("HOLD:", True, (255, 255, 255)), (sx, py))
        draw_mini(hold_id, sx, py + 26)
        screen.blit(font.render("NEXT:", True, (255, 255, 255)), (sx, py + 96))
        for k, nid in enumerate(preview_ids):
            draw_mini(nid, sx + k * 52, py + 122)
        if used_hold:
            screen.blit(big_font.render("HOLD!", True, (255, 120, 0)),
                        (WIDTH // 2 - 30, HEIGHT // 2))

        pygame.display.flip()

    pygame.quit()
