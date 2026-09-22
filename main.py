ENGINE_MODE      = 'hold_aware'
CROSSOVER_MODE   = 'blx'
HEIGHT_KNEE      = 15
LOOKAHEAD_PLY    = 2
GENERATIONS      = 40
POP_SIZE         = 24
EVAL_MAX_PIECES  = 100
SEED_BASE        = 10_000
GA_SEED          = 42
TAG              = 'v33'
WATCH_FIRST      = True


from tetris_ai import config

OVERRIDE = True
if OVERRIDE:
    config.HEIGHT_KNEE = HEIGHT_KNEE
    config.LOOKAHEAD_PLY = LOOKAHEAD_PLY
    config.ENGINE_MODE = ENGINE_MODE
    config.CROSSOVER_MODE = CROSSOVER_MODE
    config.GENERATIONS = GENERATIONS
    config.POP_SIZE = POP_SIZE
    config.EVAL_MAX_PIECES = EVAL_MAX_PIECES
    config.SEED_BASE = SEED_BASE
    config.GA_SEED = GA_SEED
    config.TAG = TAG



if WATCH_FIRST:
    from tetris_ai.testing import DEFAULT_CHAMPION, watch_champion
    DEFAULT_CHAMPION = [12.50, -2.26, -12.96, 2.62, -19.75, -53.59, 39.53, -7.03, -24.10, 3.51, -10.27, -3.60]
    watch_champion(list(DEFAULT_CHAMPION))

from tetris_ai.ga.evolution import run_evolution, save_run_stats_csv
from tetris_ai.ga.genome import fmt_weights
from tetris_ai.search import engine

champ = run_evolution(population_size=config.POP_SIZE,
                      generations=config.GENERATIONS, seed=config.GA_SEED)
save_run_stats_csv(config.out_path(f'{config.TAG}_gens{config.GENERATIONS}.csv'))
print(f"\nchampion weights: {fmt_weights(champ.weights)}", flush=True)
print(f"champion bar r20: {engine.BAR['r20']:.2f}", flush=True)
