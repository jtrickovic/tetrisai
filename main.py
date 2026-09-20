ENGINE_MODE      = 'hold_aware'
CROSSOVER_MODE   = 'blx'
HEIGHT_KNEE      = 15
LOOKAHEAD_PLY    = 2
GENERATIONS      = 40
POP_SIZE         = 24
EVAL_MAX_PIECES  = 400_000
SEED_BASE        = 10_000
GA_SEED          = 42
TAG              = 'v33'
WATCH_FIRST      = True


from tetris_ai import config

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
    watch_champion(list(DEFAULT_CHAMPION))

from tetris_ai.ga.evolution import run_evolution, save_run_stats_csv
from tetris_ai.ga.genome import fmt_weights
from tetris_ai.search import engine

champ = run_evolution(population_size=config.POP_SIZE,
                      generations=config.GENERATIONS, seed=config.GA_SEED)
save_run_stats_csv(config.out_path(f'{config.TAG}_gens{config.GENERATIONS}.csv'))
print(f"\nchampion weights: {fmt_weights(champ.weights)}", flush=True)
print(f"champion bar r20: {engine.BAR['r20']:.2f}", flush=True)
