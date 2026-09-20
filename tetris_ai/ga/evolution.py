import csv
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import config
from ..search import engine as _eng
from .genome import Genome, fmt_fit, fmt_weights

EVAL_SEEDS = tuple(config.SEED_BASE + k for k in range(config.TIER1_SEEDS))

_last_evaluated_population = None
_stagnant = 0
_best_ever = -1e18
_run_stats_log = []
_GENOMES_TXT = None


def evaluate_single_genome(genome):
    if getattr(genome, 'v32_pre', False):
        return genome
    bases = []
    tet = b2b = dbl = ihold = 0
    for seed in EVAL_SEEDS:
        st = _eng.play_game(genome, max_pieces=config.EVAL_MAX_PIECES, seed=seed)
        bases.append(st['pieces'])
        if st['survived'] or config.EVAL_MAX_PIECES > 500:
            tet += st['tetrises']
            b2b += st['b2b']
            dbl += st['double_tetris']
            ihold += st['i_hold_turns']
    tet_per_100 = 100.0 * tet / max(1, sum(bases))
    genome.fitness = tet_per_100 * sum(bases)
    genome.v32v = (genome.fitness, sum(bases), min(bases) if bases else 0,
                   tet, b2b, dbl, ihold)
    return genome


def _breed_next_gen(population, pop_size, mut_rate, mut_step):
    elite_count = max(1, int(pop_size * 0.1))
    nxt = list(population[:elite_count])
    while len(nxt) < pop_size:
        parent1 = max(random.sample(population, 3), key=lambda g: g.fitness)
        parent2 = max(random.sample(population, 3), key=lambda g: g.fitness)
        if config.CROSSOVER_MODE == 'blx':
            child = parent1.crossover_blx(parent2, config.BLX_ALPHA)
        else:
            child = parent1.crossover(parent2)
        child.mutate(mutation_rate=mut_rate, mutation_step=mut_step)
        nxt.append(child)
    return nxt


def _mut_schedule(gen, generations):
    if not generations:
        return 0.15, 0.20
    prog = min(1.0, max(0.0, gen / max(1, generations)))
    return (0.30 - 0.20 * prog,
            0.30 - 0.18 * prog)


def _diversity_restart(elites, n, champion_weights):
    out = list(elites)
    while len(out) < n:
        if random.random() < 0.5:
            out.append(Genome())
        else:
            gg = Genome(champion_weights)
            gg.mutate(mutation_rate=0.5, mutation_step=0.4, signflip_rate=0.1)
            out.append(gg)
    return out


def _dump_gen(gen, population):
    global _GENOMES_TXT
    if _GENOMES_TXT is None:
        _GENOMES_TXT = open(config.out_path(config.TAG + '_genomes.txt'), 'w')
    _GENOMES_TXT.write(f"==== Generation {gen + 1} ====\n")
    for i, g in enumerate(population):
        s = getattr(g, 'v32v', None)
        if s is None:
            continue
        fit_raw, pcs, min_pc, tet, b2b, dbl, ihold = s
        _GENOMES_TXT.write(
            f"  {i:>2}  fit_raw={fit_raw:>12,.1f}  fit_now={g.fitness:>12,.1f}  "
            f"pcs={pcs:>6,}  tets={tet:>5}  b2b={b2b:>5}  dbl={dbl:>4}  "
            f"ihold={ihold:>5}  tet/100={100.0 * tet / max(1, pcs):4.2f}\n"
            f"      W={list(g.weights)!r}\n")
    _GENOMES_TXT.flush()


def _pre_eval(genome):
    if getattr(genome, 'v32_pre', False):
        return
    evaluate_single_genome(genome)
    genome.v32_pre = True
    print(f"  [champ-first] bar r20={_eng.BAR['r20']:.2f}", flush=True)


def _evolve_one_gen(population, population_size, gen, generations=None):
    global _last_evaluated_population, _stagnant, _best_ever, EVAL_SEEDS
    population.sort(key=lambda g: g.fitness, reverse=True)
    _last_evaluated_population = population
    _dump_gen(gen, population)

    fits = np.array([g.fitness for g in population])
    print(f"--- Generation {gen + 1} ---")
    print(f"best={fmt_fit(fits[0])}  mean={fmt_fit(fits.mean())} "
          f"(sd {fmt_fit(fits.std())})  median={fmt_fit(np.median(fits))} "
          f"(worst {fmt_fit(fits[-1])})")
    print(f"  weights={fmt_weights(population[0].weights)}")

    _rs = random.getstate()
    random.seed(12345)
    _bs = _eng.play_game(population[0], max_pieces=config.DISPLAY_MAX_PIECES, seed=12345)
    random.setstate(_rs)
    print(f"  plays (validation cap {config.DISPLAY_MAX_PIECES}): "
          f"pieces={_bs['pieces']} lines={_bs['lines']} "
          f"tetrises={_bs['tetrises']}")
    _run_stats_log.append({
        'gen': gen + 1, 'ply': config.LOOKAHEAD_PLY, 'max_pieces': config.EVAL_MAX_PIECES,
        'best': float(fits[0]), 'mean': float(fits.mean()),
        'median': float(np.median(fits)), 'worst': float(fits[-1]),
        'sd': float(fits.std()),
        'pieces': _bs['pieces'], 'lines': _bs['lines'],
        'tetrises': _bs['tetrises'], 'b2b': _bs['b2b'],
        'double_tetris': _bs['double_tetris'], 'max_combo': _bs['max_combo'],
        'i_hold_turns': _bs['i_hold_turns'],
        'tet_per_100': round(100.0 * _bs['tetrises'] / (_bs['pieces'] or 1), 2),
    })

    EVAL_SEEDS = tuple(config.SEED_BASE + 100 * (gen + 1) + k for k in range(config.TIER1_SEEDS))
    with _eng.BAR_LOCK:
        _eng.BAR['r20'] = -1.0
    for g in population:
        if getattr(g, 'v32_pre', False):
            g.v32_pre = False
    _pre_eval(population[0])

    cur_best = population[0].fitness
    if cur_best > _best_ever * config.IMPROVE_THRESHOLD:
        _best_ever = cur_best
        _stagnant = 0
    else:
        _stagnant += 1
    if _stagnant >= config.STAGNATION_LIMIT:
        print(f"  [stagnant {config.STAGNATION_LIMIT} gens -> injecting diversity]", flush=True)
        _stagnant = 0
        return _diversity_restart(population[:2], population_size, population[0].weights)

    rate, step = _mut_schedule(gen, generations)
    return _breed_next_gen(population, population_size, rate, step)


def run_evolution(population_size=50, generations=100, seed_weights=None, seed=None):
    global _last_evaluated_population, _stagnant, _best_ever, _run_stats_log
    _stagnant = 0
    _best_ever = -1e18
    _run_stats_log = []
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    population = []
    if seed_weights:
        for weights in seed_weights:
            population.append(Genome(weights))
    while len(population) < population_size:
        population.append(Genome())

    num_cores = config.EVAL_WORKERS if config.EVAL_WORKERS else max(1, os.cpu_count() - 1)

    random.seed(0)
    evaluate_single_genome(Genome(population[0].weights))

    with ThreadPoolExecutor(max_workers=num_cores) as pool:
        for gen in range(generations):
            population = list(pool.map(evaluate_single_genome, population))
            population = _evolve_one_gen(population, population_size, gen, generations)
    return population[0]


def save_run_stats_csv(path='run_stats.csv'):
    if not _run_stats_log:
        print("no stats logged yet")
        return
    keys = list(_run_stats_log[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in _run_stats_log:
            w.writerow({k: row.get(k, '') for k in keys})
    print(f"wrote {len(_run_stats_log)} generations -> {path}")
