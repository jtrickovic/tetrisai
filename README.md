# TetrisAI

Tetris bot: SRS engine + BFS enumeration of all legal placements + K-ply beam
search with hold slot, weights tuned by a genetic algorithm.

## Run

```
pip install numpy numba pygame
python main.py        
```

## Structure

- `tetris_ai/` — package: `config.py` (all knobs), `tetris_base/` (shapes, SRS
  tables, board), `search/` (BFS, beam, metrics, engine), `ga/` (evolution),
  `testing.py` (game viewer + stress test)
- `runs/` — archived evolution runs
- `a.tex` — paper (compile: `tectonic a.tex`)
