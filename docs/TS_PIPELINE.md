# TS Pipeline (Train → Merge → Optimize)

This pipeline trains the baseline model, generates TS guesses from SDFs, and (optionally) optimizes them.
All outputs are recorded in a single SQLite DB.

## Quick Start

```bash
source /home/calvin/miniforge3/etc/profile.d/conda.sh
conda activate chemprop_rocm

# End-to-end (train + merge + optimize)
python scripts/ts_pipeline.py \
  --train \
  --swap-by-ts \
  --remap-to-ts \
  --optimize \
  --opt-use-xtb
```

## What It Does

1) **Train** (optional)
   - Runs `scripts/ts_basic.py` and writes predictions into `ts_predictions.sqlite`.
   - The newest `runs.run_id` is linked into the merge table.

2) **Merge**
   - Runs `scripts/batch_simple_ts_merge` logic on every `.sdf` in `DATA/SDF`.
   - Writes guesses to `ts_guesses/<rxn>_ts_guess.xyz`.
   - Writes per-atom metadata to `ts_guesses/<rxn>_ts_props.json`.
   - Adds `swap_by_ts`/`remap_to_ts` info to the DB.

3) **Optimize** (optional)
   - Uses `scripts/ts_geom_opt.py` via `ts_pipeline.py`.
   - Writes optimized guesses to `ts_guesses_opt/<rxn>_ts_opt.xyz`.
   - Derives Hookean constraints from each `.ts_props.json`, writes `ts_guesses/<rxn>_hook_constraints.json`, and logs the filename in the `ts_optimizations` table.

## Outputs

- **DB:** `ts_predictions.sqlite`
  - `runs` / `test_predictions` (from `ts_basic.py`)
  - `ts_guesses` (merge outputs)
  - `ts_optimizations` (opt outputs, now records `constraints_json_path`)
- **Guesses:** `ts_guesses/*.xyz` and `ts_guesses/*.ts_props.json`
- **Constraints:** `ts_guesses/*.hook_constraints.json` (distances between `*1/*2/*3`)
- **Optimized:** `ts_guesses_opt/*.xyz`
- **Failure log:** `ts_guesses_opt/opt_failures.log` (one line per failed optimization)

## Notes

- `--swap-by-ts` uses the TS block in the SDF to decide if r1h/r2h should be swapped.
- `--remap-to-ts` reorders output atoms only when mapnums are unique across fragments.
- Use `--limit N` to smoke-test on a subset.

## Minimal Merge Only

```bash
python scripts/ts_pipeline.py --swap-by-ts --remap-to-ts
```
