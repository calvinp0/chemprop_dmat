# Gaussian09 benchmark workflow (local prep)

This script generates Gaussian09 inputs from `ts_guesses_opt/` and writes/updates
`ts_guesses_g09_results.csv`. It defaults to **append/update** mode so running
with `--top 5` won’t wipe earlier rows.

## Common usage

```bash
# Generate PBS job folders for all reactions (append/update results)
python scripts/ts_g09_benchmark.py --all --make-pbs

# Generate PBS job folders for top 5 (append/update results)
python scripts/ts_g09_benchmark.py --top 5 --make-pbs

# Overwrite the results CSV if you really want a fresh file
python scripts/ts_g09_benchmark.py --top 5 --make-pbs --overwrite-results

# Include reactions already present in results CSV (rerun selection)
python scripts/ts_g09_benchmark.py --top 5 --make-pbs --include-existing
```

## Notes

- Baseline comparison defaults to `job_type=opt` only.
- Default route: `#p opt=(ts,calcfc,noeigentest,maxcycles=200) IOp(2/9=2000) wb97xd/def2tzvp`
- Per-reaction `charge`, `multiplicity`, `gjf_nprocshared`, `gjf_mem_bytes`,
  `pbs_ncpus`, and `pbs_mem_bytes` are pulled from `ts0_opt_resources.csv`.
- `submit.sh` and `input.gjf` are written to `ts_guesses_g09_jobs/<reaction_name>/` when using `--make-pbs`.
