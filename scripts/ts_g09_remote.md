# Gaussian09 remote workflow (Zeus)

This is a quick reference for uploading, submitting, and syncing Gaussian09 jobs.

Assumes:
- Local job folders in `ts_guesses_g09_jobs/` (created by `scripts/ts_g09_benchmark.py --make-pbs`)
- Remote target: `calvin.p@zeus.technion.ac.il:/home/calvin.p/ts_jobs/chemprop_cmpnn/ts_guesses_g09_jobs`

## Upload + submit

```bash
python scripts/ts_g09_remote.py \
  --remote calvin.p@zeus.technion.ac.il \
  --remote-base /home/calvin.p/ts_jobs/chemprop_cmpnn/ts_guesses_g09_jobs \
  --sync-to --submit
```

This writes a submission log in `ts_guesses_g09_submit_log.csv` and skips any job that
already has a `job_id` in the log. Use `--resubmit` to force re-submission.

## Common options

- `--quiet`: suppress per-job progress output (verbose is default)
- `--resubmit`: force re-submit jobs even if already logged
- `--sync-to`: upload local job dirs
- `--sync-from`: download remote job dirs
- `--update-results`: update `real_time_seconds` in `ts_guesses_g09_results.csv`
- `--use-remote-times`: compute times from server-side `initial_time`/`final_time`
- `--backfill-results`: add missing rows for any job folders not in results CSV

## Pull results + update runtimes

```bash
python scripts/ts_g09_remote.py \
  --remote calvin.p@zeus.technion.ac.il \
  --remote-base /home/calvin.p/ts_jobs/chemprop_cmpnn/ts_guesses_g09_jobs \
  --sync-from --update-results --use-remote-times
```

This updates `ts_guesses_g09_results.csv` with:
- `real_time_seconds` (remote `final_time` - `initial_time`)
- `job_id` and `submit_epoch` from the submit log
