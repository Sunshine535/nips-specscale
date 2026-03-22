# Quick Start — nips-specscale

## One-Command Setup & Run

```bash
# 1. Clone
git clone https://github.com/Sunshine535/nips-specscale.git
cd nips-specscale

# 2. Setup environment (uv + PyTorch 2.10 + CUDA 12.8)
bash setup.sh

# 3. Run ALL experiments (production mode, auto-detects 4-8 GPUs)
nohup bash scripts/run_all_experiments.sh > run.log 2>&1 &

# 4. Monitor progress
tail -f run.log

# 5. After completion, collect results
bash collect_results.sh
```

## Key Points

- **GPU**: Automatically detects and uses 4-8 A100 GPUs
- **Resume**: If interrupted, re-run the same command — completed phases are skipped
- **Results**: All outputs in `results/` directory
- **Logs**: Per-phase logs in `logs/` directory
- **Archive**: `collect_results.sh` creates timestamped tarball in `results_archive/`
- **Completion**: `results/.pipeline_done` marker created when all phases finish

## Verify Completion

```bash
cat results/.pipeline_done  # Should show PIPELINE_COMPLETE
```
