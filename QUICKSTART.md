# nips-specscale — One-Command Experiment Runner

## Run Everything (Single Command)

```bash
git clone https://github.com/Sunshine535/nips-specscale.git
cd nips-specscale
bash run.sh
```

This single command will:
1. **Auto-setup** environment (uv + PyTorch 2.10 + CUDA 12.8) — skipped if already done
2. **Run ALL experiments** in full production mode (no quick/smoke mode)
3. **Show real-time progress** in terminal + save to `run.log`
4. **Skip completed phases** if interrupted and re-run (automatic resume)

## Background Execution (Recommended for Long Runs)

```bash
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

## After Completion

```bash
# Verify pipeline completed
cat results/.pipeline_done

# Package results for transfer
bash collect_results.sh
# Output: results_archive/nips-specscale_results_TIMESTAMP.tar.gz
```

## Key Facts

- **GPU**: Auto-detects and uses 4-8 A100 GPUs
- **Resume**: Re-run `bash run.sh` — completed phases are automatically skipped
- **Force Re-run**: `FORCE_RERUN=1 bash run.sh` to re-run all phases
- **Results**: All outputs in `results/` directory
- **Logs**: Per-phase logs in `logs/` directory
