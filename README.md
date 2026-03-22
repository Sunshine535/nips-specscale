# SpecScale: Universal Scaling Laws for Speculative Inference Across Modalities

---

## Quick Start

```bash
# 1. Clone and enter project
git clone https://github.com/Sunshine535/nips-specscale.git
cd nips-specscale

# 2. One-command setup + run all experiments
bash run.sh

# 3. (Optional) Run in background for long experiments
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

### Check Completion

```bash
cat results/.pipeline_done   # Shows PIPELINE_COMPLETE when all phases finish
ls results/.phase_markers/   # See which individual phases completed
```

### Save and Send Results

```bash
# Option A: Push to GitHub
git add results/ logs/
git commit -m "Experiment results"
git push origin main

# Option B: Package as tarball
bash collect_results.sh
# Output: results_archive/nips-specscale_results_YYYYMMDD_HHMMSS.tar.gz
```

### Resume After Interruption

Re-run `bash run.sh` — completed phases are automatically skipped.
To force re-run all phases: `FORCE_RERUN=1 bash run.sh`
