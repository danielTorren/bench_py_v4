#!/bin/bash
# =============================================================================
# Slurm batch script — BENCH v4 ES full scenario run
# 4 scenarios x 64 seeds, n_households=637486 (Navarre population)
#
# Submit from project root on the cluster:
#   cd ~/BENCH_py_v.4
#   sbatch run_es_full.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f es_full_<jobid>.log
# =============================================================================

#SBATCH --job-name=bench_es_full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64         # parallel workers
#SBATCH --time=04:00:00            # 256 runs * ~20s / 64 cores ~ 5 min; 4h is generous
#SBATCH --mem=96gb                 # 64 workers * ~200 MB/model + 97 MB pop_df overhead
#SBATCH --output=es_full_%j.log
#SBATCH --partition=generic

# ---- User configuration (edit if needed) ------------------------------------
PROJECT_DIR="$HOME/BENCH_py_v.4"
VENV_DIR="$HOME/bench_v4_venv"
CONFIG="configs/bench_v4_scenarios_ES_full.yaml"
WORKERS=$SLURM_CPUS_PER_TASK

# ---- Environment -------------------------------------------------------------
module purge
module load Python/3.12.3-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a

source "$VENV_DIR/bin/activate"

export MPLBACKEND=Agg   # non-interactive matplotlib backend (no display on cluster)

cd "$PROJECT_DIR"

# ---- Job info ----------------------------------------------------------------
echo "===================================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "Config    : $CONFIG"
echo "Started   : $(date)"
echo "===================================================="
echo ""

# ---- Run --------------------------------------------------------------------
srun python main.py \
    --config "$CONFIG" \
    --jobs "$WORKERS" \
    --no-plot

echo ""
echo "===================================================="
echo "Finished  : $(date)"
echo "===================================================="
