#!/bin/bash
# =============================================================================
# Slurm batch script for BENCH v4 Sobol sensitivity analysis
#
# Submit from your project root directory on the cluster:
#   cd ~/BENCH_py_v.4
#   sbatch sensitivity/run_sa.slurm
#
# Monitor the job:
#   squeue -u $USER
#   tail -f sa_<jobid>.log
# =============================================================================

# ---- Slurm resource requests ------------------------------------------------
#SBATCH --job-name=bench_v4_sa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # parallel workers; adjust to available cores
#SBATCH --time=18:00:00           # wall-clock limit; increase for large SA runs
#SBATCH --mem=32gb                # total RAM for the job
#SBATCH --output=sa_%j.log        # log file in the directory where sbatch is run
#SBATCH --partition=generic

# ---- User configuration (edit these) ----------------------------------------
PROJECT_DIR="$HOME/BENCH_py_v.4"          # path to the project on the cluster
VENV_DIR="$HOME/bench_v4_venv"            # virtual environment created by setup_cluster_env.sh
CONFIG="configs/sa_config.yaml"        # SA config file (relative to PROJECT_DIR)
WORKERS=$SLURM_CPUS_PER_TASK              # use all allocated CPUs as workers

# ---- Environment setup -------------------------------------------------------
module purge
module load Python/3.12.3-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a

source "$VENV_DIR/bin/activate"

cd "$PROJECT_DIR"

# ---- Job info ----------------------------------------------------------------
echo "===================================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "Workers   : $WORKERS"
echo "Config    : $CONFIG"
echo "Started   : $(date)"
echo "===================================================="
echo ""

# ---- Run the analysis -------------------------------------------------------
srun python sensitivity/run_sa.py \
    --config "$CONFIG" \
    --workers "$WORKERS"

echo ""
echo "===================================================="
echo "Finished  : $(date)"
echo "===================================================="
