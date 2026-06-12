#!/bin/bash
# =============================================================================
# One-time setup script for the IIASA UnICC cluster (JupyterHub terminal).
#
# Run this once before submitting any Slurm jobs:
#   bash sensitivity/setup_cluster_env.sh
#
# After this you can submit jobs with:
#   cd ~/BENCH_py_v.4
#   sbatch sensitivity/run_sa.slurm
# =============================================================================

VENV_DIR="$HOME/bench_v4_venv"

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a   # provides numpy, scipy, pandas

echo "Python version:"
python --version
echo ""

echo "Creating virtual environment at $VENV_DIR ..."
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
pip install --upgrade pip --quiet
# numpy, scipy, pandas come from the SciPy-bundle module — only install extras
pip install matplotlib pyyaml salib joblib


echo ""
echo "===================================================="
echo "Setup complete."
echo "Virtual environment : $VENV_DIR"
echo ""
echo "To submit the SA job:"
echo "  cd ~/BENCH_py_v.4"
echo "  sbatch sensitivity/run_sa.slurm"
echo ""
echo "To check job status:"
echo "  squeue -u \$USER"
echo ""
echo "To watch the log live:"
echo "  tail -f sa_<jobid>.log"
echo "===================================================="
