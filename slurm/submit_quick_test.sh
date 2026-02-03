#!/bin/bash
#SBATCH --job-name=leakage_test
#SBATCH --account=shakeri-lab
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/test_%j.out
#SBATCH --error=slurm_logs/test_%j.err

# =============================================================================
# Quick Test - Verify Setup and Run 10 Replicates
# =============================================================================

echo "=============================================="
echo "Quick Test Run"
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "=============================================="

cd /project/shakeri-lab/b/leakage

# Load modules
module purge
module load miniforge/24.3.0-py3.11

# Activate venv if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create directories
mkdir -p results/tables results/figures slurm_logs

# Run quick test
python experiments/run_simulation.py --quick --output_dir results

echo ""
echo "Quick test completed at $(date)"
