#!/bin/bash
#SBATCH --job-name=temporal_leakage_sim
#SBATCH --account=shakeri-lab
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/simulation_%j.out
#SBATCH --error=slurm_logs/simulation_%j.err

# =============================================================================
# Temporal Leakage Simulation - SLURM Job Script
# =============================================================================

echo "=============================================="
echo "Starting Temporal Leakage Simulation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "=============================================="

# Change to project directory
cd /project/shakeri-lab/b/leakage

# Load required modules (adjust based on available modules)
module purge
module load miniforge/24.3.0-py3.11

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated virtual environment"
fi

# Print Python info
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Create output directories
mkdir -p results/tables results/figures slurm_logs

# Run the simulation
echo ""
echo "Running simulation with 100 replicates..."
echo ""

python experiments/run_simulation.py \
    --n_replicates 100 \
    --start_seed 42 \
    --output_dir results

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Simulation completed successfully!"
    echo "Results saved to: results/"
    echo "Time: $(date)"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "ERROR: Simulation failed!"
    echo "Check the error log for details."
    echo "=============================================="
    exit 1
fi
