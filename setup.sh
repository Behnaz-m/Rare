#!/bin/bash
# =============================================================================
# Setup script for Temporal Leakage project
# =============================================================================

echo "Setting up Temporal Leakage project..."

# Load Python module
module purge
module load miniforge/24.3.0-py3.11

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create output directories
mkdir -p results/tables results/figures slurm_logs

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run a quick test:"
echo "  python experiments/run_simulation.py --quick"
echo ""
echo "To submit to SLURM:"
echo "  sbatch slurm/submit_simulation.sh"
