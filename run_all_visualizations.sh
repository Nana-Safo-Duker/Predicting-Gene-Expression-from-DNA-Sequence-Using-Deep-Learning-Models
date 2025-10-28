#!/bin/bash

# =============================================================================
# Run All Visualizations
# =============================================================================
# This script generates all visualizations for gene expression prediction
# analysis using both Python and R.
#
# Usage: bash run_all_visualizations.sh
# =============================================================================

echo "=========================================="
echo "Gene Expression Prediction Visualizations"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed. Please install Python 3.8+"
    exit 1
fi

# Check if R is installed
if ! command -v Rscript &> /dev/null; then
    echo "WARNING: R is not installed. Skipping R visualizations."
    echo "To install R, visit: https://www.r-project.org/"
    R_AVAILABLE=false
else
    R_AVAILABLE=true
fi

echo "Step 1: Checking Python dependencies..."
if ! python3 -c "import numpy, matplotlib, seaborn, scipy, sklearn, pandas" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
else
    echo "✓ All Python dependencies are installed"
fi

echo ""
echo "Step 2: Running Python visualizations..."
python3 visualizations.py

if [ $? -eq 0 ]; then
    echo "✓ Python visualizations completed successfully"
else
    echo "✗ Python visualizations failed"
    exit 1
fi

if [ "$R_AVAILABLE" = true ]; then
    echo ""
    echo "Step 3: Running R visualizations..."
    Rscript visualizations.R
    
    if [ $? -eq 0 ]; then
        echo "✓ R visualizations completed successfully"
    else
        echo "✗ R visualizations failed (non-fatal)"
    fi
else
    echo ""
    echo "Step 3: Skipping R visualizations (R not installed)"
fi

echo ""
echo "=========================================="
echo "All visualizations completed!"
echo "Output saved to: figures/"
echo "=========================================="
echo ""
echo "Generated files:"
ls -lh figures/

echo ""
echo "Next steps:"
echo "  1. Launch interactive notebook: jupyter notebook visualizations.ipynb"
echo "  2. Check visualizations in the figures/ directory"
echo "  3. See README.md for detailed documentation"

