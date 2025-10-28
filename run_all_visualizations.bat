@echo off
REM =============================================================================
REM Run All Visualizations (Windows)
REM =============================================================================
REM This script generates all visualizations for the gene expression prediction
REM blog post using both Python and R.
REM
REM Usage: run_all_visualizations.bat
REM =============================================================================

echo ==========================================
echo Gene Expression Prediction Visualizations
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed. Please install Python 3.8+
    exit /b 1
)

REM Check if R is installed
Rscript --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: R is not installed. Skipping R visualizations.
    echo To install R, visit: https://www.r-project.org/
    set R_AVAILABLE=false
) else (
    set R_AVAILABLE=true
)

echo Step 1: Checking Python dependencies...
python -c "import numpy, matplotlib, seaborn, scipy, sklearn, pandas" 2>nul
if errorlevel 1 (
    echo Installing Python dependencies...
    pip install -r requirements.txt
) else (
    echo [OK] All Python dependencies are installed
)

echo.
echo Step 2: Running Python visualizations...
python visualizations.py

if errorlevel 1 (
    echo [ERROR] Python visualizations failed
    exit /b 1
) else (
    echo [OK] Python visualizations completed successfully
)

if "%R_AVAILABLE%"=="true" (
    echo.
    echo Step 3: Running R visualizations...
    Rscript visualizations.R
    
    if errorlevel 1 (
        echo [WARNING] R visualizations failed (non-fatal)
    ) else (
        echo [OK] R visualizations completed successfully
    )
) else (
    echo.
    echo Step 3: Skipping R visualizations (R not installed)
)

echo.
echo ==========================================
echo All visualizations completed!
echo Output saved to: figures\
echo ==========================================
echo.
echo Generated files:
dir figures\

echo.
echo Next steps:
echo   1. Check visualizations in the figures\ directory
echo   2. Read documentation: README.md or QUICK_START.md
echo   3. Use figures in your research or presentations

pause



