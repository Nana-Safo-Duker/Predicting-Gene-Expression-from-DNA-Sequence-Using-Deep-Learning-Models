@echo off
REM Git commands to update GitHub repository
REM Cardiovascular Risk Prediction Project
REM Date: October 27, 2024

echo ========================================================
echo GitHub Update Script
echo Cardiovascular Risk Prediction from Retinal Images
echo ========================================================
echo.

echo Current directory:
cd
echo.

REM Check git status
echo Checking git status...
git status
echo.

echo ========================================================
echo STEP 1: Remove blog_post.md
echo ========================================================
echo.

REM Check if blog_post.md exists (might be already deleted locally)
if exist blog_post.md (
    echo Removing blog_post.md from repository...
    git rm blog_post.md
) else (
    echo blog_post.md already deleted locally, staging deletion...
    git add -A
)

echo.
echo Committing removal...
git commit -m "Remove blog_post.md - focusing on code and visualizations"
echo.
echo ✓ blog_post.md removed!
echo.

pause

echo ========================================================
echo STEP 2: Add cardiovascular_visualization_complete.py
echo ========================================================
echo.

echo Adding cardiovascular_visualization_complete.py...
git add cardiovascular_visualization_complete.py

echo.
echo Committing with detailed message...
git commit -m "Add cardiovascular_visualization_complete.py - advanced OOP implementation

Features:
- Object-oriented CardiovascularRiskAnalyzer class (1,150+ lines)
- Comprehensive visualization suite matching Nature paper
- Interactive Plotly dashboards (optional)
- 4-panel risk-based attention maps with fallback logic
- Comprehensive text report generation
- Graceful handling of optional dependencies (cv2, plotly)
- Both functional and OOP interfaces for flexibility
- Production-ready with extensive error handling
- All visualizations synchronized with simple script
- 10,000 patient dataset matching published metrics

Technical Details:
- Uses pd.cut() with fallback for robust patient selection
- Panel hiding logic for consistent 2x2 grid display
- Advanced statistical analysis and reporting
- Interactive HTML dashboard generation
- Attention map risk score display across risk groups"

echo.
echo ✓ cardiovascular_visualization_complete.py added!
echo.

pause

echo ========================================================
echo STEP 3: Update Documentation
echo ========================================================
echo.

echo Staging updated documentation files...
git add README.md
git add PROJECT_SUMMARY.md
git add VISUALIZATION_COMPARISON.md
git add GIT_COMMANDS_GUIDE.md

echo.
echo Committing documentation updates...
git commit -m "Update documentation for complete script addition

Changes:
- README.md: Removed blog_post.md references, added complete script section
- PROJECT_SUMMARY.md: Restructured deliverables, updated file structure
- VISUALIZATION_COMPARISON.md: Updated comparison with current scripts
- GIT_COMMANDS_GUIDE.md: Added comprehensive Git workflow guide
- Updated project structure diagrams
- Revised file counts and metrics
- Enhanced usage instructions"

echo.
echo ✓ Documentation updated!
echo.

pause

echo ========================================================
echo STEP 4: Push to GitHub
echo ========================================================
echo.

echo Pushing all commits to remote repository...
echo Repository: https://github.com/Nana-Safo-Duker/Cardiovascular-Risk-Prediction-from-Retinal-Images
echo.

git push origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo ✓ SUCCESS! All changes pushed to GitHub
    echo ========================================================
    echo.
    echo Summary:
    echo - Removed: blog_post.md
    echo - Added: cardiovascular_visualization_complete.py
    echo - Updated: README.md, PROJECT_SUMMARY.md, VISUALIZATION_COMPARISON.md
    echo.
    echo View your changes:
    echo https://github.com/Nana-Safo-Duker/Cardiovascular-Risk-Prediction-from-Retinal-Images
    echo.
) else (
    echo.
    echo ========================================================
    echo ✗ ERROR: Push failed
    echo ========================================================
    echo.
    echo Possible causes:
    echo 1. Authentication issue - check your GitHub credentials
    echo 2. Network connection problem
    echo 3. Remote repository conflicts
    echo.
    echo Please check the error message above and try again.
    echo.
)

echo.
echo ========================================================
echo Commit History (last 5 commits):
echo ========================================================
git log --oneline -5

echo.
echo ========================================================
echo Current Repository Status:
echo ========================================================
git status

echo.
pause




