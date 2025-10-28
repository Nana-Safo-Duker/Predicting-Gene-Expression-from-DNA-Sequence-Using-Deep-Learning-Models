# Cardiovascular Risk Prediction from Retinal Images

## Overview

This project provides a comprehensive analysis and visualization suite for understanding deep learning-based cardiovascular risk prediction from retinal fundus photographs. The work is based on the groundbreaking research published in *Nature Biomedical Engineering* by Poplin et al. (2018): **"Prediction of cardiovascular risk factors from retinal fundus photographs via deep learning"**.

## Research Background

Cardiovascular diseases (CVD) are the leading cause of death globally. Traditional risk assessment requires invasive blood tests and expensive clinical procedures. This research demonstrates that deep learning algorithms can analyze retinal fundus photographs to predict:

- **Cardiovascular Risk Factors:**
  - Age (MAE ≈ 3.26 years)
  - Gender (AUC ≈ 0.97)
  - Smoking status (AUC ≈ 0.71)
  - Systolic blood pressure (correlation r ≈ 0.33)
  - Body Mass Index (correlation r ≈ 0.25)

- **Clinical Outcomes:**
  - Major Adverse Cardiac Events (MACE) within 5 years (AUC ≈ 0.70)

## Repository Contents

### 1. **cardiovascular_prediction_visualization.ipynb**
Interactive Jupyter notebook featuring:
- Simulated dataset generation matching published statistics
- Age prediction performance analysis
- ROC curves for binary classifications
- Calibration plots for probability predictions
- Continuous variable prediction analysis (SBP, BMI)
- Risk stratification visualizations
- Simulated attention map demonstrations
- Statistical significance testing
- Clinical decision support dashboard
- Comprehensive performance summaries

### 2. **cardiovascular_visualization.py**
Standalone Python script for automated visualization generation. Includes modular functions for:
- Data simulation
- Performance metric calculation
- Multi-panel figure generation
- Statistical analysis
- Batch processing of all visualizations
- **Robust fallback** for edge cases (ensures all 4 panels display)

### 3. **cardiovascular_visualization_complete.py**
Advanced object-oriented Python script with comprehensive features:
- CardiovascularRiskAnalyzer class for systematic analysis
- All features from the simple script plus:
  - Interactive Plotly dashboards
  - Comprehensive reporting
  - Advanced data generation matching paper statistics
  - Graceful fallbacks for optional dependencies (cv2, plotly)
- Production-ready with extensive error handling

### 4. **cardiovascular_visualization.R**
R script providing equivalent functionality for R users:
- All visualizations from Python scripts
- Uses ggplot2 for publication-quality figures
- dplyr for data manipulation
- Generates separate output files (e.g., `attention_maps_R.png`)
- Includes fallback mechanisms for robust execution

### 5. **README.md** (this file)
Complete project documentation and usage instructions.

## Recent Updates & Improvements

### ✅ Attention Map Fix (October 2024)
**Issue Resolved:** Fixed blank 4th panel in attention map visualizations

**Root Cause:** Patient risk grouping could sometimes create fewer than 4 groups, resulting in blank panels.

**Solution Implemented:**
- **Fallback Mechanism:** All scripts now automatically select evenly-spaced patients if binning fails
- **Panel Management:** Unused panels are hidden gracefully
- **Robust Execution:** Guaranteed 4-panel output in all scenarios

**Files Updated:**
- ✅ `cardiovascular_visualization.py` - Added fallback logic (lines 410-416, 437-448)
- ✅ `cardiovascular_visualization_complete.py` - Changed binning method + fallback (lines 185-201, 608-612)
- ✅ `cardiovascular_visualization.R` - Added fallback + empty plot handling (lines 397-404, 437-441)
- ✅ `cardiovascular_prediction_visualization.ipynb` - Added fallback to Cell 15

**Result:** All attention map visualizations now display 4 complete panels with risk scores!

## Installation & Dependencies

### Required Python Packages

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```

### Detailed Requirements

- **Python:** 3.7 or higher
- **NumPy:** ≥ 1.19.0 (numerical computing)
- **Pandas:** ≥ 1.2.0 (data manipulation)
- **Matplotlib:** ≥ 3.3.0 (visualization)
- **Seaborn:** ≥ 0.11.0 (statistical visualization)
- **SciPy:** ≥ 1.6.0 (statistical analysis)
- **Scikit-learn:** ≥ 0.24.0 (machine learning metrics)
- **Jupyter:** ≥ 1.0.0 (notebook interface, optional)

### Installation via pip

```bash
# Create virtual environment (recommended)
python -m venv cardio_env
source cardio_env/bin/activate  # On Windows: cardio_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Installation via conda

```bash
# Create conda environment
conda create -n cardio_env python=3.9
conda activate cardio_env

# Install packages
conda install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```

## Usage Instructions

### Option 1: Jupyter Notebook (Interactive)

1. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook cardiovascular_prediction_visualization.ipynb
   ```

2. **Execute cells sequentially** using Shift+Enter or Cell → Run All

3. **Explore interactive features:**
   - Modify simulation parameters (n_patients, random_seed)
   - Adjust visualization styles
   - Generate custom risk assessment dashboards
   - Export figures in various formats

### Option 2: Python Script (Automated)

1. **Run the complete visualization pipeline:**
   ```bash
   python cardiovascular_visualization.py
   ```

2. **Generated output files:**
   - `age_prediction_performance.png`
   - `roc_curves.png`
   - `calibration_curves.png`
   - `continuous_predictions.png`
   - `risk_stratification.png`
   - `attention_map.png`

3. **Customize execution:**
   ```python
   from cardiovascular_visualization import *
   
   # Generate data with custom parameters
   df = generate_simulated_data(n_patients=50000, random_seed=123)
   
   # Generate specific visualizations
   plot_age_prediction(df, save=True)
   plot_roc_curves(df, save=True)
   ```

### Option 3: R Integration (Optional)

For researchers preferring R, a conversion script is available:

```r
# Install required packages
install.packages(c("reticulate", "ggplot2", "pROC", "caret"))

# Load Python script via reticulate
library(reticulate)
source_python("cardiovascular_visualization.py")

# Or use R-native implementation (see cardiovascular_analysis.R)
source("cardiovascular_analysis.R")
run_analysis()
```

## Key Visualizations Explained

### 1. Age Prediction Performance
- **Scatter plot:** Shows correlation between true and predicted age
- **Error distribution:** Histogram of prediction errors
- **Statistics:** Mean Absolute Error (MAE), Pearson correlation coefficient

### 2. ROC Curves
- **Purpose:** Evaluate binary classification performance
- **Metrics:** Area Under the Curve (AUC-ROC)
- **Interpretation:** 
  - AUC = 1.0: Perfect classification
  - AUC = 0.5: Random guessing
  - AUC > 0.7: Clinically useful

### 3. Calibration Curves
- **Purpose:** Assess whether predicted probabilities match observed frequencies
- **Ideal result:** Points fall along diagonal line
- **Clinical importance:** Ensures risk predictions are reliable for decision-making

### 4. Risk Stratification
- **Purpose:** Demonstrate clinical utility by grouping patients into risk categories
- **Categories:** Low (<5%), Moderate (5-10%), High (10-20%), Very High (>20%)
- **Validation:** Observed MACE rates should increase with risk category

### 5. Attention Maps
- **Purpose:** Visualize which retinal regions the model focuses on
- **Biological plausibility:** Should highlight vascular architecture, optic disc
- **Clinical value:** Builds trust by showing interpretable patterns

### 6. Continuous Variable Predictions
- **Bland-Altman plots:** Assess agreement between predicted and true values
- **Scatter plots:** Show linear relationships
- **Applications:** Systolic blood pressure, BMI prediction

## Data Simulation Details

Since the actual UK Biobank and EyePACS datasets are not publicly available, this project uses **realistic simulated data** that matches the statistical properties reported in the original paper:

### Simulation Parameters

| Variable | Distribution | Parameters | Source |
|----------|-------------|------------|---------|
| Age | Normal | μ=60, σ=12 years | UK Biobank demographics |
| Gender | Binomial | p=0.52 (male) | Population statistics |
| Smoking | Binomial | p=0.18 | CDC prevalence data |
| SBP | Normal | μ=135, σ=18 mmHg | NHANES database |
| BMI | Normal | μ=27, σ=4.5 kg/m² | WHO obesity data |
| MACE | Risk model | See code | Cox regression simulation |

### Prediction Error Injection

To simulate model performance:
- **Age:** Add Gaussian noise (σ=3.26 years) to achieve MAE reported in paper
- **Gender:** High accuracy (AUC ≈ 0.97) via minimal noise
- **Smoking:** Moderate accuracy (AUC ≈ 0.71) via increased noise
- **SBP/BMI:** Correlation-based prediction with residual variation
- **MACE:** Logistic model with age, smoking, SBP, BMI as risk factors

## Statistical Methods

### Implemented Tests

1. **Pearson Correlation Coefficient**
   - Measures linear relationship strength
   - Range: -1 (perfect negative) to +1 (perfect positive)
   - Used for: Age, SBP, BMI predictions

2. **Mean Absolute Error (MAE)**
   - Average absolute difference between predictions and truth
   - Units: Same as measured variable
   - Interpretation: Lower = better

3. **ROC-AUC (Receiver Operating Characteristic)**
   - Evaluates classifier performance across all thresholds
   - Range: 0.5 (random) to 1.0 (perfect)
   - Used for: Gender, smoking, MACE predictions

4. **Calibration Curves**
   - Compares predicted probabilities to observed frequencies
   - Good calibration: Points near diagonal
   - Clinical importance: Reliable risk estimates

5. **Independent T-tests**
   - Compares means between two groups
   - Null hypothesis: No difference
   - Used for: Age differences between MACE vs non-MACE groups

6. **Chi-square Test**
   - Assesses association between categorical variables
   - Null hypothesis: Variables are independent
   - Used for: Smoking status vs MACE association

7. **Odds Ratios**
   - Quantifies relative risk between groups
   - OR > 1: Increased risk
   - Used for: High risk vs low risk MACE comparison

## Clinical Implications

### Potential Applications

1. **Primary Care Screening**
   - Cardiovascular risk assessment during routine checkups
   - No blood tests required
   - Accessible in resource-limited settings

2. **Optometry Integration**
   - Dual-purpose screening (eye health + cardiovascular risk)
   - Existing imaging infrastructure
   - Opportunity for early detection

3. **Population Health Programs**
   - Large-scale screening campaigns
   - Risk stratification for intervention targeting
   - Cost-effective prevention strategies

4. **Telehealth & Mobile Health**
   - Portable retinal cameras with embedded AI
   - Remote screening in underserved areas
   - Smartphone-based applications (future development)

### Clinical Decision Support

The project includes a **Clinical Decision Support Dashboard** that demonstrates how AI predictions could integrate into clinical workflows:

- **Risk gauge:** Visual representation of MACE probability
- **Risk factor comparison:** Predicted vs actual values
- **Actionable recommendations:** Risk-stratified clinical guidance
  - Low risk: Routine maintenance
  - Moderate risk: Comprehensive assessment
  - High risk: Cardiology referral
  - Very high risk: Urgent intervention

## Limitations & Considerations

### Data Limitations
- **Simulated data:** Not real patient data; for educational purposes only
- **Selection bias:** Original study participants were healthy enough for imaging
- **Generalizability:** May not perform equally across all demographics

### Model Limitations
- **Observational design:** Cannot prove causation
- **Static images:** Don't capture dynamic cardiovascular changes
- **Interpretability:** Deep learning models are partially "black boxes"

### Clinical Limitations
- **Validation needed:** Prospective clinical trials required before deployment
- **Image quality:** Requires high-quality retinal photographs
- **Complementary tool:** Should augment, not replace, clinical judgment

### Ethical Considerations
- **Algorithmic bias:** Performance may vary across ethnic groups
- **Privacy concerns:** Biometric data security
- **Access equity:** Risk of widening healthcare disparities
- **Informed consent:** Patients must understand AI involvement

## Future Directions

### Technical Enhancements
1. **Explainable AI:** Develop methods to interpret model decisions
2. **Multi-modal integration:** Combine retinal images with EHR data
3. **Longitudinal analysis:** Predict trajectory of cardiovascular health
4. **Transfer learning:** Adapt models to diverse populations

### Clinical Translation
1. **Prospective trials:** Validate effectiveness in real-world settings
2. **Standardization:** Develop imaging protocols optimized for AI
3. **Workflow integration:** Seamless EHR incorporation
4. **Regulatory approval:** FDA/CE marking for clinical use

### Research Extensions
1. **Other diseases:** Chronic kidney disease, neurodegenerative disorders
2. **Treatment response:** Predict medication effectiveness
3. **Biological mechanisms:** Identify causal pathways linking retina to CVD
4. **Federated learning:** Train models across institutions while preserving privacy

## References

### Primary Research Article

Poplin, R., Varadarajan, A. V., Blumer, K., Liu, Y., McConnell, M. V., Corrado, G. S., Peng, L., & Webster, D. R. (2018). **Prediction of cardiovascular risk factors from retinal fundus photographs via deep learning.** *Nature Biomedical Engineering*, 2(3), 158-164. https://doi.org/10.1038/s41551-018-0195-0

### Supporting Literature

1. **Deep Learning Foundations:**
   - LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444.
   - Szegedy, C., et al. (2016). Rethinking the Inception architecture for computer vision. *CVPR*.

2. **Medical AI Applications:**
   - Gulshan, V., et al. (2016). Development and validation of a deep learning algorithm for detection of diabetic retinopathy. *JAMA*, 316(22), 2402-2410.
   - Topol, E. J. (2019). High-performance medicine: the convergence of human and artificial intelligence. *Nature Medicine*, 25(1), 44-56.

3. **Cardiovascular Epidemiology:**
   - D'Agostino, R. B., et al. (2008). General cardiovascular risk profile for use in primary care: the Framingham Heart Study. *Circulation*, 117(6), 743-753.
   - Wong, T. Y., & Mitchell, P. (2007). The eye in hypertension. *The Lancet*, 369(9559), 425-435.

4. **Statistics & Methodology:**
   - Hanley, J. A., & McNeil, B. J. (1982). The meaning and use of the area under a ROC curve. *Radiology*, 143(1), 29-36.
   - Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet*, 327(8476), 307-310.

## Project Structure

```
cardiovascular-risk-prediction/
│
├── cardiovascular_prediction_visualization.ipynb   # Interactive notebook
├── cardiovascular_visualization.py                 # Simple Python script
├── cardiovascular_visualization_complete.py        # Advanced Python script (OOP)
├── cardiovascular_visualization.R                  # R implementation
├── README.md                                       # This file
├── requirements.txt                                # Python dependencies
├── VISUALIZATION_COMPARISON.md                     # Script comparison guide
│
├── outputs/                                        # Generated visualizations
│   ├── age_prediction_performance.png
│   ├── roc_curves.png
│   ├── calibration_curves.png
│   ├── continuous_predictions.png
│   ├── risk_stratification.png
│   ├── attention_maps.png
│   ├── interactive_dashboard.html
│   └── cardiovascular_analysis_report.txt
│
└── docs/                                           # Documentation
    ├── PROJECT_SUMMARY.md
    ├── FINAL_SYNCHRONIZATION_SUMMARY.md
    ├── ALL_SCRIPTS_FIX_SUMMARY.md
    └── COMPLETE_PROJECT_CHECKLIST.md
```

## Troubleshooting

### Common Issues

**Issue:** `ImportError: No module named 'sklearn'`
```bash
# Solution: Install scikit-learn
pip install scikit-learn
```

**Issue:** Matplotlib style warning `'seaborn-v0_8-darkgrid' not found`
```python
# Solution: Use alternative style
plt.style.use('seaborn-darkgrid')  # Older seaborn
# or
plt.style.use('default')  # Default matplotlib
```

**Issue:** Figures not displaying in Jupyter
```python
# Solution: Add magic command
%matplotlib inline
```

**Issue:** Memory error with large datasets
```python
# Solution: Reduce sample size
df = generate_simulated_data(n_patients=5000)  # Instead of 10000
```

### Performance Optimization

For large-scale analyses:
```python
# Use vectorized operations
import numpy as np
# Prefer np.array operations over pandas.apply()

# Parallelize computations
from joblib import Parallel, delayed
# Use for independent visualizations

# Reduce figure DPI for faster rendering
plt.savefig('figure.png', dpi=150)  # Instead of 300
```

## Contributing

This project was created as part of a bioinformatics research paper summary assignment. Contributions, suggestions, and feedback are welcome!

### How to Contribute

1. **Report Issues:** Found a bug? Open an issue on GitHub
2. **Suggest Enhancements:** Ideas for new visualizations or analyses
3. **Submit Pull Requests:** Code improvements, documentation updates
4. **Share Applications:** Novel uses of the visualization toolkit

### Contribution Guidelines

- Follow PEP 8 style for Python code
- Include docstrings for all functions
- Add comments for complex logic
- Test visualizations across different dataset sizes
- Update README.md for new features

## License

This project is provided for **educational and research purposes**. The code is released under the MIT License:

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Note:** The original research paper is copyrighted by Nature Publishing Group. This project only provides analysis and visualization tools inspired by that research.

## Acknowledgments

- **Original Research:** Poplin et al. (2018) and the Google Research team
- **Data Sources:** UK Biobank, EyePACS (simulated data used in this project)
- **Tools & Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn communities
- **Educational Support:** Bioinformatics course instructors and teaching assistants
- **AI Assistance:** Claude (Anthropic) and ChatGPT (OpenAI) for writing support

## Contact & Support

For questions, suggestions, or collaboration inquiries:

- **Email:** [your.email@example.com]
- **GitHub:** [github.com/yourusername/cardiovascular-risk-prediction]
- **LinkedIn:** [linkedin.com/in/yourprofile]

## Citation

If you use this code or visualization toolkit in your research or educational materials, please cite:

```bibtex
@misc{cardiovascular_viz_2024,
  author = {[Your Name]},
  title = {Cardiovascular Risk Prediction from Retinal Images: Visualization Toolkit},
  year = {2024},
  howpublished = {GitHub repository},
  url = {https://github.com/yourusername/cardiovascular-risk-prediction}
}
```

And please cite the original research:

```bibtex
@article{poplin2018prediction,
  title={Prediction of cardiovascular risk factors from retinal fundus photographs via deep learning},
  author={Poplin, Ryan and Varadarajan, Avinash V and Blumer, Katy and Liu, Yun and McConnell, Michael V and Corrado, Greg S and Peng, Lily and Webster, Dale R},
  journal={Nature Biomedical Engineering},
  volume={2},
  number={3},
  pages={158--164},
  year={2018},
  publisher={Nature Publishing Group}
}
```

---

**Last Updated:** October 27, 2024  
**Version:** 1.0.0  
**Status:** Educational Project - Not for Clinical Use

---

## Additional Resources

### Online Tools & Datasets

- **UK Biobank:** https://www.ukbiobank.ac.uk/
- **EyePACS:** https://www.eyepacs.com/
- **Kaggle Diabetic Retinopathy Dataset:** https://www.kaggle.com/c/diabetic-retinopathy-detection
- **DRIVE Retinal Dataset:** https://drive.grand-challenge.org/

### Educational Materials

- **Coursera - AI for Medicine:** https://www.coursera.org/specializations/ai-for-medicine
- **Fast.ai Deep Learning Course:** https://www.fast.ai/
- **Stanford CS231n:** http://cs231n.stanford.edu/

### Related Tools

- **TensorFlow/Keras:** Deep learning frameworks
- **PyTorch:** Alternative deep learning framework
- **OpenCV:** Image processing
- **ITK-SNAP:** Medical image visualization

---

**Thank you for exploring this project! We hope these tools enhance your understanding of AI applications in cardiovascular medicine.**

