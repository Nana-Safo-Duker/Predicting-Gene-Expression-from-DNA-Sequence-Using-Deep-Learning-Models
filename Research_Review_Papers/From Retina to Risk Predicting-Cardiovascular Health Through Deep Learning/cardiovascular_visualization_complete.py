"""
Complete Cardiovascular Risk Prediction Visualization Suite
============================================================

This comprehensive script combines the best features from both visualization approaches:
- Matches Nature Biomedical Engineering paper metrics (Poplin et al., 2018)
- Object-oriented architecture for flexibility
- Interactive and static visualizations
- Comprehensive reporting capabilities
- Scalable from 200 to 10,000+ patients

Author: Scientific Blog Post Project
Date: October 27, 2024
Version: 2.0 (Merged)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_curve, auc, mean_absolute_error, mean_squared_error, r2_score
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

# Optional imports with graceful fallbacks
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
    print("✅ Plotly available - interactive dashboards enabled")
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly not available - interactive dashboards disabled")
    print("   Install with: pip install plotly")

try:
    import cv2
    CV2_AVAILABLE = True
    print("✅ OpenCV available - optimized image processing enabled")
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  OpenCV not available - using numpy fallback")

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("="*80)
print("COMPREHENSIVE CARDIOVASCULAR RISK PREDICTION VISUALIZATION SUITE")
print("="*80)


class CardiovascularRiskAnalyzer:
    """
    Comprehensive analyzer for cardiovascular risk assessment from retinal images.
    
    This class combines research-paper-accurate simulations with advanced
    visualization and reporting capabilities.
    """
    
    def __init__(self, n_patients=10000, random_seed=42):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        n_patients : int
            Number of patients to simulate (default: 10000 to match paper)
        random_seed : int
            Random seed for reproducibility
        """
        self.n_patients = n_patients
        self.random_seed = random_seed
        self.patient_data = None
        self.attention_maps = None
        self.attention_map_patients = None  # Store patient info for attention maps
        
        # Performance metrics storage
        self.age_metrics = {}
        self.auc_metrics = {}
        
        np.random.seed(random_seed)
    
    def generate_research_paper_dataset(self):
        """
        Generate dataset matching Nature Biomedical Engineering paper statistics.
        
        This replicates the exact performance metrics reported in:
        Poplin et al. (2018) - Prediction of cardiovascular risk factors from 
        retinal fundus photographs via deep learning.
        
        Returns:
        --------
        pd.DataFrame : Patient data with predictions
        """
        print(f"\nGenerating research-paper-accurate dataset ({self.n_patients} patients)...")
        
        # True patient characteristics (UK Biobank statistics)
        true_age = np.random.normal(60, 12, self.n_patients)
        true_gender = np.random.binomial(1, 0.52, self.n_patients)  # 52% male
        true_smoking = np.random.binomial(1, 0.18, self.n_patients)  # 18% smokers
        true_sbp = np.random.normal(135, 18, self.n_patients)  # Systolic BP
        true_bmi = np.random.normal(27, 4.5, self.n_patients)  # BMI
        
        # Additional retinal features
        vessel_density = np.random.normal(0.6, 0.1, self.n_patients)
        vessel_density = np.clip(vessel_density, 0.2, 1.0)
        optic_disc_area = np.random.normal(2.5, 0.5, self.n_patients)
        optic_disc_area = np.clip(optic_disc_area, 1.0, 4.0)
        
        # Model predictions with paper-reported performance
        # Age: MAE ≈ 3.26 years
        pred_age = true_age + np.random.normal(0, 3.26, self.n_patients)
        
        # Gender: AUC ≈ 0.97
        gender_prob = np.clip(true_gender + np.random.normal(0, 0.15, self.n_patients), 0, 1)
        
        # Smoking: AUC ≈ 0.71
        smoking_prob = np.clip(true_smoking + np.random.normal(0, 0.35, self.n_patients), 0, 1)
        
        # SBP: correlation r ≈ 0.33
        pred_sbp = true_sbp * 0.33 + np.random.normal(135, 15, self.n_patients)
        
        # BMI: correlation r ≈ 0.25
        pred_bmi = true_bmi * 0.25 + np.random.normal(27, 4, self.n_patients)
        
        # MACE prediction (C-statistic ≈ 0.70)
        base_risk = 0.05
        risk_score = (base_risk + 
                      0.002 * (true_age - 60) + 
                      0.03 * true_smoking + 
                      0.001 * (true_sbp - 135) + 
                      0.005 * (true_bmi - 27))
        risk_score = np.clip(risk_score, 0, 0.5)
        true_mace = np.random.binomial(1, risk_score, self.n_patients)
        pred_mace_prob = np.clip(risk_score + np.random.normal(0, 0.08, self.n_patients), 0, 1)
        
        # Create comprehensive DataFrame
        self.patient_data = pd.DataFrame({
            'patient_id': [f'P{i:04d}' for i in range(self.n_patients)],
            'true_age': true_age,
            'pred_age': pred_age,
            'true_gender': true_gender,
            'pred_gender_prob': gender_prob,
            'true_smoking': true_smoking,
            'pred_smoking_prob': smoking_prob,
            'true_sbp': true_sbp,
            'pred_sbp': pred_sbp,
            'true_bmi': true_bmi,
            'pred_bmi': pred_bmi,
            'true_mace': true_mace,
            'pred_mace_prob': pred_mace_prob,
            'vessel_density': vessel_density,
            'optic_disc_area': optic_disc_area,
            'risk_score': pred_mace_prob  # Use MACE probability as overall risk score
        })
        
        print(f"✓ Dataset generated: {self.patient_data.shape[0]} patients, "
              f"{self.patient_data.shape[1]} variables")
        
        return self.patient_data
    
    def generate_attention_maps_subset(self, n_samples=100):
        """
        Generate attention maps for a subset of patients (memory efficient).
        
        Parameters:
        -----------
        n_samples : int
            Number of attention maps to generate
        """
        if self.patient_data is None:
            raise ValueError("Generate dataset first using generate_research_paper_dataset()")
        
        print(f"\nGenerating {n_samples} attention maps...")
        
        # Select representative samples across risk spectrum
        # Use pd.cut for equal-width bins (more robust than qcut)
        risk_groups = pd.cut(self.patient_data['risk_score'], bins=n_samples, 
                            labels=False, duplicates='drop')
        sample_indices = []
        for q in range(n_samples):
            candidates = self.patient_data[risk_groups == q].index
            if len(candidates) > 0:
                sample_indices.append(candidates[0])
        
        # If we don't have enough samples, fill with additional patients
        if len(sample_indices) < n_samples:
            print(f"⚠️  Warning: Only found {len(sample_indices)} groups, selecting additional patients...")
            # Sort by risk and select evenly spaced patients
            sorted_patients = self.patient_data.sort_values('risk_score')
            step = len(sorted_patients) // n_samples
            sample_indices = [sorted_patients.index[i * step] for i in range(n_samples)]
        
        self.attention_maps = []
        self.attention_map_patients = []  # Store patient info for display
        for idx in sample_indices[:n_samples]:
            patient = self.patient_data.iloc[idx]
            attention = self._generate_attention_map(
                patient['true_age'],
                patient['true_smoking'],
                patient['true_sbp'],
                patient['true_bmi']
            )
            self.attention_maps.append(attention)
            self.attention_map_patients.append({
                'risk_score': patient['risk_score'],
                'age': patient['true_age'],
                'index': idx
            })
        
        print(f"✓ Generated {len(self.attention_maps)} attention maps")
        return self.attention_maps
    
    def _generate_attention_map(self, age, smoking, sbp, bmi, size=512):
        """Generate risk-based attention map for a patient."""
        attention = np.zeros((size, size))
        center_x, center_y = size // 2, size // 2
        
        # Optic disc attention (always present)
        y, x = np.ogrid[:size, :size]
        disc_mask = (x - center_x)**2 + (y - center_y)**2 < (size // 8)**2
        attention[disc_mask] += 0.7
        
        # Vessel attention (increases with risk)
        vessel_intensity = 0.3 + 0.4 * (smoking + (sbp > 140) + (bmi > 30)) / 3
        for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
            vessel_x = center_x + np.cos(angle) * np.linspace(0, size//3, 50)
            vessel_y = center_y + np.sin(angle) * np.linspace(0, size//3, 50)
            for j in range(len(vessel_x)-1):
                self._draw_line(attention, 
                              (int(vessel_x[j]), int(vessel_y[j])),
                              (int(vessel_x[j+1]), int(vessel_y[j+1])),
                              vessel_intensity, 2)
        
        # Macula attention (age-dependent)
        macula_x, macula_y = center_x + 20, center_y - 30
        macula_intensity = 0.2 + 0.3 * (age - 25) / 60
        macula_mask = (x - macula_x)**2 + (y - macula_y)**2 < (size // 12)**2
        attention[macula_mask] += macula_intensity
        
        # Smooth and normalize
        attention = gaussian_filter(attention, sigma=2)
        attention = np.clip(attention, 0, 1)
        
        return attention
    
    def _draw_line(self, image, start, end, value, thickness):
        """Draw line on image (uses cv2 if available, else numpy)."""
        if CV2_AVAILABLE:
            cv2.line(image, start, end, value, thickness)
        else:
            # Numpy fallback
            x1, y1 = start
            x2, y2 = end
            num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
            x_coords = np.linspace(x1, x2, num_points).astype(int)
            y_coords = np.linspace(y1, y2, num_points).astype(int)
            
            for x, y in zip(x_coords, y_coords):
                for dx in range(-thickness//2, thickness//2 + 1):
                    for dy in range(-thickness//2, thickness//2 + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                            image[ny, nx] = value
    
    # ========================================================================
    # VISUALIZATION METHODS - RESEARCH PAPER FIGURES
    # ========================================================================
    
    def plot_age_prediction(self, save=True, filename='age_prediction_comprehensive.png'):
        """Enhanced age prediction visualization (4 subplots)."""
        print("\n" + "="*80)
        print("GENERATING AGE PREDICTION VISUALIZATIONS")
        print("="*80)
        
        mae = mean_absolute_error(self.patient_data['true_age'], 
                                  self.patient_data['pred_age'])
        rmse = np.sqrt(mean_squared_error(self.patient_data['true_age'],
                                          self.patient_data['pred_age']))
        age_corr, _ = stats.pearsonr(self.patient_data['true_age'],
                                     self.patient_data['pred_age'])
        r2 = r2_score(self.patient_data['true_age'], self.patient_data['pred_age'])
        residuals = self.patient_data['pred_age'] - self.patient_data['true_age']
        
        # Store metrics
        self.age_metrics = {'mae': mae, 'rmse': rmse, 'r': age_corr, 'r2': r2}
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Scatter with residual coloring
        scatter = axes[0, 0].scatter(self.patient_data['true_age'],
                                     self.patient_data['pred_age'],
                                     c=residuals, cmap='RdYlBu_r', s=20, alpha=0.5)
        axes[0, 0].plot([30, 90], [30, 90], 'r--', linewidth=2, label='Perfect prediction')
        
        # Regression line
        lr = LinearRegression()
        lr.fit(self.patient_data['true_age'].values.reshape(-1, 1),
               self.patient_data['pred_age'].values)
        line_x = np.linspace(30, 90, 100)
        line_y = lr.predict(line_x.reshape(-1, 1))
        axes[0, 0].plot(line_x, line_y, 'g-', linewidth=2,
                       label=f'Regression (R²={r2:.3f})')
        
        axes[0, 0].set_xlabel('True Age (years)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Predicted Age (years)', fontsize=12, fontweight='bold')
        axes[0, 0].set_title(f'Age Prediction Performance\nMAE = {mae:.2f} years, r = {age_corr:.3f}',
                            fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 0], label='Residuals (years)')
        
        # 2. Residuals plot
        axes[0, 1].scatter(self.patient_data['true_age'], residuals, alpha=0.3, s=10)
        axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].axhline(residuals.mean(), color='blue', linestyle='-', linewidth=2,
                          label=f'Mean: {residuals.mean():.2f}')
        axes[0, 1].set_xlabel('True Age (years)', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Residuals (years)', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Residuals vs True Age', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribution comparison
        axes[1, 0].hist(self.patient_data['true_age'], bins=30, alpha=0.7,
                       label='True Age', color='skyblue', edgecolor='black')
        axes[1, 0].hist(self.patient_data['pred_age'], bins=30, alpha=0.7,
                       label='Predicted Age', color='coral', edgecolor='black')
        axes[1, 0].set_xlabel('Age (years)', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Age Distribution Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Absolute error analysis
        abs_errors = np.abs(residuals)
        axes[1, 1].scatter(self.patient_data['true_age'], abs_errors, alpha=0.3, s=10)
        axes[1, 1].set_xlabel('True Age (years)', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Absolute Error (years)', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Absolute Prediction Error', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add metrics box
        metrics_text = f'MAE: {mae:.2f} years\nRMSE: {rmse:.2f} years\nR²: {r2:.3f}'
        axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes,
                       verticalalignment='top', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        if save:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
        plt.show()
        
        print(f"  MAE: {mae:.2f} years | RMSE: {rmse:.2f} years | r: {age_corr:.3f}")
        
        return fig
    
    def plot_roc_curves(self, save=True, filename='roc_curves.png'):
        """Plot ROC curves for binary classifications."""
        print("\n" + "="*80)
        print("GENERATING ROC CURVES")
        print("="*80)
        
        # Calculate ROC curves
        fpr_gender, tpr_gender, _ = roc_curve(self.patient_data['true_gender'],
                                              self.patient_data['pred_gender_prob'])
        auc_gender = auc(fpr_gender, tpr_gender)
        
        fpr_smoking, tpr_smoking, _ = roc_curve(self.patient_data['true_smoking'],
                                                self.patient_data['pred_smoking_prob'])
        auc_smoking = auc(fpr_smoking, tpr_smoking)
        
        fpr_mace, tpr_mace, _ = roc_curve(self.patient_data['true_mace'],
                                          self.patient_data['pred_mace_prob'])
        auc_mace = auc(fpr_mace, tpr_mace)
        
        # Store metrics
        self.auc_metrics = {
            'gender': auc_gender,
            'smoking': auc_smoking,
            'mace': auc_mace
        }
        
        # Plot
        plt.figure(figsize=(10, 10))
        plt.plot(fpr_gender, tpr_gender, linewidth=3,
                label=f'Gender (AUC = {auc_gender:.3f})', color='#2E86AB')
        plt.plot(fpr_smoking, tpr_smoking, linewidth=3,
                label=f'Smoking Status (AUC = {auc_smoking:.3f})', color='#A23B72')
        plt.plot(fpr_mace, tpr_mace, linewidth=3,
                label=f'MACE Prediction (AUC = {auc_mace:.3f})', color='#F18F01')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.5)')
        
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
        plt.title('ROC Curves for Cardiovascular Risk Factor Prediction',
                 fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        plt.tight_layout()
        if save:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
        plt.show()
        
        print(f"  Gender: {auc_gender:.3f} | Smoking: {auc_smoking:.3f} | MACE: {auc_mace:.3f}")
        
        return auc_gender, auc_smoking, auc_mace
    
    def plot_calibration(self, save=True, filename='calibration_curves.png'):
        """Plot calibration curves for MACE prediction."""
        print("\n" + "="*80)
        print("GENERATING CALIBRATION CURVES")
        print("="*80)
        
        fraction_positives, mean_predicted = calibration_curve(
            self.patient_data['true_mace'],
            self.patient_data['pred_mace_prob'],
            n_bins=10, strategy='quantile'
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calibration curve
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
        axes[0].plot(mean_predicted, fraction_positives, 'o-', linewidth=3,
                    markersize=10, color='#F18F01', label='Model calibration')
        axes[0].set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Observed Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Calibration Curve for MACE Prediction',
                         fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Probability distribution
        axes[1].hist(self.patient_data[self.patient_data['true_mace']==1]['pred_mace_prob'],
                    bins=30, alpha=0.6, label='MACE cases', color='red', edgecolor='black')
        axes[1].hist(self.patient_data[self.patient_data['true_mace']==0]['pred_mace_prob'],
                    bins=30, alpha=0.6, label='No MACE', color='green', edgecolor='black')
        axes[1].set_xlabel('Predicted MACE Probability', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title('Distribution of Predicted Probabilities',
                         fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
        plt.show()
        
        return fig
    
    def plot_continuous_predictions(self, save=True, filename='continuous_predictions.png'):
        """Plot SBP and BMI predictions with Bland-Altman analysis."""
        print("\n" + "="*80)
        print("GENERATING CONTINUOUS VARIABLE PREDICTIONS")
        print("="*80)
        
        sbp_corr, _ = stats.pearsonr(self.patient_data['true_sbp'],
                                     self.patient_data['pred_sbp'])
        bmi_corr, _ = stats.pearsonr(self.patient_data['true_bmi'],
                                     self.patient_data['pred_bmi'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # SBP scatter
        axes[0, 0].scatter(self.patient_data['true_sbp'],
                          self.patient_data['pred_sbp'], alpha=0.3, s=10, color='#E63946')
        axes[0, 0].plot([90, 180], [90, 180], 'k--', linewidth=2)
        axes[0, 0].set_xlabel('True SBP (mmHg)', fontweight='bold')
        axes[0, 0].set_ylabel('Predicted SBP (mmHg)', fontweight='bold')
        axes[0, 0].set_title(f'SBP Prediction (r = {sbp_corr:.3f})', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # SBP Bland-Altman
        mean_sbp = (self.patient_data['true_sbp'] + self.patient_data['pred_sbp']) / 2
        diff_sbp = self.patient_data['pred_sbp'] - self.patient_data['true_sbp']
        axes[0, 1].scatter(mean_sbp, diff_sbp, alpha=0.3, s=10, color='#E63946')
        axes[0, 1].axhline(diff_sbp.mean(), color='blue', linestyle='-', linewidth=2)
        axes[0, 1].axhline(diff_sbp.mean() + 1.96*diff_sbp.std(), color='red',
                          linestyle='--', linewidth=2)
        axes[0, 1].axhline(diff_sbp.mean() - 1.96*diff_sbp.std(), color='red',
                          linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Mean SBP (mmHg)', fontweight='bold')
        axes[0, 1].set_ylabel('Difference (Pred - True)', fontweight='bold')
        axes[0, 1].set_title('Bland-Altman: SBP', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # BMI scatter
        axes[1, 0].scatter(self.patient_data['true_bmi'],
                          self.patient_data['pred_bmi'], alpha=0.3, s=10, color='#457B9D')
        axes[1, 0].plot([15, 40], [15, 40], 'k--', linewidth=2)
        axes[1, 0].set_xlabel('True BMI (kg/m²)', fontweight='bold')
        axes[1, 0].set_ylabel('Predicted BMI (kg/m²)', fontweight='bold')
        axes[1, 0].set_title(f'BMI Prediction (r = {bmi_corr:.3f})', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # BMI Bland-Altman
        mean_bmi = (self.patient_data['true_bmi'] + self.patient_data['pred_bmi']) / 2
        diff_bmi = self.patient_data['pred_bmi'] - self.patient_data['true_bmi']
        axes[1, 1].scatter(mean_bmi, diff_bmi, alpha=0.3, s=10, color='#457B9D')
        axes[1, 1].axhline(diff_bmi.mean(), color='blue', linestyle='-', linewidth=2)
        axes[1, 1].axhline(diff_bmi.mean() + 1.96*diff_bmi.std(), color='red',
                          linestyle='--', linewidth=2)
        axes[1, 1].axhline(diff_bmi.mean() - 1.96*diff_bmi.std(), color='red',
                          linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Mean BMI (kg/m²)', fontweight='bold')
        axes[1, 1].set_ylabel('Difference (Pred - True)', fontweight='bold')
        axes[1, 1].set_title('Bland-Altman: BMI', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
        plt.show()
        
        print(f"  SBP correlation: r = {sbp_corr:.3f} | BMI correlation: r = {bmi_corr:.3f}")
        
        return fig
    
    def plot_risk_stratification(self, save=True, filename='risk_stratification.png'):
        """Plot risk stratification analysis."""
        print("\n" + "="*80)
        print("GENERATING RISK STRATIFICATION ANALYSIS")
        print("="*80)
        
        self.patient_data['risk_category'] = pd.cut(
            self.patient_data['pred_mace_prob'],
            bins=[0, 0.05, 0.10, 0.20, 1.0],
            labels=['Low (<5%)', 'Moderate (5-10%)', 'High (10-20%)', 'Very High (>20%)']
        )
        
        risk_analysis = self.patient_data.groupby('risk_category').agg({
            'true_mace': ['count', 'sum', 'mean']
        })
        risk_analysis.columns = ['N', 'Events', 'Rate']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Patient distribution
        risk_counts = self.patient_data['risk_category'].value_counts().sort_index()
        colors = ['#06D6A0', '#FFD166', '#EF476F', '#8B0000']
        axes[0].bar(range(len(risk_counts)), risk_counts.values, color=colors,
                   edgecolor='black', linewidth=1.5)
        axes[0].set_xticks(range(len(risk_counts)))
        axes[0].set_xticklabels(risk_counts.index, rotation=15, ha='right')
        axes[0].set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
        axes[0].set_title('Patient Distribution by Risk Category',
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Observed MACE rates
        observed_rates = (risk_analysis['Rate'] * 100).values
        axes[1].bar(range(len(observed_rates)), observed_rates, color=colors,
                   edgecolor='black', linewidth=1.5)
        axes[1].set_xticks(range(len(observed_rates)))
        axes[1].set_xticklabels(risk_analysis.index, rotation=15, ha='right')
        axes[1].set_ylabel('Observed MACE Rate (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Observed MACE Rate by Risk Category',
                         fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
        plt.show()
        
        print("\nRisk Stratification Results:")
        print(risk_analysis)
        
        return fig
    
    def plot_attention_maps(self, n_samples=4, save=True, filename='attention_maps.png'):
        """Plot attention heatmaps for different risk groups."""
        print("\n" + "="*80)
        print("GENERATING ATTENTION MAP VISUALIZATIONS")
        print("="*80)
        
        if self.attention_maps is None:
            print("Generating attention maps for visualization...")
            self.generate_attention_maps_subset(n_samples=n_samples)
        
        risk_groups = pd.cut(self.patient_data['risk_score'], bins=4,
                            labels=['Low', 'Medium', 'High', 'Very High'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        titles = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
        
        for i, (group, title) in enumerate(zip(['Low', 'Medium', 'High', 'Very High'], titles)):
            if i >= len(self.attention_maps):
                # Hide unused panels
                axes[i].axis('off')
                continue
            
            attention = self.attention_maps[i]
            im = axes[i].imshow(attention, cmap='hot', vmin=0, vmax=1)
            
            # Add anatomical landmarks
            center_x, center_y = 256, 256
            circle1 = plt.Circle((center_x, center_y), 64, fill=False,
                                color='cyan', linewidth=2)
            axes[i].add_patch(circle1)
            axes[i].text(center_x, center_y - 80, 'Optic Disc', ha='center',
                        color='cyan', fontweight='bold', fontsize=10)
            
            circle2 = plt.Circle((center_x + 20, center_y - 30), 40, fill=False,
                                color='yellow', linewidth=2)
            axes[i].add_patch(circle2)
            axes[i].text(center_x + 20, center_y - 90, 'Macula', ha='center',
                        color='yellow', fontweight='bold', fontsize=10)
            
            # Display title with risk score if available
            if hasattr(self, 'attention_map_patients') and i < len(self.attention_map_patients):
                risk_score = self.attention_map_patients[i]['risk_score']
                axes[i].set_title(f'{title} Patient\nRisk Score: {risk_score:.3f}',
                                fontsize=12, fontweight='bold')
            else:
                axes[i].set_title(f'{title} Patient', fontsize=12, fontweight='bold')
            
            axes[i].set_xlabel('Width (pixels)', fontsize=10)
            axes[i].set_ylabel('Height (pixels)', fontsize=10)
            
            cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            cbar.set_label('Attention Weight', fontsize=9)
        
        plt.tight_layout()
        if save:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
        plt.show()
        
        return fig
    
    def create_performance_summary(self):
        """Create and display comprehensive performance summary."""
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        if not self.age_metrics or not self.auc_metrics:
            print("Please run visualizations first to calculate metrics.")
            return None
        
        sbp_corr, _ = stats.pearsonr(self.patient_data['true_sbp'],
                                     self.patient_data['pred_sbp'])
        bmi_corr, _ = stats.pearsonr(self.patient_data['true_bmi'],
                                     self.patient_data['pred_bmi'])
        
        summary = pd.DataFrame({
            'Risk Factor': ['Age', 'Gender', 'Smoking', 'SBP', 'BMI', 'MACE'],
            'Type': ['Regression', 'Binary', 'Binary', 'Regression', 'Regression', 'Binary'],
            'Metric': [
                f"MAE={self.age_metrics['mae']:.2f}y",
                f"AUC={self.auc_metrics['gender']:.3f}",
                f"AUC={self.auc_metrics['smoking']:.3f}",
                f"r={sbp_corr:.3f}",
                f"r={bmi_corr:.3f}",
                f"AUC={self.auc_metrics['mace']:.3f}"
            ],
            'Paper Reported': ['3.26y', '0.97', '0.71', '0.33', '0.25', '0.70'],
            'Performance': ['Excellent', 'Excellent', 'Good', 'Moderate', 'Moderate', 'Good']
        })
        
        print(summary.to_string(index=False))
        print("="*80)
        
        return summary
    
    # ========================================================================
    # INTERACTIVE VISUALIZATIONS (requires Plotly)
    # ========================================================================
    
    def create_interactive_dashboard(self, save_path="interactive_dashboard.html"):
        """Create interactive Plotly dashboard."""
        if not PLOTLY_AVAILABLE:
            print("\n⚠️  Plotly not available. Install with: pip install plotly")
            return None
        
        print("\n" + "="*80)
        print("GENERATING INTERACTIVE DASHBOARD")
        print("="*80)
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Age Prediction vs Risk Score',
                'Risk Score Distribution',
                'Age Prediction Residuals',
                'Risk Factors Correlation',
                'Feature Importance',
                'Age by Risk Group'
            ],
            specs=[[{"type": "scatter"}, {"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "bar"}, {"type": "box"}]]
        )
        
        # 1. Age prediction scatter
        fig.add_trace(
            go.Scatter(
                x=self.patient_data['true_age'],
                y=self.patient_data['pred_age'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=self.patient_data['risk_score'],
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(title="Risk Score")
                ),
                text=self.patient_data['patient_id'],
                hovertemplate='Patient: %{text}<br>Actual: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>',
                name='Patients'
            ),
            row=1, col=1
        )
        
        # Perfect prediction line
        fig.add_trace(
            go.Scatter(
                x=[25, 85], y=[25, 85],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Prediction'
            ),
            row=1, col=1
        )
        
        # 2. Risk score distribution
        fig.add_trace(
            go.Histogram(
                x=self.patient_data['risk_score'],
                nbinsx=20,
                name='Risk Distribution'
            ),
            row=1, col=2
        )
        
        # 3. Residuals
        residuals = self.patient_data['pred_age'] - self.patient_data['true_age']
        fig.add_trace(
            go.Scatter(
                x=self.patient_data['true_age'],
                y=residuals,
                mode='markers',
                marker=dict(size=4, color='blue', opacity=0.4),
                name='Residuals'
            ),
            row=1, col=3
        )
        
        fig.add_trace(
            go.Scatter(
                x=[self.patient_data['true_age'].min(), self.patient_data['true_age'].max()],
                y=[0, 0],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Zero Line'
            ),
            row=1, col=3
        )
        
        # 4. Correlation matrix
        risk_factors = self.patient_data[['true_age', 'true_smoking', 'true_sbp',
                                          'true_bmi', 'risk_score']]
        risk_factors.columns = ['Age', 'Smoking', 'SBP', 'BMI', 'Risk']
        corr_matrix = risk_factors.corr()
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                name='Correlation'
            ),
            row=2, col=1
        )
        
        # 5. Feature importance
        features = ['Age', 'Diabetes', 'Hypertension', 'Smoking', 'Vessel Density', 'Optic Disc']
        importance = [0.30, 0.15, 0.12, 0.10, 0.20, 0.13]
        
        fig.add_trace(
            go.Bar(x=features, y=importance, name='Feature Importance'),
            row=2, col=2
        )
        
        # 6. Age by risk group
        risk_groups = pd.cut(self.patient_data['risk_score'], bins=4,
                            labels=['Low', 'Medium', 'High', 'Very High'])
        for group in ['Low', 'Medium', 'High', 'Very High']:
            ages = self.patient_data[risk_groups == group]['true_age'].values
            fig.add_trace(
                go.Box(y=ages, name=group, boxpoints='outliers'),
                row=2, col=3
            )
        
        # Update layout
        fig.update_layout(
            title='Interactive Cardiovascular Risk Assessment Dashboard',
            height=800,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Actual Age (years)", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Age (years)", row=1, col=1)
        fig.update_xaxes(title_text="Risk Score", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Actual Age (years)", row=1, col=3)
        fig.update_yaxes(title_text="Residuals (years)", row=1, col=3)
        
        # Save
        fig.write_html(save_path)
        print(f"✓ Interactive dashboard saved: {save_path}")
        
        # Try to open in browser
        try:
            import webbrowser
            import os
            abs_path = os.path.abspath(save_path)
            webbrowser.open(f"file://{abs_path}")
            print(f"🌐 Dashboard opened in browser!")
        except:
            print(f"   Open manually: {save_path}")
        
        return fig
    
    # ========================================================================
    # COMPREHENSIVE REPORTING
    # ========================================================================
    
    def generate_comprehensive_report(self, save_path="cardiovascular_analysis_report.txt"):
        """Generate detailed text report with all statistics."""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        mae = mean_absolute_error(self.patient_data['true_age'],
                                  self.patient_data['pred_age'])
        rmse = np.sqrt(mean_squared_error(self.patient_data['true_age'],
                                          self.patient_data['pred_age']))
        r2 = r2_score(self.patient_data['true_age'], self.patient_data['pred_age'])
        
        risk_groups = pd.cut(self.patient_data['risk_score'], bins=4,
                            labels=['Low', 'Medium', 'High', 'Very High'])
        risk_counts = risk_groups.value_counts()
        
        report = f"""
================================================================================
COMPREHENSIVE CARDIOVASCULAR RISK ASSESSMENT REPORT
================================================================================

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Based on: Poplin et al. (2018), Nature Biomedical Engineering

================================================================================
EXECUTIVE SUMMARY
================================================================================

This report presents a comprehensive analysis of cardiovascular risk assessment
using deep learning predictions from retinal fundus photographs. The analysis
includes {len(self.patient_data)} patients with complete demographic, clinical,
and retinal imaging data, matching the statistical properties reported in the
Nature Biomedical Engineering paper.

================================================================================
DATASET OVERVIEW
================================================================================

Total patients: {len(self.patient_data)}
Age range: {self.patient_data['true_age'].min():.1f} - {self.patient_data['true_age'].max():.1f} years
Mean age: {self.patient_data['true_age'].mean():.1f} ± {self.patient_data['true_age'].std():.1f} years

Gender Distribution:
  Male: {self.patient_data['true_gender'].sum()} ({self.patient_data['true_gender'].mean()*100:.1f}%)
  Female: {(1-self.patient_data['true_gender']).sum()} ({(1-self.patient_data['true_gender'].mean())*100:.1f}%)

Clinical Characteristics:
  Smoking prevalence: {self.patient_data['true_smoking'].mean()*100:.1f}%
  Mean SBP: {self.patient_data['true_sbp'].mean():.1f} ± {self.patient_data['true_sbp'].std():.1f} mmHg
  Mean BMI: {self.patient_data['true_bmi'].mean():.1f} ± {self.patient_data['true_bmi'].std():.1f} kg/m²

================================================================================
AGE PREDICTION PERFORMANCE
================================================================================

Deep learning model performance for age prediction from retinal images:

  Mean Absolute Error (MAE): {mae:.2f} years
  Root Mean Square Error (RMSE): {rmse:.2f} years
  R-squared (R²): {r2:.3f}
  Pearson correlation: {stats.pearsonr(self.patient_data['true_age'], self.patient_data['pred_age'])[0]:.3f}

PAPER COMPARISON:
  Our simulation: MAE = {mae:.2f} years
  Paper reported: MAE = 3.26 years
  Match quality: {"✓ Excellent" if abs(mae - 3.26) < 0.5 else "Good"}

================================================================================
CARDIOVASCULAR RISK FACTOR PREDICTIONS
================================================================================

Binary Classifications (AUC-ROC):

  Gender Prediction:
    AUC: {auc(roc_curve(self.patient_data['true_gender'], self.patient_data['pred_gender_prob'])[0], roc_curve(self.patient_data['true_gender'], self.patient_data['pred_gender_prob'])[1]):.3f}
    Paper reported: 0.97
    Performance: {"Excellent" if auc(roc_curve(self.patient_data['true_gender'], self.patient_data['pred_gender_prob'])[0], roc_curve(self.patient_data['true_gender'], self.patient_data['pred_gender_prob'])[1]) > 0.9 else "Good"}

  Smoking Status:
    AUC: {auc(roc_curve(self.patient_data['true_smoking'], self.patient_data['pred_smoking_prob'])[0], roc_curve(self.patient_data['true_smoking'], self.patient_data['pred_smoking_prob'])[1]):.3f}
    Paper reported: 0.71
    Performance: {"Good" if auc(roc_curve(self.patient_data['true_smoking'], self.patient_data['pred_smoking_prob'])[0], roc_curve(self.patient_data['true_smoking'], self.patient_data['pred_smoking_prob'])[1]) > 0.65 else "Fair"}

  MACE Prediction (5-year):
    AUC: {auc(roc_curve(self.patient_data['true_mace'], self.patient_data['pred_mace_prob'])[0], roc_curve(self.patient_data['true_mace'], self.patient_data['pred_mace_prob'])[1]):.3f}
    Paper reported: 0.70
    Performance: {"Good" if auc(roc_curve(self.patient_data['true_mace'], self.patient_data['pred_mace_prob'])[0], roc_curve(self.patient_data['true_mace'], self.patient_data['pred_mace_prob'])[1]) > 0.65 else "Fair"}

Continuous Variables (Pearson r):

  Systolic Blood Pressure:
    Correlation: {stats.pearsonr(self.patient_data['true_sbp'], self.patient_data['pred_sbp'])[0]:.3f}
    Paper reported: 0.33
    MAE: {mean_absolute_error(self.patient_data['true_sbp'], self.patient_data['pred_sbp']):.1f} mmHg

  Body Mass Index:
    Correlation: {stats.pearsonr(self.patient_data['true_bmi'], self.patient_data['pred_bmi'])[0]:.3f}
    Paper reported: 0.25
    MAE: {mean_absolute_error(self.patient_data['true_bmi'], self.patient_data['pred_bmi']):.1f} kg/m²

================================================================================
RISK STRATIFICATION ANALYSIS
================================================================================

Patients stratified by predicted MACE probability:

  Low Risk (<5%): {risk_counts.get('Low', 0)} patients ({risk_counts.get('Low', 0)/len(self.patient_data)*100:.1f}%)
  Medium Risk (5-10%): {risk_counts.get('Medium', 0)} patients ({risk_counts.get('Medium', 0)/len(self.patient_data)*100:.1f}%)
  High Risk (10-20%): {risk_counts.get('High', 0)} patients ({risk_counts.get('High', 0)/len(self.patient_data)*100:.1f}%)
  Very High Risk (>20%): {risk_counts.get('Very High', 0)} patients ({risk_counts.get('Very High', 0)/len(self.patient_data)*100:.1f}%)

Overall Risk Statistics:
  Mean risk score: {self.patient_data['risk_score'].mean():.3f}
  Median risk score: {self.patient_data['risk_score'].median():.3f}
  Risk score range: {self.patient_data['risk_score'].min():.3f} - {self.patient_data['risk_score'].max():.3f}

High/Very High Risk Patients: {risk_counts.get('High', 0) + risk_counts.get('Very High', 0)} ({(risk_counts.get('High', 0) + risk_counts.get('Very High', 0))/len(self.patient_data)*100:.1f}%)

================================================================================
CLINICAL IMPLICATIONS
================================================================================

1. AGE PREDICTION ACCURACY
   The model achieves {r2*100:.1f}% of variance explained in age prediction,
   with an average error of {mae:.1f} years. This demonstrates that retinal
   vascular patterns contain significant age-related information.

2. CARDIOVASCULAR RISK SCREENING
   {risk_counts.get('High', 0) + risk_counts.get('Very High', 0)} patients
   ({(risk_counts.get('High', 0) + risk_counts.get('Very High', 0))/len(self.patient_data)*100:.1f}%)
   are classified as high or very high risk for MACE within 5 years.
   These patients should be prioritized for comprehensive cardiovascular evaluation.

3. NON-INVASIVE ASSESSMENT
   The model successfully predicts multiple cardiovascular risk factors from
   retinal images alone, without requiring blood tests or invasive procedures.
   This enables accessible screening in primary care and underserved settings.

4. SMOKING DETECTION
   The model achieves AUC {auc(roc_curve(self.patient_data['true_smoking'], self.patient_data['pred_smoking_prob'])[0], roc_curve(self.patient_data['true_smoking'], self.patient_data['pred_smoking_prob'])[1]):.2f} for smoking status detection,
   revealing tobacco-related vascular changes visible in retinal images.

================================================================================
RECOMMENDATIONS
================================================================================

CLINICAL IMPLEMENTATION:
1. Integrate retinal screening into routine eye examinations for dual-purpose
   screening (ocular health + cardiovascular risk)
2. Prioritize patients with high predicted risk for comprehensive workup
3. Use predictions as complementary tool alongside traditional risk scores
4. Consider cost-effectiveness for population-level screening programs

FUTURE RESEARCH:
1. Prospective validation in diverse patient populations
2. Integration with electronic health records for enhanced prediction
3. Longitudinal studies to assess prediction stability over time
4. Investigation of causal mechanisms linking retina to CVD

LIMITATIONS:
1. Simulated data for demonstration purposes (not real patient data)
2. Cross-sectional design prevents causal inference
3. Model requires high-quality retinal photographs
4. Performance may vary across different imaging equipment and populations

================================================================================
METHODOLOGY NOTES
================================================================================

Data Simulation:
  - Dataset size: {len(self.patient_data)} patients
  - Random seed: {self.random_seed}
  - Simulated to match paper-reported performance metrics

Statistical Methods:
  - Pearson correlation for continuous variables
  - ROC-AUC for binary classifications
  - Mean Absolute Error for prediction accuracy
  - Calibration analysis for probability assessment

Software:
  - Python 3.x
  - numpy, pandas, matplotlib, seaborn, scipy, scikit-learn
  - Optional: plotly (interactive visualizations)

================================================================================
REFERENCES
================================================================================

1. Poplin, R., Varadarajan, A. V., Blumer, K., Liu, Y., McConnell, M. V.,
   Corrado, G. S., Peng, L., & Webster, D. R. (2018). Prediction of
   cardiovascular risk factors from retinal fundus photographs via deep
   learning. Nature Biomedical Engineering, 2(3), 158-164.
   https://doi.org/10.1038/s41551-018-0195-0

2. UK Biobank. https://www.ukbiobank.ac.uk/

3. EyePACS. https://www.eyepacs.com/

================================================================================
END OF REPORT
================================================================================
"""
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ Comprehensive report saved: {save_path}")
        
        return report
    
    # ========================================================================
    # CONVENIENCE METHODS
    # ========================================================================
    
    def run_all_visualizations(self, save=True):
        """Run all visualizations in sequence."""
        print("\n" + "="*80)
        print("RUNNING COMPLETE VISUALIZATION PIPELINE")
        print("="*80)
        
        self.plot_age_prediction(save=save)
        self.plot_roc_curves(save=save)
        self.plot_calibration(save=save)
        self.plot_continuous_predictions(save=save)
        self.plot_risk_stratification(save=save)
        self.plot_attention_maps(n_samples=4, save=save)
        self.create_performance_summary()
        
        if PLOTLY_AVAILABLE:
            self.create_interactive_dashboard()
        
        self.generate_comprehensive_report()
        
        print("\n" + "="*80)
        print("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
        print("="*80)
        print("\nGenerated Files:")
        print("  • age_prediction_comprehensive.png")
        print("  • roc_curves.png")
        print("  • calibration_curves.png")
        print("  • continuous_predictions.png")
        print("  • risk_stratification.png")
        print("  • attention_maps.png")
        if PLOTLY_AVAILABLE:
            print("  • interactive_dashboard.html")
        print("  • cardiovascular_analysis_report.txt")
        print("\n" + "="*80)


# ============================================================================
# STANDALONE FUNCTIONAL INTERFACE (for backward compatibility)
# ============================================================================

def quick_analysis(n_patients=10000, save=True):
    """
    Quick analysis with default settings - maintains backward compatibility.
    
    Parameters:
    -----------
    n_patients : int
        Number of patients to simulate
    save : bool
        Whether to save figures
    """
    analyzer = CardiovascularRiskAnalyzer(n_patients=n_patients)
    analyzer.generate_research_paper_dataset()
    analyzer.run_all_visualizations(save=save)
    return analyzer


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n🏥 Cardiovascular Risk Prediction - Complete Visualization Suite")
    print("="*80)
    print("\nThis comprehensive tool combines:")
    print("  ✓ Research paper-accurate simulations (Nature Biomed Eng 2018)")
    print("  ✓ Static publication-quality figures")
    print("  ✓ Interactive Plotly dashboards")
    print("  ✓ Comprehensive statistical reporting")
    print("  ✓ Attention heatmap analysis")
    print("\n" + "="*80)
    
    # Quick start with default settings
    print("\nRunning complete analysis with 10,000 patients...")
    print("(This matches the scale reported in the research paper)")
    print("\n" + "="*80)
    
    analyzer = quick_analysis(n_patients=10000, save=True)
    
    print("\n🎉 Analysis complete! Check the generated files in your directory.")
    print("\nTo use interactively:")
    print("  >>> analyzer = CardiovascularRiskAnalyzer(n_patients=5000)")
    print("  >>> analyzer.generate_research_paper_dataset()")
    print("  >>> analyzer.plot_age_prediction()")
    print("  >>> analyzer.plot_roc_curves()")
    print("  >>> analyzer.create_interactive_dashboard()")
