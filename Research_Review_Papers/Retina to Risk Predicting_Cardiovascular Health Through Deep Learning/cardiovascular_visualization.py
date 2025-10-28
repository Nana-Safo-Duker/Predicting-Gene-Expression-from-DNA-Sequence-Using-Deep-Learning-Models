"""
Cardiovascular Risk Prediction from Retinal Images - Simple Visualization Script
==================================================================================

This standalone Python script generates comprehensive visualizations for deep learning-based
cardiovascular risk prediction from retinal fundus photographs.

Based on: Poplin et al. (2018), Nature Biomedical Engineering
"Prediction of cardiovascular risk factors from retinal fundus photographs via deep learning"

Features:
- Simulated dataset generation (10,000 patients)
- Age prediction analysis with comprehensive metrics
- ROC curves for binary classifications (Gender, Smoking, MACE)
- Calibration curves for probability predictions
- Continuous variable predictions (SBP, BMI) with Bland-Altman plots
- Risk stratification analysis
- Simulated attention map visualizations
- Performance summary tables

Author: Scientific Blog Post Project
Date: October 27, 2024
Version: 2.0 (Simplified with Fallback)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_curve, auc, mean_absolute_error, mean_squared_error
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("CARDIOVASCULAR RISK PREDICTION VISUALIZATION SCRIPT")
print("="*80)


# =============================================================================
# 1. DATA GENERATION
# =============================================================================

def generate_simulated_data(n_patients=10000, random_seed=42):
    """
    Generate simulated patient data matching published study statistics.
    
    Parameters:
    -----------
    n_patients : int
        Number of patients to simulate (default: 10000)
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame : Patient data with true and predicted values
    """
    np.random.seed(random_seed)
    print(f"\nGenerating simulated dataset ({n_patients} patients)...")
    
    # True patient characteristics
    true_age = np.random.normal(60, 12, n_patients)
    true_gender = np.random.binomial(1, 0.52, n_patients)  # 52% male
    true_smoking = np.random.binomial(1, 0.18, n_patients)  # 18% smokers
    true_sbp = np.random.normal(135, 18, n_patients)  # Systolic BP
    true_bmi = np.random.normal(27, 4.5, n_patients)  # BMI
    
    # Model predictions with realistic error
    pred_age = true_age + np.random.normal(0, 3.26, n_patients)  # MAE = 3.26
    pred_gender_prob = np.clip(true_gender + np.random.normal(0, 0.15, n_patients), 0, 1)
    pred_smoking_prob = np.clip(true_smoking + np.random.normal(0, 0.35, n_patients), 0, 1)
    pred_sbp = true_sbp * 0.33 + np.random.normal(135, 15, n_patients)  # r = 0.33
    pred_bmi = true_bmi * 0.25 + np.random.normal(27, 4, n_patients)  # r = 0.25
    
    # MACE prediction (Major Adverse Cardiac Events)
    base_risk = 0.05
    risk_score = (base_risk + 
                  0.002 * (true_age - 60) + 
                  0.03 * true_smoking + 
                  0.001 * (true_sbp - 135) + 
                  0.005 * (true_bmi - 27))
    risk_score = np.clip(risk_score, 0, 0.5)
    true_mace = np.random.binomial(1, risk_score, n_patients)
    pred_mace_prob = np.clip(risk_score + np.random.normal(0, 0.08, n_patients), 0, 1)
    
    # Create DataFrame
    df = pd.DataFrame({
        'true_age': true_age,
        'pred_age': pred_age,
        'true_gender': true_gender,
        'pred_gender_prob': pred_gender_prob,
        'true_smoking': true_smoking,
        'pred_smoking_prob': pred_smoking_prob,
        'true_sbp': true_sbp,
        'pred_sbp': pred_sbp,
        'true_bmi': true_bmi,
        'pred_bmi': pred_bmi,
        'true_mace': true_mace,
        'pred_mace_prob': pred_mace_prob
    })
    
    print(f"✓ Dataset generated: {df.shape[0]} patients, {df.shape[1]} variables")
    return df


# =============================================================================
# 2. AGE PREDICTION VISUALIZATION
# =============================================================================

def plot_age_prediction(df, save=True, filename='age_prediction_performance.png'):
    """
    Plot age prediction performance with error analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Patient data with true_age and pred_age columns
    save : bool
        Whether to save the figure
    filename : str
        Output filename
    """
    print("\n" + "="*80)
    print("GENERATING AGE PREDICTION VISUALIZATIONS")
    print("="*80)
    
    # Calculate metrics
    mae = mean_absolute_error(df['true_age'], df['pred_age'])
    rmse = np.sqrt(mean_squared_error(df['true_age'], df['pred_age']))
    age_corr, p_value = stats.pearsonr(df['true_age'], df['pred_age'])
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot with regression line
    axes[0].scatter(df['true_age'], df['pred_age'], alpha=0.3, s=10, color='steelblue')
    axes[0].plot([30, 90], [30, 90], 'r--', linewidth=2, label='Perfect prediction')
    
    # Fit regression line
    lr = LinearRegression()
    lr.fit(df['true_age'].values.reshape(-1, 1), df['pred_age'].values)
    line_x = np.linspace(30, 90, 100)
    line_y = lr.predict(line_x.reshape(-1, 1))
    axes[0].plot(line_x, line_y, 'g-', linewidth=2, label='Regression line')
    
    axes[0].set_xlabel('True Age (years)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Predicted Age (years)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Age Prediction Performance\nMAE = {mae:.2f} years, r = {age_corr:.3f}',
                      fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error distribution
    errors = df['pred_age'] - df['true_age']
    axes[1].hist(errors, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[1].axvline(errors.mean(), color='blue', linestyle='-', linewidth=2,
                    label=f'Mean: {errors.mean():.2f}')
    axes[1].set_xlabel('Prediction Error (years)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Distribution of Age Prediction Errors', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
    plt.show()
    
    print(f"  MAE: {mae:.2f} years | RMSE: {rmse:.2f} years | r: {age_corr:.3f}")


# =============================================================================
# 3. ROC CURVES
# =============================================================================

def plot_roc_curves(df, save=True, filename='roc_curves.png'):
    """
    Plot ROC curves for binary classifications.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Patient data with true and predicted values
    save : bool
        Whether to save the figure
    filename : str
        Output filename
    
    Returns:
    --------
    tuple : (auc_gender, auc_smoking, auc_mace)
    """
    print("\n" + "="*80)
    print("GENERATING ROC CURVES")
    print("="*80)
    
    # Calculate ROC curves
    fpr_gender, tpr_gender, _ = roc_curve(df['true_gender'], df['pred_gender_prob'])
    auc_gender = auc(fpr_gender, tpr_gender)
    
    fpr_smoking, tpr_smoking, _ = roc_curve(df['true_smoking'], df['pred_smoking_prob'])
    auc_smoking = auc(fpr_smoking, tpr_smoking)
    
    fpr_mace, tpr_mace, _ = roc_curve(df['true_mace'], df['pred_mace_prob'])
    auc_mace = auc(fpr_mace, tpr_mace)
    
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


# =============================================================================
# 4. CALIBRATION CURVES
# =============================================================================

def plot_calibration(df, save=True, filename='calibration_curves.png'):
    """
    Plot calibration curves for MACE prediction.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Patient data
    save : bool
        Whether to save the figure
    filename : str
        Output filename
    """
    print("\n" + "="*80)
    print("GENERATING CALIBRATION CURVES")
    print("="*80)
    
    fraction_positives, mean_predicted = calibration_curve(
        df['true_mace'],
        df['pred_mace_prob'],
        n_bins=10, strategy='quantile'
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calibration curve
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    axes[0].plot(mean_predicted, fraction_positives, 'o-', linewidth=3,
                 markersize=10, color='#F18F01', label='Model calibration')
    axes[0].set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Observed Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Calibration Curve for MACE Prediction', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Probability distribution
    axes[1].hist(df[df['true_mace']==1]['pred_mace_prob'],
                 bins=30, alpha=0.6, label='MACE cases', color='red', edgecolor='black')
    axes[1].hist(df[df['true_mace']==0]['pred_mace_prob'],
                 bins=30, alpha=0.6, label='No MACE', color='green', edgecolor='black')
    axes[1].set_xlabel('Predicted MACE Probability', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
    plt.show()


# =============================================================================
# 5. CONTINUOUS PREDICTIONS (SBP, BMI)
# =============================================================================

def plot_continuous_predictions(df, save=True, filename='continuous_predictions.png'):
    """
    Plot SBP and BMI predictions with Bland-Altman analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Patient data
    save : bool
        Whether to save the figure
    filename : str
        Output filename
    """
    print("\n" + "="*80)
    print("GENERATING CONTINUOUS VARIABLE PREDICTIONS")
    print("="*80)
    
    sbp_corr, _ = stats.pearsonr(df['true_sbp'], df['pred_sbp'])
    bmi_corr, _ = stats.pearsonr(df['true_bmi'], df['pred_bmi'])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # SBP scatter
    axes[0, 0].scatter(df['true_sbp'], df['pred_sbp'], alpha=0.3, s=10, color='#E63946')
    axes[0, 0].plot([90, 180], [90, 180], 'k--', linewidth=2)
    axes[0, 0].set_xlabel('True SBP (mmHg)', fontweight='bold')
    axes[0, 0].set_ylabel('Predicted SBP (mmHg)', fontweight='bold')
    axes[0, 0].set_title(f'SBP Prediction (r = {sbp_corr:.3f})', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # SBP Bland-Altman
    mean_sbp = (df['true_sbp'] + df['pred_sbp']) / 2
    diff_sbp = df['pred_sbp'] - df['true_sbp']
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
    axes[1, 0].scatter(df['true_bmi'], df['pred_bmi'], alpha=0.3, s=10, color='#457B9D')
    axes[1, 0].plot([15, 40], [15, 40], 'k--', linewidth=2)
    axes[1, 0].set_xlabel('True BMI (kg/m²)', fontweight='bold')
    axes[1, 0].set_ylabel('Predicted BMI (kg/m²)', fontweight='bold')
    axes[1, 0].set_title(f'BMI Prediction (r = {bmi_corr:.3f})', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # BMI Bland-Altman
    mean_bmi = (df['true_bmi'] + df['pred_bmi']) / 2
    diff_bmi = df['pred_bmi'] - df['true_bmi']
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


# =============================================================================
# 6. RISK STRATIFICATION
# =============================================================================

def plot_risk_stratification(df, save=True, filename='risk_stratification.png'):
    """
    Plot risk stratification analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Patient data
    save : bool
        Whether to save the figure
    filename : str
        Output filename
    """
    print("\n" + "="*80)
    print("GENERATING RISK STRATIFICATION ANALYSIS")
    print("="*80)
    
    df['risk_category'] = pd.cut(
        df['pred_mace_prob'],
        bins=[0, 0.05, 0.10, 0.20, 1.0],
        labels=['Low (<5%)', 'Moderate (5-10%)', 'High (10-20%)', 'Very High (>20%)']
    )
    
    risk_analysis = df.groupby('risk_category').agg({
        'true_mace': ['count', 'sum', 'mean']
    })
    risk_analysis.columns = ['N', 'Events', 'Rate']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Patient distribution
    risk_counts = df['risk_category'].value_counts().sort_index()
    colors = ['#06D6A0', '#FFD166', '#EF476F', '#8B0000']
    axes[0].bar(range(len(risk_counts)), risk_counts.values, color=colors,
                edgecolor='black', linewidth=1.5)
    axes[0].set_xticks(range(len(risk_counts)))
    axes[0].set_xticklabels(risk_counts.index, rotation=15, ha='right')
    axes[0].set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
    axes[0].set_title('Patient Distribution by Risk Category', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Observed MACE rates
    observed_rates = (risk_analysis['Rate'] * 100).values
    axes[1].bar(range(len(observed_rates)), observed_rates, color=colors,
                edgecolor='black', linewidth=1.5)
    axes[1].set_xticks(range(len(observed_rates)))
    axes[1].set_xticklabels(risk_analysis.index, rotation=15, ha='right')
    axes[1].set_ylabel('Observed MACE Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Observed MACE Rate by Risk Category', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
    plt.show()
    
    print("\nRisk Stratification Results:")
    print(risk_analysis)


# =============================================================================
# 7. ATTENTION MAP VISUALIZATION
# =============================================================================

def generate_attention_map(age, risk_prob, size=512):
    """
    Generate a risk-based attention map.
    
    Parameters:
    -----------
    age : float
        Patient age
    risk_prob : float
        MACE risk probability
    size : int
        Image size (pixels)
    
    Returns:
    --------
    np.ndarray : Attention map
    """
    attention = np.zeros((size, size))
    center_x, center_y = size // 2, size // 2
    
    # Optic disc attention
    y, x = np.ogrid[:size, :size]
    disc_mask = (x - center_x)**2 + (y - center_y)**2 < (size // 8)**2
    attention[disc_mask] += 0.7
    
    # Vessel attention (increases with risk)
    vessel_intensity = 0.3 + 0.5 * risk_prob
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        vessel_x = center_x + np.cos(angle) * np.linspace(0, size//3, 50)
        vessel_y = center_y + np.sin(angle) * np.linspace(0, size//3, 50)
        for i in range(len(vessel_x)-1):
            x1, y1 = int(vessel_x[i]), int(vessel_y[i])
            x2, y2 = int(vessel_x[i+1]), int(vessel_y[i+1])
            # Simple line drawing
            num_points = max(abs(x2-x1), abs(y2-y1)) + 1
            xs = np.linspace(x1, x2, num_points).astype(int)
            ys = np.linspace(y1, y2, num_points).astype(int)
            for x, y in zip(xs, ys):
                if 0 <= x < size and 0 <= y < size:
                    attention[y, x] = vessel_intensity
    
    # Macula attention (age-dependent)
    macula_x, macula_y = center_x + 20, center_y - 30
    macula_intensity = 0.2 + 0.3 * (age - 25) / 60
    macula_mask = (x - macula_x)**2 + (y - macula_y)**2 < (size // 12)**2
    attention[macula_mask] += macula_intensity
    
    # Smooth and normalize
    attention = gaussian_filter(attention, sigma=2)
    attention = np.clip(attention, 0, 1)
    
    return attention


def plot_attention_map(df, save=True, filename='attention_map.png'):
    """
    Plot attention heatmaps for different risk groups.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Patient data
    save : bool
        Whether to save the figure
    filename : str
        Output filename
    """
    print("\n" + "="*80)
    print("GENERATING ATTENTION MAP VISUALIZATIONS")
    print("="*80)
    
    # Categorize patients by risk
    df['risk_group'] = pd.cut(df['pred_mace_prob'], bins=4,
                               labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Select representative patients from each risk group
    representative_patients = []
    for group in ['Low', 'Medium', 'High', 'Very High']:
        patients_in_group = df[df['risk_group'] == group]
        if len(patients_in_group) > 0:
            representative_patients.append(patients_in_group.iloc[0])
    
    # Fallback: If we don't have 4 groups, select evenly-spaced patients
    if len(representative_patients) < 4:
        print(f"⚠️  Warning: Only found {len(representative_patients)} risk groups")
        print("    Selecting evenly-spaced patients across risk spectrum...")
        sorted_df = df.sort_values('pred_mace_prob').reset_index(drop=True)
        step = len(sorted_df) // 4
        representative_patients = [sorted_df.iloc[i * step] for i in range(4)]
    
    # Generate attention maps
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    titles = ['Low Risk Patient', 'Medium Risk Patient', 
              'High Risk Patient', 'Very High Risk Patient']
    
    for i in range(min(4, len(representative_patients))):
        patient = representative_patients[i]
        attention = generate_attention_map(patient['true_age'], patient['pred_mace_prob'])
        
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
        
        axes[i].set_title(f'{titles[i]}\nRisk Score: {patient["pred_mace_prob"]:.3f}',
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Width (pixels)', fontsize=10)
        axes[i].set_ylabel('Height (pixels)', fontsize=10)
        
        cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', fontsize=9)
    
    # Hide any unused panels
    for i in range(len(representative_patients), 4):
        axes[i].axis('off')
    
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
    plt.show()


# =============================================================================
# 8. PERFORMANCE SUMMARY
# =============================================================================

def create_performance_summary(df, auc_gender, auc_smoking, auc_mace):
    """
    Create and display comprehensive performance summary.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Patient data
    auc_gender, auc_smoking, auc_mace : float
        AUC values from ROC analysis
    """
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    age_mae = mean_absolute_error(df['true_age'], df['pred_age'])
    sbp_corr, _ = stats.pearsonr(df['true_sbp'], df['pred_sbp'])
    bmi_corr, _ = stats.pearsonr(df['true_bmi'], df['pred_bmi'])
    
    summary = pd.DataFrame({
        'Risk Factor': ['Age', 'Gender', 'Smoking', 'SBP', 'BMI', 'MACE'],
        'Type': ['Regression', 'Binary', 'Binary', 'Regression', 'Regression', 'Binary'],
        'Metric': [
            f"MAE={age_mae:.2f}y",
            f"AUC={auc_gender:.3f}",
            f"AUC={auc_smoking:.3f}",
            f"r={sbp_corr:.3f}",
            f"r={bmi_corr:.3f}",
            f"AUC={auc_mace:.3f}"
        ],
        'Paper Reported': ['3.26y', '0.97', '0.71', '0.33', '0.25', '0.70'],
        'Performance': ['Excellent', 'Excellent', 'Good', 'Moderate', 'Moderate', 'Good']
    })
    
    print(summary.to_string(index=False))
    print("="*80)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function - runs all visualizations."""
    print("\n🏥 Cardiovascular Risk Prediction - Visualization Suite")
    print("="*80)
    print("\nThis tool replicates visualizations from:")
    print("Poplin et al. (2018), Nature Biomedical Engineering")
    print("="*80)
    
    # Generate data
    df = generate_simulated_data(n_patients=10000)
    
    # Generate all visualizations
    plot_age_prediction(df)
    auc_gender, auc_smoking, auc_mace = plot_roc_curves(df)
    plot_calibration(df)
    plot_continuous_predictions(df)
    plot_risk_stratification(df)
    plot_attention_map(df)
    create_performance_summary(df, auc_gender, auc_smoking, auc_mace)
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated Files:")
    print("  • age_prediction_performance.png")
    print("  • roc_curves.png")
    print("  • calibration_curves.png")
    print("  • continuous_predictions.png")
    print("  • risk_stratification.png")
    print("  • attention_map.png")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
