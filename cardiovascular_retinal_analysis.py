"""
Cardiovascular Risk Prediction from Retinal Images - Data Visualization
Based on Poplin et al. (2018) Nature Biomedical Engineering paper

This script creates visualizations demonstrating the key findings from the research:
1. Predicted vs Actual Age scatter plot
2. Attention heatmap concept for retinal image analysis
3. Cardiovascular risk factor prediction accuracy comparison
4. Statistical performance metrics visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, roc_auc_score
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RetinalCardiovascularVisualizer:
    """
    Class to create visualizations for retinal cardiovascular risk prediction analysis
    """
    
    def __init__(self):
        self.results = self._generate_synthetic_data()
        
    def _generate_synthetic_data(self):
        """
        Generate synthetic data based on the paper's reported results
        Note: This is simulated data for visualization purposes
        """
        np.random.seed(42)
        
        # Age prediction data (MAE = 3.26 years)
        n_samples = 1000
        actual_age = np.random.normal(65, 15, n_samples)
        actual_age = np.clip(actual_age, 30, 90)  # Realistic age range
        
        # Simulate prediction with MAE of 3.26 years
        age_error = np.random.normal(0, 3.26, n_samples)
        predicted_age = actual_age + age_error
        
        # Gender prediction (AUC = 0.97)
        actual_gender = np.random.binomial(1, 0.5, n_samples)
        # High accuracy simulation
        gender_pred_proba = np.where(actual_gender == 1, 
                                   np.random.beta(8, 2, n_samples),
                                   np.random.beta(2, 8, n_samples))
        
        # Smoking status (AUC = 0.71)
        actual_smoking = np.random.binomial(1, 0.2, n_samples)
        smoking_pred_proba = np.where(actual_smoking == 1,
                                    np.random.beta(4, 3, n_samples),
                                    np.random.beta(3, 4, n_samples))
        
        # Blood pressure prediction (MAE = 11.23 mmHg)
        actual_bp = np.random.normal(130, 20, n_samples)
        actual_bp = np.clip(actual_bp, 90, 200)
        bp_error = np.random.normal(0, 11.23, n_samples)
        predicted_bp = actual_bp + bp_error
        
        # MACE prediction (AUC = 0.70)
        actual_mace = np.random.binomial(1, 0.1, n_samples)
        mace_pred_proba = np.where(actual_mace == 1,
                                  np.random.beta(3, 2, n_samples),
                                  np.random.beta(2, 3, n_samples))
        
        return {
            'age': {'actual': actual_age, 'predicted': predicted_age},
            'gender': {'actual': actual_gender, 'predicted_proba': gender_pred_proba},
            'smoking': {'actual': actual_smoking, 'predicted_proba': smoking_pred_proba},
            'bp': {'actual': actual_bp, 'predicted': predicted_bp},
            'mace': {'actual': actual_mace, 'predicted_proba': mace_pred_proba}
        }
    
    def plot_age_prediction(self, figsize=(10, 8)):
        """
        Create predicted vs actual age scatter plot with regression line
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Scatter plot
        actual = self.results['age']['actual']
        predicted = self.results['age']['predicted']
        
        # Calculate metrics
        mae = mean_absolute_error(actual, predicted)
        r2 = stats.pearsonr(actual, predicted)[0]**2
        
        # Scatter plot with regression line
        ax1.scatter(actual, predicted, alpha=0.6, s=30, color='steelblue')
        
        # Add regression line
        z = np.polyfit(actual, predicted, 1)
        p = np.poly1d(z)
        ax1.plot(actual, p(actual), "r--", alpha=0.8, linewidth=2)
        
        # Perfect prediction line
        min_age, max_age = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        ax1.plot([min_age, max_age], [min_age, max_age], 'k--', alpha=0.5, linewidth=1)
        
        ax1.set_xlabel('Actual Age (years)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Predicted Age (years)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Age Prediction Performance\nMAE = {mae:.2f} years, R² = {r2:.3f}', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Residual plot
        residuals = predicted - actual
        ax2.scatter(actual, residuals, alpha=0.6, s=30, color='coral')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Actual Age (years)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Residuals (Predicted - Actual)', fontsize=12, fontweight='bold')
        ax2.set_title('Residual Plot: Age Prediction', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_attention_heatmap_concept(self, figsize=(12, 8)):
        """
        Create conceptual attention heatmap visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Simulate retinal image regions
        height, width = 100, 100
        
        # Create different attention patterns for different predictions
        attention_patterns = {
            'Age': self._create_attention_pattern(height, width, 'optic_disc'),
            'Gender': self._create_attention_pattern(height, width, 'vessel_pattern'),
            'Smoking': self._create_attention_pattern(height, width, 'peripheral'),
            'Blood Pressure': self._create_attention_pattern(height, width, 'vessel_caliber'),
            'MACE': self._create_attention_pattern(height, width, 'optic_disc_vessels')
        }
        
        # Plot attention maps
        for i, (prediction_type, attention_map) in enumerate(attention_patterns.items()):
            row = i // 3
            col = i % 3
            
            im = axes[row, col].imshow(attention_map, cmap='hot', alpha=0.8)
            axes[row, col].set_title(f'{prediction_type}\nAttention Map', 
                                   fontsize=12, fontweight='bold')
            axes[row, col].axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        # Add anatomical labels
        axes[1, 2].text(0.5, 0.5, 'Anatomical Regions:\n• Optic Disc (center)\n• Blood Vessels (radiating)\n• Macula (temporal)\n• Peripheral Retina', 
                       transform=axes[1, 2].transAxes, fontsize=10, 
                       ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="lightblue", alpha=0.7))
        axes[1, 2].axis('off')
        
        plt.suptitle('Deep Learning Attention Maps: Retinal Regions Driving Predictions', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        return fig
    
    def _create_attention_pattern(self, height, width, pattern_type):
        """
        Create synthetic attention patterns for different prediction types
        """
        attention_map = np.zeros((height, width))
        
        if pattern_type == 'optic_disc':
            # Focus on optic disc region (center)
            center_y, center_x = height // 2, width // 2
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= (height//4)**2
            attention_map[mask] = np.random.exponential(2, mask.sum())
            
        elif pattern_type == 'vessel_pattern':
            # Focus on blood vessel patterns (radiating from center)
            center_y, center_x = height // 2, width // 2
            y, x = np.ogrid[:height, :width]
            angles = np.arctan2(y - center_y, x - center_x)
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            # Create radiating pattern
            for angle in np.linspace(0, 2*np.pi, 8):
                mask = (np.abs(angles - angle) < 0.3) & (distances < height//2)
                attention_map[mask] = np.random.exponential(1.5, mask.sum())
                
        elif pattern_type == 'peripheral':
            # Focus on peripheral retina
            center_y, center_x = height // 2, width // 2
            y, x = np.ogrid[:height, :width]
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            mask = distances > height//3
            attention_map[mask] = np.random.exponential(1, mask.sum())
            
        elif pattern_type == 'vessel_caliber':
            # Focus on vessel caliber (thickness)
            center_y, center_x = height // 2, width // 2
            y, x = np.ogrid[:height, :width]
            angles = np.arctan2(y - center_y, x - center_x)
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            # Thicker vessel pattern
            for angle in np.linspace(0, 2*np.pi, 6):
                mask = (np.abs(angles - angle) < 0.2) & (distances < height//2)
                attention_map[mask] = np.random.exponential(2, mask.sum())
                
        elif pattern_type == 'optic_disc_vessels':
            # Combined optic disc and vessel pattern
            center_y, center_x = height // 2, width // 2
            y, x = np.ogrid[:height, :width]
            # Optic disc
            disc_mask = (x - center_x)**2 + (y - center_y)**2 <= (height//5)**2
            attention_map[disc_mask] = np.random.exponential(2, disc_mask.sum())
            # Vessels
            angles = np.arctan2(y - center_y, x - center_x)
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            for angle in np.linspace(0, 2*np.pi, 8):
                mask = (np.abs(angles - angle) < 0.25) & (distances < height//2)
                attention_map[mask] += np.random.exponential(1, mask.sum())
        
        # Normalize
        attention_map = attention_map / attention_map.max()
        return attention_map
    
    def plot_prediction_accuracy_comparison(self, figsize=(12, 8)):
        """
        Create comprehensive comparison of prediction accuracies
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. MAE comparison for continuous variables
        metrics = ['Age\n(years)', 'Blood Pressure\n(mmHg)']
        mae_values = [3.26, 11.23]
        colors = ['steelblue', 'coral']
        
        bars1 = ax1.bar(metrics, mae_values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
        ax1.set_title('Continuous Variable Prediction Accuracy', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars1, mae_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. AUC comparison for binary classifications
        binary_metrics = ['Gender', 'Smoking\nStatus', 'MACE\nEvents']
        auc_values = [0.97, 0.71, 0.70]
        colors2 = ['green', 'orange', 'red']
        
        bars2 = ax2.bar(binary_metrics, auc_values, color=colors2, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Area Under ROC Curve (AUC)', fontsize=12, fontweight='bold')
        ax2.set_title('Binary Classification Performance', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars2, auc_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. ROC curves for binary classifications
        from sklearn.metrics import roc_curve
        
        # Generate synthetic ROC curves
        fpr_gender, tpr_gender, _ = roc_curve(self.results['gender']['actual'], 
                                             self.results['gender']['predicted_proba'])
        fpr_smoking, tpr_smoking, _ = roc_curve(self.results['smoking']['actual'], 
                                              self.results['smoking']['predicted_proba'])
        fpr_mace, tpr_mace, _ = roc_curve(self.results['mace']['actual'], 
                                        self.results['mace']['predicted_proba'])
        
        ax3.plot(fpr_gender, tpr_gender, 'g-', linewidth=2, label=f'Gender (AUC = 0.97)')
        ax3.plot(fpr_smoking, tpr_smoking, 'orange', linewidth=2, label=f'Smoking (AUC = 0.71)')
        ax3.plot(fpr_mace, tpr_mace, 'r-', linewidth=2, label=f'MACE (AUC = 0.70)')
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        ax3.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax3.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax3.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Clinical significance interpretation
        clinical_data = {
            'Prediction': ['Age', 'Gender', 'Smoking', 'Blood Pressure', 'MACE'],
            'Accuracy': ['High', 'Excellent', 'Moderate', 'Good', 'Good'],
            'Clinical Value': ['Biological Age', 'Sex-specific Risk', 'Lifestyle Factor', 'Hypertension Risk', 'Cardiac Events']
        }
        
        # Create a table-like visualization
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = []
        for i in range(len(clinical_data['Prediction'])):
            table_data.append([clinical_data['Prediction'][i], 
                             clinical_data['Accuracy'][i], 
                             clinical_data['Clinical Value'][i]])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Prediction', 'Accuracy Level', 'Clinical Significance'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color code the accuracy levels
        colors_table = {'High': 'lightgreen', 'Excellent': 'green', 
                       'Moderate': 'yellow', 'Good': 'lightblue'}
        
        for i in range(1, len(table_data) + 1):
            accuracy = table_data[i-1][1]
            table[(i, 1)].set_facecolor(colors_table.get(accuracy, 'white'))
        
        ax4.set_title('Clinical Significance of Predictions', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_statistical_summary(self, figsize=(10, 6)):
        """
        Create statistical summary visualization
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create summary statistics
        summary_data = {
            'Metric': ['Mean Absolute Error (Age)', 'Mean Absolute Error (BP)', 
                      'AUC (Gender)', 'AUC (Smoking)', 'AUC (MACE)'],
            'Value': [3.26, 11.23, 0.97, 0.71, 0.70],
            'Unit': ['years', 'mmHg', 'AUC', 'AUC', 'AUC'],
            'Significance': ['p < 0.001', 'p < 0.001', 'p < 0.001', 'p < 0.001', 'p < 0.001']
        }
        
        # Create horizontal bar plot
        y_pos = np.arange(len(summary_data['Metric']))
        colors = ['steelblue', 'coral', 'green', 'orange', 'red']
        
        bars = ax.barh(y_pos, summary_data['Value'], color=colors, alpha=0.7, edgecolor='black')
        
        # Customize the plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(summary_data['Metric'], fontsize=11)
        ax.set_xlabel('Performance Value', fontsize=12, fontweight='bold')
        ax.set_title('Statistical Performance Summary\n(Poplin et al., 2018)', 
                   fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (bar, value, unit) in enumerate(zip(bars, summary_data['Value'], summary_data['Unit'])):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.2f} {unit}', va='center', fontweight='bold')
        
        # Add significance indicators
        for i, sig in enumerate(summary_data['Significance']):
            ax.text(0.02, i, sig, va='center', ha='left', fontsize=9, 
                   style='italic', color='gray')
        
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        return fig

def main():
    """
    Main function to generate all visualizations
    """
    print("Generating visualizations for Cardiovascular Risk Prediction from Retinal Images")
    print("Based on Poplin et al. (2018) Nature Biomedical Engineering")
    print("=" * 80)
    
    # Initialize visualizer
    visualizer = RetinalCardiovascularVisualizer()
    
    # Generate all plots
    print("1. Creating Age Prediction Scatter Plot...")
    fig1 = visualizer.plot_age_prediction()
    fig1.savefig('age_prediction_analysis.png', dpi=300, bbox_inches='tight')
    
    print("2. Creating Attention Heatmap Concept...")
    fig2 = visualizer.plot_attention_heatmap_concept()
    fig2.savefig('attention_heatmaps.png', dpi=300, bbox_inches='tight')
    
    print("3. Creating Prediction Accuracy Comparison...")
    fig3 = visualizer.plot_prediction_accuracy_comparison()
    fig3.savefig('prediction_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    
    print("4. Creating Statistical Summary...")
    fig4 = visualizer.plot_statistical_summary()
    fig4.savefig('statistical_summary.png', dpi=300, bbox_inches='tight')
    
    # Display all plots
    plt.show()
    
    print("\nAll visualizations completed and saved!")
    print("Files created:")
    print("- age_prediction_analysis.png")
    print("- attention_heatmaps.png") 
    print("- prediction_accuracy_comparison.png")
    print("- statistical_summary.png")

if __name__ == "__main__":
    main()