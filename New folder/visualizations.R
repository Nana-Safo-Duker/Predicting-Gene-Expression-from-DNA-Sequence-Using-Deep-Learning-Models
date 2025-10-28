# Gene Expression Prediction Visualizations in R
# =================================================
# 
# This script generates visualizations for analyzing deep learning model
# performance in predicting gene expression from DNA sequences.
#
# Author: Scientific Blog Post Series
# Date: October 26, 2025
#
# Required packages: ggplot2, dplyr, gridExtra, viridis

# Load required libraries
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("dplyr")) install.packages("dplyr")
if (!require("gridExtra")) install.packages("gridExtra")
if (!require("viridis")) install.packages("viridis")

library(ggplot2)
library(dplyr)
library(gridExtra)
library(viridis)

# Create output directory
dir.create("figures", showWarnings = FALSE)

# Set seed for reproducibility
set.seed(42)

# Generate simulated data
generate_data <- function(n = 2000, correlation = 0.85) {
  true_expr <- rlnorm(n, meanlog = 2, sdlog = 1.2)
  noise_std <- sqrt((1 - correlation^2) / correlation^2) * sd(true_expr)
  pred_expr <- correlation * true_expr + rnorm(n, 0, noise_std)
  pred_expr <- pmax(pred_expr, 0.1)
  
  data.frame(
    true_expression = true_expr,
    predicted_expression = pred_expr
  )
}

# Figure 1: Model Performance
plot_model_performance <- function(data) {
  cor_val <- cor(data$true_expression, data$predicted_expression, method = "pearson")
  r2_val <- cor_val^2
  
  p <- ggplot(data, aes(x = true_expression, y = predicted_expression)) +
    geom_hex(bins = 50) +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", size = 1) +
    scale_fill_viridis(name = "Count") +
    labs(
      x = "Experimental Gene Expression (log₂ FPKM)",
      y = "Predicted Gene Expression (log₂ FPKM)",
      title = "Deep Learning Model Performance",
      subtitle = sprintf("Pearson r = %.3f, R² = %.3f", cor_val, r2_val)
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      axis.title = element_text(face = "bold", size = 12)
    )
  
  ggsave("figures/figure1_model_performance_R.png", p, width = 8, height = 7, dpi = 300)
  print("Saved: figures/figure1_model_performance_R.png")
  
  return(list(correlation = cor_val, r2 = r2_val))
}

# Figure 2: Error Analysis
plot_error_analysis <- function(data) {
  data$error <- data$predicted_expression - data$true_expression
  data$relative_error <- (data$error / data$true_expression) * 100
  
  p1 <- ggplot(data, aes(x = error)) +
    geom_histogram(bins = 60, fill = "lightcoral", color = "black", alpha = 0.7) +
    geom_vline(aes(xintercept = mean(error)), color = "red", linetype = "dashed", size = 1) +
    geom_vline(aes(xintercept = median(error)), color = "blue", linetype = "dashed", size = 1) +
    labs(
      x = "Prediction Error",
      y = "Frequency",
      title = "A. Distribution of Prediction Errors"
    ) +
    theme_minimal()
  
  p2 <- ggplot(data, aes(x = true_expression, y = error)) +
    geom_point(alpha = 0.4, size = 1, color = "steelblue") +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed", size = 1) +
    geom_smooth(method = "loess", color = "orange", size = 1.5) +
    labs(
      x = "True Expression Level",
      y = "Prediction Error",
      title = "B. Error vs. Expression Level"
    ) +
    theme_minimal()
  
  p3 <- ggplot(data, aes(x = relative_error)) +
    geom_histogram(bins = 60, fill = "lightgreen", color = "black", alpha = 0.7) +
    geom_vline(aes(xintercept = median(relative_error)), 
               color = "darkgreen", linetype = "dashed", size = 1) +
    labs(
      x = "Relative Error (%)",
      y = "Frequency",
      title = "C. Distribution of Relative Errors"
    ) +
    theme_minimal()
  
  p4 <- ggplot(data, aes(sample = scale(error))) +
    stat_qq() +
    stat_qq_line(color = "red", size = 1) +
    labs(
      title = "D. Q-Q Plot of Standardized Residuals",
      x = "Theoretical Quantiles",
      y = "Sample Quantiles"
    ) +
    theme_minimal()
  
  combined <- grid.arrange(p1, p2, p3, p4, ncol = 2)
  ggsave("figures/figure2_error_analysis_R.png", combined, width = 14, height = 10, dpi = 300)
  print("Saved: figures/figure2_error_analysis_R.png")
}

# Figure 3: Cell Type Performance
plot_cell_type_performance <- function() {
  cell_types <- c('K562', 'HepG2', 'GM12878', 'H1-ESC', 'MCF7', 'HeLa-S3', 'A549', 'Jurkat')
  
  df <- data.frame(
    CellType = factor(cell_types, levels = rev(cell_types)),
    Pearson = pmax(0.7, pmin(0.92, rnorm(length(cell_types), 0.82, 0.05))),
    MSE = runif(length(cell_types), 0.15, 0.35),
    Samples = sample(400:800, length(cell_types), replace = TRUE)
  )
  df$R2 <- df$Pearson^2 + rnorm(length(cell_types), 0, 0.02)
  
  p1 <- ggplot(df, aes(x = Pearson, y = CellType, fill = CellType)) +
    geom_col(alpha = 0.8, color = "black") +
    geom_vline(xintercept = 0.8, linetype = "dashed", color = "red", size = 1) +
    scale_fill_viridis_d() +
    labs(
      x = "Pearson Correlation",
      y = NULL,
      title = "A. Performance Across Cell Types"
    ) +
    theme_minimal() +
    theme(legend.position = "none")
  
  p2 <- ggplot(df, aes(x = R2, y = CellType, fill = CellType)) +
    geom_col(alpha = 0.8, color = "black") +
    scale_fill_viridis_d() +
    labs(
      x = "R² Score",
      y = NULL,
      title = "B. R² Scores by Cell Type"
    ) +
    theme_minimal() +
    theme(legend.position = "none")
  
  combined <- grid.arrange(p1, p2, ncol = 2)
  ggsave("figures/figure3_cell_type_performance_R.png", combined, width = 14, height = 5, dpi = 300)
  print("Saved: figures/figure3_cell_type_performance_R.png")
}

# Figure 4: Model Comparison
plot_model_comparison <- function() {
  models <- c('Linear\nRegression', 'Random\nForest', 'SVM', 'Shallow\nNeural Net', 
              'CNN', 'RNN', 'CNN+RNN\n(This Study)')
  
  df <- data.frame(
    Model = factor(models, levels = models),
    Correlation = c(0.45, 0.58, 0.62, 0.69, 0.75, 0.78, 0.85),
    R2 = c(0.20, 0.34, 0.38, 0.48, 0.56, 0.61, 0.72),
    TrainingTime = c(0.5, 2, 5, 10, 30, 35, 45),
    Highlight = c(rep("Baseline", 6), "This Study")
  )
  
  p1 <- ggplot(df, aes(x = Model, y = Correlation, fill = Highlight)) +
    geom_col(alpha = 0.8, color = "black", size = 1) +
    scale_fill_manual(values = c("Baseline" = "lightgray", "This Study" = "steelblue")) +
    geom_hline(yintercept = 0.8, linetype = "dashed", color = "red", size = 1) +
    labs(
      y = "Pearson Correlation (r)",
      x = NULL,
      title = "Model Comparison: Prediction Accuracy"
    ) +
    theme_minimal() +
    theme(
      legend.position = "none",
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(face = "bold")
    )
  
  p2 <- ggplot(df, aes(x = TrainingTime, y = Correlation, color = Highlight, size = Highlight)) +
    geom_point(alpha = 0.7) +
    scale_color_manual(values = c("Baseline" = "lightcoral", "This Study" = "darkgreen")) +
    scale_size_manual(values = c("Baseline" = 3, "This Study" = 6)) +
    labs(
      x = "Training Time (hours)",
      y = "Pearson Correlation (r)",
      title = "Performance vs. Computational Cost"
    ) +
    theme_minimal() +
    theme(legend.position = "none", plot.title = element_text(face = "bold"))
  
  combined <- grid.arrange(p1, p2, ncol = 2)
  ggsave("figures/figure4_model_comparison_R.png", combined, width = 14, height = 5, dpi = 300)
  print("Saved: figures/figure4_model_comparison_R.png")
}

# Main execution
cat("========================================\n")
cat("Gene Expression Prediction Visualizations\n")
cat("========================================\n\n")

cat("Generating data...\n")
data <- generate_data(n = 2000, correlation = 0.85)

cat("\nGenerating visualizations...\n\n")
metrics <- plot_model_performance(data)
plot_error_analysis(data)
plot_cell_type_performance()
plot_model_comparison()

cat("\n========================================\n")
cat("Summary Statistics:\n")
cat("========================================\n")
cat(sprintf("Pearson Correlation: %.3f\n", metrics$correlation))
cat(sprintf("R² Score: %.3f\n", metrics$r2))
cat("\nAll visualizations saved to 'figures/' directory\n")
cat("========================================\n")

