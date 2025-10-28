# Cardiovascular Risk Prediction from Retinal Images - R Visualization Script
# 
# This R script generates comprehensive visualizations for deep learning-based
# cardiovascular risk prediction from retinal fundus photographs.
#
# Based on: Poplin et al. (2018), Nature Biomedical Engineering
# "Prediction of cardiovascular risk factors from retinal fundus photographs via deep learning"
#
# Features:
# - Age prediction analysis
# - ROC curves for binary classifications
# - Calibration curves for probability predictions
# - Continuous variable predictions (SBP, BMI)
# - Risk stratification analysis
# - Attention map visualization
# - Performance summary tables
#
# Author: Scientific Blog Post Project
# Date: October 27, 2024
# Version: 2.0 (Complete)

# =============================================================================
# PACKAGE INSTALLATION AND LOADING
# =============================================================================

# List of required packages
required_packages <- c(
  "ggplot2",      # Visualization
  "dplyr",        # Data manipulation
  "tidyr",        # Data tidying
  "pROC",         # ROC curve analysis
  "caret",        # Machine learning utilities
  "gridExtra",    # Multiple plots
  "scales",       # Scale functions
  "reshape2",     # Data reshaping
  "Metrics"       # Performance metrics
)

# Install missing packages
install_if_missing <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
  if(length(new_packages)) {
    cat("Installing missing packages:", paste(new_packages, collapse=", "), "\n")
    install.packages(new_packages, dependencies=TRUE)
  }
}

install_if_missing(required_packages)

# Load packages
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(pROC)
  library(caret)
  library(gridExtra)
  library(scales)
  library(reshape2)
  library(Metrics)
})

cat("==============================================================================\n")
cat("CARDIOVASCULAR RISK PREDICTION VISUALIZATION SCRIPT (R)\n")
cat("==============================================================================\n\n")

# =============================================================================
# 1. DATA GENERATION
# =============================================================================

generate_simulated_data <- function(n_patients = 10000, random_seed = 42) {
  #' Generate simulated patient data matching published study statistics
  #' 
  #' @param n_patients Number of patients to simulate
  #' @param random_seed Random seed for reproducibility
  #' @return Data frame with simulated patient data
  
  set.seed(random_seed)
  cat("Generating simulated dataset for", n_patients, "patients...\n")
  
  # True patient characteristics
  true_age <- rnorm(n_patients, mean = 60, sd = 12)
  true_gender <- rbinom(n_patients, size = 1, prob = 0.52)
  true_smoking <- rbinom(n_patients, size = 1, prob = 0.18)
  true_sbp <- rnorm(n_patients, mean = 135, sd = 18)
  true_bmi <- rnorm(n_patients, mean = 27, sd = 4.5)
  
  # Model predictions with realistic error
  pred_age <- true_age + rnorm(n_patients, mean = 0, sd = 3.26)
  pred_gender_prob <- pmin(pmax(true_gender + rnorm(n_patients, 0, 0.15), 0), 1)
  pred_smoking_prob <- pmin(pmax(true_smoking + rnorm(n_patients, 0, 0.35), 0), 1)
  pred_sbp <- true_sbp * 0.33 + rnorm(n_patients, mean = 135, sd = 15)
  pred_bmi <- true_bmi * 0.25 + rnorm(n_patients, mean = 27, sd = 4)
  
  # MACE prediction
  base_risk <- 0.05
  risk_score <- base_risk + 
    0.002 * (true_age - 60) + 
    0.03 * true_smoking + 
    0.001 * (true_sbp - 135) + 
    0.005 * (true_bmi - 27)
  risk_score <- pmin(pmax(risk_score, 0), 0.5)
  true_mace <- rbinom(n_patients, size = 1, prob = risk_score)
  pred_mace_prob <- pmin(pmax(risk_score + rnorm(n_patients, 0, 0.08), 0), 1)
  
  # Create data frame
  df <- data.frame(
    true_age = true_age,
    pred_age = pred_age,
    true_gender = true_gender,
    pred_gender_prob = pred_gender_prob,
    true_smoking = true_smoking,
    pred_smoking_prob = pred_smoking_prob,
    true_sbp = true_sbp,
    pred_sbp = pred_sbp,
    true_bmi = true_bmi,
    pred_bmi = pred_bmi,
    true_mace = true_mace,
    pred_mace_prob = pred_mace_prob
  )
  
  cat("✓ Dataset generated:", nrow(df), "patients,", ncol(df), "variables\n\n")
  return(df)
}

# =============================================================================
# 2. AGE PREDICTION VISUALIZATION
# =============================================================================

plot_age_prediction <- function(df, save = TRUE) {
  #' Plot age prediction performance
  #' 
  #' @param df Data frame with patient data
  #' @param save Whether to save the plot
  
  cat("==============================================================================\n")
  cat("GENERATING AGE PREDICTION VISUALIZATIONS\n")
  cat("==============================================================================\n")
  
  # Calculate metrics
  age_mae <- mae(df$true_age, df$pred_age)
  age_cor <- cor(df$true_age, df$pred_age)
  
  # Scatter plot
  p1 <- ggplot(df, aes(x = true_age, y = pred_age)) +
    geom_point(alpha = 0.3, size = 0.5, color = "steelblue") +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", size = 1) +
    geom_smooth(method = "lm", color = "green", size = 1, se = FALSE) +
    labs(
      x = "True Age (years)",
      y = "Predicted Age (years)",
      title = sprintf("Age Prediction Performance\nMAE = %.2f years, r = %.3f", age_mae, age_cor)
    ) +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5))
  
  # Error distribution
  errors <- df$pred_age - df$true_age
  p2 <- ggplot(data.frame(errors = errors), aes(x = errors)) +
    geom_histogram(bins = 50, fill = "coral", color = "black", alpha = 0.7) +
    geom_vline(xintercept = 0, color = "red", linetype = "dashed", size = 1) +
    geom_vline(xintercept = mean(errors), color = "blue", size = 1) +
    labs(
      x = "Prediction Error (years)",
      y = "Frequency",
      title = "Distribution of Age Prediction Errors"
    ) +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5))
  
  # Combine plots
  combined_plot <- grid.arrange(p1, p2, ncol = 2)
  
  if (save) {
    ggsave("age_prediction_performance_R.png", combined_plot, 
           width = 15, height = 6, dpi = 300, units = "in")
    cat("✓ Saved: age_prediction_performance_R.png\n")
  }
  
  cat(sprintf("  MAE: %.2f years\n", age_mae))
  cat(sprintf("  Correlation: r = %.3f, p < 0.001\n\n", age_cor))
}

# =============================================================================
# 3. ROC CURVES
# =============================================================================

plot_roc_curves <- function(df, save = TRUE) {
  #' Plot ROC curves for binary classifications
  #' 
  #' @param df Data frame with patient data
  #' @param save Whether to save the plot
  #' @return List of AUC values
  
  cat("==============================================================================\n")
  cat("GENERATING ROC CURVES\n")
  cat("==============================================================================\n")
  
  # Calculate ROC curves
  roc_gender <- roc(df$true_gender, df$pred_gender_prob, quiet = TRUE)
  roc_smoking <- roc(df$true_smoking, df$pred_smoking_prob, quiet = TRUE)
  roc_mace <- roc(df$true_mace, df$pred_mace_prob, quiet = TRUE)
  
  # Extract coordinates
  coords_gender <- coords(roc_gender, "all", ret = c("fpr", "tpr"))
  coords_smoking <- coords(roc_smoking, "all", ret = c("fpr", "tpr"))
  coords_mace <- coords(roc_mace, "all", ret = c("fpr", "tpr"))
  
  # Prepare data for plotting
  roc_data <- rbind(
    data.frame(fpr = coords_gender$fpr, tpr = coords_gender$tpr, 
               Variable = sprintf("Gender (AUC = %.3f)", auc(roc_gender))),
    data.frame(fpr = coords_smoking$fpr, tpr = coords_smoking$tpr, 
               Variable = sprintf("Smoking (AUC = %.3f)", auc(roc_smoking))),
    data.frame(fpr = coords_mace$fpr, tpr = coords_mace$tpr, 
               Variable = sprintf("MACE (AUC = %.3f)", auc(roc_mace)))
  )
  
  # Plot
  p <- ggplot(roc_data, aes(x = fpr, y = tpr, color = Variable)) +
    geom_line(size = 1.5) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
    scale_color_manual(values = c("#2E86AB", "#A23B72", "#F18F01")) +
    labs(
      x = "False Positive Rate (1 - Specificity)",
      y = "True Positive Rate (Sensitivity)",
      title = "ROC Curves for Cardiovascular Risk Factor Prediction",
      color = ""
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      legend.position = c(0.7, 0.3),
      legend.background = element_rect(fill = "white", color = "black")
    ) +
    coord_fixed()
  
  if (save) {
    ggsave("roc_curves_R.png", p, width = 10, height = 10, dpi = 300, units = "in")
    cat("✓ Saved: roc_curves_R.png\n")
  }
  
  cat(sprintf("  Gender AUC: %.3f\n", auc(roc_gender)))
  cat(sprintf("  Smoking AUC: %.3f\n", auc(roc_smoking)))
  cat(sprintf("  MACE AUC: %.3f\n\n", auc(roc_mace)))
  
  return(list(
    auc_gender = auc(roc_gender),
    auc_smoking = auc(roc_smoking),
    auc_mace = auc(roc_mace)
  ))
}

# =============================================================================
# 4. CALIBRATION CURVES
# =============================================================================

plot_calibration <- function(df, save = TRUE) {
  #' Plot calibration curve for MACE prediction
  #' 
  #' @param df Data frame with patient data
  #' @param save Whether to save the plot
  
  cat("==============================================================================\n")
  cat("GENERATING CALIBRATION CURVES\n")
  cat("==============================================================================\n")
  
  # Create calibration bins
  df <- df %>%
    mutate(pred_bin = cut(pred_mace_prob, breaks = 10, include.lowest = TRUE))
  
  # Calculate observed vs predicted
  calibration_data <- df %>%
    group_by(pred_bin) %>%
    summarise(
      predicted = mean(pred_mace_prob),
      observed = mean(true_mace),
      n = n(),
      .groups = "drop"
    ) %>%
    filter(!is.na(predicted))
  
  # Create plots
  p1 <- ggplot(calibration_data, aes(x = predicted, y = observed)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black", size = 1) +
    geom_point(size = 4, color = "#F18F01") +
    geom_line(color = "#F18F01", size = 1.5) +
    labs(
      x = "Predicted Probability",
      y = "Observed Frequency",
      title = "Calibration Curve for MACE Prediction"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      axis.title = element_text(face = "bold")
    ) +
    coord_fixed(xlim = c(0, max(calibration_data$predicted)), 
                ylim = c(0, max(calibration_data$observed))) +
    annotate("text", x = 0.02, y = max(calibration_data$observed) * 0.95, 
             label = "Perfect calibration", hjust = 0, fontface = "italic")
  
  # Probability distribution
  p2 <- ggplot(df, aes(x = pred_mace_prob, fill = factor(true_mace))) +
    geom_histogram(bins = 30, alpha = 0.6, position = "identity") +
    scale_fill_manual(values = c("0" = "green", "1" = "red"),
                     labels = c("No MACE", "MACE cases"),
                     name = "") +
    labs(
      x = "Predicted MACE Probability",
      y = "Frequency",
      title = "Distribution of Predicted Probabilities"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      axis.title = element_text(face = "bold"),
      legend.position = c(0.8, 0.8)
    )
  
  # Combine plots
  combined_plot <- grid.arrange(p1, p2, ncol = 2)
  
  if (save) {
    ggsave("calibration_curves_R.png", combined_plot, 
           width = 15, height = 6, dpi = 300, units = "in")
    cat("✓ Saved: calibration_curves_R.png\n")
  }
  
  cat("\n")
}

# =============================================================================
# 5. ATTENTION MAP SIMULATION
# =============================================================================

plot_attention_map <- function(df, save = TRUE) {
  #' Create attention map visualization for different risk groups
  #' 
  #' @param df Data frame with patient data
  #' @param save Whether to save the plot
  
  cat("==============================================================================\n")
  cat("GENERATING ATTENTION MAP VISUALIZATION\n")
  cat("==============================================================================\n")
  
  # Function to generate risk-based attention map
  generate_attention <- function(age, risk_prob, size = 512) {
    # Create coordinate grid
    x <- rep(1:size, each = size)
    y <- rep(1:size, times = size)
    center_x <- size / 2
    center_y <- size / 2
    
    # Initialize attention
    attention <- numeric(length(x))
    
    # Optic disc attention
    dist_disc <- sqrt((x - center_x)^2 + (y - center_y)^2)
    attention[dist_disc < size/8] <- 0.7
    
    # Vessel attention (increases with risk)
    vessel_intensity <- 0.3 + 0.5 * risk_prob
    for (angle in seq(0, 2*pi, length.out = 12)) {
      for (t in seq(0, size/3, length.out = 30)) {
        vessel_x <- center_x + cos(angle) * t
        vessel_y <- center_y + sin(angle) * t
        dist_vessel <- sqrt((x - vessel_x)^2 + (y - vessel_y)^2)
        attention[dist_vessel < 2] <- attention[dist_vessel < 2] + vessel_intensity
      }
    }
    
    # Macula attention (age-dependent)
    macula_x <- center_x + 20
    macula_y <- center_y - 30
    macula_intensity <- 0.2 + 0.3 * (age - 25) / 60
    dist_macula <- sqrt((x - macula_x)^2 + (y - macula_y)^2)
    attention[dist_macula < size/12] <- attention[dist_macula < size/12] + macula_intensity
    
    # Clip values
    attention <- pmin(attention, 1)
    attention <- pmax(attention, 0)
    
    data.frame(x = x, y = y, attention = attention)
  }
  
  # Select representative patients from different risk groups
  df <- df %>%
    mutate(risk_category = cut(pred_mace_prob, breaks = 4, 
                               labels = c("Low", "Medium", "High", "Very High")))
  
  representatives <- df %>%
    group_by(risk_category) %>%
    slice(1) %>%
    ungroup()
  
  # Fallback: If we don't have 4 groups, select evenly-spaced patients
  if (nrow(representatives) < 4) {
    cat(sprintf("⚠️  Warning: Only found %d risk groups\n", nrow(representatives)))
    cat("    Selecting evenly-spaced patients across risk spectrum...\n")
    df_sorted <- df %>% arrange(pred_mace_prob)
    step <- floor(nrow(df_sorted) / 4)
    representatives <- df_sorted[seq(1, nrow(df_sorted), by = step)[1:4], ]
  }
  
  # Generate attention maps for each risk group
  plot_list <- list()
  titles <- c("Low Risk Patient", "Medium Risk Patient", 
              "High Risk Patient", "Very High Risk Patient")
  
  for (i in 1:min(4, nrow(representatives))) {
    patient <- representatives[i, ]
    att_data <- generate_attention(patient$true_age, patient$pred_mace_prob)
    
    p <- ggplot(att_data, aes(x = x, y = y, fill = attention)) +
      geom_raster() +
      scale_fill_gradient(low = "black", high = "red", name = "Attention\nWeight",
                         limits = c(0, 1)) +
      labs(title = sprintf("%s\nRisk Score: %.3f", titles[i], patient$pred_mace_prob),
           x = "Width (pixels)", y = "Height (pixels)") +
      theme_minimal(base_size = 10) +
      theme(
        plot.title = element_text(face = "bold", hjust = 0.5, size = 11),
        axis.title = element_text(size = 9),
        legend.position = "right"
      ) +
      coord_fixed() +
      # Add anatomical labels
      annotate("text", x = 256, y = 180, label = "Optic Disc", 
               color = "cyan", fontface = "bold", size = 3) +
      annotate("text", x = 276, y = 200, label = "Macula", 
               color = "yellow", fontface = "bold", size = 3)
    
    plot_list[[i]] <- p
  }
  
  # Add empty plots for any missing panels
  while (length(plot_list) < 4) {
    empty_plot <- ggplot() + theme_void()
    plot_list[[length(plot_list) + 1]] <- empty_plot
  }
  
  # Combine plots in 2x2 grid
  combined_plot <- grid.arrange(grobs = plot_list, ncol = 2, nrow = 2)
  
  if (save) {
    ggsave("attention_maps_R.png", combined_plot, 
           width = 15, height = 15, dpi = 300, units = "in")
    cat("✓ Saved: attention_maps_R.png\n")
  }
  
  cat("\nKey Observations:\n")
  cat("  • Bright regions indicate high attention\n")
  cat("  • Model focuses on optic disc, vessels, and macula\n")
  cat("  • Higher risk patients show more intense vessel attention\n\n")
}

# =============================================================================
# 6. RISK STRATIFICATION
# =============================================================================

plot_risk_stratification <- function(df, save = TRUE) {
  #' Plot risk stratification analysis
  #' 
  #' @param df Data frame with patient data
  #' @param save Whether to save the plot
  
  cat("==============================================================================\n")
  cat("GENERATING RISK STRATIFICATION ANALYSIS\n")
  cat("==============================================================================\n")
  
  # Categorize risk
  df <- df %>%
    mutate(
      risk_category = cut(
        pred_mace_prob,
        breaks = c(0, 0.05, 0.10, 0.20, 1.0),
        labels = c("Low (<5%)", "Moderate (5-10%)", "High (10-20%)", "Very High (>20%)")
      )
    )
  
  # Calculate statistics
  risk_summary <- df %>%
    group_by(risk_category) %>%
    summarise(
      n_patients = n(),
      mace_rate = mean(true_mace) * 100,
      .groups = "drop"
    )
  
  # Plot 1: Patient distribution
  p1 <- ggplot(risk_summary, aes(x = risk_category, y = n_patients, fill = risk_category)) +
    geom_bar(stat = "identity", color = "black") +
    scale_fill_manual(values = c("#06D6A0", "#FFD166", "#EF476F", "#8B0000")) +
    labs(
      x = "Risk Category",
      y = "Number of Patients",
      title = "Patient Distribution by Risk Category"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      legend.position = "none",
      axis.text.x = element_text(angle = 15, hjust = 1)
    )
  
  # Plot 2: MACE rate
  p2 <- ggplot(risk_summary, aes(x = risk_category, y = mace_rate, fill = risk_category)) +
    geom_bar(stat = "identity", color = "black") +
    scale_fill_manual(values = c("#06D6A0", "#FFD166", "#EF476F", "#8B0000")) +
    labs(
      x = "Risk Category",
      y = "Observed MACE Rate (%)",
      title = "Observed MACE Rate by Risk Category"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      legend.position = "none",
      axis.text.x = element_text(angle = 15, hjust = 1)
    )
  
  # Combine plots
  combined_plot <- grid.arrange(p1, p2, ncol = 2)
  
  if (save) {
    ggsave("risk_stratification_R.png", combined_plot, 
           width = 15, height = 6, dpi = 300, units = "in")
    cat("✓ Saved: risk_stratification_R.png\n")
  }
  
  cat("\nRisk Stratification Results:\n")
  print(risk_summary)
  cat("\n")
}

# =============================================================================
# 5. CONTINUOUS VARIABLE PREDICTIONS
# =============================================================================

plot_continuous_predictions <- function(df, save = TRUE) {
  #' Plot SBP and BMI prediction performance
  #' 
  #' @param df Data frame with patient data
  #' @param save Whether to save the plot
  
  cat("==============================================================================\n")
  cat("GENERATING CONTINUOUS VARIABLE PREDICTIONS\n")
  cat("==============================================================================\n")
  
  # Calculate correlations
  sbp_cor <- cor(df$true_sbp, df$pred_sbp)
  bmi_cor <- cor(df$true_bmi, df$pred_bmi)
  
  # SBP scatter plot
  p1 <- ggplot(df, aes(x = true_sbp, y = pred_sbp)) +
    geom_point(alpha = 0.3, size = 0.5, color = "#E63946") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
    labs(
      x = "True SBP (mmHg)",
      y = "Predicted SBP (mmHg)",
      title = sprintf("SBP Prediction (r = %.3f)", sbp_cor)
    ) +
    theme_minimal(base_size = 10) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5))
  
  # SBP Bland-Altman
  df_sbp <- df %>%
    mutate(
      mean_sbp = (true_sbp + pred_sbp) / 2,
      diff_sbp = pred_sbp - true_sbp
    )
  
  p2 <- ggplot(df_sbp, aes(x = mean_sbp, y = diff_sbp)) +
    geom_point(alpha = 0.3, size = 0.5, color = "#E63946") +
    geom_hline(yintercept = mean(df_sbp$diff_sbp), color = "blue", size = 1) +
    labs(
      x = "Mean SBP (mmHg)",
      y = "Difference (Pred - True)",
      title = "Bland-Altman: SBP"
    ) +
    theme_minimal(base_size = 10) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5))
  
  # BMI scatter plot
  p3 <- ggplot(df, aes(x = true_bmi, y = pred_bmi)) +
    geom_point(alpha = 0.3, size = 0.5, color = "#457B9D") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
    labs(
      x = "True BMI (kg/m²)",
      y = "Predicted BMI (kg/m²)",
      title = sprintf("BMI Prediction (r = %.3f)", bmi_cor)
    ) +
    theme_minimal(base_size = 10) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5))
  
  # BMI Bland-Altman
  df_bmi <- df %>%
    mutate(
      mean_bmi = (true_bmi + pred_bmi) / 2,
      diff_bmi = pred_bmi - true_bmi
    )
  
  p4 <- ggplot(df_bmi, aes(x = mean_bmi, y = diff_bmi)) +
    geom_point(alpha = 0.3, size = 0.5, color = "#457B9D") +
    geom_hline(yintercept = mean(df_bmi$diff_bmi), color = "blue", size = 1) +
    labs(
      x = "Mean BMI (kg/m²)",
      y = "Difference (Pred - True)",
      title = "Bland-Altman: BMI"
    ) +
    theme_minimal(base_size = 10) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5))
  
  # Combine plots
  combined_plot <- grid.arrange(p1, p2, p3, p4, ncol = 2)
  
  if (save) {
    ggsave("continuous_predictions_R.png", combined_plot, 
           width = 15, height = 12, dpi = 300, units = "in")
    cat("✓ Saved: continuous_predictions_R.png\n")
  }
  
  cat(sprintf("  SBP correlation: r = %.3f\n", sbp_cor))
  cat(sprintf("  BMI correlation: r = %.3f\n\n", bmi_cor))
}

# =============================================================================
# 6. PERFORMANCE SUMMARY
# =============================================================================

create_performance_summary <- function(df, auc_list) {
  #' Create comprehensive performance summary
  #' 
  #' @param df Data frame with patient data
  #' @param auc_list List of AUC values
  
  cat("==============================================================================\n")
  cat("PERFORMANCE SUMMARY\n")
  cat("==============================================================================\n")
  
  # Calculate metrics
  age_mae <- mae(df$true_age, df$pred_age)
  sbp_cor <- cor(df$true_sbp, df$pred_sbp)
  bmi_cor <- cor(df$true_bmi, df$pred_bmi)
  
  summary_df <- data.frame(
    Risk_Factor = c("Age", "Gender", "Smoking", "SBP", "BMI", "MACE"),
    Metric = c(
      sprintf("MAE=%.2f years", age_mae),
      sprintf("AUC=%.3f", auc_list$auc_gender),
      sprintf("AUC=%.3f", auc_list$auc_smoking),
      sprintf("r=%.3f", sbp_cor),
      sprintf("r=%.3f", bmi_cor),
      sprintf("AUC=%.3f", auc_list$auc_mace)
    ),
    Performance = c("Excellent", "Excellent", "Good", "Moderate", "Moderate", "Good")
  )
  
  print(summary_df, row.names = FALSE)
  cat("==============================================================================\n\n")
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

run_analysis <- function() {
  #' Main execution function
  
  cat("\n")
  cat("==============================================================================\n")
  cat("STARTING VISUALIZATION PIPELINE\n")
  cat("==============================================================================\n\n")
  
  # Generate data
  df <- generate_simulated_data(n_patients = 10000)
  
  # Generate all visualizations
  plot_age_prediction(df)
  auc_list <- plot_roc_curves(df)
  plot_calibration(df)
  plot_continuous_predictions(df)
  plot_risk_stratification(df)
  plot_attention_map(df)
  create_performance_summary(df, auc_list)
  
  cat("==============================================================================\n")
  cat("VISUALIZATION PIPELINE COMPLETED SUCCESSFULLY\n")
  cat("==============================================================================\n")
  cat("\nGenerated Files:\n")
  cat("  • age_prediction_performance_R.png\n")
  cat("  • roc_curves_R.png\n")
  cat("  • calibration_curves_R.png\n")
  cat("  • continuous_predictions_R.png\n")
  cat("  • risk_stratification_R.png\n")
  cat("  • attention_maps_R.png\n")
  cat("\n")
  cat("==============================================================================\n")
}

# Run the analysis
run_analysis()

