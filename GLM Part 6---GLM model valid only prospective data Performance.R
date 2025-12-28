# Complete Prospective Data Validation with Trained GLM Model
# Load required packages
library(readxl)
library(caret)
library(pROC)
library(openxlsx)
library(ggplot2)
library(dplyr)
library(reshape2)
library(gridExtra)

# ==================== Configuration ====================
cat("=== Prospective Data Validation Configuration ===\n")

# File paths
model_path <- "final_glm_model.rds"
features_path <- "final_features.rds"
train_data_path <- "C:/Users/Bo Wang/Desktop/Training Cohort.xlsx"
prospective_data_path <- "C:/Users/Bo Wang/Desktop/Prospective Cohort.xlsx"

# Output directories
output_dir <- "Prospective_Validation_Results"
plots_dir <- file.path(output_dir, "Plots")

# Create directories
if(!dir.exists(output_dir)) dir.create(output_dir)
if(!dir.exists(plots_dir)) dir.create(plots_dir)

# ==================== Load Model and Data ====================
cat("Step 1: Loading Model and Data...\n")

# Load trained model and features
final_glm_model <- readRDS(model_path)
final_features <- readRDS(features_path)

cat("Model loaded successfully\n")
cat("Number of features:", length(final_features), "\n")
cat("Model type: GLM (Logistic Regression)\n")
cat("Model family:", final_glm_model$family$family, "\n")
cat("Model link function:", final_glm_model$family$link, "\n")

# Load prospective data
prospective_data <- read_excel(prospective_data_path)
cat("Prospective data dimensions:", dim(prospective_data), "\n")

# Check for Diagnosis column
has_labels <- "Diagnosis" %in% names(prospective_data)
if(has_labels) {
  prospective_data$Diagnosis <- as.factor(prospective_data$Diagnosis)
  cat("Diagnosis distribution in prospective data:\n")
  print(table(prospective_data$Diagnosis))
} else {
  cat("No Diagnosis column found - performing unsupervised prediction only\n")
  prospective_data$Diagnosis <- NA
}

# Load training data for reference (if needed for imputation)
train_data <- read_excel(train_data_path)
train_data$Diagnosis <- as.factor(train_data$Diagnosis)

# ==================== Data Preprocessing ====================
cat("Step 2: Data Preprocessing...\n")

# Check for missing values
missing_count <- colSums(is.na(prospective_data))
if(sum(missing_count) > 0) {
  cat("Missing values found. Imputing using training data statistics...\n")
  
  # Simple imputation using training data
  for(feature in names(prospective_data)) {
    if(feature %in% names(train_data) && feature != "Diagnosis") {
      na_indices <- is.na(prospective_data[[feature]])
      if(any(na_indices)) {
        if(is.numeric(train_data[[feature]])) {
          # Use median for numeric features
          impute_value <- median(train_data[[feature]], na.rm = TRUE)
          prospective_data[na_indices, feature] <- impute_value
        } else {
          # Use mode for categorical features
          freq_table <- table(train_data[[feature]])
          impute_value <- names(freq_table)[which.max(freq_table)]
          prospective_data[na_indices, feature] <- impute_value
        }
      }
    }
  }
}

# Check and align features
missing_features <- setdiff(final_features, names(prospective_data))
if(length(missing_features) > 0) {
  cat("Adding missing features:", paste(missing_features, collapse = ", "), "\n")
  for(feature in missing_features) {
    if(feature %in% names(train_data)) {
      if(is.numeric(train_data[[feature]])) {
        prospective_data[[feature]] <- median(train_data[[feature]], na.rm = TRUE)
      } else {
        freq_table <- table(train_data[[feature]])
        prospective_data[[feature]] <- names(freq_table)[which.max(freq_table)]
      }
    } else {
      prospective_data[[feature]] <- 0  # Default value
    }
  }
}

# Prepare features for prediction
prospective_features <- prospective_data[, final_features, drop = FALSE]

# Ensure factor levels match training data
for(feature in final_features) {
  if(is.factor(train_data[[feature]]) && is.character(prospective_features[[feature]])) {
    prospective_features[[feature]] <- factor(prospective_features[[feature]], 
                                              levels = levels(train_data[[feature]]))
  }
}

cat("Data preprocessing completed. Final feature dimensions:", dim(prospective_features), "\n")

# ==================== Model Prediction ====================
cat("Step 3: Making Predictions...\n")

# GLM prediction function
predict_glm <- function(model, newdata) {
  # Get probabilities
  probabilities <- predict(model, newdata = newdata, type = "response")
  
  # Get class predictions (threshold = 0.5)
  predictions <- ifelse(probabilities > 0.5, 
                        levels(train_data$Diagnosis)[2], 
                        levels(train_data$Diagnosis)[1])
  predictions <- factor(predictions, levels = levels(train_data$Diagnosis))
  
  return(list(probabilities = probabilities, predictions = predictions))
}

# Make predictions
pred_results <- predict_glm(final_glm_model, prospective_features)
predictions <- pred_results$predictions
probabilities <- pred_results$probabilities

# Create results dataframe
results <- data.frame(
  SampleID = if("SampleID" %in% names(prospective_data)) prospective_data$SampleID else 1:nrow(prospective_data),
  Actual_Diagnosis = if(has_labels) prospective_data$Diagnosis else NA,
  Predicted_Diagnosis = predictions,
  Predicted_Probability = probabilities
)

cat("Predictions completed successfully\n")

# ==================== Performance Evaluation ====================
cat("Step 4: Performance Evaluation...\n")

performance_metrics <- NULL
confusion_matrix <- NULL
roc_obj <- NULL

if(has_labels && !all(is.na(prospective_data$Diagnosis))) {
  # Ensure factor levels match
  actual <- factor(prospective_data$Diagnosis)
  predicted <- factor(predictions, levels = levels(actual))
  
  # Calculate confusion matrix
  confusion_matrix <- confusionMatrix(predicted, actual)
  cat("Confusion Matrix:\n")
  print(confusion_matrix$table)
  
  # Calculate performance metrics
  calculate_metrics <- function(true, pred, probs) {
    true_numeric <- as.numeric(true) - 1
    pred_numeric <- as.numeric(pred) - 1
    
    cm <- table(Predicted = pred_numeric, Actual = true_numeric)
    
    # Ensure 2x2 matrix
    if(nrow(cm) == 1) {
      if(rownames(cm) == "0") {
        cm <- rbind(cm, c(0, 0))
      } else {
        cm <- rbind(c(0, 0), cm)
      }
      rownames(cm) <- c("0", "1")
    }
    if(ncol(cm) == 1) {
      if(colnames(cm) == "0") {
        cm <- cbind(cm, c(0, 0))
      } else {
        cm <- cbind(c(0, 0), cm)
      }
      colnames(cm) <- c("0", "1")
    }
    
    TP <- cm[2, 2]
    TN <- cm[1, 1]
    FP <- cm[2, 1]
    FN <- cm[1, 2]
    
    accuracy <- (TP + TN) / (TP + TN + FP + FN)
    sensitivity <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
    specificity <- ifelse((TN + FP) > 0, TN / (TN + FP), 0)
    ppv <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
    npv <- ifelse((TN + FN) > 0, TN / (TN + FN), 0)
    f1 <- ifelse((sensitivity + ppv) > 0, 2 * sensitivity * ppv / (sensitivity + ppv), 0)
    balanced_accuracy <- (sensitivity + specificity) / 2
    
    # AUC calculation
    auc_value <- tryCatch({
      if(length(unique(true_numeric)) == 2) {
        roc_obj <- roc(true_numeric, probs)
        auc(roc_obj)
      } else {
        0.5
      }
    }, error = function(e) 0.5)
    
    return(list(
      Accuracy = accuracy,
      Sensitivity = sensitivity,
      Specificity = specificity,
      PPV = ppv,
      NPV = npv,
      F1_Score = f1,
      Balanced_Accuracy = balanced_accuracy,
      AUC = auc_value,
      Confusion_Matrix = cm
    ))
  }
  
  performance_metrics <- calculate_metrics(actual, predicted, results$Predicted_Probability)
  
  # ROC curve for binary classification
  if(length(levels(actual)) == 2) {
    actual_numeric <- as.numeric(actual) - 1
    roc_obj <- roc(actual_numeric, results$Predicted_Probability)
    cat("AUC:", performance_metrics$AUC, "\n")
  }
  
  cat("Performance Metrics:\n")
  print(data.frame(Metric = names(performance_metrics)[1:8], 
                   Value = unlist(performance_metrics[1:8])))
} else {
  cat("No ground truth labels available for performance evaluation\n")
}

# ==================== Comprehensive Visualization ====================
cat("Step 5: Generating Visualizations...\n")

# 1. Prediction Distribution
p1 <- ggplot(results, aes(x = Predicted_Diagnosis, fill = Predicted_Diagnosis)) +
  geom_bar(alpha = 0.7, show.legend = FALSE) +
  geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5) +
  labs(title = "Prediction Distribution - Prospective Data (GLM)",
       x = "Predicted Diagnosis",
       y = "Count") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

# 2. Probability Distribution
p2 <- ggplot(results, aes(x = Predicted_Probability)) +
  geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7, color = "black") +
  labs(title = "Prediction Probability Distribution (GLM)",
       x = "Predicted Probability",
       y = "Count") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

# 3. Confusion Matrix Heatmap (if labels available)
if(!is.null(confusion_matrix)) {
  cm_data <- as.data.frame(confusion_matrix$table)
  p3 <- ggplot(cm_data, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
    scale_fill_gradient(low = "lightblue", high = "darkblue") +
    labs(title = "Confusion Matrix Heatmap (GLM)",
         x = "Actual Diagnosis",
         y = "Predicted Diagnosis") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
} else {
  p3 <- ggplot() + 
    annotate("text", x = 0.5, y = 0.5, label = "No Confusion Matrix\n(No ground truth labels)", 
             size = 6) +
    theme_void()
}

# 4. ROC Curve (if binary classification and labels available)
if(!is.null(roc_obj)) {
  roc_data <- data.frame(
    Sensitivity = roc_obj$sensitivities,
    Specificity = roc_obj$specificities
  )
  roc_data$FPR <- 1 - roc_data$Specificity
  
  p4 <- ggplot(roc_data, aes(x = FPR, y = Sensitivity)) +
    geom_line(color = "blue", size = 1.2) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    labs(title = paste("ROC Curve (AUC =", round(performance_metrics$AUC, 3), ") - GLM"),
         x = "False Positive Rate (1 - Specificity)",
         y = "True Positive Rate (Sensitivity)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold")) +
    annotate("text", x = 0.7, y = 0.2, 
             label = paste("AUC =", round(performance_metrics$AUC, 3)),
             size = 5, color = "blue")
} else {
  p4 <- ggplot() + 
    annotate("text", x = 0.5, y = 0.5, 
             label = "No ROC Curve\n(Not binary classification or no labels)", 
             size = 5) +
    theme_void()
}

# 5. Calibration Curve (if labels available)
if(has_labels && !all(is.na(prospective_data$Diagnosis)) && length(unique(prospective_data$Diagnosis)) == 2) {
  calibration_data <- data.frame(
    Probability = results$Predicted_Probability,
    True_Label = as.numeric(prospective_data$Diagnosis) - 1
  )
  
  # Create bins
  calibration_data$Bin <- cut(calibration_data$Probability, 
                              breaks = seq(0, 1, by = 0.1), 
                              include.lowest = TRUE)
  
  calibration_summary <- calibration_data %>%
    group_by(Bin) %>%
    summarise(
      Mean_Predicted = mean(Probability),
      Mean_Actual = mean(True_Label),
      Count = n()
    ) %>%
    filter(Count > 0)
  
  p5 <- ggplot(calibration_summary, aes(x = Mean_Predicted, y = Mean_Actual)) +
    geom_point(aes(size = Count), color = "blue", alpha = 0.7) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    geom_smooth(method = "loess", color = "darkgreen", se = TRUE) +
    labs(title = "Calibration Plot (GLM)",
         x = "Mean Predicted Probability",
         y = "Observed Fraction Positive") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
} else {
  p5 <- ggplot() + 
    annotate("text", x = 0.5, y = 0.5, 
             label = "No Calibration Plot\n(Not binary classification or no labels)", 
             size = 5) +
    theme_void()
}

# 6. Threshold Analysis (if labels available)
if(has_labels && !all(is.na(prospective_data$Diagnosis)) && length(unique(prospective_data$Diagnosis)) == 2) {
  thresholds <- seq(0.1, 0.9, by = 0.05)
  threshold_analysis <- data.frame(
    Threshold = thresholds,
    Sensitivity = numeric(length(thresholds)),
    Specificity = numeric(length(thresholds)),
    Accuracy = numeric(length(thresholds))
  )
  
  actual_numeric <- as.numeric(prospective_data$Diagnosis) - 1
  
  for(i in 1:length(thresholds)) {
    pred_binary <- ifelse(results$Predicted_Probability > thresholds[i], 1, 0)
    
    cm <- table(Predicted = pred_binary, Actual = actual_numeric)
    
    # Ensure 2x2 matrix
    if(nrow(cm) == 1) {
      if(rownames(cm) == "0") {
        cm <- rbind(cm, c(0, 0))
      } else {
        cm <- rbind(c(0, 0), cm)
      }
      rownames(cm) <- c("0", "1")
    }
    if(ncol(cm) == 1) {
      if(colnames(cm) == "0") {
        cm <- cbind(cm, c(0, 0))
      } else {
        cm <- cbind(c(0, 0), cm)
      }
      colnames(cm) <- c("0", "1")
    }
    
    TP <- cm[2, 2]
    TN <- cm[1, 1]
    FP <- cm[2, 1]
    FN <- cm[1, 2]
    
    threshold_analysis$Sensitivity[i] <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
    threshold_analysis$Specificity[i] <- ifelse((TN + FP) > 0, TN / (TN + FP), 0)
    threshold_analysis$Accuracy[i] <- (TP + TN) / (TP + TN + FP + FN)
  }
  
  threshold_long <- melt(threshold_analysis, id.vars = "Threshold", 
                         variable.name = "Metric", value.name = "Value")
  
  p6 <- ggplot(threshold_long, aes(x = Threshold, y = Value, color = Metric)) +
    geom_line(size = 1.2) +
    geom_point(size = 1) +
    labs(title = "Performance Metrics vs Decision Threshold (GLM)",
         x = "Decision Threshold",
         y = "Performance Value") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          legend.position = "top")
  
  # Find optimal threshold (Youden's J statistic)
  youden_j <- threshold_analysis$Sensitivity + threshold_analysis$Specificity - 1
  optimal_idx <- which.max(youden_j)
  optimal_threshold <- threshold_analysis$Threshold[optimal_idx]
  
  # Add optimal threshold to plot
  p6 <- p6 + 
    geom_vline(xintercept = optimal_threshold, linetype = "dashed", color = "red") +
    annotate("text", x = optimal_threshold, y = 0.1, 
             label = paste("Optimal =", round(optimal_threshold, 2)), 
             hjust = -0.1, color = "red")
} else {
  p6 <- ggplot() + 
    annotate("text", x = 0.5, y = 0.5, 
             label = "No Threshold Analysis\n(Not binary classification or no labels)", 
             size = 5) +
    theme_void()
}

# 7. Feature Importance Visualization (GLM coefficients)
if(length(final_features) > 0) {
  # Extract coefficients from GLM model
  coefficients <- coef(final_glm_model)
  # Remove intercept
  feature_coefs <- coefficients[-1]
  
  feature_imp_df <- data.frame(
    Feature = names(feature_coefs),
    Coefficient = as.numeric(feature_coefs),
    Importance = abs(as.numeric(feature_coefs))
  ) %>%
    arrange(desc(Importance)) %>%
    head(15)  # Top 15 features
  
  p7 <- ggplot(feature_imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
    coord_flip() +
    labs(title = "Top 15 Features by Coefficient Importance (GLM)",
         x = "Features",
         y = "Absolute Coefficient Value") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
} else {
  p7 <- ggplot() + 
    annotate("text", x = 0.5, y = 0.5, label = "No Feature Importance Available", size = 5) +
    theme_void()
}

# Save individual plots
ggsave(file.path(plots_dir, "01_Prediction_Distribution.png"), p1, width = 10, height = 6, dpi = 300)
ggsave(file.path(plots_dir, "02_Probability_Distribution.png"), p2, width = 10, height = 6, dpi = 300)
ggsave(file.path(plots_dir, "03_Confusion_Matrix.png"), p3, width = 8, height = 6, dpi = 300)
ggsave(file.path(plots_dir, "04_ROC_Curve.png"), p4, width = 8, height = 6, dpi = 300)
ggsave(file.path(plots_dir, "05_Calibration_Plot.png"), p5, width = 8, height = 6, dpi = 300)
ggsave(file.path(plots_dir, "06_Threshold_Analysis.png"), p6, width = 10, height = 6, dpi = 300)
ggsave(file.path(plots_dir, "07_Feature_Importance.png"), p7, width = 10, height = 8, dpi = 300)

# ==================== Generate Comprehensive Report ====================
cat("Step 6: Generating Comprehensive Report...\n")

# Save results to Excel
wb <- createWorkbook()

# Predictions sheet
addWorksheet(wb, "Predictions")
writeData(wb, "Predictions", results, startRow = 1)

# Performance metrics sheet
if(!is.null(performance_metrics)) {
  addWorksheet(wb, "Performance_Metrics")
  metrics_df <- data.frame(
    Metric = names(performance_metrics)[1:8],
    Value = unlist(performance_metrics[1:8])
  )
  writeData(wb, "Performance_Metrics", metrics_df, startRow = 1)
  
  # Confusion matrix sheet
  addWorksheet(wb, "Confusion_Matrix")
  cm_df <- as.data.frame(performance_metrics$Confusion_Matrix)
  writeData(wb, "Confusion_Matrix", cm_df, startRow = 1)
}

# Data summary sheet
addWorksheet(wb, "Data_Summary")
summary_df <- data.frame(
  Item = c("Prospective Samples", "Features Used", "Has Ground Truth", 
           "Model Type", "Model Family", "Prediction Date"),
  Value = c(nrow(prospective_data), length(final_features), has_labels,
            "GLM (Logistic Regression)", 
            paste(final_glm_model$family$family, "(", final_glm_model$family$link, ")"),
            as.character(Sys.Date()))
)
writeData(wb, "Data_Summary", summary_df, startRow = 1)

# GLM model coefficients sheet
if(length(final_features) > 0) {
  addWorksheet(wb, "GLM_Coefficients")
  coefficients <- coef(final_glm_model)
  coef_df <- data.frame(
    Term = names(coefficients),
    Coefficient = as.numeric(coefficients),
    Odds_Ratio = exp(as.numeric(coefficients))
  )
  writeData(wb, "GLM_Coefficients", coef_df, startRow = 1)
}

# Save threshold analysis if available
if(exists("threshold_analysis")) {
  addWorksheet(wb, "Threshold_Analysis")
  writeData(wb, "Threshold_Analysis", threshold_analysis, startRow = 1)
}

# Model statistics sheet
addWorksheet(wb, "Model_Statistics")
model_stats <- data.frame(
  Statistic = c("AIC", "Null Deviance", "Residual Deviance", "Pseudo R-squared"),
  Value = c(
    round(AIC(final_glm_model), 2),
    round(final_glm_model$null.deviance, 2),
    round(final_glm_model$deviance, 2),
    round(1 - (final_glm_model$deviance / final_glm_model$null.deviance), 4)
  )
)
writeData(wb, "Model_Statistics", model_stats, startRow = 1)

saveWorkbook(wb, file.path(output_dir, "Prospective_Validation_Results.xlsx"), overwrite = TRUE)

# Save R objects
saveRDS(results, file.path(output_dir, "prospective_predictions.rds"))
if(!is.null(performance_metrics)) {
  saveRDS(performance_metrics, file.path(output_dir, "performance_metrics.rds"))
}
if(!is.null(roc_obj)) {
  saveRDS(roc_obj, file.path(output_dir, "roc_object.rds"))
}

# ==================== Final Summary ====================
cat("\n=== PROSPECTIVE VALIDATION COMPLETE ===\n")
cat("Results saved to:", output_dir, "\n")
cat("Plots saved to:", plots_dir, "\n")

cat("\n=== SUMMARY ===\n")
cat("Samples processed:", nrow(prospective_data), "\n")
cat("Features used:", length(final_features), "\n")
cat("Model type: GLM (Logistic Regression)\n")
cat("Model family:", final_glm_model$family$family, "\n")
cat("Model link function:", final_glm_model$family$link, "\n")

if(has_labels && !is.null(performance_metrics)) {
  cat("\n=== PERFORMANCE METRICS ===\n")
  cat("Accuracy:", round(performance_metrics$Accuracy, 3), "\n")
  cat("Sensitivity:", round(performance_metrics$Sensitivity, 3), "\n")
  cat("Specificity:", round(performance_metrics$Specificity, 3), "\n")
  cat("AUC:", round(performance_metrics$AUC, 3), "\n")
  cat("F1 Score:", round(performance_metrics$F1_Score, 3), "\n")
  
  if(exists("optimal_threshold")) {
    cat("Optimal Threshold (Youden):", round(optimal_threshold, 3), "\n")
  }
} else {
  cat("\nNo performance metrics calculated (no ground truth labels)\n")
}

cat("\n=== MODEL STATISTICS ===\n")
cat("AIC:", round(AIC(final_glm_model), 2), "\n")
cat("Null Deviance:", round(final_glm_model$null.deviance, 2), "\n")
cat("Residual Deviance:", round(final_glm_model$deviance, 2), "\n")
cat("Pseudo R-squared:", round(1 - (final_glm_model$deviance / final_glm_model$null.deviance), 4), "\n")

cat("\n=== FILES GENERATED ===\n")
cat("- Prospective_Validation_Results.xlsx (Excel report)\n")
cat("- prospective_predictions.rds (R data)\n")
cat("- 7 visualization plots in Plots/ directory\n")

cat("\nFirst few predictions:\n")
print(head(results[, c("SampleID", "Actual_Diagnosis", "Predicted_Diagnosis", "Predicted_Probability")]))

cat("\n=== VALIDATION COMPLETE ===\n")