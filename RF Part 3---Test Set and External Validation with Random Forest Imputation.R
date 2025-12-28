# Part 3: Test Set and External Validation Set Validation with Random Forest Imputation
# Load required packages
library(readxl)
library(caret)
library(randomForest)
library(mice)
library(pROC)
library(openxlsx)
library(ggplot2)

# Load data and objects from previous parts
train_data <- read_excel("C:/Users/Bo Wang/Desktop/Training Cohort.xlsx")
test_data <- read_excel("C:/Users/Bo Wang/Desktop/Internal Validation Cohort.xlsx")
train_data$Diagnosis <- as.factor(train_data$Diagnosis)
test_data$Diagnosis <- as.factor(test_data$Diagnosis)

final_rf_model <- readRDS("final_rf_model.rds")
final_features <- readRDS("final_features.rds")
train_data_balanced <- readRDS("train_data_balanced.rds")
cv_summary <- readRDS("cv_summary.rds")

# ==================== Load External Validation Set ====================
cat("Loading external validation set...\n")
external_data <- read_excel("C:/Users/Bo Wang/Desktop/External Validation Cohort.xlsx")
external_data$Diagnosis <- as.factor(external_data$Diagnosis)
cat("External validation set dimensions:", dim(external_data), "\n")
cat("External validation set Diagnosis distribution:\n")
print(table(external_data$Diagnosis))

# Create workbook for results - 使用缩短的工作表名称
wb3 <- createWorkbook()
addWorksheet(wb3, "Test_Performance")
addWorksheet(wb3, "External_Performance")
addWorksheet(wb3, "Misclassified_Test")
addWorksheet(wb3, "Misclassified_External")
addWorksheet(wb3, "Prob_Dist_Test")
addWorksheet(wb3, "Prob_Dist_External")
addWorksheet(wb3, "Perf_Comparison")

# Define metrics function
calculate_all_metrics <- function(true_labels, pred_probs, threshold = 0.5) {
  pred_labels <- ifelse(pred_probs > threshold, 1, 0)
  true_labels_numeric <- as.numeric(true_labels) - 1
  
  cm <- table(Predicted = pred_labels, Actual = true_labels_numeric)
  
  if(nrow(cm) == 1) {
    if(rownames(cm) == "0") {
      cm <- rbind(cm, c(0, 0))
      rownames(cm) <- c("0", "1")
    } else {
      cm <- rbind(c(0, 0), cm)
      rownames(cm) <- c("0", "1")
    }
  }
  if(ncol(cm) == 1) {
    if(colnames(cm) == "0") {
      cm <- cbind(cm, c(0, 0))
      colnames(cm) <- c("0", "1")
    } else {
      cm <- cbind(c(0, 0), cm)
      colnames(cm) <- c("0", "1")
    }
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
  f1_score <- ifelse((sensitivity + ppv) > 0, 2 * sensitivity * ppv / (sensitivity + ppv), 0)
  balanced_accuracy <- (sensitivity + specificity) / 2
  
  auc_result <- tryCatch({
    roc_obj <- roc(true_labels_numeric, pred_probs)
    auc_value <- auc(roc_obj)
    ci <- ci.auc(roc_obj, method = "bootstrap")
    list(auc = auc_value, ci_lower = ci[1], ci_upper = ci[3])
  }, error = function(e) {
    list(auc = 0.5, ci_lower = 0.5, ci_upper = 0.5)
  })
  
  return(list(
    AUC = auc_result$auc,
    AUC_CI_Lower = auc_result$ci_lower,
    AUC_CI_Upper = auc_result$ci_upper,
    Sensitivity = sensitivity,
    Specificity = specificity,
    PPV = ppv,
    NPV = npv,
    Accuracy = accuracy,
    F1 = f1_score,
    Balanced_Accuracy = balanced_accuracy,
    ROC_Obj = if(exists("roc_obj")) roc_obj else NULL,
    Confusion_Matrix = cm
  ))
}

# ==================== Common Data Preparation Function ====================
prepare_validation_data <- function(validation_data, train_data, final_features, train_data_balanced) {
  # Prepare features and labels
  validation_features_all <- validation_data[, !names(validation_data) %in% "Diagnosis"]
  validation_labels_all <- validation_data$Diagnosis
  
  # Random Forest Imputation
  cat("Performing RF imputation on validation set...\n")
  if(any(is.na(validation_features_all))) {
    # Combine train and validation for consistent imputation
    combined_data <- rbind(
      cbind(train_data[, !names(train_data) %in% "Diagnosis"], Source = "train"),
      cbind(validation_features_all, Source = "validation")
    )
    
    mice_args_combined <- list(
      data = combined_data[, !names(combined_data) %in% "Source"],
      m = 5,
      maxit = 10,
      printFlag = FALSE,
      seed = 999,
      method = "rf"  # Random Forest imputation
    )
    
    imp_model_combined <- do.call(mice, mice_args_combined)
    combined_imputed <- complete(imp_model_combined, 1)
    
    # Separate imputed validation set
    validation_start_idx <- nrow(train_data) + 1
    validation_end_idx <- nrow(combined_data)
    validation_features_imputed <- combined_imputed[validation_start_idx:validation_end_idx, ]
  } else {
    validation_features_imputed <- validation_features_all
  }
  
  # Ensure no missing values
  for(feature in names(validation_features_imputed)) {
    if(any(is.na(validation_features_imputed[[feature]]))) {
      if(is.numeric(validation_features_imputed[[feature]])) {
        mean_val <- mean(train_data_balanced[[feature]], na.rm = TRUE)
        validation_features_imputed[[feature]][is.na(validation_features_imputed[[feature]])] <- mean_val
      } else {
        mode_val <- names(sort(table(train_data_balanced[[feature]]), decreasing = TRUE))[1]
        validation_features_imputed[[feature]][is.na(validation_features_imputed[[feature]])] <- mode_val
      }
    }
  }
  
  # Check for missing features
  missing_features <- setdiff(final_features, names(validation_features_imputed))
  if(length(missing_features) > 0) {
    cat("Adding missing features:", paste(missing_features, collapse = ", "), "\n")
    for(feature in missing_features) {
      if(is.numeric(train_data_balanced[[feature]])) {
        validation_features_imputed[[feature]] <- mean(train_data_balanced[[feature]], na.rm = TRUE)
      } else {
        mode_val <- names(sort(table(train_data_balanced[[feature]]), decreasing = TRUE))[1]
        validation_features_imputed[[feature]] <- mode_val
      }
    }
  }
  
  # Prepare validation data for Random Forest
  validation_features_subset <- validation_features_imputed[, final_features, drop = FALSE]
  
  return(list(
    features = validation_features_subset,
    labels = validation_labels_all,
    features_imputed = validation_features_imputed
  ))
}

# ==================== Test Set Validation ====================
cat("=== Part 3: Test Set Validation with RF Imputation ===\n")
test_prepared <- prepare_validation_data(test_data, train_data, final_features, train_data_balanced)

# Predict using final Random Forest model
cat("Making predictions on test set with Random Forest model...\n")
test_probs <- predict(final_rf_model, test_prepared$features, type = "prob")[, 2]
test_metrics <- calculate_all_metrics(test_prepared$labels, test_probs)
test_roc <- test_metrics$ROC_Obj

# ==================== External Validation Set Validation ====================
cat("=== External Validation Set Validation with RF Imputation ===\n")
external_prepared <- prepare_validation_data(external_data, train_data, final_features, train_data_balanced)

# Predict using final Random Forest model
cat("Making predictions on external validation set with Random Forest model...\n")
external_probs <- predict(final_rf_model, external_prepared$features, type = "prob")[, 2]
external_metrics <- calculate_all_metrics(external_prepared$labels, external_probs)
external_roc <- external_metrics$ROC_Obj

# ==================== Generate Visualizations ====================
cat("Generating Visualizations...\n")
if(!dir.exists("Part3_Results")) dir.create("Part3_Results")

# 1. ROC Curves Comparison
png("Part3_Results/ROC_Curves_Comparison.png", width = 800, height = 600)
par(mar = c(5, 5, 4, 2) + 0.1)
plot(test_roc, main = "ROC Curves Comparison", 
     col = "blue", lwd = 3, print.auc = TRUE, print.auc.x = 0.6, print.auc.y = 0.4)
plot(external_roc, add = TRUE, col = "red", lwd = 3, print.auc = TRUE, print.auc.x = 0.6, print.auc.y = 0.3)
abline(0, 1, lty = 3, col = "gray")
legend("bottomright", 
       legend = c(paste("Test Set (AUC =", round(test_metrics$AUC, 3), ")"), 
                  paste("External Set (AUC =", round(external_metrics$AUC, 3), ")")),
       col = c("blue", "red"), lwd = 3, cex = 1.2)
dev.off()

# 2. Individual ROC Curves
# Test Set ROC
png("Part3_Results/Test_Set_ROC_Curve.png", width = 800, height = 600)
par(mar = c(5, 5, 4, 2) + 0.1)
plot(test_roc, main = "ROC Curve - Test Set (Random Forest)", 
     col = "blue", lwd = 3, print.auc = TRUE, print.auc.x = 0.6, print.auc.y = 0.4)
abline(0, 1, lty = 3, col = "gray")
legend("bottomright", legend = paste("AUC =", round(test_metrics$AUC, 3)), 
       col = "blue", lwd = 3, cex = 1.2)
dev.off()

# External Set ROC
png("Part3_Results/External_Set_ROC_Curve.png", width = 800, height = 600)
par(mar = c(5, 5, 4, 2) + 0.1)
plot(external_roc, main = "ROC Curve - External Validation Set (Random Forest)", 
     col = "red", lwd = 3, print.auc = TRUE, print.auc.x = 0.6, print.auc.y = 0.4)
abline(0, 1, lty = 3, col = "gray")
legend("bottomright", legend = paste("AUC =", round(external_metrics$AUC, 3)), 
       col = "red", lwd = 3, cex = 1.2)
dev.off()

# 3. Calibration Curves Comparison
create_calibration_plot <- function(probs, true_labels, title, color) {
  calibration_data <- data.frame(
    Probability = probs,
    True_Label = as.numeric(true_labels) - 1
  )
  
  calibration_data$Bin <- cut(calibration_data$Probability, 
                              breaks = seq(0, 1, by = 0.1), 
                              include.lowest = TRUE)
  
  calibration_summary <- aggregate(Probability ~ Bin, data = calibration_data, FUN = mean)
  names(calibration_summary)[2] <- "Mean_Predicted"
  
  actual_means <- aggregate(True_Label ~ Bin, data = calibration_data, FUN = mean)
  names(actual_means)[2] <- "Mean_Actual"
  
  counts <- aggregate(True_Label ~ Bin, data = calibration_data, FUN = length)
  names(counts)[2] <- "Count"
  
  calibration_summary <- merge(calibration_summary, actual_means, by = "Bin")
  calibration_summary <- merge(calibration_summary, counts, by = "Bin")
  
  p <- ggplot(calibration_summary, aes(x = Mean_Predicted, y = Mean_Actual)) +
    geom_point(aes(size = Count), color = color, alpha = 0.7) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    geom_smooth(method = "loess", color = "darkgreen", se = FALSE) +
    labs(title = title,
         x = "Mean Predicted Probability",
         y = "Observed Fraction Positive",
         size = "Number of Cases") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
  
  return(p)
}

p1_test <- create_calibration_plot(test_probs, test_prepared$labels, 
                                   "Calibration Plot - Test Set (Random Forest)", "blue")
p1_external <- create_calibration_plot(external_probs, external_prepared$labels, 
                                       "Calibration Plot - External Set (Random Forest)", "red")

ggsave("Part3_Results/Calibration_Plot_Test.png", p1_test, width = 8, height = 6, dpi = 300)
ggsave("Part3_Results/Calibration_Plot_External.png", p1_external, width = 8, height = 6, dpi = 300)

# 4. Probability Distribution Comparison
create_probability_distribution <- function(probs, true_labels, title) {
  prob_data <- data.frame(
    Probability = probs,
    True_Label = as.factor(as.numeric(true_labels) - 1)
  )
  
  p <- ggplot(prob_data, aes(x = Probability, fill = True_Label)) +
    geom_density(alpha = 0.6) +
    labs(title = title,
         x = "Predicted Probability",
         y = "Density",
         fill = "True Class") +
    scale_fill_manual(values = c("0" = "red", "1" = "blue"), 
                      labels = c("Control", "Case")) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
  
  return(p)
}

p2_test <- create_probability_distribution(test_probs, test_prepared$labels, 
                                           "Probability Distribution - Test Set (Random Forest)")
p2_external <- create_probability_distribution(external_probs, external_prepared$labels, 
                                               "Probability Distribution - External Set (Random Forest)")

ggsave("Part3_Results/Probability_Distribution_Test.png", p2_test, width = 10, height = 6, dpi = 300)
ggsave("Part3_Results/Probability_Distribution_External.png", p2_external, width = 10, height = 6, dpi = 300)

# 5. Performance Comparison (CV vs Test vs External)
comparison_data <- data.frame(
  Dataset = rep(c("Cross-Validation", "Test", "External"), each = 8),
  Metric = rep(c("AUC", "Sensitivity", "Specificity", "PPV", "NPV", "Accuracy", "F1", "Balanced_Accuracy"), 3),
  Value = c(
    cv_summary$Mean[1], cv_summary$Mean[2], cv_summary$Mean[3],
    cv_summary$Mean[4], cv_summary$Mean[5], cv_summary$Mean[6],
    cv_summary$Mean[7], cv_summary$Mean[8],
    test_metrics$AUC, test_metrics$Sensitivity, test_metrics$Specificity,
    test_metrics$PPV, test_metrics$NPV, test_metrics$Accuracy,
    test_metrics$F1, test_metrics$Balanced_Accuracy,
    external_metrics$AUC, external_metrics$Sensitivity, external_metrics$Specificity,
    external_metrics$PPV, external_metrics$NPV, external_metrics$Accuracy,
    external_metrics$F1, external_metrics$Balanced_Accuracy
  )
)

p3 <- ggplot(comparison_data, aes(x = Metric, y = Value, fill = Dataset)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  labs(title = "Performance Comparison: Cross-Validation vs Test Set vs External Set (Random Forest)",
       y = "Performance Value", x = "Metric") +
  scale_fill_manual(values = c("Cross-Validation" = "blue", "Test" = "red", "External" = "green")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "top") +
  ylim(0, 1)

ggsave("Part3_Results/Performance_Comparison.png", p3, width = 12, height = 6, dpi = 300)

# 6. Misclassification Analysis
analyze_misclassification <- function(probs, true_labels, features_imputed, dataset_name) {
  pred_labels <- ifelse(probs > 0.5, 1, 0)
  true_labels_numeric <- as.numeric(true_labels) - 1
  misclassified_indices <- which(pred_labels != true_labels_numeric)
  
  if(length(misclassified_indices) > 0) {
    misclassified_cases <- data.frame(
      Case_ID = rownames(features_imputed)[misclassified_indices],
      True_Label = true_labels_numeric[misclassified_indices],
      Predicted_Label = pred_labels[misclassified_indices],
      Probability = probs[misclassified_indices]
    )
    
    p <- ggplot(misclassified_cases, aes(x = as.factor(True_Label), fill = as.factor(Predicted_Label))) +
      geom_bar(alpha = 0.8) +
      labs(title = paste(dataset_name, "Misclassification Analysis (Random Forest)"),
           x = "True Label",
           y = "Count",
           fill = "Predicted Label") +
      scale_fill_manual(values = c("0" = "red", "1" = "blue"),
                        labels = c("Control", "Case")) +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
    
    ggsave(paste0("Part3_Results/", dataset_name, "_Misclassification.png"), p, width = 8, height = 6, dpi = 300)
    
    return(misclassified_cases)
  }
  return(NULL)
}

test_misclassified <- analyze_misclassification(test_probs, test_prepared$labels, 
                                                test_prepared$features_imputed, "Test")
external_misclassified <- analyze_misclassification(external_probs, external_prepared$labels, 
                                                    external_prepared$features_imputed, "External")

# 7. Confusion Matrix Visualization
create_confusion_matrix_plot <- function(metrics, title) {
  cm_data <- as.data.frame(metrics$Confusion_Matrix)
  p <- ggplot(cm_data, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "white", size = 6) +
    scale_fill_gradient(low = "lightblue", high = "darkblue") +
    labs(title = title,
         x = "Actual Class",
         y = "Predicted Class") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          legend.position = "none")
  
  return(p)
}

p5_test <- create_confusion_matrix_plot(test_metrics, "Confusion Matrix - Test Set (Random Forest)")
p5_external <- create_confusion_matrix_plot(external_metrics, "Confusion Matrix - External Set (Random Forest)")

ggsave("Part3_Results/Confusion_Matrix_Test.png", p5_test, width = 6, height = 5, dpi = 300)
ggsave("Part3_Results/Confusion_Matrix_External.png", p5_external, width = 6, height = 5, dpi = 300)

# 8. Decision Threshold Analysis for Both Sets
perform_threshold_analysis <- function(probs, true_labels, dataset_name) {
  thresholds <- seq(0.1, 0.9, by = 0.1)
  threshold_analysis <- data.frame(
    Threshold = thresholds,
    Sensitivity = numeric(length(thresholds)),
    Specificity = numeric(length(thresholds)),
    Accuracy = numeric(length(thresholds))
  )
  
  for(i in 1:length(thresholds)) {
    pred_labels_thresh <- ifelse(probs > thresholds[i], 1, 0)
    true_labels_numeric <- as.numeric(true_labels) - 1
    
    cm_thresh <- table(Predicted = pred_labels_thresh, Actual = true_labels_numeric)
    
    # 处理可能的维度问题
    if(nrow(cm_thresh) == 1) {
      if(rownames(cm_thresh) == "0") {
        cm_thresh <- rbind(cm_thresh, c(0, 0))
        rownames(cm_thresh) <- c("0", "1")
      } else {
        cm_thresh <- rbind(c(0, 0), cm_thresh)
        rownames(cm_thresh) <- c("0", "1")
      }
    }
    if(ncol(cm_thresh) == 1) {
      if(colnames(cm_thresh) == "0") {
        cm_thresh <- cbind(cm_thresh, c(0, 0))
        colnames(cm_thresh) <- c("0", "1")
      } else {
        cm_thresh <- cbind(c(0, 0), cm_thresh)
        colnames(cm_thresh) <- c("0", "1")
      }
    }
    
    TP <- cm_thresh[2, 2]
    TN <- cm_thresh[1, 1]
    FP <- cm_thresh[2, 1]
    FN <- cm_thresh[1, 2]
    
    threshold_analysis$Sensitivity[i] <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
    threshold_analysis$Specificity[i] <- ifelse((TN + FP) > 0, TN / (TN + FP), 0)
    threshold_analysis$Accuracy[i] <- (TP + TN) / (TP + TN + FP + FN)
  }
  
  # Plot threshold analysis
  threshold_long <- reshape(threshold_analysis, 
                            direction = "long",
                            varying = c("Sensitivity", "Specificity", "Accuracy"),
                            v.names = "Value",
                            timevar = "Metric",
                            times = c("Sensitivity", "Specificity", "Accuracy"),
                            idvar = "Threshold")
  
  p <- ggplot(threshold_long, aes(x = Threshold, y = Value, color = Metric)) +
    geom_line(size = 1.2) +
    geom_point(size = 2) +
    labs(title = paste("Performance Metrics vs Decision Threshold -", dataset_name, "(Random Forest)"),
         x = "Decision Threshold",
         y = "Performance Value") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          legend.position = "top") +
    scale_x_continuous(breaks = thresholds)
  
  ggsave(paste0("Part3_Results/Threshold_Analysis_", dataset_name, ".png"), p, width = 10, height = 6, dpi = 300)
  
  return(threshold_analysis)
}

test_threshold_analysis <- perform_threshold_analysis(test_probs, test_prepared$labels, "Test")
external_threshold_analysis <- perform_threshold_analysis(external_probs, external_prepared$labels, "External")

# ==================== Save Results ====================
# Prepare performance results
test_performance <- data.frame(
  Metric = c("AUC", "Sensitivity", "Specificity", "PPV", "NPV", "Accuracy", "F1", "Balanced_Accuracy"),
  Value = c(
    test_metrics$AUC,
    test_metrics$Sensitivity,
    test_metrics$Specificity,
    test_metrics$PPV,
    test_metrics$NPV,
    test_metrics$Accuracy,
    test_metrics$F1,
    test_metrics$Balanced_Accuracy
  ),
  AUC_CI_Lower = c(test_metrics$AUC_CI_Lower, rep(NA, 7)),
  AUC_CI_Upper = c(test_metrics$AUC_CI_Upper, rep(NA, 7))
)

external_performance <- data.frame(
  Metric = c("AUC", "Sensitivity", "Specificity", "PPV", "NPV", "Accuracy", "F1", "Balanced_Accuracy"),
  Value = c(
    external_metrics$AUC,
    external_metrics$Sensitivity,
    external_metrics$Specificity,
    external_metrics$PPV,
    external_metrics$NPV,
    external_metrics$Accuracy,
    external_metrics$F1,
    external_metrics$Balanced_Accuracy
  ),
  AUC_CI_Lower = c(external_metrics$AUC_CI_Lower, rep(NA, 7)),
  AUC_CI_Upper = c(external_metrics$AUC_CI_Upper, rep(NA, 7))
)

writeData(wb3, "Test_Performance", test_performance, startRow = 1)
writeData(wb3, "External_Performance", external_performance, startRow = 1)

if(!is.null(test_misclassified)) {
  writeData(wb3, "Misclassified_Test", test_misclassified, startRow = 1)
}
if(!is.null(external_misclassified)) {
  writeData(wb3, "Misclassified_External", external_misclassified, startRow = 1)
}

# Save probability distributions
test_prob_distribution <- data.frame(
  Case_ID = rownames(test_prepared$features_imputed),
  True_Label = as.numeric(test_prepared$labels) - 1,
  Predicted_Probability = test_probs,
  Predicted_Label = ifelse(test_probs > 0.5, 1, 0)
)
external_prob_distribution <- data.frame(
  Case_ID = rownames(external_prepared$features_imputed),
  True_Label = as.numeric(external_prepared$labels) - 1,
  Predicted_Probability = external_probs,
  Predicted_Label = ifelse(external_probs > 0.5, 1, 0)
)

writeData(wb3, "Prob_Dist_Test", test_prob_distribution, startRow = 1)
writeData(wb3, "Prob_Dist_External", external_prob_distribution, startRow = 1)

# Save performance comparison
writeData(wb3, "Perf_Comparison", comparison_data, startRow = 1)

# Save threshold analysis
addWorksheet(wb3, "Threshold_Test")
addWorksheet(wb3, "Threshold_External")
writeData(wb3, "Threshold_Test", test_threshold_analysis, startRow = 1)
writeData(wb3, "Threshold_External", external_threshold_analysis, startRow = 1)

saveWorkbook(wb3, "Part3_Test_and_External_Validation_Results.xlsx", overwrite = TRUE)

# Save test and external objects
saveRDS(test_metrics, "test_metrics.rds")
saveRDS(test_probs, "test_probs.rds")
saveRDS(test_threshold_analysis, "test_threshold_analysis.rds")
saveRDS(external_metrics, "external_metrics.rds")
saveRDS(external_probs, "external_probs.rds")
saveRDS(external_threshold_analysis, "external_threshold_analysis.rds")

cat("\n=== Part 3 Complete ===\n")
cat("Results saved to:\n")
cat("- Part3_Test_and_External_Validation_Results.xlsx\n")
cat("- Part3_Results/ directory\n")
cat("- test_metrics.rds, external_metrics.rds\n")
cat("- test_probs.rds, external_probs.rds\n")
cat("- test_threshold_analysis.rds, external_threshold_analysis.rds\n")

cat("\n=== Performance Summary ===\n")
cat("\nTest Set Performance:\n")
print(test_performance)
cat("\nExternal Validation Set Performance:\n")
print(external_performance)

cat("\nTest Set Confusion Matrix:\n")
print(test_metrics$Confusion_Matrix)
cat("\nExternal Set Confusion Matrix:\n")
print(external_metrics$Confusion_Matrix)

# Optimal Threshold Analysis
cat("\n=== Optimal Threshold Analysis ===\n")
calculate_optimal_threshold <- function(threshold_analysis, dataset_name) {
  youden_j <- threshold_analysis$Sensitivity + threshold_analysis$Specificity - 1
  optimal_idx <- which.max(youden_j)
  cat(dataset_name, "Optimal Threshold:", threshold_analysis$Threshold[optimal_idx], "\n")
  cat(dataset_name, "Sensitivity at optimal threshold:", round(threshold_analysis$Sensitivity[optimal_idx], 3), "\n")
  cat(dataset_name, "Specificity at optimal threshold:", round(threshold_analysis$Specificity[optimal_idx], 3), "\n\n")
}

calculate_optimal_threshold(test_threshold_analysis, "Test Set")
calculate_optimal_threshold(external_threshold_analysis, "External Set")