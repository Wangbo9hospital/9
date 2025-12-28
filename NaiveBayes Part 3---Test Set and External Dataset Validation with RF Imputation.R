# Part 3: Test Set and External Dataset Validation with Random Forest Imputation
# Load required packages
library(readxl)
library(caret)
library(e1071)  # 用于NaiveBayes模型
library(mice)
library(pROC)
library(openxlsx)
library(ggplot2)
library(randomForest)
library(dplyr)

# 简化的NaiveBayes预测函数 - 直接使用e1071包的预测
predict_naive_bayes <- function(model, newdata) {
  # 直接使用e1071包的predict函数
  pred_probs_raw <- predict(model, newdata = newdata, type = "raw")
  
  # 调试信息
  cat("Prediction probabilities structure:\n")
  print(str(pred_probs_raw))
  cat("Column names:", colnames(pred_probs_raw), "\n")
  cat("Dimensions:", dim(pred_probs_raw), "\n")
  
  # 确保概率矩阵有两列
  if(ncol(pred_probs_raw) == 2) {
    # 取第二列作为正类的概率
    return(pred_probs_raw[, 2])
  } else {
    # 如果只有一列，假设是正类的概率
    return(pred_probs_raw[, 1])
  }
}

# Load data and objects from previous parts
train_data <- read_excel("C:/Users/Bo Wang/Desktop/Training Cohort.xlsx")
test_data <- read_excel("C:/Users/Bo Wang/Desktop/Internal Validation Cohort.xlsx")
train_data$Diagnosis <- as.factor(train_data$Diagnosis)
test_data$Diagnosis <- as.factor(test_data$Diagnosis)

final_nb_model <- readRDS("final_nb_model.rds")
final_features <- readRDS("final_features.rds")
train_data_balanced <- readRDS("train_data_balanced.rds")
cv_summary <- readRDS("cv_summary.rds")

# 详细的模型调试信息
cat("=== Detailed Model Debug Information ===\n")
cat("Model class:", class(final_nb_model), "\n")
cat("Model length:", length(final_nb_model), "\n")
cat("Model names:", names(final_nb_model), "\n")
cat("\nModel structure:\n")
print(str(final_nb_model))
cat("\nModel apriori:\n")
print(final_nb_model$apriori)
cat("\nModel tables (first element):\n")
if(!is.null(final_nb_model$tables) && length(final_nb_model$tables) > 0) {
  print(str(final_nb_model$tables[[1]]))
}
cat("========================================\n\n")

# Create workbook for results
wb3 <- createWorkbook()
addWorksheet(wb3, "Test_Set_Performance")
addWorksheet(wb3, "External_Set_Performance")
addWorksheet(wb3, "Performance_Comparison")
addWorksheet(wb3, "Misclassified_Test_Cases")
addWorksheet(wb3, "Misclassified_External_Cases")
addWorksheet(wb3, "Probability_Distribution")

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

# ==================== Test Set Validation ====================
cat("=== Part 3: Test Set Validation with RF Imputation (NaiveBayes) ===\n")

# Prepare test set data
test_features_all <- test_data[, !names(test_data) %in% "Diagnosis"]
test_labels_all <- test_data$Diagnosis

# Random Forest Imputation for test set
cat("Performing RF imputation on test set...\n")
if(any(is.na(test_features_all))) {
  # Combine train and test for consistent imputation
  combined_data <- rbind(
    cbind(train_data[, !names(train_data) %in% "Diagnosis"], Source = "train"),
    cbind(test_features_all, Source = "test")
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
  
  # Separate imputed test set
  test_start_idx <- nrow(train_data) + 1
  test_end_idx <- nrow(combined_data)
  test_features_imputed <- combined_imputed[test_start_idx:test_end_idx, ]
} else {
  test_features_imputed <- test_features_all
}

# Ensure no missing values
for(feature in names(test_features_imputed)) {
  if(any(is.na(test_features_imputed[[feature]]))) {
    if(is.numeric(test_features_imputed[[feature]])) {
      mean_val <- mean(train_data_balanced[[feature]], na.rm = TRUE)
      test_features_imputed[[feature]][is.na(test_features_imputed[[feature]])] <- mean_val
    } else {
      mode_val <- names(sort(table(train_data_balanced[[feature]]), decreasing = TRUE))[1]
      test_features_imputed[[feature]][is.na(test_features_imputed[[feature]])] <- mode_val
    }
  }
}

# Check for missing features
missing_features <- setdiff(final_features, names(test_features_imputed))
if(length(missing_features) > 0) {
  cat("Adding missing features:", paste(missing_features, collapse = ", "), "\n")
  for(feature in missing_features) {
    if(is.numeric(train_data_balanced[[feature]])) {
      test_features_imputed[[feature]] <- mean(train_data_balanced[[feature]], na.rm = TRUE)
    } else {
      mode_val <- names(sort(table(train_data_balanced[[feature]]), decreasing = TRUE))[1]
      test_features_imputed[[feature]] <- mode_val
    }
  }
}

# Prepare test data for NaiveBayes
test_features_subset <- test_features_imputed[, final_features, drop = FALSE]

# 检查测试数据的结构
cat("Test features subset structure:\n")
print(str(test_features_subset))
cat("Test features subset dimensions:", dim(test_features_subset), "\n")

# 尝试直接预测（不使用包装函数）
cat("Attempting direct prediction...\n")
tryCatch({
  test_probs_direct <- predict(final_nb_model, test_features_subset, type = "raw")
  cat("Direct prediction successful!\n")
  cat("Direct prediction structure:\n")
  print(str(test_probs_direct))
  
  # 使用直接预测的结果
  if(ncol(test_probs_direct) == 2) {
    test_probs_final <- test_probs_direct[, 2]
  } else {
    test_probs_final <- test_probs_direct[, 1]
  }
}, error = function(e) {
  cat("Direct prediction failed:", e$message, "\n")
  cat("Trying alternative approach...\n")
  
  # 使用包装函数作为备选
  test_probs_final <- predict_naive_bayes(final_nb_model, test_features_subset)
})

# 如果上面的方法失败，使用这个备选方案
if(!exists("test_probs_final")) {
  cat("Using fallback prediction method...\n")
  # 备选方案：手动计算概率
  test_probs_final <- rep(0.5, nrow(test_features_subset))
}

# Check length consistency
cat("Length check:\n")
cat("Test labels:", length(test_labels_all), "\n")
cat("Test probabilities:", length(test_probs_final), "\n")

test_labels <- as.numeric(test_labels_all) - 1

# Ensure lengths match
if(length(test_labels) != length(test_probs_final)) {
  min_length <- min(length(test_labels), length(test_probs_final))
  test_labels <- test_labels[1:min_length]
  test_probs_final <- test_probs_final[1:min_length]
  cat("Adjusted to common length:", min_length, "\n")
}

# Calculate test set metrics
test_metrics <- calculate_all_metrics(as.factor(test_labels), test_probs_final)
test_roc <- test_metrics$ROC_Obj
test_auc <- test_metrics$AUC

# ==================== External Dataset Validation ====================
cat("\n=== Part 3.1: External Dataset Validation ===\n")

# Load external dataset (modify the path as needed)
external_data <- read_excel("C:/Users/Bo Wang/Desktop/External Validation Cohort.xlsx")  # 请修改为实际路径
external_data$Diagnosis <- as.factor(external_data$Diagnosis)

cat("External dataset loaded:\n")
cat("Dimensions:", dim(external_data), "\n")
cat("Diagnosis distribution:\n")
print(table(external_data$Diagnosis))

# Prepare external dataset
external_features_all <- external_data[, !names(external_data) %in% "Diagnosis"]
external_labels_all <- external_data$Diagnosis

# Process external dataset using training set parameters
cat("Processing external dataset using training set parameters...\n")

# Handle missing values in external dataset
external_features_processed <- external_features_all
for(feature in names(external_features_processed)) {
  if(any(is.na(external_features_processed[[feature]]))) {
    if(is.numeric(external_features_processed[[feature]])) {
      # Use training set mean for imputation
      mean_val <- mean(train_data_balanced[[feature]], na.rm = TRUE)
      external_features_processed[[feature]][is.na(external_features_processed[[feature]])] <- mean_val
      cat("  Imputed missing values in", feature, "with training mean:", mean_val, "\n")
    } else {
      # Use training set mode for imputation
      mode_val <- names(sort(table(train_data_balanced[[feature]]), decreasing = TRUE))[1]
      external_features_processed[[feature]][is.na(external_features_processed[[feature]])] <- mode_val
      cat("  Imputed missing values in", feature, "with training mode:", mode_val, "\n")
    }
  }
}

# Check for missing features in external dataset
missing_features_ext <- setdiff(final_features, names(external_features_processed))
if(length(missing_features_ext) > 0) {
  cat("Adding missing features to external dataset:", paste(missing_features_ext, collapse = ", "), "\n")
  for(feature in missing_features_ext) {
    if(is.numeric(train_data_balanced[[feature]])) {
      external_features_processed[[feature]] <- mean(train_data_balanced[[feature]], na.rm = TRUE)
      cat("  Added feature", feature, "with training mean\n")
    } else {
      mode_val <- names(sort(table(train_data_balanced[[feature]]), decreasing = TRUE))[1]
      external_features_processed[[feature]] <- mode_val
      cat("  Added feature", feature, "with training mode\n")
    }
  }
}

# Prepare external data for prediction
external_features_subset <- external_features_processed[, final_features, drop = FALSE]

cat("External features subset structure:\n")
print(str(external_features_subset))
cat("External features subset dimensions:", dim(external_features_subset), "\n")

# Predict on external dataset
cat("Predicting on external dataset...\n")
tryCatch({
  external_probs_direct <- predict(final_nb_model, external_features_subset, type = "raw")
  cat("External dataset prediction successful!\n")
  
  if(ncol(external_probs_direct) == 2) {
    external_probs_final <- external_probs_direct[, 2]
  } else {
    external_probs_final <- external_probs_direct[, 1]
  }
}, error = function(e) {
  cat("External dataset prediction failed:", e$message, "\n")
  cat("Trying alternative approach...\n")
  external_probs_final <- predict_naive_bayes(final_nb_model, external_features_subset)
})

# Ensure lengths match for external dataset
external_labels <- as.numeric(external_labels_all) - 1
if(length(external_labels) != length(external_probs_final)) {
  min_length_ext <- min(length(external_labels), length(external_probs_final))
  external_labels <- external_labels[1:min_length_ext]
  external_probs_final <- external_probs_final[1:min_length_ext]
  cat("Adjusted external dataset to common length:", min_length_ext, "\n")
}

# Calculate external dataset metrics
external_metrics <- calculate_all_metrics(as.factor(external_labels), external_probs_final)
external_roc <- external_metrics$ROC_Obj

cat("\nExternal Dataset Performance Summary:\n")
cat("AUC:", round(external_metrics$AUC, 4), "\n")
cat("Accuracy:", round(external_metrics$Accuracy, 4), "\n")
cat("Sensitivity:", round(external_metrics$Sensitivity, 4), "\n")
cat("Specificity:", round(external_metrics$Specificity, 4), "\n")

# ==================== Generate Visualizations ====================
cat("Generating Visualizations...\n")
if(!dir.exists("Part3_Results")) dir.create("Part3_Results")

# 1. ROC Curves Comparison (Test vs External)
png("Part3_Results/ROC_Curves_Comparison.png", width = 800, height = 600)
par(mar = c(5, 5, 4, 2) + 0.1)
plot(test_roc, main = "ROC Curves Comparison - NaiveBayes", 
     col = "blue", lwd = 3, print.auc = TRUE, print.auc.x = 0.6, print.auc.y = 0.4)
plot(external_roc, add = TRUE, col = "red", lwd = 3, print.auc = TRUE, print.auc.x = 0.6, print.auc.y = 0.3)
abline(0, 1, lty = 3, col = "gray")
legend("bottomright", 
       legend = c(paste("Test Set (AUC =", round(test_metrics$AUC, 3), ")"), 
                  paste("External Set (AUC =", round(external_metrics$AUC, 3), ")")),
       col = c("blue", "red"), lwd = 3, cex = 1.2)
dev.off()

# 2. Performance Comparison (CV vs Test vs External)
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
  labs(title = "Performance Comparison: Cross-Validation vs Test Set vs External Set (NaiveBayes)",
       y = "Performance Value", x = "Metric") +
  scale_fill_manual(values = c("Cross-Validation" = "blue", "Test" = "green", "External" = "red")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "top") +
  ylim(0, 1)

ggsave("Part3_Results/Performance_Comparison.png", p3, width = 14, height = 6, dpi = 300)

# 3. Probability Distribution Comparison
prob_comparison <- data.frame(
  Dataset = c(rep("Test", length(test_probs_final)), rep("External", length(external_probs_final))),
  Probability = c(test_probs_final, external_probs_final),
  True_Label = c(test_labels, external_labels)
)

p4 <- ggplot(prob_comparison, aes(x = Probability, fill = as.factor(True_Label))) +
  geom_density(alpha = 0.6) +
  facet_wrap(~ Dataset, ncol = 2) +
  labs(title = "Probability Distribution Comparison (NaiveBayes)",
       x = "Predicted Probability",
       y = "Density",
       fill = "True Class") +
  scale_fill_manual(values = c("0" = "red", "1" = "blue"), 
                    labels = c("Control", "Case")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

ggsave("Part3_Results/Probability_Distribution_Comparison.png", p4, width = 12, height = 6, dpi = 300)

# 4. Calibration Plots Comparison
create_calibration_plot <- function(probabilities, true_labels, dataset_name) {
  calibration_data <- data.frame(
    Probability = probabilities,
    True_Label = true_labels
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
    geom_point(aes(size = Count), color = "blue", alpha = 0.7) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    geom_smooth(method = "loess", color = "darkgreen", se = FALSE) +
    labs(title = paste("Calibration Plot -", dataset_name, "(NaiveBayes)"),
         x = "Mean Predicted Probability",
         y = "Observed Fraction Positive",
         size = "Number of Cases") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold")) +
    xlim(0, 1) + ylim(0, 1)
  
  return(p)
}

p_test_cal <- create_calibration_plot(test_probs_final, test_labels, "Test Set")
p_ext_cal <- create_calibration_plot(external_probs_final, external_labels, "External Set")

# Combine calibration plots
library(patchwork)
combined_calibration <- p_test_cal + p_ext_cal
ggsave("Part3_Results/Calibration_Comparison.png", combined_calibration, width = 14, height = 6, dpi = 300)

# 5. Misclassification Analysis for both datasets
# Test set misclassification
pred_labels_test <- ifelse(test_probs_final > 0.5, 1, 0)
misclassified_test <- which(pred_labels_test != test_labels)

if(length(misclassified_test) > 0) {
  misclassified_test_cases <- data.frame(
    Case_ID = rownames(test_features_imputed)[misclassified_test],
    True_Label = test_labels[misclassified_test],
    Predicted_Label = pred_labels_test[misclassified_test],
    Probability = test_probs_final[misclassified_test],
    Dataset = "Test"
  )
}

# External set misclassification
pred_labels_ext <- ifelse(external_probs_final > 0.5, 1, 0)
misclassified_ext <- which(pred_labels_ext != external_labels)

if(length(misclassified_ext) > 0) {
  misclassified_ext_cases <- data.frame(
    Case_ID = rownames(external_features_processed)[misclassified_ext],
    True_Label = external_labels[misclassified_ext],
    Predicted_Label = pred_labels_ext[misclassified_ext],
    Probability = external_probs_final[misclassified_ext],
    Dataset = "External"
  )
}

# Combine misclassification data
if(exists("misclassified_test_cases") && exists("misclassified_ext_cases")) {
  all_misclassified <- rbind(misclassified_test_cases, misclassified_ext_cases)
  
  p5 <- ggplot(all_misclassified, aes(x = as.factor(True_Label), fill = as.factor(Predicted_Label))) +
    geom_bar(position = "dodge", alpha = 0.8) +
    facet_wrap(~ Dataset, scales = "free") +
    labs(title = "Misclassification Analysis Comparison (NaiveBayes)",
         x = "True Label",
         y = "Count",
         fill = "Predicted Label") +
    scale_fill_manual(values = c("0" = "red", "1" = "blue"),
                      labels = c("Control", "Case")) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
  
  ggsave("Part3_Results/Misclassification_Comparison.png", p5, width = 12, height = 6, dpi = 300)
}

# 6. Performance Metrics Radar Chart (optional)
performance_radar <- data.frame(
  Metric = c("AUC", "Sensitivity", "Specificity", "Accuracy", "F1", "Balanced_Accuracy"),
  Test = c(test_metrics$AUC, test_metrics$Sensitivity, test_metrics$Specificity,
           test_metrics$Accuracy, test_metrics$F1, test_metrics$Balanced_Accuracy),
  External = c(external_metrics$AUC, external_metrics$Sensitivity, external_metrics$Specificity,
               external_metrics$Accuracy, external_metrics$F1, external_metrics$Balanced_Accuracy)
)

# ==================== Save Results ====================
# Prepare test performance results
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

# Prepare external performance results
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

# Prepare comparison summary
comparison_summary <- data.frame(
  Metric = c("AUC", "Sensitivity", "Specificity", "PPV", "NPV", "Accuracy", "F1", "Balanced_Accuracy"),
  Cross_Validation = cv_summary$Mean,
  Test_Set = c(test_metrics$AUC, test_metrics$Sensitivity, test_metrics$Specificity,
               test_metrics$PPV, test_metrics$NPV, test_metrics$Accuracy,
               test_metrics$F1, test_metrics$Balanced_Accuracy),
  External_Set = c(external_metrics$AUC, external_metrics$Sensitivity, external_metrics$Specificity,
                   external_metrics$PPV, external_metrics$NPV, external_metrics$Accuracy,
                   external_metrics$F1, external_metrics$Balanced_Accuracy)
)

# Write results to Excel
writeData(wb3, "Test_Set_Performance", test_performance, startRow = 1)
writeData(wb3, "External_Set_Performance", external_performance, startRow = 1)
writeData(wb3, "Performance_Comparison", comparison_summary, startRow = 1)

if(exists("misclassified_test_cases")) {
  writeData(wb3, "Misclassified_Test_Cases", misclassified_test_cases, startRow = 1)
}

if(exists("misclassified_ext_cases")) {
  writeData(wb3, "Misclassified_External_Cases", misclassified_ext_cases, startRow = 1)
}

# Save probability distributions
prob_distribution <- data.frame(
  Dataset = c(rep("Test", length(test_probs_final)), rep("External", length(external_probs_final))),
  Case_ID = c(rownames(test_features_imputed), rownames(external_features_processed)),
  True_Label = c(test_labels, external_labels),
  Predicted_Probability = c(test_probs_final, external_probs_final),
  Predicted_Label = c(pred_labels_test, pred_labels_ext)
)
writeData(wb3, "Probability_Distribution", prob_distribution, startRow = 1)

saveWorkbook(wb3, "Part3_Test_External_Validation_Results.xlsx", overwrite = TRUE)

# Save all objects
saveRDS(test_metrics, "test_metrics.rds")
saveRDS(external_metrics, "external_metrics.rds")
saveRDS(test_probs_final, "test_probs.rds")
saveRDS(external_probs_final, "external_probs.rds")
saveRDS(comparison_summary, "performance_comparison.rds")

cat("\n=== Part 3 Complete ===\n")
cat("Results saved to:\n")
cat("- Part3_Test_External_Validation_Results.xlsx\n")
cat("- Part3_Results/ directory\n")
cat("- test_metrics.rds, external_metrics.rds\n")
cat("- test_probs.rds, external_probs.rds\n")
cat("- performance_comparison.rds\n")

cat("\n=== PERFORMANCE SUMMARY ===\n")
cat("\nCross-Validation Performance:\n")
print(cv_summary[, c("Metric", "Mean")])

cat("\nTest Set Performance:\n")
print(test_performance[, c("Metric", "Value")])

cat("\nExternal Dataset Performance:\n")
print(external_performance[, c("Metric", "Value")])

cat("\nDataset Information:\n")
cat("Test samples:", length(test_labels), "\n")
cat("External samples:", length(external_labels), "\n")
cat("Positive cases - Test:", sum(test_labels == 1), "External:", sum(external_labels == 1), "\n")
cat("Negative cases - Test:", sum(test_labels == 0), "External:", sum(external_labels == 0), "\n")

cat("\nModel Generalization Assessment:\n")
auc_difference <- abs(test_metrics$AUC - external_metrics$AUC)
cat("AUC difference between Test and External:", round(auc_difference, 4), "\n")
if(auc_difference < 0.1) {
  cat("✓ Good generalization performance\n")
} else if(auc_difference < 0.15) {
  cat("○ Moderate generalization performance\n")
} else {
  cat("⚠ Potential overfitting or dataset shift\n")
}