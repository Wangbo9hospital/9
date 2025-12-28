# Part 3: Test Set and External Validation Set Validation with Random Forest Imputation
# Load required packages
library(readxl)
library(caret)
library(xgboost)
library(mice)
library(pROC)
library(openxlsx)
library(ggplot2)
library(randomForest)
library(dplyr)

# Load data and objects from previous parts
train_data <- read_excel("C:/Users/Bo Wang/Desktop/Training Cohort.xlsx")
test_data <- read_excel("C:/Users/Bo Wang/Desktop/Internal Validation Cohort.xlsx")
# 添加外部验证集加载
external_data <- read_excel("C:/Users/Bo Wang/Desktop/External Validation Cohort.xlsx") 

train_data$Diagnosis <- as.factor(train_data$Diagnosis)
test_data$Diagnosis <- as.factor(test_data$Diagnosis)
external_data$Diagnosis <- as.factor(external_data$Diagnosis)

final_xgb_model <- readRDS("final_xgb_model.rds")
final_features <- readRDS("final_features.rds")
train_data_balanced <- readRDS("train_data_balanced.rds")
cv_summary <- readRDS("cv_summary.rds")

# Create workbook for results
wb3 <- createWorkbook()
addWorksheet(wb3, "Test_Set_Performance")
addWorksheet(wb3, "External_Validation_Performance")
addWorksheet(wb3, "Performance_Comparison")
addWorksheet(wb3, "Misclassified_Test_Cases")
addWorksheet(wb3, "Misclassified_External_Cases")
addWorksheet(wb3, "Probability_Distribution")

# Define metrics function (保持不变)
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

# ==================== 改进的数据预处理函数 ====================
preprocess_validation_data <- function(validation_data, train_data_balanced, final_features) {
  cat("Starting data preprocessing...\n")
  
  # 准备验证集数据
  validation_features_all <- validation_data[, !names(validation_data) %in% "Diagnosis", drop = FALSE]
  validation_labels_all <- validation_data$Diagnosis
  
  cat("Original validation data dimensions:", dim(validation_features_all), "\n")
  cat("Features in validation data:", names(validation_features_all), "\n")
  cat("Final features required:", final_features, "\n")
  
  # 首先确保所有必需的特征都存在
  missing_features <- setdiff(final_features, names(validation_features_all))
  if(length(missing_features) > 0) {
    cat("Adding missing features:", paste(missing_features, collapse = ", "), "\n")
    for(feature in missing_features) {
      if(feature %in% names(train_data_balanced)) {
        if(is.numeric(train_data_balanced[[feature]])) {
          validation_features_all[[feature]] <- mean(train_data_balanced[[feature]], na.rm = TRUE)
        } else {
          mode_val <- names(sort(table(train_data_balanced[[feature]]), decreasing = TRUE))[1]
          validation_features_all[[feature]] <- mode_val
        }
      } else {
        validation_features_all[[feature]] <- 0
      }
    }
  }
  
  # 只保留需要的特征
  validation_features_all <- validation_features_all[, final_features, drop = FALSE]
  
  # 处理缺失值 - 使用训练数据的统计量
  cat("Handling missing values...\n")
  for(feature in final_features) {
    if(any(is.na(validation_features_all[[feature]]))) {
      if(is.numeric(train_data_balanced[[feature]])) {
        mean_val <- mean(train_data_balanced[[feature]], na.rm = TRUE)
        validation_features_all[[feature]][is.na(validation_features_all[[feature]])] <- mean_val
      } else {
        mode_val <- names(sort(table(train_data_balanced[[feature]]), decreasing = TRUE))[1]
        validation_features_all[[feature]][is.na(validation_features_all[[feature]])] <- mode_val
      }
    }
  }
  
  # 确保没有缺失值
  if(any(is.na(validation_features_all))) {
    cat("Warning: Still have NA values after imputation\n")
    # 使用简单方法填充剩余的NA
    for(col in names(validation_features_all)) {
      na_indices <- is.na(validation_features_all[[col]])
      if(any(na_indices)) {
        if(is.numeric(validation_features_all[[col]])) {
          validation_features_all[[col]][na_indices] <- mean(validation_features_all[[col]], na.rm = TRUE)
        } else {
          validation_features_all[[col]][na_indices] <- names(sort(table(validation_features_all[[col]]), decreasing = TRUE))[1]
        }
      }
    }
  }
  
  # ========== 转换为数值类型 ==========
  cat("Converting features to numeric...\n")
  validation_features_numeric <- validation_features_all
  
  # 将因子和字符变量转换为数值
  for(col in final_features) {
    if(col %in% names(validation_features_numeric)) {
      if(is.factor(validation_features_numeric[[col]])) {
        validation_features_numeric[[col]] <- as.numeric(validation_features_numeric[[col]])
      } else if(is.character(validation_features_numeric[[col]])) {
        validation_features_numeric[[col]] <- as.numeric(as.factor(validation_features_numeric[[col]]))
      } else if(!is.numeric(validation_features_numeric[[col]])) {
        validation_features_numeric[[col]] <- as.numeric(validation_features_numeric[[col]])
      }
    }
  }
  
  # 检查是否有任何非数值数据
  if(any(sapply(validation_features_numeric, function(x) !is.numeric(x)))) {
    cat("Warning: Some features are still not numeric after conversion, forcing conversion...\n")
    validation_features_numeric <- as.data.frame(lapply(validation_features_numeric, function(x) as.numeric(as.character(x))))
  }
  
  # 转换为矩阵
  validation_matrix <- as.matrix(validation_features_numeric)
  
  # 最终检查NA值
  if(any(is.na(validation_matrix))) {
    cat("Final NA imputation...\n")
    for(j in 1:ncol(validation_matrix)) {
      na_indices <- is.na(validation_matrix[, j])
      if(any(na_indices)) {
        validation_matrix[na_indices, j] <- mean(validation_matrix[, j], na.rm = TRUE)
      }
    }
  }
  
  cat("Final processed data dimensions:", dim(validation_matrix), "\n")
  
  return(list(
    features = validation_matrix,
    labels = validation_labels_all,
    features_imputed = validation_features_all
  ))
}

# ==================== Test Set Validation ====================
cat("=== Part 3: Test Set Validation ===\n")
test_processed <- preprocess_validation_data(test_data, train_data_balanced, final_features)
test_features_subset <- test_processed$features
test_labels_all <- test_processed$labels
test_features_imputed <- test_processed$features_imputed

cat("Test set processed successfully. Dimensions:", dim(test_features_subset), "\n")

# Predict using final model
test_probs <- predict(final_xgb_model, test_features_subset)
test_metrics <- calculate_all_metrics(test_labels_all, test_probs)
test_roc <- test_metrics$ROC_Obj

# ==================== External Validation Set Validation ====================
cat("=== External Validation Set Validation ===\n")
external_processed <- preprocess_validation_data(external_data, train_data_balanced, final_features)
external_features_subset <- external_processed$features
external_labels_all <- external_processed$labels
external_features_imputed <- external_processed$features_imputed

cat("External set processed successfully. Dimensions:", dim(external_features_subset), "\n")

# Predict using final model
external_probs <- predict(final_xgb_model, external_features_subset)
external_metrics <- calculate_all_metrics(external_labels_all, external_probs)
external_roc <- external_metrics$ROC_Obj

# ==================== Generate Visualizations ====================
cat("Generating Visualizations...\n")
if(!dir.exists("Part3_Results")) dir.create("Part3_Results")

# 1. ROC Curve (Test + External)
png("Part3_Results/Combined_ROC_Curve.png", width = 800, height = 600)
par(mar = c(5, 5, 4, 2) + 0.1)
plot(test_roc, main = "ROC Curves - Test vs External Validation", 
     col = "blue", lwd = 3, print.auc = TRUE, print.auc.x = 0.6, print.auc.y = 0.4)
plot(external_roc, add = TRUE, col = "red", lwd = 3, print.auc = TRUE, print.auc.x = 0.6, print.auc.y = 0.3)
abline(0, 1, lty = 3, col = "gray")
legend("bottomright", 
       legend = c(paste("Test Set (AUC =", round(test_metrics$AUC, 3), ")"), 
                  paste("External (AUC =", round(external_metrics$AUC, 3), ")")), 
       col = c("blue", "red"), lwd = 3, cex = 1.2)
dev.off()

# 2. Calibration Curves
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

p1_test <- create_calibration_plot(test_probs, test_labels_all, 
                                   "Calibration Plot - Test Set", "blue")
p1_external <- create_calibration_plot(external_probs, external_labels_all, 
                                       "Calibration Plot - External Validation Set", "red")

ggsave("Part3_Results/Calibration_Plot_Test.png", p1_test, width = 8, height = 6, dpi = 300)
ggsave("Part3_Results/Calibration_Plot_External.png", p1_external, width = 8, height = 6, dpi = 300)

# 3. Probability Distribution by True Class (Combined)
combined_calibration_data <- rbind(
  data.frame(Probability = test_probs, True_Label = as.numeric(test_labels_all) - 1, Dataset = "Test"),
  data.frame(Probability = external_probs, True_Label = as.numeric(external_labels_all) - 1, Dataset = "External")
)

p2 <- ggplot(combined_calibration_data, aes(x = Probability, fill = as.factor(True_Label))) +
  geom_density(alpha = 0.6) +
  facet_wrap(~ Dataset, ncol = 2) +
  labs(title = "Probability Distribution by True Class",
       x = "Predicted Probability",
       y = "Density",
       fill = "True Class") +
  scale_fill_manual(values = c("0" = "red", "1" = "blue"), 
                    labels = c("Control", "Case")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

ggsave("Part3_Results/Combined_Probability_Distribution.png", p2, width = 12, height = 6, dpi = 300)

# 4. Performance Comparison (CV vs Test vs External)
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
  labs(title = "Performance Comparison: Cross-Validation vs Test Set vs External Validation",
       y = "Performance Value", x = "Metric") +
  scale_fill_manual(values = c("Cross-Validation" = "blue", "Test" = "red", "External" = "green")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "top") +
  ylim(0, 1)

ggsave("Part3_Results/Performance_Comparison.png", p3, width = 14, height = 6, dpi = 300)

# 5. Misclassification Analysis
create_misclassification_analysis <- function(probs, true_labels, features_imputed, dataset_name) {
  pred_labels <- ifelse(probs > 0.5, 1, 0)
  true_labels_numeric <- as.numeric(true_labels) - 1
  misclassified_indices <- which(pred_labels != true_labels_numeric)
  
  if(length(misclassified_indices) > 0) {
    misclassified_cases <- data.frame(
      Case_ID = if(!is.null(rownames(features_imputed))) rownames(features_imputed)[misclassified_indices] else misclassified_indices,
      True_Label = true_labels_numeric[misclassified_indices],
      Predicted_Label = pred_labels[misclassified_indices],
      Probability = probs[misclassified_indices]
    )
    
    p <- ggplot(misclassified_cases, aes(x = as.factor(True_Label), fill = as.factor(Predicted_Label))) +
      geom_bar(alpha = 0.8) +
      labs(title = paste(dataset_name, "Misclassification Analysis"),
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

test_misclassified <- create_misclassification_analysis(test_probs, test_labels_all, test_features_imputed, "Test")
external_misclassified <- create_misclassification_analysis(external_probs, external_labels_all, external_features_imputed, "External")

# 6. Confusion Matrix Visualization
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

p5_test <- create_confusion_matrix_plot(test_metrics, "Confusion Matrix - Test Set")
p5_external <- create_confusion_matrix_plot(external_metrics, "Confusion Matrix - External Validation Set")

ggsave("Part3_Results/Confusion_Matrix_Test.png", p5_test, width = 6, height = 5, dpi = 300)
ggsave("Part3_Results/Confusion_Matrix_External.png", p5_external, width = 6, height = 5, dpi = 300)

# 7. Decision Threshold Analysis (Combined)
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
  
  threshold_analysis$Dataset <- dataset_name
  return(threshold_analysis)
}

test_threshold <- perform_threshold_analysis(test_probs, test_labels_all, "Test")
external_threshold <- perform_threshold_analysis(external_probs, external_labels_all, "External")
combined_threshold <- rbind(test_threshold, external_threshold)

# Plot combined threshold analysis
threshold_long <- reshape(combined_threshold, 
                          direction = "long",
                          varying = c("Sensitivity", "Specificity", "Accuracy"),
                          v.names = "Value",
                          timevar = "Metric",
                          times = c("Sensitivity", "Specificity", "Accuracy"),
                          idvar = c("Threshold", "Dataset"))

p6 <- ggplot(threshold_long, aes(x = Threshold, y = Value, color = Dataset, linetype = Metric)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +
  labs(title = "Performance Metrics vs Decision Threshold",
       x = "Decision Threshold",
       y = "Performance Value") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        legend.position = "top") +
  scale_x_continuous(breaks = seq(0.1, 0.9, by = 0.1)) +
  scale_color_manual(values = c("Test" = "blue", "External" = "red"))

ggsave("Part3_Results/Combined_Threshold_Analysis.png", p6, width = 12, height = 6, dpi = 300)

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

# Performance comparison summary
performance_comparison <- data.frame(
  Metric = c("AUC", "Sensitivity", "Specificity", "PPV", "NPV", "Accuracy", "F1", "Balanced_Accuracy"),
  Cross_Validation = c(
    cv_summary$Mean[1], cv_summary$Mean[2], cv_summary$Mean[3],
    cv_summary$Mean[4], cv_summary$Mean[5], cv_summary$Mean[6],
    cv_summary$Mean[7], cv_summary$Mean[8]
  ),
  Test_Set = c(
    test_metrics$AUC, test_metrics$Sensitivity, test_metrics$Specificity,
    test_metrics$PPV, test_metrics$NPV, test_metrics$Accuracy,
    test_metrics$F1, test_metrics$Balanced_Accuracy
  ),
  External_Validation = c(
    external_metrics$AUC, external_metrics$Sensitivity, external_metrics$Specificity,
    external_metrics$PPV, external_metrics$NPV, external_metrics$Accuracy,
    external_metrics$F1, external_metrics$Balanced_Accuracy
  )
)

writeData(wb3, "Test_Set_Performance", test_performance, startRow = 1)
writeData(wb3, "External_Validation_Performance", external_performance, startRow = 1)
writeData(wb3, "Performance_Comparison", performance_comparison, startRow = 1)

if(!is.null(test_misclassified)) {
  writeData(wb3, "Misclassified_Test_Cases", test_misclassified, startRow = 1)
}
if(!is.null(external_misclassified)) {
  writeData(wb3, "Misclassified_External_Cases", external_misclassified, startRow = 1)
}

# Save probability distribution
test_prob_distribution <- data.frame(
  Case_ID = if(!is.null(rownames(test_features_imputed))) rownames(test_features_imputed) else 1:nrow(test_features_imputed),
  True_Label = as.numeric(test_labels_all) - 1,
  Predicted_Probability = test_probs,
  Predicted_Label = ifelse(test_probs > 0.5, 1, 0),
  Dataset = "Test"
)

external_prob_distribution <- data.frame(
  Case_ID = if(!is.null(rownames(external_features_imputed))) rownames(external_features_imputed) else 1:nrow(external_features_imputed),
  True_Label = as.numeric(external_labels_all) - 1,
  Predicted_Probability = external_probs,
  Predicted_Label = ifelse(external_probs > 0.5, 1, 0),
  Dataset = "External"
)

combined_prob_distribution <- rbind(test_prob_distribution, external_prob_distribution)
writeData(wb3, "Probability_Distribution", combined_prob_distribution, startRow = 1)

# Save threshold analysis
addWorksheet(wb3, "Threshold_Analysis")
writeData(wb3, "Threshold_Analysis", combined_threshold, startRow = 1)

saveWorkbook(wb3, "Part3_Test_and_External_Validation_Results.xlsx", overwrite = TRUE)

# Save objects
saveRDS(test_metrics, "test_metrics.rds")
saveRDS(external_metrics, "external_metrics.rds")
saveRDS(test_probs, "test_probs.rds")
saveRDS(external_probs, "external_probs.rds")
saveRDS(combined_threshold, "combined_threshold_analysis.rds")

cat("\n=== Part 3 Complete ===\n")
cat("Results saved to:\n")
cat("- Part3_Test_and_External_Validation_Results.xlsx\n")
cat("- Part3_Results/ directory\n")
cat("- test_metrics.rds, external_metrics.rds\n")
cat("- test_probs.rds, external_probs.rds\n")
cat("- combined_threshold_analysis.rds\n")

cat("\nTest Set Performance:\n")
print(test_performance)
cat("\nExternal Validation Set Performance:\n")
print(external_performance)
cat("\nPerformance Comparison:\n")
print(performance_comparison)

cat("\nOptimal Threshold Analysis (based on Youden's J statistic):\n")
youden_j_test <- test_threshold$Sensitivity + test_threshold$Specificity - 1
optimal_idx_test <- which.max(youden_j_test)
cat("Test Set - Optimal Threshold:", test_threshold$Threshold[optimal_idx_test], "\n")
cat("Test Set - Sensitivity at optimal threshold:", round(test_threshold$Sensitivity[optimal_idx_test], 3), "\n")
cat("Test Set - Specificity at optimal threshold:", round(test_threshold$Specificity[optimal_idx_test], 3), "\n")

youden_j_external <- external_threshold$Sensitivity + external_threshold$Specificity - 1
optimal_idx_external <- which.max(youden_j_external)
cat("External Set - Optimal Threshold:", external_threshold$Threshold[optimal_idx_external], "\n")
cat("External Set - Sensitivity at optimal threshold:", round(external_threshold$Sensitivity[optimal_idx_external], 3), "\n")
cat("External Set - Specificity at optimal threshold:", round(external_threshold$Specificity[optimal_idx_external], 3), "\n")