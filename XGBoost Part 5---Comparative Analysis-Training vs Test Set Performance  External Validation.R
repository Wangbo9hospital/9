# Comparative Analysis: Training (CV) vs Test Set vs External Validation Performance
# Load required packages
library(ggplot2)
library(pROC)
library(openxlsx)
library(gridExtra)
library(patchwork)
library(dplyr)
library(grid)

# Load results from previous parts
cv_results <- readRDS("cv_results.rds")
cv_summary <- readRDS("cv_summary.rds")
cv_predictions <- readRDS("cv_predictions.rds")  # 加载CV预测结果
test_metrics <- readRDS("test_metrics.rds")
test_probs <- readRDS("test_probs.rds")
# 添加外部验证集结果
external_metrics <- readRDS("external_metrics.rds")
external_probs <- readRDS("external_probs.rds")

train_data_balanced <- readRDS("train_data_balanced.rds")
final_xgb_model <- readRDS("final_xgb_model.rds")
final_features <- readRDS("final_features.rds")

# ==================== Prepare Data for Comparison ====================
cat("=== Comparative Analysis: Training (CV) vs Test Set vs External Validation Performance ===\n")

# 使用CV预测结果作为训练集性能
train_cv_probs <- cv_predictions$Predicted_Probability
train_cv_labels <- cv_predictions$True_Label

# 计算训练集CV的ROC和性能指标
train_cv_roc <- roc(train_cv_labels, train_cv_probs)
train_cv_auc <- auc(train_cv_roc)

# 计算训练集CV的详细性能指标
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

train_cv_metrics <- calculate_all_metrics(
  as.factor(train_cv_labels), 
  train_cv_probs
)

cat("Training Set (CV) Performance:\n")
cat("AUC:", round(train_cv_auc, 4), "\n")
cat("Accuracy:", round(train_cv_metrics$Accuracy, 4), "\n")
cat("Samples:", length(train_cv_labels), "\n")

# 修复测试集数据问题
cat("Loading and preparing test data...\n")

# 重新加载测试数据以确保一致性
test_data <- read_excel("C:/Users/Bo Wang/Desktop/Internal Validation Cohort.xlsx")
test_data$Diagnosis <- as.factor(test_data$Diagnosis)

# 准备测试集特征
test_features_all <- test_data[, !names(test_data) %in% "Diagnosis"]

# 使用与训练集相同的插补方法
for(feature in names(test_features_all)) {
  if(any(is.na(test_features_all[[feature]]))) {
    if(is.numeric(test_features_all[[feature]])) {
      mean_val <- mean(train_data_balanced[[feature]], na.rm = TRUE)
      test_features_all[[feature]][is.na(test_features_all[[feature]])] <- mean_val
    } else {
      mode_val <- names(sort(table(train_data_balanced[[feature]]), decreasing = TRUE))[1]
      test_features_all[[feature]][is.na(test_features_all[[feature]])] <- mode_val
    }
  }
}

# 确保所有特征都存在
missing_features <- setdiff(final_features, names(test_features_all))
if(length(missing_features) > 0) {
  cat("Adding missing features:", paste(missing_features, collapse = ", "), "\n")
  for(feature in missing_features) {
    if(is.numeric(train_data_balanced[[feature]])) {
      test_features_all[[feature]] <- mean(train_data_balanced[[feature]], na.rm = TRUE)
    } else {
      mode_val <- names(sort(table(train_data_balanced[[feature]]), decreasing = TRUE))[1]
      test_features_all[[feature]] <- mode_val
    }
  }
}

# ========== 修复：确保所有特征都是数值类型 ==========
cat("Converting test features to numeric...\n")
test_features_subset <- test_features_all[, final_features, drop = FALSE]

# 将因子和字符变量转换为数值
for(col in final_features) {
  if(col %in% names(test_features_subset)) {
    if(is.factor(test_features_subset[[col]])) {
      cat("Converting factor variable", col, "to numeric\n")
      test_features_subset[[col]] <- as.numeric(test_features_subset[[col]])
    } else if(is.character(test_features_subset[[col]])) {
      cat("Converting character variable", col, "to numeric\n")
      # 先转换为因子，再转换为数值
      test_features_subset[[col]] <- as.numeric(as.factor(test_features_subset[[col]]))
    } else if(!is.numeric(test_features_subset[[col]])) {
      cat("Converting non-numeric variable", col, "to numeric\n")
      test_features_subset[[col]] <- as.numeric(test_features_subset[[col]])
    }
  }
}

# 检查是否有任何非数值数据
if(any(sapply(test_features_subset, function(x) !is.numeric(x)))) {
  cat("Warning: Some test features are still not numeric after conversion\n")
  # 强制转换为数值
  test_features_subset <- as.data.frame(lapply(test_features_subset, function(x) as.numeric(as.factor(x))))
}

# 转换为矩阵
test_features_subset <- as.matrix(test_features_subset)

# 检查是否有NA值并处理
if(any(is.na(test_features_subset))) {
  cat("Warning: NA values found in test features, imputing with mean...\n")
  for(j in 1:ncol(test_features_subset)) {
    na_indices <- is.na(test_features_subset[, j])
    if(any(na_indices)) {
      test_features_subset[na_indices, j] <- mean(test_features_subset[, j], na.rm = TRUE)
    }
  }
}

# 调试信息
cat("Test features dimensions:", dim(test_features_subset), "\n")
cat("Test features type:", class(test_features_subset), "\n")
cat("Test features data type:", typeof(test_features_subset), "\n")

# 准备测试标签
test_labels <- as.numeric(test_data$Diagnosis) - 1

# 重新计算测试集预测概率以确保一致性
test_probs_recalc <- predict(final_xgb_model, test_features_subset)

# 检查长度是否一致
cat("Length check:\n")
cat("Test labels:", length(test_labels), "\n")
cat("Test probabilities:", length(test_probs_recalc), "\n")
cat("Original test probabilities:", length(test_probs), "\n")

# 使用重新计算的概率或原始概率（确保长度一致）
if(length(test_labels) == length(test_probs_recalc)) {
  test_probs_final <- test_probs_recalc
  cat("Using recalculated test probabilities\n")
} else if(length(test_labels) == length(test_probs)) {
  test_probs_final <- test_probs
  cat("Using original test probabilities\n")
} else {
  # 如果都不匹配，使用重新计算的并截断到最小长度
  min_length <- min(length(test_labels), length(test_probs_recalc))
  test_labels <- test_labels[1:min_length]
  test_probs_final <- test_probs_recalc[1:min_length]
  cat("Adjusted to common length:", min_length, "\n")
}

# 计算测试集ROC
test_roc <- roc(test_labels, test_probs_final)
test_auc <- auc(test_roc)

# ==================== 准备外部验证集数据 ====================
cat("Preparing external validation data...\n")

# 重新加载外部验证数据以确保一致性
external_data <- read_excel("C:/Users/Bo Wang/Desktop/External Validation Cohort.xlsx") # 请修改为您的实际路径
external_data$Diagnosis <- as.factor(external_data$Diagnosis)

# 准备外部验证集特征
external_features_all <- external_data[, !names(external_data) %in% "Diagnosis"]

# 使用与训练集相同的插补方法
for(feature in names(external_features_all)) {
  if(any(is.na(external_features_all[[feature]]))) {
    if(is.numeric(external_features_all[[feature]])) {
      mean_val <- mean(train_data_balanced[[feature]], na.rm = TRUE)
      external_features_all[[feature]][is.na(external_features_all[[feature]])] <- mean_val
    } else {
      mode_val <- names(sort(table(train_data_balanced[[feature]]), decreasing = TRUE))[1]
      external_features_all[[feature]][is.na(external_features_all[[feature]])] <- mode_val
    }
  }
}

# 确保所有特征都存在
missing_features_external <- setdiff(final_features, names(external_features_all))
if(length(missing_features_external) > 0) {
  cat("Adding missing features in external set:", paste(missing_features_external, collapse = ", "), "\n")
  for(feature in missing_features_external) {
    if(is.numeric(train_data_balanced[[feature]])) {
      external_features_all[[feature]] <- mean(train_data_balanced[[feature]], na.rm = TRUE)
    } else {
      mode_val <- names(sort(table(train_data_balanced[[feature]]), decreasing = TRUE))[1]
      external_features_all[[feature]] <- mode_val
    }
  }
}

# ========== 修复：确保所有特征都是数值类型 ==========
cat("Converting external features to numeric...\n")
external_features_subset <- external_features_all[, final_features, drop = FALSE]

# 将因子和字符变量转换为数值
for(col in final_features) {
  if(col %in% names(external_features_subset)) {
    if(is.factor(external_features_subset[[col]])) {
      cat("Converting factor variable", col, "to numeric\n")
      external_features_subset[[col]] <- as.numeric(external_features_subset[[col]])
    } else if(is.character(external_features_subset[[col]])) {
      cat("Converting character variable", col, "to numeric\n")
      external_features_subset[[col]] <- as.numeric(as.factor(external_features_subset[[col]]))
    } else if(!is.numeric(external_features_subset[[col]])) {
      cat("Converting non-numeric variable", col, "to numeric\n")
      external_features_subset[[col]] <- as.numeric(external_features_subset[[col]])
    }
  }
}

# 检查是否有任何非数值数据
if(any(sapply(external_features_subset, function(x) !is.numeric(x)))) {
  cat("Warning: Some external features are still not numeric after conversion\n")
  external_features_subset <- as.data.frame(lapply(external_features_subset, function(x) as.numeric(as.factor(x))))
}

# 转换为矩阵
external_features_subset <- as.matrix(external_features_subset)

# 检查是否有NA值并处理
if(any(is.na(external_features_subset))) {
  cat("Warning: NA values found in external features, imputing with mean...\n")
  for(j in 1:ncol(external_features_subset)) {
    na_indices <- is.na(external_features_subset[, j])
    if(any(na_indices)) {
      external_features_subset[na_indices, j] <- mean(external_features_subset[, j], na.rm = TRUE)
    }
  }
}

# 准备外部验证标签
external_labels <- as.numeric(external_data$Diagnosis) - 1

# 重新计算外部验证集预测概率以确保一致性
external_probs_recalc <- predict(final_xgb_model, external_features_subset)

# 检查长度是否一致
cat("External validation length check:\n")
cat("External labels:", length(external_labels), "\n")
cat("External probabilities:", length(external_probs_recalc), "\n")
cat("Original external probabilities:", length(external_probs), "\n")

# 使用重新计算的概率或原始概率（确保长度一致）
if(length(external_labels) == length(external_probs_recalc)) {
  external_probs_final <- external_probs_recalc
  cat("Using recalculated external probabilities\n")
} else if(length(external_labels) == length(external_probs)) {
  external_probs_final <- external_probs
  cat("Using original external probabilities\n")
} else {
  # 如果都不匹配，使用重新计算的并截断到最小长度
  min_length <- min(length(external_labels), length(external_probs_recalc))
  external_labels <- external_labels[1:min_length]
  external_probs_final <- external_probs_recalc[1:min_length]
  cat("Adjusted to common length:", min_length, "\n")
}

# 计算外部验证集ROC
external_roc <- roc(external_labels, external_probs_final)
external_auc <- auc(external_roc)

# ==================== Create Comprehensive Comparison Plot ====================
cat("Creating comprehensive comparison plots...\n")
if(!dir.exists("Comparative_Analysis")) dir.create("Comparative_Analysis")

# 计算AUC的置信区间
train_cv_auc_ci <- ci.auc(train_cv_roc)
test_auc_ci <- ci.auc(test_roc)
external_auc_ci <- ci.auc(external_roc)

# 1. ROC Curve Comparison (三个数据集) - 修改样式
png("Comparative_Analysis/ROC_Comparison_Three_Sets.png", width = 1000, height = 800)
par(mar = c(5, 5, 4, 2) + 0.1, mfrow = c(1, 1))

# 设置空的绘图区域，确保从(0,0)开始
plot(1, type = "n", 
     xlim = c(0, 1), ylim = c(0, 1),
     xlab = "False positive rate", 
     ylab = "True positive rate",
     main = "ROC Curves: Train vs Internal vs External Validation",
     cex.lab = 1.3, cex.axis = 1.2, cex.main = 1.5)

# 添加对角线
abline(0, 1, lty = 2, col = "gray", lwd = 2)

# 绘制ROC曲线 - 确保从(0,0)开始
lines(1 - test_roc$specificities, test_roc$sensitivities, 
      col = "red", lwd = 3, type = "l")
lines(1 - train_cv_roc$specificities, train_cv_roc$sensitivities, 
      col = "blue", lwd = 3, lty = 1)
lines(1 - external_roc$specificities, external_roc$sensitivities, 
      col = "green", lwd = 3, lty = 1)

# 添加图例，显示AUC和95%CI
legend("bottomright", 
       legend = c(
         sprintf("Train (AUC = %.3f [%.3f-%.3f])", 
                 train_cv_auc, train_cv_auc_ci[1], train_cv_auc_ci[3]),
         sprintf("Internal (AUC = %.3f [%.3f-%.3f])", 
                 test_auc, test_auc_ci[1], test_auc_ci[3]),
         sprintf("External (AUC = %.3f [%.3f-%.3f])", 
                 external_auc, external_auc_ci[1], external_auc_ci[3])
       ),
       col = c("blue", "red", "green"), 
       lwd = 3, 
       lty = 1,
       cex = 1.2,
       bg = "white", 
       box.lwd = 1)

dev.off()

# 2. Performance Metrics Comparison Bar Chart (三个数据集)
performance_comparison <- data.frame(
  Metric = c("AUC", "Sensitivity", "Specificity", "Accuracy", "F1", "Balanced_Accuracy"),
  Training_CV = c(
    train_cv_auc,
    train_cv_metrics$Sensitivity,
    train_cv_metrics$Specificity,
    train_cv_metrics$Accuracy,
    train_cv_metrics$F1,
    train_cv_metrics$Balanced_Accuracy
  ),
  Test = c(
    test_auc,
    test_metrics$Sensitivity,
    test_metrics$Specificity,
    test_metrics$Accuracy,
    test_metrics$F1,
    test_metrics$Balanced_Accuracy
  ),
  External = c(
    external_auc,
    external_metrics$Sensitivity,
    external_metrics$Specificity,
    external_metrics$Accuracy,
    external_metrics$F1,
    external_metrics$Balanced_Accuracy
  )
)

comparison_long <- data.frame(
  Metric = rep(performance_comparison$Metric, 3),
  Dataset = rep(c("Training_CV", "Test", "External"), each = nrow(performance_comparison)),
  Value = c(performance_comparison$Training_CV, performance_comparison$Test, performance_comparison$External)
)

# 计算CV的标准差（使用十折交叉验证结果）
cv_sd <- data.frame(
  Metric = c("AUC", "Sensitivity", "Specificity", "Accuracy", "F1", "Balanced_Accuracy"),
  SD = c(
    sd(cv_results$AUC),
    sd(cv_results$Sensitivity),
    sd(cv_results$Specificity),
    sd(cv_results$Accuracy),
    sd(cv_results$F1),
    sd(cv_results$Balanced_Accuracy)
  )
)

comparison_long <- merge(comparison_long, cv_sd, by = "Metric", all.x = TRUE)
comparison_long$SD[comparison_long$Dataset != "Training_CV"] <- NA

p_bar <- ggplot(comparison_long, aes(x = Metric, y = Value, fill = Dataset)) +
  geom_bar(stat = "identity", position = position_dodge(0.8), width = 0.7, alpha = 0.8) +
  geom_errorbar(aes(ymin = Value - SD, ymax = Value + SD), 
                position = position_dodge(0.8), width = 0.2, 
                color = "black", alpha = 0.7, na.rm = TRUE) +
  labs(title = "Performance Metrics Comparison: Training (CV) vs Test vs External Validation",
       y = "Score", x = "Metric") +
  scale_fill_manual(values = c("Training_CV" = "blue", "Test" = "red", "External" = "green"),
                    labels = c("Training_CV" = "Training (CV)", "Test" = "Test", "External" = "External Validation")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        axis.text = element_text(size = 11),
        axis.title = element_text(size = 13),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "top") +
  ylim(0, 1)

ggsave("Comparative_Analysis/Performance_Bar_Comparison_Three_Sets.png", p_bar, width = 14, height = 8, dpi = 300)

# 3. Probability Distribution Comparison (三个数据集)
prob_comparison <- data.frame(
  Probability = c(train_cv_probs, test_probs_final, external_probs_final),
  Dataset = c(rep("Training_CV", length(train_cv_probs)), 
              rep("Test", length(test_probs_final)),
              rep("External", length(external_probs_final))),
  True_Label = c(train_cv_labels, test_labels, external_labels)
)

p_prob <- ggplot(prob_comparison, aes(x = Probability, fill = Dataset)) +
  geom_density(alpha = 0.6) +
  facet_wrap(~ True_Label, labeller = as_labeller(c("0" = "Control", "1" = "Case"))) +
  labs(title = "Predicted Probability Distribution by Class and Dataset",
       x = "Predicted Probability", y = "Density") +
  scale_fill_manual(values = c("Training_CV" = "blue", "Test" = "red", "External" = "green"),
                    labels = c("Training_CV" = "Training (CV)", "Test" = "Test", "External" = "External Validation")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        strip.text = element_text(size = 12, face = "bold"))

ggsave("Comparative_Analysis/Probability_Distribution_Comparison_Three_Sets.png", p_prob, width = 14, height = 6, dpi = 300)

# 4. Calibration Curve Comparison (三个数据集)
create_calibration_data <- function(probs, labels, dataset_name) {
  calibration_data <- data.frame(
    Probability = probs,
    True_Label = labels
  )
  
  calibration_data$Bin <- cut(calibration_data$Probability, 
                              breaks = seq(0, 1, by = 0.1), 
                              include.lowest = TRUE)
  
  cal_summary <- aggregate(Probability ~ Bin, data = calibration_data, FUN = mean)
  names(cal_summary)[2] <- "Mean_Predicted"
  actual <- aggregate(True_Label ~ Bin, data = calibration_data, FUN = mean)
  names(actual)[2] <- "Mean_Actual"
  counts <- aggregate(True_Label ~ Bin, data = calibration_data, FUN = length)
  names(counts)[2] <- "Count"
  
  cal_summary <- merge(cal_summary, actual, by = "Bin")
  cal_summary <- merge(cal_summary, counts, by = "Bin")
  cal_summary$Dataset <- dataset_name
  
  return(cal_summary)
}

# Create calibration data for all datasets
train_cv_cal_summary <- create_calibration_data(train_cv_probs, train_cv_labels, "Training_CV")
test_cal_summary <- create_calibration_data(test_probs_final, test_labels, "Test")
external_cal_summary <- create_calibration_data(external_probs_final, external_labels, "External")

# Combine calibration data
calibration_combined <- rbind(train_cv_cal_summary, test_cal_summary, external_cal_summary)

p_cal <- ggplot(calibration_combined, aes(x = Mean_Predicted, y = Mean_Actual, color = Dataset)) +
  geom_point(aes(size = Count), alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  geom_smooth(method = "loess", se = FALSE) +
  labs(title = "Calibration Curve Comparison: Training (CV) vs Test vs External Validation",
       x = "Mean Predicted Probability",
       y = "Observed Fraction Positive",
       size = "Number of Cases",
       color = "Dataset") +
  scale_color_manual(values = c("Training_CV" = "blue", "Test" = "red", "External" = "green"),
                     labels = c("Training_CV" = "Training (CV)", "Test" = "Test", "External" = "External Validation")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

ggsave("Comparative_Analysis/Calibration_Comparison_Three_Sets.png", p_cal, width = 12, height = 8, dpi = 300)

# 5. Create Comprehensive Summary Figure (三个数据集)
cat("Creating comprehensive summary figure...\n")

# 使用patchwork包组合图表
comprehensive_plot <- (p_bar + p_cal) / p_prob +
  plot_annotation(
    title = "Comprehensive Model Performance: Training (CV) vs Test vs External Validation Sets",
    subtitle = paste("Training (CV):", length(train_cv_labels), "samples |",
                     "Test:", length(test_labels), "samples |", 
                     "External:", length(external_labels), "samples"),
    theme = theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
                  plot.subtitle = element_text(size = 12, hjust = 0.5))
  )

ggsave("Comparative_Analysis/Comprehensive_Comparison_Three_Sets.png", comprehensive_plot, 
       width = 18, height = 14, dpi = 300)

# ==================== Save Comparative Results ====================
cat("Saving comparative analysis results...\n")

wb_compare <- createWorkbook()
addWorksheet(wb_compare, "Performance_Comparison")
addWorksheet(wb_compare, "ROC_Statistics")
addWorksheet(wb_compare, "Detailed_Metrics")
addWorksheet(wb_compare, "CV_Test_External_Predictions")

# Save performance comparison table
performance_table <- data.frame(
  Metric = performance_comparison$Metric,
  Training_CV_Mean = round(performance_comparison$Training_CV, 3),
  Training_CV_SD = round(cv_sd$SD, 3),
  Test_Value = round(performance_comparison$Test, 3),
  External_Value = round(performance_comparison$External, 3),
  Train_Test_Difference = round(performance_comparison$Training_CV - performance_comparison$Test, 3),
  Train_External_Difference = round(performance_comparison$Training_CV - performance_comparison$External, 3)
)

writeData(wb_compare, "Performance_Comparison", performance_table, startRow = 1)

# Save ROC statistics
roc_stats <- data.frame(
  Dataset = c("Training_CV", "Test", "External"),
  AUC = c(train_cv_auc, test_auc, external_auc),
  AUC_CI_Lower = c(ci.auc(train_cv_roc)[1], ci.auc(test_roc)[1], ci.auc(external_roc)[1]),
  AUC_CI_Upper = c(ci.auc(train_cv_roc)[3], ci.auc(test_roc)[3], ci.auc(external_roc)[3]),
  Samples = c(length(train_cv_labels), length(test_labels), length(external_labels)),
  Positive_Cases = c(sum(train_cv_labels == 1), sum(test_labels == 1), sum(external_labels == 1)),
  Negative_Cases = c(sum(train_cv_labels == 0), sum(test_labels == 0), sum(external_labels == 0))
)

writeData(wb_compare, "ROC_Statistics", roc_stats, startRow = 1)

# Save detailed metrics
detailed_metrics <- data.frame(
  Metric = c("AUC", "Sensitivity", "Specificity", "PPV", "NPV", "Accuracy", "F1", "Balanced_Accuracy"),
  Training_CV_Mean = c(
    train_cv_auc,
    train_cv_metrics$Sensitivity,
    train_cv_metrics$Specificity,
    train_cv_metrics$PPV,
    train_cv_metrics$NPV,
    train_cv_metrics$Accuracy,
    train_cv_metrics$F1,
    train_cv_metrics$Balanced_Accuracy
  ),
  Training_CV_SD = c(
    sd(cv_results$AUC),
    sd(cv_results$Sensitivity),
    sd(cv_results$Specificity),
    sd(cv_results$PPV),
    sd(cv_results$NPV),
    sd(cv_results$Accuracy),
    sd(cv_results$F1),
    sd(cv_results$Balanced_Accuracy)
  ),
  Test_Value = c(
    test_auc,
    test_metrics$Sensitivity,
    test_metrics$Specificity,
    test_metrics$PPV,
    test_metrics$NPV,
    test_metrics$Accuracy,
    test_metrics$F1,
    test_metrics$Balanced_Accuracy
  ),
  External_Value = c(
    external_auc,
    external_metrics$Sensitivity,
    external_metrics$Specificity,
    external_metrics$PPV,
    external_metrics$NPV,
    external_metrics$Accuracy,
    external_metrics$F1,
    external_metrics$Balanced_Accuracy
  )
)

writeData(wb_compare, "Detailed_Metrics", detailed_metrics, startRow = 1)

# Save prediction comparisons
prediction_comparison <- data.frame(
  Dataset = c(rep("Training_CV", length(train_cv_labels)), 
              rep("Test", length(test_labels)),
              rep("External", length(external_labels))),
  True_Label = c(train_cv_labels, test_labels, external_labels),
  Predicted_Probability = c(train_cv_probs, test_probs_final, external_probs_final),
  Sample_Type = c(rep("Training", length(train_cv_labels)), 
                  rep("Test", length(test_labels)),
                  rep("External", length(external_labels)))
)

writeData(wb_compare, "CV_Test_External_Predictions", prediction_comparison, startRow = 1)

saveWorkbook(wb_compare, "Comparative_Analysis/Comparative_Results_Three_Sets.xlsx", overwrite = TRUE)

# ==================== Print Summary ====================
cat("\n=== Comparative Analysis Complete ===\n")
cat("Results saved to Comparative_Analysis/ directory\n")
cat("\nPerformance Summary:\n")
cat("Training Set (CV) AUC:", round(train_cv_auc, 3), "\n")
cat("Test Set AUC:", round(test_auc, 3), "\n")
cat("External Validation AUC:", round(external_auc, 3), "\n")
cat("AUC Differences:\n")
cat("  Train-Test:", round(abs(train_cv_auc - test_auc), 3), "\n")
cat("  Train-External:", round(abs(train_cv_auc - external_auc), 3), "\n")
cat("  Test-External:", round(abs(test_auc - external_auc), 3), "\n")

cat("\nSample Sizes:\n")
cat("Training (CV):", length(train_cv_labels), "samples (", sum(train_cv_labels == 1), "cases,", sum(train_cv_labels == 0), "controls)\n")
cat("Test:", length(test_labels), "samples (", sum(test_labels == 1), "cases,", sum(test_labels == 0), "controls)\n")
cat("External:", length(external_labels), "samples (", sum(external_labels == 1), "cases,", sum(external_labels == 0), "controls)\n")

# Calculate performance stability
performance_stability <- data.frame(
  Metric = detailed_metrics$Metric,
  Training_CV_CV = round(detailed_metrics$Training_CV_SD / detailed_metrics$Training_CV_Mean * 100, 1),
  Train_Test_Difference = round(abs(detailed_metrics$Training_CV_Mean - detailed_metrics$Test_Value), 3),
  Train_External_Difference = round(abs(detailed_metrics$Training_CV_Mean - detailed_metrics$External_Value), 3)
)

cat("\nPerformance Stability (Coefficient of Variation % in Training CV):\n")
print(performance_stability[, c("Metric", "Training_CV_CV")])

cat("\nLargest performance differences:\n")
largest_diffs_train_test <- performance_stability[order(-performance_stability$Train_Test_Difference), ]
largest_diffs_train_external <- performance_stability[order(-performance_stability$Train_External_Difference), ]

cat("Between Training and Test:\n")
print(head(largest_diffs_train_test, 3))
cat("Between Training and External:\n")
print(head(largest_diffs_train_external, 3))

# 生成模型泛化能力评估
cat("\n=== Model Generalization Assessment ===\n")
auc_difference_test <- abs(train_cv_auc - test_auc)
auc_difference_external <- abs(train_cv_auc - external_auc)

assess_generalization <- function(auc_diff, dataset_name) {
  if(auc_diff < 0.05) {
    cat("✓ Excellent generalization to", dataset_name, ": AUC difference < 0.05\n")
  } else if(auc_diff < 0.1) {
    cat("○ Good generalization to", dataset_name, ": AUC difference < 0.1\n")
  } else {
    cat("⚠ Potential overfitting for", dataset_name, ": AUC difference >=", round(auc_diff, 3), "\n")
  }
}

assess_generalization(auc_difference_test, "test set")
assess_generalization(auc_difference_external, "external validation set")

# 计算平均性能下降
mean_performance_drop_test <- mean(abs(performance_comparison$Training_CV - performance_comparison$Test))
mean_performance_drop_external <- mean(abs(performance_comparison$Training_CV - performance_comparison$External))
cat("Average performance drop from training (CV) to test:", round(mean_performance_drop_test, 3), "\n")
cat("Average performance drop from training (CV) to external:", round(mean_performance_drop_external, 3), "\n")

# 保存修复后的测试概率和训练CV概率
saveRDS(test_probs_final, "test_probs_final.rds")
saveRDS(test_labels, "test_labels_final.rds")
saveRDS(external_probs_final, "external_probs_final.rds")
saveRDS(external_labels, "external_labels_final.rds")
saveRDS(train_cv_probs, "train_cv_probs.rds")
saveRDS(train_cv_labels, "train_cv_labels.rds")

cat("\nFixed probabilities and labels saved:\n")
cat("- test_probs_final.rds and test_labels_final.rds\n")
cat("- external_probs_final.rds and external_labels_final.rds\n")
cat("- train_cv_probs.rds and train_cv_labels.rds\n")

# ==================== 新增：详细性能对比分析 ====================
cat("\n=== Detailed Performance Analysis ===\n")

# 计算统计显著性（使用DeLong检验比较AUC）
roc_test_train_test <- roc.test(train_cv_roc, test_roc, method = "delong")
roc_test_train_external <- roc.test(train_cv_roc, external_roc, method = "delong")
roc_test_test_external <- roc.test(test_roc, external_roc, method = "delong")

cat("AUC Comparison (DeLong test):\n")
cat("Training vs Test - p-value:", round(roc_test_train_test$p.value, 4), "\n")
cat("Training vs External - p-value:", round(roc_test_train_external$p.value, 4), "\n")
cat("Test vs External - p-value:", round(roc_test_test_external$p.value, 4), "\n")

# 计算性能下降百分比
performance_drop_pct <- data.frame(
  Metric = performance_comparison$Metric,
  Training_CV = performance_comparison$Training_CV,
  Test = performance_comparison$Test,
  External = performance_comparison$External,
  Drop_Train_Test_Pct = round((performance_comparison$Training_CV - performance_comparison$Test) / performance_comparison$Training_CV * 100, 1),
  Drop_Train_External_Pct = round((performance_comparison$Training_CV - performance_comparison$External) / performance_comparison$Training_CV * 100, 1)
)

cat("\nPerformance Drop Analysis (%):\n")
print(performance_drop_pct)

# 保存详细分析结果
write.csv(performance_drop_pct, "Comparative_Analysis/Performance_Drop_Analysis_Three_Sets.csv", row.names = FALSE)

# 生成综合泛化能力报告
cat("\n=== Comprehensive Generalization Report ===\n")
cat("Model demonstrates:\n")

if(mean_performance_drop_test < 0.05 && mean_performance_drop_external < 0.05) {
  cat("✓ EXCELLENT generalization across both test and external validation sets\n")
} else if(mean_performance_drop_test < 0.1 && mean_performance_drop_external < 0.1) {
  cat("○ GOOD generalization across both test and external validation sets\n")
} else if(mean_performance_drop_test < 0.1) {
  cat("○ REASONABLE generalization to test set but POOR to external validation\n")
} else if(mean_performance_drop_external < 0.1) {
  cat("○ REASONABLE generalization to external validation but POOR to test set\n")
} else {
  cat("⚠ POOR generalization to both test and external validation sets\n")
}

cat("Recommendations:\n")
if(auc_difference_external > 0.1) {
  cat("- Consider collecting more diverse training data to improve external validity\n")
}
if(mean_performance_drop_test > 0.1) {
  cat("- Model may be overfitting to training data, consider regularization\n")
}
if(abs(test_auc - external_auc) > 0.1) {
  cat("- Significant difference between test and external performance suggests dataset distribution differences\n")
}

# 保存综合报告
generalization_report <- data.frame(
  Assessment = c("Train-Test AUC Difference", "Train-External AUC Difference", "Test-External AUC Difference",
                 "Mean Performance Drop (Train-Test)", "Mean Performance Drop (Train-External)",
                 "Overall Generalization Assessment"),
  Value = c(round(auc_difference_test, 3), round(auc_difference_external, 3), round(abs(test_auc - external_auc), 3),
            round(mean_performance_drop_test, 3), round(mean_performance_drop_external, 3),
            if(mean_performance_drop_test < 0.05 && mean_performance_drop_external < 0.05) "Excellent" 
            else if(mean_performance_drop_test < 0.1 && mean_performance_drop_external < 0.1) "Good"
            else if(mean_performance_drop_test < 0.1) "Reasonable to test only"
            else if(mean_performance_drop_external < 0.1) "Reasonable to external only"
            else "Poor")
)

write.csv(generalization_report, "Comparative_Analysis/Generalization_Assessment_Report.csv", row.names = FALSE)

cat("\n=== Analysis Complete ===\n")
cat("Comprehensive comparison of Training (CV), Test, and External Validation sets completed successfully!\n")