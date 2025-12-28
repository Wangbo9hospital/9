# Part 2: 10-Fold Cross-Validation with Random Forest Imputation
# Load required packages
library(readxl)
library(caret)
library(e1071)
library(mice)
library(pROC)
library(openxlsx)
library(ggplot2)
library(randomForest)
library(reshape2)

# Load data and objects from Part 1
train_data <- read_excel("C:/Users/Bo Wang/Desktop/Training Cohort.xlsx")
train_data$Diagnosis <- as.factor(train_data$Diagnosis)
final_glm_model <- readRDS("final_glm_model.rds")
final_features <- readRDS("final_features.rds")
train_data_balanced <- readRDS("train_data_balanced.rds")

# 检查特征一致性
cat("检查特征一致性...\n")
cat("Final features from Part 1:", length(final_features), "\n")
cat("Features in balanced data:", length(names(train_data_balanced)), "\n")

# 找出缺失的特征
missing_features <- setdiff(final_features, names(train_data_balanced))
if(length(missing_features) > 0) {
  cat("警告: 以下特征在平衡数据中缺失:", paste(missing_features, collapse = ", "), "\n")
  # 使用可用的特征
  final_features <- intersect(final_features, names(train_data_balanced))
  cat("使用可用的特征:", length(final_features), "\n")
}

# 检查原始数据中的特征
missing_in_original <- setdiff(final_features, names(train_data))
if(length(missing_in_original) > 0) {
  cat("警告: 以下特征在原始数据中缺失:", paste(missing_in_original, collapse = ", "), "\n")
  final_features <- intersect(final_features, names(train_data))
  cat("最终使用的特征:", length(final_features), "\n")
}

# Create workbook for results
wb2 <- createWorkbook()
addWorksheet(wb2, "CV_Detailed_Results")
addWorksheet(wb2, "CV_Summary_Statistics")
addWorksheet(wb2, "CV_Performance_Metrics")
addWorksheet(wb2, "CV_Predictions")

# Define metrics function (same as Part 1)
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
    ROC_Obj = if(exists("roc_obj")) roc_obj else NULL
  ))
}

# ==================== 10-Fold Cross-Validation ====================
cat("=== Part 2: 10-Fold Cross-Validation with RF Imputation (GLM) ===\n")

set.seed(123)
folds <- createFolds(train_data$Diagnosis, k = 10, returnTrain = TRUE)

cv_results <- data.frame(
  Fold = 1:10,
  AUC = numeric(10),
  Sensitivity = numeric(10),
  Specificity = numeric(10),
  PPV = numeric(10),
  NPV = numeric(10),
  Accuracy = numeric(10),
  F1 = numeric(10),
  Balanced_Accuracy = numeric(10)
)

cv_rocs <- list()
misclassified_cases <- list()

# 初始化cv_predictions数据框
cv_predictions <- data.frame(
  Sample_ID = character(),
  Fold = numeric(),
  True_Label = numeric(),
  Predicted_Probability = numeric(),
  Predicted_Label = numeric(),
  stringsAsFactors = FALSE
)

for(i in 1:10) {
  cat("Processing fold", i, "...\n")
  
  # Split data
  fold_train_indices <- folds[[i]]
  fold_train_raw <- train_data[fold_train_indices, ]
  fold_test_raw <- train_data[-fold_train_indices, ]
  
  # 检查当前fold的特征可用性
  available_features <- intersect(final_features, names(fold_train_raw))
  if(length(available_features) < length(final_features)) {
    cat("  警告: Fold", i, "中缺少", length(final_features) - length(available_features), "个特征\n")
    cat("  使用", length(available_features), "个可用特征\n")
  }
  
  # Random Forest Imputation for current fold
  cat("  Performing RF imputation for fold", i, "...\n")
  missing_count_fold <- colSums(is.na(fold_train_raw))
  
  if(sum(missing_count_fold) > 0) {
    mice_args_fold <- list(
      data = fold_train_raw,
      m = 5,
      maxit = 10,
      printFlag = FALSE,
      seed = 123 + i,
      method = "rf"  # Random Forest imputation
    )
    
    imp_model_fold <- do.call(mice, mice_args_fold)
    fold_train_imputed <- complete(imp_model_fold, 1)
  } else {
    fold_train_imputed <- fold_train_raw
  }
  
  # Impute validation set using training set parameters
  if(any(is.na(fold_test_raw))) {
    for(feature in names(fold_test_raw)) {
      if(any(is.na(fold_test_raw[[feature]]))) {
        if(is.numeric(fold_test_raw[[feature]])) {
          mean_val <- mean(fold_train_imputed[[feature]], na.rm = TRUE)
          fold_test_raw[[feature]][is.na(fold_test_raw[[feature]])] <- mean_val
        } else {
          mode_val <- names(sort(table(fold_train_imputed[[feature]]), decreasing = TRUE))[1]
          fold_test_raw[[feature]][is.na(fold_test_raw[[feature]])] <- mode_val
        }
      }
    }
    fold_test_imputed <- fold_test_raw
  } else {
    fold_test_imputed <- fold_test_raw
  }
  
  # Handle class imbalance
  class_ratio_fold <- min(table(fold_train_imputed$Diagnosis)) / nrow(fold_train_imputed)
  if(class_ratio_fold < 0.3) {
    cat("  Applying SMOTE for fold", i, "...\n")
    if(!require("smotefamily")) install.packages("smotefamily"); library(smotefamily)
    
    # 确保SMOTE数据包含所有可用特征
    smote_features <- intersect(final_features, names(fold_train_imputed))
    smote_data_prep <- fold_train_imputed[, c(smote_features, "Diagnosis")]
    
    tryCatch({
      smote_data_fold <- SMOTE(
        smote_data_prep[, smote_features], 
        smote_data_prep$Diagnosis, 
        K = 5
      )
      fold_train_balanced <- smote_data_fold$data
      fold_train_balanced$class <- as.factor(fold_train_balanced$class)
      names(fold_train_balanced)[ncol(fold_train_balanced)] <- "Diagnosis"
      cat("  SMOTE applied successfully for fold", i, "\n")
    }, error = function(e) {
      cat("  SMOTE failed for fold", i, ":", e$message, "\n")
      cat("  Using original data without SMOTE\n")
      fold_train_balanced <- fold_train_imputed
    })
  } else {
    fold_train_balanced <- fold_train_imputed
  }
  
  # 再次检查平衡数据中的特征可用性
  available_features_balanced <- intersect(final_features, names(fold_train_balanced))
  if(length(available_features_balanced) < length(final_features)) {
    cat("  警告: 平衡数据中缺少", length(final_features) - length(available_features_balanced), "个特征\n")
    cat("  使用", length(available_features_balanced), "个可用特征\n")
  }
  
  # Prepare data for GLM
  fold_train_features <- fold_train_balanced[, available_features_balanced, drop = FALSE]
  fold_train_labels <- fold_train_balanced$Diagnosis
  fold_test_features <- fold_test_imputed[, available_features_balanced, drop = FALSE]
  fold_test_labels <- fold_test_imputed$Diagnosis
  
  # Train GLM Model
  cat("  Training GLM model for fold", i, "...\n")
  
  # 准备GLM公式
  glm_formula <- as.formula(paste("Diagnosis ~", paste(available_features_balanced, collapse = " + ")))
  
  tryCatch({
    fold_model <- glm(
      formula = glm_formula,
      data = fold_train_balanced,
      family = binomial(link = "logit")
    )
    cat("  GLM model trained successfully for fold", i, "\n")
  }, error = function(e) {
    cat("  Failed to train GLM model for fold", i, ":", e$message, "\n")
    # 如果训练失败，跳过这个fold
    cv_results[i, "AUC"] <- 0.5
    cv_results[i, "Sensitivity"] <- 0.5
    cv_results[i, "Specificity"] <- 0.5
    cv_results[i, "PPV"] <- 0.5
    cv_results[i, "NPV"] <- 0.5
    cv_results[i, "Accuracy"] <- 0.5
    cv_results[i, "F1"] <- 0.5
    cv_results[i, "Balanced_Accuracy"] <- 0.5
    next
  })
  
  # 确保模型已成功训练
  if(!exists("fold_model") || is.null(fold_model)) {
    cat("  GLM model is NULL for fold", i, ", skipping...\n")
    next
  }
  
  # Predict and evaluate
  cat("  Predicting with fold model...\n")
  tryCatch({
    fold_probs <- predict(fold_model, newdata = fold_test_imputed, type = "response")
    fold_metrics <- calculate_all_metrics(fold_test_imputed$Diagnosis, fold_probs)
    
    # Store results
    cv_results[i, "AUC"] <- fold_metrics$AUC
    cv_results[i, "Sensitivity"] <- fold_metrics$Sensitivity
    cv_results[i, "Specificity"] <- fold_metrics$Specificity
    cv_results[i, "PPV"] <- fold_metrics$PPV
    cv_results[i, "NPV"] <- fold_metrics$NPV
    cv_results[i, "Accuracy"] <- fold_metrics$Accuracy
    cv_results[i, "F1"] <- fold_metrics$F1
    cv_results[i, "Balanced_Accuracy"] <- fold_metrics$Balanced_Accuracy
    
    # Store ROC
    if(!is.null(fold_metrics$ROC_Obj)) {
      cv_rocs[[i]] <- fold_metrics$ROC_Obj
    }
    
    # 保存当前fold的预测结果
    fold_pred_labels <- ifelse(fold_probs > 0.5, 1, 0)
    fold_true_labels <- as.numeric(fold_test_imputed$Diagnosis) - 1
    
    # 创建当前fold的预测结果数据框
    fold_pred_df <- data.frame(
      Sample_ID = rownames(fold_test_imputed),
      Fold = i,
      True_Label = fold_true_labels,
      Predicted_Probability = fold_probs,
      Predicted_Label = fold_pred_labels,
      stringsAsFactors = FALSE
    )
    
    # 合并到总的cv_predictions
    cv_predictions <- rbind(cv_predictions, fold_pred_df)
    
    # Identify misclassified cases
    misclassified_indices <- which(fold_pred_labels != fold_true_labels)
    
    if(length(misclassified_indices) > 0) {
      misclassified_cases[[i]] <- data.frame(
        Fold = i,
        Case_ID = rownames(fold_test_imputed)[misclassified_indices],
        True_Label = fold_true_labels[misclassified_indices],
        Predicted_Label = fold_pred_labels[misclassified_indices],
        Probability = fold_probs[misclassified_indices]
      )
    }
  }, error = function(e) {
    cat("  Prediction failed for fold", i, ":", e$message, "\n")
  })
}

# 计算整体CV预测性能
cat("Calculating overall CV performance...\n")
if(nrow(cv_predictions) > 0) {
  overall_cv_metrics <- calculate_all_metrics(
    as.factor(cv_predictions$True_Label), 
    cv_predictions$Predicted_Probability
  )
} else {
  cat("Warning: No predictions available for overall CV performance calculation\n")
  overall_cv_metrics <- list(
    AUC = 0.5,
    Sensitivity = 0.5,
    Specificity = 0.5,
    PPV = 0.5,
    NPV = 0.5,
    Accuracy = 0.5,
    F1 = 0.5,
    Balanced_Accuracy = 0.5
  )
}

# 添加整体性能到cv_summary
cv_summary <- data.frame(
  Metric = c("AUC", "Sensitivity", "Specificity", "PPV", "NPV", "Accuracy", "F1", "Balanced_Accuracy"),
  Mean = c(
    mean(cv_results$AUC, na.rm = TRUE),
    mean(cv_results$Sensitivity, na.rm = TRUE),
    mean(cv_results$Specificity, na.rm = TRUE),
    mean(cv_results$PPV, na.rm = TRUE),
    mean(cv_results$NPV, na.rm = TRUE),
    mean(cv_results$Accuracy, na.rm = TRUE),
    mean(cv_results$F1, na.rm = TRUE),
    mean(cv_results$Balanced_Accuracy, na.rm = TRUE)
  ),
  SD = c(
    sd(cv_results$AUC, na.rm = TRUE),
    sd(cv_results$Sensitivity, na.rm = TRUE),
    sd(cv_results$Specificity, na.rm = TRUE),
    sd(cv_results$PPV, na.rm = TRUE),
    sd(cv_results$NPV, na.rm = TRUE),
    sd(cv_results$Accuracy, na.rm = TRUE),
    sd(cv_results$F1, na.rm = TRUE),
    sd(cv_results$Balanced_Accuracy, na.rm = TRUE)
  ),
  Median = c(
    median(cv_results$AUC, na.rm = TRUE),
    median(cv_results$Sensitivity, na.rm = TRUE),
    median(cv_results$Specificity, na.rm = TRUE),
    median(cv_results$PPV, na.rm = TRUE),
    median(cv_results$NPV, na.rm = TRUE),
    median(cv_results$Accuracy, na.rm = TRUE),
    median(cv_results$F1, na.rm = TRUE),
    median(cv_results$Balanced_Accuracy, na.rm = TRUE)
  ),
  IQR = c(
    IQR(cv_results$AUC, na.rm = TRUE),
    IQR(cv_results$Sensitivity, na.rm = TRUE),
    IQR(cv_results$Specificity, na.rm = TRUE),
    IQR(cv_results$PPV, na.rm = TRUE),
    IQR(cv_results$NPV, na.rm = TRUE),
    IQR(cv_results$Accuracy, na.rm = TRUE),
    IQR(cv_results$F1, na.rm = TRUE),
    IQR(cv_results$Balanced_Accuracy, na.rm = TRUE)
  ),
  Overall_CV = c(
    overall_cv_metrics$AUC,
    overall_cv_metrics$Sensitivity,
    overall_cv_metrics$Specificity,
    overall_cv_metrics$PPV,
    overall_cv_metrics$NPV,
    overall_cv_metrics$Accuracy,
    overall_cv_metrics$F1,
    overall_cv_metrics$Balanced_Accuracy
  )
)

# ==================== Generate Visualizations ====================
cat("Generating Visualizations...\n")
if(!dir.exists("Part2_Results")) dir.create("Part2_Results")

# 1. CV Performance Boxplot
cv_long <- reshape2::melt(cv_results, id.vars = "Fold", variable.name = "Metric", value.name = "Value")

p1 <- ggplot(cv_long, aes(x = Metric, y = Value, fill = Metric)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.6) +
  labs(title = "10-Fold Cross-Validation Performance Metrics - GLM",
       x = "Performance Metrics",
       y = "Value") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none") +
  ylim(0, 1)

ggsave("Part2_Results/CV_Performance_Boxplot.png", p1, width = 12, height = 8, dpi = 300)

# 2. ROC Curves for each fold (only if we have ROC objects)
if(length(cv_rocs) > 0 && any(!sapply(cv_rocs, is.null))) {
  png("Part2_Results/CV_ROC_Curves.png", width = 800, height = 600)
  par(mar = c(5, 5, 4, 2) + 0.1)
  plot(1, type = "n", xlim = c(0, 1), ylim = c(0, 1), 
       xlab = "1 - Specificity", ylab = "Sensitivity",
       main = "10-Fold Cross-Validation ROC Curves - GLM",
       cex.lab = 1.2, cex.axis = 1.1, cex.main = 1.3)
  
  colors <- rainbow(length(cv_rocs))
  for(i in 1:length(cv_rocs)) {
    if(!is.null(cv_rocs[[i]])) {
      roc_obj <- cv_rocs[[i]]
      lines(1 - roc_obj$specificities, roc_obj$sensitivities, 
            type = "l", col = colors[i], lwd = 2, lty = i)
    }
  }
  
  # Add mean ROC
  if(length(cv_rocs) > 0) {
    mean_sensitivity <- numeric(100)
    mean_specificity <- seq(0, 1, length.out = 100)
    
    for(i in 1:100) {
      spec <- 1 - mean_specificity[i]
      sens_values <- sapply(cv_rocs, function(roc_obj) {
        if(!is.null(roc_obj)) {
          idx <- which.min(abs(roc_obj$specificities - spec))
          if(length(idx) > 0) roc_obj$sensitivities[idx] else NA
        } else NA
      })
      mean_sensitivity[i] <- mean(sens_values, na.rm = TRUE)
    }
    
    lines(mean_specificity, mean_sensitivity, type = "l", col = "black", lwd = 4)
  }
  
  abline(0, 1, lty = 3, col = "gray")
  legend("bottomright", legend = c(paste("Fold", 1:10), "Mean ROC"), 
         col = c(colors, "black"), lty = c(1:10, 1), lwd = c(rep(2, 10), 4),
         cex = 0.8, bg = "white")
  dev.off()
}

# 3. Performance Distribution
p2 <- ggplot(cv_results, aes(x = AUC)) +
  geom_histogram(binwidth = 0.02, fill = "lightblue", color = "black", alpha = 0.7) +
  geom_vline(aes(xintercept = mean(AUC, na.rm = TRUE)), color = "red", linetype = "dashed", size = 1) +
  labs(title = "Distribution of AUC across 10-Fold CV - GLM",
       x = "AUC",
       y = "Frequency") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

ggsave("Part2_Results/AUC_Distribution.png", p2, width = 8, height = 6, dpi = 300)

# 4. Misclassification Analysis (only if we have misclassified cases)
if(length(misclassified_cases) > 0 && nrow(do.call(rbind, misclassified_cases)) > 0) {
  all_misclassified <- do.call(rbind, misclassified_cases)
  
  p3 <- ggplot(all_misclassified, aes(x = as.factor(True_Label), fill = as.factor(Predicted_Label))) +
    geom_bar(position = "dodge", alpha = 0.8) +
    labs(title = "Misclassification Analysis - GLM",
         x = "True Label",
         y = "Count",
         fill = "Predicted Label") +
    scale_fill_manual(values = c("0" = "red", "1" = "blue")) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
  
  ggsave("Part2_Results/Misclassification_Analysis.png", p3, width = 8, height = 6, dpi = 300)
}

# 5. CV预测概率分布 (only if we have predictions)
if(nrow(cv_predictions) > 0) {
  p4 <- ggplot(cv_predictions, aes(x = Predicted_Probability, fill = as.factor(True_Label))) +
    geom_histogram(alpha = 0.6, position = "identity", bins = 30) +
    labs(title = "Distribution of CV Predicted Probabilities by True Label - GLM",
         x = "Predicted Probability",
         y = "Count",
         fill = "True Label") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
  
  ggsave("Part2_Results/CV_Prediction_Distribution.png", p4, width = 10, height = 6, dpi = 300)
  
  # 6. 各Fold预测性能比较
  p5 <- ggplot(cv_predictions, aes(x = as.factor(Fold), y = Predicted_Probability, fill = as.factor(True_Label))) +
    geom_boxplot(alpha = 0.7) +
    labs(title = "Predicted Probabilities Distribution by Fold - GLM",
         x = "Fold",
         y = "Predicted Probability",
         fill = "True Label") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
  
  ggsave("Part2_Results/CV_Predictions_by_Fold.png", p5, width = 12, height = 8, dpi = 300)
}

# ==================== Save Results ====================
writeData(wb2, "CV_Detailed_Results", cv_results, startRow = 1)
writeData(wb2, "CV_Summary_Statistics", cv_summary, startRow = 1)

if(nrow(cv_predictions) > 0) {
  writeData(wb2, "CV_Predictions", cv_predictions, startRow = 1)
}

# Save misclassified cases if they exist
if(length(misclassified_cases) > 0 && nrow(do.call(rbind, misclassified_cases)) > 0) {
  addWorksheet(wb2, "Misclassified_Cases")
  writeData(wb2, "Misclassified_Cases", do.call(rbind, misclassified_cases), startRow = 1)
}

saveWorkbook(wb2, "Part2_Cross_Validation_Results.xlsx", overwrite = TRUE)

# Save CV objects for future use
saveRDS(cv_results, "cv_results.rds")
saveRDS(cv_summary, "cv_summary.rds")
saveRDS(cv_predictions, "cv_predictions.rds")

cat("\n=== Part 2 Complete ===\n")
cat("Results saved to:\n")
cat("- Part2_Cross_Validation_Results.xlsx\n")
cat("- Part2_Results/ directory\n")
cat("- cv_results.rds\n")
cat("- cv_summary.rds\n")
cat("- cv_predictions.rds\n")
cat("\nCross-Validation Performance Summary:\n")
print(cv_summary)
cat("\nOverall CV Performance (from cv_predictions):\n")
cat("AUC:", round(overall_cv_metrics$AUC, 4), "\n")
cat("Accuracy:", round(overall_cv_metrics$Accuracy, 4), "\n")
cat("Sensitivity:", round(overall_cv_metrics$Sensitivity, 4), "\n")
cat("Specificity:", round(overall_cv_metrics$Specificity, 4), "\n")
cat("\nModel Information:\n")
cat("Model type: GLM (Logistic Regression)\n")
cat("Number of folds: 10\n")
cat("Total training samples (CV):", nrow(cv_predictions), "\n")