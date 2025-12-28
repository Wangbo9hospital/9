# Part 2: 10-Fold Cross-Validation with Random Forest Imputation
# Load required packages
library(readxl)
library(caret)
library(e1071)  # 用于NaiveBayes模型
library(mice)
library(pROC)
library(openxlsx)
library(ggplot2)
library(randomForest)

# 修正的NaiveBayes预测函数
predict_naive_bayes <- function(model, newdata) {
  # 首先尝试从模型对象中获取类型信息
  if(!is.null(model$model_type)) {
    model_type <- model$model_type
  } else {
    # 如果模型没有保存类型信息，通过类名判断
    model_class <- class(model)[1]
    if(model_class == "naiveBayes") {
      model_type <- "e1071"
    } else if(model_class == "NaiveBayes") {
      model_type <- "klaR"
    } else {
      model_type <- model_class
    }
  }
  
  cat("  Model type detected:", model_type, "\n")
  
  if(model_type == "e1071") {
    # e1071包的预测
    pred_probs_raw <- predict(model, newdata = newdata, type = "raw")
    # 确保概率矩阵有两列
    if(ncol(pred_probs_raw) == 2) {
      # 取第二列作为正类的概率
      return(pred_probs_raw[, 2])
    } else {
      # 如果只有一列，假设是正类的概率
      return(pred_probs_raw[, 1])
    }
  } else if(model_type == "klaR") {
    # klaR包的预测
    pred <- predict(model, newdata = newdata)
    return(pred$posterior[, 2])  # 取第二列作为正类的概率
  } else {
    stop("Unknown NaiveBayes model type: ", model_type)
  }
}

# Load data and objects from Part 1
train_data <- read_excel("C:/Users/Bo Wang/Desktop/Training Cohort.xlsx")
train_data$Diagnosis <- as.factor(train_data$Diagnosis)
final_nb_model <- readRDS("final_nb_model.rds")
final_features <- readRDS("final_features.rds")
train_data_balanced <- readRDS("train_data_balanced.rds")

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
cat("=== Part 2: 10-Fold Cross-Validation with RF Imputation (NaiveBayes) ===\n")

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
    
    smote_data_fold <- SMOTE(
      fold_train_imputed[, c(final_features, "Diagnosis")], 
      fold_train_imputed$Diagnosis, 
      K = 5
    )
    fold_train_balanced <- smote_data_fold$data
    fold_train_balanced$class <- as.factor(fold_train_balanced$class)
    names(fold_train_balanced)[ncol(fold_train_balanced)] <- "Diagnosis"
  } else {
    fold_train_balanced <- fold_train_imputed
  }
  
  # Prepare data for NaiveBayes
  fold_train_features <- fold_train_balanced[, final_features, drop = FALSE]
  fold_train_labels <- fold_train_balanced$Diagnosis
  fold_test_features <- fold_test_imputed[, final_features, drop = FALSE]
  fold_test_labels <- fold_test_imputed$Diagnosis
  
  # Train NaiveBayes Model
  cat("  Training NaiveBayes model for fold", i, "...\n")
  
  fold_model <- NULL
  model_type <- NULL
  
  tryCatch({
    # 方法1: 使用e1071包的naiveBayes函数
    fold_model <- naiveBayes(
      x = fold_train_features,
      y = fold_train_labels
    )
    model_type <- "e1071"
    fold_model$model_type <- "e1071"  # 明确设置模型类型
    cat("  NaiveBayes model trained successfully using e1071 package.\n")
  }, error = function(e) {
    cat("  Error with e1071 naiveBayes:", e$message, "\n")
    cat("  Trying klaR package...\n")
    
    # 方法2: 如果e1071失败，尝试klaR包
    tryCatch({
      fold_model <- NaiveBayes(
        x = fold_train_features,
        grouping = fold_train_labels,
        usekernel = TRUE  # 使用核密度估计处理连续变量
      )
      model_type <- "klaR"
      fold_model$model_type <- "klaR"  # 明确设置模型类型
      cat("  NaiveBayes model trained successfully using klaR package.\n")
    }, error = function(e2) {
      stop("Both e1071 and klaR packages failed to train NaiveBayes model: ", e2$message)
    })
  })
  
  # 确保模型已成功训练
  if(is.null(fold_model)) {
    stop("Failed to train model for fold ", i)
  }
  
  # Predict and evaluate using unified prediction function
  cat("  Predicting with fold model...\n")
  fold_probs <- predict_naive_bayes(fold_model, fold_test_features)
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
}

# 计算整体CV预测性能
cat("Calculating overall CV performance...\n")
overall_cv_metrics <- calculate_all_metrics(
  as.factor(cv_predictions$True_Label), 
  cv_predictions$Predicted_Probability
)

# 添加整体性能到cv_summary
cv_summary <- data.frame(
  Metric = c("AUC", "Sensitivity", "Specificity", "PPV", "NPV", "Accuracy", "F1", "Balanced_Accuracy"),
  Mean = c(
    mean(cv_results$AUC),
    mean(cv_results$Sensitivity),
    mean(cv_results$Specificity),
    mean(cv_results$PPV),
    mean(cv_results$NPV),
    mean(cv_results$Accuracy),
    mean(cv_results$F1),
    mean(cv_results$Balanced_Accuracy)
  ),
  SD = c(
    sd(cv_results$AUC),
    sd(cv_results$Sensitivity),
    sd(cv_results$Specificity),
    sd(cv_results$PPV),
    sd(cv_results$NPV),
    sd(cv_results$Accuracy),
    sd(cv_results$F1),
    sd(cv_results$Balanced_Accuracy)
  ),
  Median = c(
    median(cv_results$AUC),
    median(cv_results$Sensitivity),
    median(cv_results$Specificity),
    median(cv_results$PPV),
    median(cv_results$NPV),
    median(cv_results$Accuracy),
    median(cv_results$F1),
    median(cv_results$Balanced_Accuracy)
  ),
  IQR = c(
    IQR(cv_results$AUC),
    IQR(cv_results$Sensitivity),
    IQR(cv_results$Specificity),
    IQR(cv_results$PPV),
    IQR(cv_results$NPV),
    IQR(cv_results$Accuracy),
    IQR(cv_results$F1),
    IQR(cv_results$Balanced_Accuracy)
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
  labs(title = "10-Fold Cross-Validation Performance Metrics - NaiveBayes",
       x = "Performance Metrics",
       y = "Value") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none") +
  ylim(0, 1)

ggsave("Part2_Results/CV_Performance_Boxplot.png", p1, width = 12, height = 8, dpi = 300)
# ==================== 保存结果 ====================
# 保存RDS文件
saveRDS(cv_results, "cv_results.rds")
saveRDS(cv_summary, "cv_summary.rds")  # 添加这行代码
saveRDS(cv_predictions, "cv_predictions.rds")
saveRDS(cv_rocs, "cv_rocs.rds")  # 可选：保存ROC曲线对象

# 保存工作簿
saveWorkbook(wb2, "Part2_Cross_Validation_Results.xlsx", overwrite = TRUE)

# ==================== 在控制台输出带有±1SD的汇总信息 ====================
cat("\n=== Part 2 Complete ===\n")
cat("Results saved to:\n")
cat("- Part2_Cross_Validation_Results.xlsx\n")
cat("- Part2_Results/ directory\n")
cat("- cv_results.rds\n")
cat("- cv_summary.rds\n")
cat("- cv_predictions.rds\n")
# 2. ROC Curves for each fold with ±1SD shaded area
png("Part2_Results/CV_ROC_Curves.png", width = 800, height = 600)
par(mar = c(5, 5, 4, 2) + 0.1)
plot(1, type = "n", xlim = c(0, 1), ylim = c(0, 1), 
     xlab = "1 - Specificity", ylab = "Sensitivity",
     main = "10-Fold Cross-Validation ROC Curves - NaiveBayes",
     cex.lab = 1.2, cex.axis = 1.1, cex.main = 1.3)

# 创建一个序列点来计算均值和标准差
n_points <- 100
fpr_seq <- seq(0, 1, length.out = n_points)

# 收集每个fold在相同FPR点上的灵敏度
sens_matrix <- matrix(NA, nrow = n_points, ncol = length(cv_rocs))

for(i in 1:length(cv_rocs)) {
  if(!is.null(cv_rocs[[i]])) {
    roc_obj <- cv_rocs[[i]]
    # 获取当前fold的ROC曲线数据
    fpr <- 1 - roc_obj$specificities
    sens <- roc_obj$sensitivities
    
    # 确保数据是单调的（ROC曲线应该是单调的）
    # 对FPR进行排序
    ord <- order(fpr)
    fpr <- fpr[ord]
    sens <- sens[ord]
    
    # 使用插值获取在标准FPR序列上的灵敏度
    sens_interp <- approx(fpr, sens, xout = fpr_seq, method = "linear", 
                          rule = 2, ties = mean)$y
    
    # 存储到矩阵中
    sens_matrix[, i] <- sens_interp
  }
}

# 计算每个FPR点上的均值和标准差
sens_mean <- rowMeans(sens_matrix, na.rm = TRUE)
sens_sd <- apply(sens_matrix, 1, sd, na.rm = TRUE)

# 计算±1SD的范围
sens_upper <- sens_mean + sens_sd
sens_lower <- sens_mean - sens_sd

# 确保范围在0-1之间
sens_upper <- pmin(pmax(sens_upper, 0), 1)
sens_lower <- pmin(pmax(sens_lower, 0), 1)

# 绘制±1SD的浅灰色背景区域
# 首先绘制上边界到下边界的区域
polygon_x <- c(fpr_seq, rev(fpr_seq))
polygon_y <- c(sens_upper, rev(sens_lower))
polygon(polygon_x, polygon_y, col = rgb(0.9, 0.9, 0.9, 0.5), border = NA)

# 绘制平均ROC曲线（黑色，粗线）
lines(fpr_seq, sens_mean, type = "l", col = "black", lwd = 4, lty = 1)

# 绘制各个fold的ROC曲线
colors <- rainbow(length(cv_rocs))
for(i in 1:length(cv_rocs)) {
  if(!is.null(cv_rocs[[i]])) {
    roc_obj <- cv_rocs[[i]]
    lines(1 - roc_obj$specificities, roc_obj$sensitivities, 
          type = "l", col = colors[i], lwd = 1.5, lty = i)
  }
}

# 重新绘制平均ROC曲线，确保它在最上面
lines(fpr_seq, sens_mean, type = "l", col = "black", lwd = 4)

abline(0, 1, lty = 3, col = "gray", lwd = 2)

# 添加图例
legend("bottomright", 
       legend = c(paste("Fold", 1:10), "Mean ROC", "Mean ± 1SD"), 
       col = c(colors, "black", "gray"), 
       lty = c(1:10, 1, 1), 
       lwd = c(rep(1.5, 10), 4, 8),
       cex = 0.8, 
       bg = "white",
       fill = c(rep(NA, 11), rgb(0.9, 0.9, 0.9, 0.5)))

# 添加AUC均值和SD信息到图中
auc_mean <- mean(cv_results$AUC, na.rm = TRUE)
auc_sd <- sd(cv_results$AUC, na.rm = TRUE)

text(0.6, 0.2, 
     sprintf("Mean AUC = %.3f ± %.3f", auc_mean, auc_sd),
     cex = 0.9, col = "black", font = 2)

text(0.6, 0.12, 
     sprintf("AUC range: %.3f - %.3f", 
             auc_mean - auc_sd, 
             auc_mean + auc_sd),
     cex = 0.8, col = "black")

dev.off()

# 3. Performance Distribution
p2 <- ggplot(cv_results, aes(x = AUC)) +
  geom_histogram(binwidth = 0.02, fill = "lightblue", color = "black", alpha = 0.7) +
  geom_vline(aes(xintercept = mean(AUC)), color = "red", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = mean(AUC) - sd(AUC)), color = "orange", linetype = "dashed", size = 0.8) +
  geom_vline(aes(xintercept = mean(AUC) + sd(AUC)), color = "orange", linetype = "dashed", size = 0.8) +
  geom_rect(aes(xmin = mean(AUC) - sd(AUC),
                xmax = mean(AUC) + sd(AUC),
                ymin = 0, ymax = Inf), 
            fill = rgb(0.9, 0.9, 0.9, 0.3), color = NA) +
  labs(title = "Distribution of AUC across 10-Fold CV - NaiveBayes",
       subtitle = sprintf("Mean AUC = %.3f ± %.3f", mean(cv_results$AUC), sd(cv_results$AUC)),
       x = "AUC",
       y = "Frequency") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 10))

ggsave("Part2_Results/AUC_Distribution.png", p2, width = 8, height = 6, dpi = 300)
# 4. Misclassification Analysis
if(length(misclassified_cases) > 0) {
  all_misclassified <- do.call(rbind, misclassified_cases)
  
  p3 <- ggplot(all_misclassified, aes(x = as.factor(True_Label), fill = as.factor(Predicted_Label))) +
    geom_bar(position = "dodge", alpha = 0.8) +
    labs(title = "Misclassification Analysis - NaiveBayes",
         x = "True Label",
         y = "Count",
         fill = "Predicted Label") +
    scale_fill_manual(values = c("0" = "red", "1" = "blue")) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
  
  ggsave("Part2_Results/Misclassification_Analysis.png", p3, width = 8, height = 6, dpi = 300)
}

# 5. CV预测概率分布
p4 <- ggplot(cv_predictions, aes(x = Predicted_Probability, fill = as.factor(True_Label))) +
  geom_histogram(alpha = 0.6, position = "identity", bins = 30) +
  labs(title = "Distribution of CV Predicted Probabilities by True Label - NaiveBayes",
       x = "Predicted Probability",
       y = "Count",
       fill = "True Label") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

ggsave("Part2_Results/CV_Prediction_Distribution.png", p4, width = 10, height = 6, dpi = 300)

# 6. 各Fold预测性能比较
p5 <- ggplot(cv_predictions, aes(x = as.factor(Fold), y = Predicted_Probability, fill = as.factor(True_Label))) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Predicted Probabilities Distribution by Fold - NaiveBayes",
       x = "Fold",
       y = "Predicted Probability",
       fill = "True Label") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

ggsave("Part2_Results/CV_Predictions_by_Fold.png", p5, width = 12, height = 8, dpi = 300)

# 在保存结果之前，修改cv_summary数据框，添加Mean ± 1SD列

# 计算Mean ± 1SD
cv_summary$Mean_minus_1SD <- cv_summary$Mean - cv_summary$SD
cv_summary$Mean_plus_1SD <- cv_summary$Mean + cv_summary$SD

# 重新排列列顺序，使结果更清晰
cv_summary <- cv_summary[, c("Metric", "Mean", "SD", "Mean_minus_1SD", "Mean_plus_1SD", 
                             "Median", "IQR", "Overall_CV")]

# 创建一个更详细的性能汇总表，包含±1SD
detailed_summary <- data.frame(
  Metric = cv_summary$Metric,
  Mean_SD = sprintf("%.3f ± %.3f", cv_summary$Mean, cv_summary$SD),
  Range_1SD = sprintf("[%.3f, %.3f]", cv_summary$Mean_minus_1SD, cv_summary$Mean_plus_1SD),
  Median_IQR = sprintf("%.3f [%.3f]", cv_summary$Median, cv_summary$IQR),
  Overall_CV = sprintf("%.3f", cv_summary$Overall_CV),
  stringsAsFactors = FALSE
)

# 将详细汇总表添加到工作簿
addWorksheet(wb2, "CV_Summary_with_1SD")
writeData(wb2, "CV_Summary_with_1SD", detailed_summary, startRow = 1)

# 修改原来的CV_Summary_Statistics工作表
writeData(wb2, "CV_Summary_Statistics", cv_summary, startRow = 1)
# 7. 新增：ROC曲线性能汇总图（包含±1SD区间）
# 创建一个新的ROC曲线图，更清晰地展示均值和±1SD
png("Part2_Results/CV_ROC_Curves_Detailed.png", width = 800, height = 600)
par(mar = c(5, 5, 4, 2) + 0.1)

# 设置绘图区域
plot(1, type = "n", xlim = c(0, 1), ylim = c(0, 1), 
     xlab = "1 - Specificity (False Positive Rate)", 
     ylab = "Sensitivity (True Positive Rate)",
     main = "10-Fold CV ROC Curves with Mean ± 1SD - NaiveBayes",
     cex.lab = 1.3, cex.axis = 1.1, cex.main = 1.4)

# 添加网格线
grid(col = "lightgray", lty = "dotted", lwd = 0.7)

# 绘制±1SD的浅灰色背景区域
polygon(polygon_x, polygon_y, col = rgb(0.85, 0.85, 0.85, 0.4), border = NA)

# 绘制各个fold的ROC曲线（用细线，浅色）
for(i in 1:length(cv_rocs)) {
  if(!is.null(cv_rocs[[i]])) {
    roc_obj <- cv_rocs[[i]]
    lines(1 - roc_obj$specificities, roc_obj$sensitivities, 
          type = "l", col = rgb(0.6, 0.6, 0.6, 0.4), lwd = 1, lty = 1)
  }
}

# 绘制平均ROC曲线（红色，粗线）
lines(fpr_seq, sens_mean, type = "l", col = "red", lwd = 3, lty = 1)

# 绘制±1SD边界线（蓝色，虚线）
lines(fpr_seq, sens_upper, type = "l", col = "blue", lwd = 1.5, lty = 2)
lines(fpr_seq, sens_lower, type = "l", col = "blue", lwd = 1.5, lty = 2)

# 对角线参考线
abline(0, 1, lty = 3, col = "darkgray", lwd = 2)

# 计算并显示AUC统计信息
auc_values <- cv_results$AUC
auc_summary_text <- c(
  sprintf("Mean AUC = %.3f", mean(auc_values)),
  sprintf("SD = %.3f", sd(auc_values)),
  sprintf("Range: %.3f - %.3f", min(auc_values), max(auc_values)),
  sprintf("95%% CI: %.3f - %.3f", 
          mean(auc_values) - 1.96*sd(auc_values)/sqrt(length(auc_values)),
          mean(auc_values) + 1.96*sd(auc_values)/sqrt(length(auc_values)))
)

# 在图中添加统计信息框
legend("bottomright", 
       legend = c("Mean ROC", "Mean ± 1SD", "Individual Folds", "Random Classifier"),
       col = c("red", "blue", rgb(0.6, 0.6, 0.6, 0.4), "darkgray"),
       lty = c(1, 2, 1, 3),
       lwd = c(3, 1.5, 1, 2),
       cex = 0.9,
       bg = "white",
       box.col = "white")

# 在左上角添加AUC统计信息
legend("topleft",
       legend = auc_summary_text,
       bty = "n",
       cex = 0.9,
       text.col = "darkblue",
       bg = rgb(1, 1, 1, 0.7))

# 添加图标题说明
mtext(sprintf("10-Fold Cross-Validation (n=%d samples)", nrow(cv_predictions)), 
      side = 3, line = 0.2, cex = 0.9, col = "darkgreen")

dev.off()

# 8. 新增：ROC曲线的箱线图，展示AUC的分布
p_auc_box <- ggplot(cv_results, aes(x = "", y = AUC)) +
  geom_boxplot(fill = "lightblue", alpha = 0.7, width = 0.5) +
  geom_jitter(width = 0.1, alpha = 0.6, size = 2) +
  geom_point(aes(y = mean(AUC)), color = "red", size = 4, shape = 18) +
  geom_errorbar(aes(ymin = mean(AUC) - sd(AUC), ymax = mean(AUC) + sd(AUC)), 
                width = 0.3, color = "blue", size = 1) +
  labs(title = "AUC Distribution across 10-Fold CV",
       subtitle = "Red diamond: Mean AUC; Blue bar: Mean ± 1SD",
       x = "",
       y = "AUC") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 10),
        axis.text.x = element_blank()) +
  ylim(0, 1)

ggsave("Part2_Results/AUC_Boxplot_with_SD.png", p_auc_box, width = 6, height = 8, dpi = 300)

# ==================== 生成包含±1SD的可视化 ====================
cat("Generating Visualizations with ±1SD...\n")

# 1. 带±1SD误差线的性能指标图
p_sd <- ggplot(cv_summary, aes(x = Metric, y = Mean)) +
  geom_point(size = 3, color = "blue") +
  geom_errorbar(aes(ymin = Mean_minus_1SD, ymax = Mean_plus_1SD), 
                width = 0.2, color = "blue", size = 0.8) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray", alpha = 0.7) +
  labs(title = "10-Fold CV Performance Metrics with ±1SD - NaiveBayes",
       subtitle = "Error bars represent Mean ± 1 Standard Deviation",
       x = "Performance Metrics",
       y = "Value") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 10),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  ylim(0, 1) +
  scale_x_discrete(labels = function(x) gsub("_", " ", x))

ggsave("Part2_Results/CV_Performance_with_1SD.png", p_sd, width = 12, height = 8, dpi = 300)

# 2. 各Fold性能与总体均值的比较图（带±1SD区间）
fold_performance <- data.frame(
  Fold = rep(1:10, 8),
  Metric = rep(cv_summary$Metric, each = 10),
  Value = c(cv_results$AUC, cv_results$Sensitivity, cv_results$Specificity,
            cv_results$PPV, cv_results$NPV, cv_results$Accuracy,
            cv_results$F1, cv_results$Balanced_Accuracy)
)

# 添加总体均值和±1SD信息
fold_performance$Overall_Mean <- rep(cv_summary$Mean, each = 10)
fold_performance$Lower_1SD <- rep(cv_summary$Mean_minus_1SD, each = 10)
fold_performance$Upper_1SD <- rep(cv_summary$Mean_plus_1SD, each = 10)

p_folds_sd <- ggplot(fold_performance, aes(x = as.factor(Fold), y = Value)) +
  geom_point(aes(color = Metric), size = 2, alpha = 0.7) +
  geom_hline(aes(yintercept = Overall_Mean, color = Metric), 
             linetype = "dashed", alpha = 0.5) +
  geom_rect(aes(ymin = Lower_1SD, ymax = Upper_1SD), 
            xmin = -Inf, xmax = Inf, alpha = 0.1, fill = "gray") +
  facet_wrap(~ Metric, scales = "free_y", ncol = 4) +
  labs(title = "Fold-wise Performance vs Overall Mean ±1SD",
       x = "Fold Number",
       y = "Value") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid.minor = element_blank())

ggsave("Part2_Results/Fold_Performance_vs_1SD.png", p_folds_sd, width = 14, height = 10, dpi = 300)

# 3. 性能指标分布与±1SD区间（直方图）
# 选择AUC作为示例
p_auc_sd <- ggplot(cv_results, aes(x = AUC)) +
  geom_histogram(binwidth = 0.02, fill = "lightblue", color = "black", alpha = 0.7) +
  geom_vline(xintercept = mean(cv_results$AUC), color = "red", 
             linetype = "solid", size = 1) +
  geom_vline(xintercept = mean(cv_results$AUC) - sd(cv_results$AUC), 
             color = "orange", linetype = "dashed", size = 0.8) +
  geom_vline(xintercept = mean(cv_results$AUC) + sd(cv_results$AUC), 
             color = "orange", linetype = "dashed", size = 0.8) +
  geom_rect(aes(xmin = mean(cv_results$AUC) - sd(cv_results$AUC),
                xmax = mean(cv_results$AUC) + sd(cv_results$AUC),
                ymin = 0, ymax = Inf), 
            fill = "orange", alpha = 0.1) +
  annotate("text", x = mean(cv_results$AUC), y = Inf, 
           label = sprintf("Mean = %.3f", mean(cv_results$AUC)),
           vjust = 2, hjust = 0.5, color = "red", size = 4) +
  annotate("text", x = mean(cv_results$AUC) - sd(cv_results$AUC), y = Inf,
           label = sprintf("-1SD = %.3f", mean(cv_results$AUC) - sd(cv_results$AUC)),
           vjust = 4, hjust = 0.5, color = "orange", size = 3.5) +
  annotate("text", x = mean(cv_results$AUC) + sd(cv_results$AUC), y = Inf,
           label = sprintf("+1SD = %.3f", mean(cv_results$AUC) + sd(cv_results$AUC)),
           vjust = 4, hjust = 0.5, color = "orange", size = 3.5) +
  labs(title = "AUC Distribution with Mean ±1SD across 10-Fold CV",
       subtitle = "Orange shaded area represents Mean ± 1 Standard Deviation",
       x = "AUC",
       y = "Frequency") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 10))

ggsave("Part2_Results/AUC_Distribution_with_1SD.png", p_auc_sd, width = 10, height = 8, dpi = 300)

# 4. 创建所有指标的±1SD区间表（用于报告）
metrics_table <- data.frame(
  Metric = cv_summary$Metric,
  `Mean ± SD` = sprintf("%.3f ± %.3f", cv_summary$Mean, cv_summary$SD),
  `Range (±1SD)` = sprintf("%.3f - %.3f", 
                           cv_summary$Mean_minus_1SD, 
                           cv_summary$Mean_plus_1SD),
  `Median (IQR)` = sprintf("%.3f (%.3f)", cv_summary$Median, cv_summary$IQR),
  `Overall CV` = sprintf("%.3f", cv_summary$Overall_CV),
  check.names = FALSE,
  stringsAsFactors = FALSE
)

addWorksheet(wb2, "Performance_Table_Formatted")
writeData(wb2, "Performance_Table_Formatted", metrics_table, startRow = 1)

# ==================== 在控制台输出带有±1SD的汇总信息 ====================
cat("\n=== Part 2 Complete ===\n")
cat("Results saved to:\n")
cat("- Part2_Cross_Validation_Results.xlsx\n")
cat("- Part2_Results/ directory\n")
cat("- cv_results.rds\n")
cat("- cv_summary.rds\n")
cat("- cv_predictions.rds\n")

cat("\n=== Cross-Validation Performance Summary (Mean ± SD) ===\n")
for(i in 1:nrow(cv_summary)) {
  cat(sprintf("%-20s: %.3f ± %.3f [%.3f - %.3f] (Median: %.3f, Overall CV: %.3f)\n",
              cv_summary$Metric[i],
              cv_summary$Mean[i],
              cv_summary$SD[i],
              cv_summary$Mean_minus_1SD[i],
              cv_summary$Mean_plus_1SD[i],
              cv_summary$Median[i],
              cv_summary$Overall_CV[i]))
}

cat("\n=== Formatted Performance Table ===\n")
print(metrics_table, row.names = FALSE)

cat("\n=== Key Statistics ===\n")
cat(sprintf("AUC (Mean ± SD): %.3f ± %.3f\n", 
            cv_summary$Mean[cv_summary$Metric == "AUC"],
            cv_summary$SD[cv_summary$Metric == "AUC"]))
cat(sprintf("Accuracy (Mean ± SD): %.3f ± %.3f\n",
            cv_summary$Mean[cv_summary$Metric == "Accuracy"],
            cv_summary$SD[cv_summary$Metric == "Accuracy"]))
cat(sprintf("Sensitivity (Mean ± SD): %.3f ± %.3f\n",
            cv_summary$Mean[cv_summary$Metric == "Sensitivity"],
            cv_summary$SD[cv_summary$Metric == "Sensitivity"]))
cat(sprintf("Specificity (Mean ± SD): %.3f ± %.3f\n",
            cv_summary$Mean[cv_summary$Metric == "Specificity"],
            cv_summary$SD[cv_summary$Metric == "Specificity"]))

cat("\n=== Model Information ===\n")
cat("Model type: NaiveBayes\n")
cat("Number of folds: 10\n")
cat("Total training samples (CV):", nrow(cv_predictions), "\n")
cat("Features used:", length(final_features), "\n")
cat("Feature names:", paste(final_features, collapse = ", "), "\n")