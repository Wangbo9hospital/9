# Part 4: Model Interpretation for NaiveBayes Model (with External Validation)
# Load required packages
library(readxl)
library(e1071)
library(ggplot2)
library(openxlsx)
library(viridis)
library(ggforce)
library(dplyr)
library(pROC)
library(caret)

# Load objects from previous parts
final_nb_model <- readRDS("final_nb_model.rds")
final_features <- readRDS("final_features.rds")
train_data_balanced <- readRDS("train_data_balanced.rds")
test_data <- read_excel("C:/Users/Bo Wang/Desktop/Internal Validation Cohort.xlsx")
test_data$Diagnosis <- as.factor(test_data$Diagnosis)

# 加载外部数据集
external_data <- read_excel("C:/Users/Bo Wang/Desktop/External Validation Cohort.xlsx")  # 请修改为实际路径
external_data$Diagnosis <- as.factor(external_data$Diagnosis)

# 统一的NaiveBayes预测函数
predict_naive_bayes <- function(model, newdata) {
  if(is.null(model$model_type)) {
    # 如果模型没有保存类型信息，尝试自动判断
    model_class <- class(model)[1]
    if(model_class == "naiveBayes") {
      model_type <- "e1071"
    } else if(model_class == "NaiveBayes") {
      model_type <- "klaR"
    } else {
      model_type <- model_class
    }
  } else {
    model_type <- model$model_type
  }
  
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

# 安全的文件名生成函数
safe_filename <- function(feature_name) {
  # 替换所有可能引起问题的字符
  safe_name <- gsub("[^a-zA-Z0-9_]", "_", feature_name)
  # 限制长度
  if(nchar(safe_name) > 50) {
    safe_name <- substr(safe_name, 1, 50)
  }
  return(safe_name)
}

# Create workbook for results
wb4 <- createWorkbook()
addWorksheet(wb4, "Feature_Importance_Test")
addWorksheet(wb4, "Feature_Importance_External")
addWorksheet(wb4, "Feature_Importance_Comparison")
addWorksheet(wb4, "Partial_Dependence_Test")
addWorksheet(wb4, "Partial_Dependence_External")
addWorksheet(wb4, "Individual_Predictions_Test")
addWorksheet(wb4, "Individual_Predictions_External")
addWorksheet(wb4, "Model_Coefficients")
addWorksheet(wb4, "Interpretation_Summary")

# ==================== Model Interpretation Analysis ====================
cat("=== Part 4: NaiveBayes Model Interpretation (Test + External) ===\n")

# Prepare data
train_features <- train_data_balanced[, final_features, drop = FALSE]
train_labels <- train_data_balanced$Diagnosis

# Prepare test data (using same imputation as Part 3)
test_features_all <- test_data[, !names(test_data) %in% "Diagnosis"]
# Simple imputation for test set
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

# Ensure all features are present
missing_features <- setdiff(final_features, names(test_features_all))
if(length(missing_features) > 0) {
  for(feature in missing_features) {
    if(is.numeric(train_data_balanced[[feature]])) {
      test_features_all[[feature]] <- mean(train_data_balanced[[feature]], na.rm = TRUE)
    } else {
      mode_val <- names(sort(table(train_data_balanced[[feature]]), decreasing = TRUE))[1]
      test_features_all[[feature]] <- mode_val
    }
  }
}

test_features <- test_features_all[, final_features, drop = FALSE]
test_labels <- test_data$Diagnosis

# Prepare external data (using training set parameters)
cat("Preparing external dataset for interpretation...\n")
external_features_all <- external_data[, !names(external_data) %in% "Diagnosis"]
external_labels_all <- external_data$Diagnosis

# Process external dataset using training set parameters
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

# Ensure all features are present in external dataset
missing_features_ext <- setdiff(final_features, names(external_features_all))
if(length(missing_features_ext) > 0) {
  for(feature in missing_features_ext) {
    if(is.numeric(train_data_balanced[[feature]])) {
      external_features_all[[feature]] <- mean(train_data_balanced[[feature]], na.rm = TRUE)
    } else {
      mode_val <- names(sort(table(train_data_balanced[[feature]]), decreasing = TRUE))[1]
      external_features_all[[feature]] <- mode_val
    }
  }
}

external_features <- external_features_all[, final_features, drop = FALSE]
external_labels <- external_data$Diagnosis

cat("Data preparation complete:\n")
cat("Test set dimensions:", dim(test_features), "\n")
cat("External set dimensions:", dim(external_features), "\n")

# ==================== Calculate Feature Importance for Both Datasets ====================
cat("Calculating feature importance for both datasets...\n")

# 方法1: 基于置换的重要性 (Permutation Importance)
calculate_permutation_importance <- function(model, features, labels, n_permutations = 10) {
  # 首先计算基准性能
  baseline_probs <- predict_naive_bayes(model, features)
  baseline_auc <- auc(roc(as.numeric(labels) - 1, baseline_probs))
  
  importance_scores <- numeric(ncol(features))
  names(importance_scores) <- colnames(features)
  
  for(i in 1:ncol(features)) {
    feature_name <- colnames(features)[i]
    permutation_drops <- numeric(n_permutations)
    
    for(j in 1:n_permutations) {
      # 置换特征值
      permuted_features <- features
      permuted_features[[feature_name]] <- sample(permuted_features[[feature_name]])
      
      # 使用置换后的数据进行预测
      perm_probs <- predict_naive_bayes(model, permuted_features)
      perm_auc <- auc(roc(as.numeric(labels) - 1, perm_probs))
      permutation_drops[j] <- baseline_auc - perm_auc
    }
    
    importance_scores[i] <- mean(permutation_drops)
  }
  
  return(importance_scores)
}

# 计算测试集置换重要性
cat("Calculating permutation importance for test set...\n")
perm_importance_test <- calculate_permutation_importance(final_nb_model, test_features, test_labels, n_permutations = 5)

# 计算外部数据集置换重要性
cat("Calculating permutation importance for external set...\n")
perm_importance_external <- calculate_permutation_importance(final_nb_model, external_features, external_labels, n_permutations = 5)

# 创建特征重要性数据框
feature_importance_test <- data.frame(
  Feature = names(perm_importance_test),
  Importance = as.numeric(perm_importance_test),
  Dataset = "Test",
  stringsAsFactors = FALSE
)
feature_importance_test <- feature_importance_test[order(-feature_importance_test$Importance), ]

feature_importance_external <- data.frame(
  Feature = names(perm_importance_external),
  Importance = as.numeric(perm_importance_external),
  Dataset = "External",
  stringsAsFactors = FALSE
)
feature_importance_external <- feature_importance_external[order(-feature_importance_external$Importance), ]

# 创建特征重要性比较数据框
importance_comparison <- merge(
  feature_importance_test[, c("Feature", "Importance")],
  feature_importance_external[, c("Feature", "Importance")],
  by = "Feature",
  suffixes = c("_Test", "_External")
)
importance_comparison$Difference <- abs(importance_comparison$Importance_Test - importance_comparison$Importance_External)
importance_comparison$Rank_Test <- match(importance_comparison$Feature, feature_importance_test$Feature)
importance_comparison$Rank_External <- match(importance_comparison$Feature, feature_importance_external$Feature)
importance_comparison$Rank_Difference <- abs(importance_comparison$Rank_Test - importance_comparison$Rank_External)

importance_comparison <- importance_comparison[order(-importance_comparison$Importance_Test), ]

# ==================== Generate Model Interpretation Visualizations ====================
cat("Generating model interpretation visualizations for both datasets...\n")
if(!dir.exists("Part4_Results")) dir.create("Part4_Results")

# 1. Feature Importance Comparison Plot (Top 20)
top_20_features_test <- head(feature_importance_test, 20)
top_20_features_external <- head(feature_importance_external, 20)

# 合并数据用于比较
importance_comparison_top <- importance_comparison[
  importance_comparison$Feature %in% 
    union(top_20_features_test$Feature, top_20_features_external$Feature),
]

importance_comparison_long <- reshape2::melt(
  importance_comparison_top[, c("Feature", "Importance_Test", "Importance_External")],
  id.vars = "Feature",
  variable.name = "Dataset",
  value.name = "Importance"
)
importance_comparison_long$Dataset <- ifelse(
  importance_comparison_long$Dataset == "Importance_Test", "Test", "External"
)

p1 <- ggplot(importance_comparison_long, aes(x = reorder(Feature, Importance), y = Importance, fill = Dataset)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  coord_flip() +
  labs(title = "Feature Importance Comparison - NaiveBayes Model",
       subtitle = "Permutation Importance (Test vs External Dataset)",
       x = "Features",
       y = "Mean AUC Decrease") +
  scale_fill_manual(values = c("Test" = "darkred", "External" = "darkblue")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 12),
        axis.text = element_text(size = 10),
        axis.title = element_text(size = 12),
        legend.position = "top")

ggsave("Part4_Results/Feature_Importance_Comparison.png", p1, width = 14, height = 10, dpi = 300)

# 2. Feature Importance Correlation Plot
p2 <- ggplot(importance_comparison, aes(x = Importance_Test, y = Importance_External)) +
  geom_point(aes(color = Rank_Difference), size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, color = "red", linetype = "dashed") +
  geom_abline(slope = 1, intercept = 0, color = "gray", linetype = "dotted") +
  geom_text(aes(label = ifelse(Rank_Difference > 5 | Difference > 0.05, Feature, "")), 
            vjust = -0.5, size = 3) +
  labs(title = "Feature Importance Correlation: Test vs External Dataset",
       x = "Importance (Test Set)",
       y = "Importance (External Set)",
       color = "Rank Difference") +
  scale_color_gradient(low = "blue", high = "red") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

ggsave("Part4_Results/Feature_Importance_Correlation.png", p2, width = 10, height = 8, dpi = 300)

# 计算重要性相关性
importance_cor <- cor(importance_comparison$Importance_Test, importance_comparison$Importance_External, 
                      use = "complete.obs")
cat("Feature importance correlation between test and external sets:", round(importance_cor, 4), "\n")

# 3. Partial Dependence Plots Comparison for Top Features
cat("Creating partial dependence plots comparison...\n")
top_10_features_combined <- head(importance_comparison$Feature, 10)

create_partial_dependence <- function(model, features, feature_name, grid_size = 20) {
  # 创建网格值
  if(is.numeric(features[[feature_name]])) {
    feature_range <- seq(min(features[[feature_name]]), max(features[[feature_name]]), length.out = grid_size)
  } else {
    feature_range <- unique(features[[feature_name]])
  }
  
  # 计算部分依赖
  pd_values <- numeric(length(feature_range))
  
  for(i in 1:length(feature_range)) {
    # 创建修改后的数据集
    modified_features <- features
    modified_features[[feature_name]] <- feature_range[i]
    
    # 预测
    pred_probs <- predict_naive_bayes(model, modified_features)
    pd_values[i] <- mean(pred_probs)
  }
  
  return(data.frame(
    Feature_Value = feature_range,
    Predicted_Probability = pd_values,
    Feature = feature_name
  ))
}

# 为每个重要特征创建部分依赖图比较
for(feature in top_10_features_combined) {
  if(feature %in% colnames(train_features)) {
    # 测试集部分依赖
    pd_data_test <- create_partial_dependence(final_nb_model, test_features, feature)
    pd_data_test$Dataset <- "Test"
    
    # 外部集部分依赖
    pd_data_external <- create_partial_dependence(final_nb_model, external_features, feature)
    pd_data_external$Dataset <- "External"
    
    # 合并数据
    pd_data_combined <- rbind(pd_data_test, pd_data_external)
    
    if(is.numeric(train_features[[feature]])) {
      p_pd <- ggplot(pd_data_combined, aes(x = Feature_Value, y = Predicted_Probability, color = Dataset)) +
        geom_line(size = 1.5) +
        geom_point(size = 2) +
        labs(title = paste("Partial Dependence Comparison:", feature),
             x = feature,
             y = "Predicted Probability") +
        scale_color_manual(values = c("Test" = "darkred", "External" = "darkblue")) +
        theme_minimal() +
        theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
              legend.position = "top")
    } else {
      p_pd <- ggplot(pd_data_combined, aes(x = Feature_Value, y = Predicted_Probability, fill = Dataset)) +
        geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
        labs(title = paste("Partial Dependence Comparison:", feature),
             x = feature,
             y = "Predicted Probability") +
        scale_fill_manual(values = c("Test" = "darkred", "External" = "darkblue")) +
        theme_minimal() +
        theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
              axis.text.x = element_text(angle = 45, hjust = 1),
              legend.position = "top")
    }
    
    # 使用安全的文件名
    safe_feature_name <- safe_filename(feature)
    ggsave(paste0("Part4_Results/Partial_Dependence_Comparison_", safe_feature_name, ".png"), 
           p_pd, width = 12, height = 6, dpi = 300)
  }
}

# 4. Individual Prediction Explanations for Both Datasets
cat("Generating individual prediction explanations for both datasets...\n")

# 选择代表性的测试案例和外部案例
set.seed(123)
test_sample_indices <- sample(1:nrow(test_features), min(3, nrow(test_features)))
external_sample_indices <- sample(1:nrow(external_features), min(3, nrow(external_features)))

# 为测试集生成个体预测解释
for(i in seq_along(test_sample_indices)) {
  idx <- test_sample_indices[i]
  case_features <- test_features[idx, , drop = FALSE]
  true_label <- as.numeric(test_labels[idx]) - 1
  
  # 获取预测概率
  pred_prob <- predict_naive_bayes(final_nb_model, case_features)
  pred_label <- ifelse(pred_prob > 0.5, 1, 0)
  
  # 创建特征贡献图
  feature_effects <- numeric(length(final_features))
  names(feature_effects) <- final_features
  
  for(feature in final_features) {
    # 计算当该特征设为均值时的预测
    modified_features <- case_features
    if(is.numeric(train_features[[feature]])) {
      modified_features[[feature]] <- mean(train_features[[feature]], na.rm = TRUE)
    } else {
      mode_val <- names(sort(table(train_features[[feature]]), decreasing = TRUE))[1]
      modified_features[[feature]] <- mode_val
    }
    
    modified_pred <- predict_naive_bayes(final_nb_model, modified_features)
    feature_effects[feature] <- pred_prob - modified_pred
  }
  
  # 创建特征贡献图
  effect_data <- data.frame(
    Feature = names(feature_effects),
    Effect = feature_effects,
    stringsAsFactors = FALSE
  )
  effect_data <- effect_data[order(-abs(effect_data$Effect)), ]
  effect_data_top <- head(effect_data, 15)
  
  p_effect <- ggplot(effect_data_top, aes(x = reorder(Feature, Effect), y = Effect, 
                                          fill = Effect > 0)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    coord_flip() +
    scale_fill_manual(values = c("TRUE" = "red", "FALSE" = "blue"), 
                      labels = c("Positive Impact", "Negative Impact")) +
    labs(title = paste("Feature Effect Plot - Test Case", idx, 
                       "\nTrue:", true_label, "Pred:", round(pred_prob, 3)),
         x = "Features",
         y = "Effect on Prediction",
         fill = "Impact Direction") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 10, face = "bold"),
          legend.position = "top")
  
  ggsave(paste0("Part4_Results/Feature_Effect_Test_Case_", idx, ".png"), 
         p_effect, width = 10, height = 6, dpi = 300)
}

# 为外部集生成个体预测解释
for(i in seq_along(external_sample_indices)) {
  idx <- external_sample_indices[i]
  case_features <- external_features[idx, , drop = FALSE]
  true_label <- as.numeric(external_labels[idx]) - 1
  
  # 获取预测概率
  pred_prob <- predict_naive_bayes(final_nb_model, case_features)
  pred_label <- ifelse(pred_prob > 0.5, 1, 0)
  
  # 创建特征贡献图
  feature_effects <- numeric(length(final_features))
  names(feature_effects) <- final_features
  
  for(feature in final_features) {
    # 计算当该特征设为均值时的预测
    modified_features <- case_features
    if(is.numeric(train_features[[feature]])) {
      modified_features[[feature]] <- mean(train_features[[feature]], na.rm = TRUE)
    } else {
      mode_val <- names(sort(table(train_features[[feature]]), decreasing = TRUE))[1]
      modified_features[[feature]] <- mode_val
    }
    
    modified_pred <- predict_naive_bayes(final_nb_model, modified_features)
    feature_effects[feature] <- pred_prob - modified_pred
  }
  
  # 创建特征贡献图
  effect_data <- data.frame(
    Feature = names(feature_effects),
    Effect = feature_effects,
    stringsAsFactors = FALSE
  )
  effect_data <- effect_data[order(-abs(effect_data$Effect)), ]
  effect_data_top <- head(effect_data, 15)
  
  p_effect <- ggplot(effect_data_top, aes(x = reorder(Feature, Effect), y = Effect, 
                                          fill = Effect > 0)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    coord_flip() +
    scale_fill_manual(values = c("TRUE" = "red", "FALSE" = "blue"), 
                      labels = c("Positive Impact", "Negative Impact")) +
    labs(title = paste("Feature Effect Plot - External Case", idx, 
                       "\nTrue:", true_label, "Pred:", round(pred_prob, 3)),
         x = "Features",
         y = "Effect on Prediction",
         fill = "Impact Direction") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 10, face = "bold"),
          legend.position = "top")
  
  ggsave(paste0("Part4_Results/Feature_Effect_External_Case_", idx, ".png"), 
         p_effect, width = 10, height = 6, dpi = 300)
}

# 5. Feature Distribution Comparison by Class
cat("Creating feature distribution comparison plots...\n")
top_5_features <- head(importance_comparison$Feature, 5)

for(feature in top_5_features) {
  if(feature %in% colnames(train_features)) {
    # 测试集分布
    dist_data_test <- data.frame(
      Value = test_features[[feature]],
      Class = test_labels,
      Dataset = "Test"
    )
    
    # 外部集分布
    dist_data_external <- data.frame(
      Value = external_features[[feature]],
      Class = external_labels,
      Dataset = "External"
    )
    
    dist_data_combined <- rbind(dist_data_test, dist_data_external)
    
    if(is.numeric(train_features[[feature]])) {
      p_dist <- ggplot(dist_data_combined, aes(x = Value, fill = Class)) +
        geom_density(alpha = 0.6) +
        facet_wrap(~ Dataset, ncol = 2) +
        labs(title = paste("Feature Distribution Comparison by Class:", feature),
             x = feature,
             y = "Density") +
        theme_minimal() +
        theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"))
    } else {
      p_dist <- ggplot(dist_data_combined, aes(x = Value, fill = Class)) +
        geom_bar(position = "dodge", alpha = 0.8) +
        facet_wrap(~ Dataset, ncol = 2) +
        labs(title = paste("Feature Distribution Comparison by Class:", feature),
             x = feature,
             y = "Count") +
        theme_minimal() +
        theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
              axis.text.x = element_text(angle = 45, hjust = 1))
    }
    
    # 使用安全的文件名
    safe_feature_name <- safe_filename(feature)
    ggsave(paste0("Part4_Results/Feature_Distribution_Comparison_", safe_feature_name, ".png"), 
           p_dist, width = 12, height = 6, dpi = 300)
  }
}

# 6. Model Stability Analysis
cat("Analyzing model stability across datasets...\n")

# 计算特征重要性的一致性
rank_correlation <- cor(importance_comparison$Rank_Test, importance_comparison$Rank_External, 
                        method = "spearman")

# 识别不稳定的特征（重要性差异大）
unstable_features <- importance_comparison[
  importance_comparison$Difference > quantile(importance_comparison$Difference, 0.75) |
    importance_comparison$Rank_Difference > 10,
]

cat("Model stability analysis:\n")
cat("Rank correlation (Spearman):", round(rank_correlation, 4), "\n")
cat("Number of unstable features (top 25% difference):", nrow(unstable_features), "\n")
cat("Most unstable features (large importance difference):\n")
print(head(unstable_features[order(-unstable_features$Difference), ], 5))

# 创建模型稳定性图
p_stability <- ggplot(importance_comparison, aes(x = Rank_Difference, y = Difference)) +
  geom_point(aes(color = Importance_Test, size = Importance_Test), alpha = 0.7) +
  geom_text(aes(label = ifelse(Difference > 0.05 | Rank_Difference > 10, Feature, "")), 
            vjust = -0.5, size = 3) +
  labs(title = "Model Stability Analysis: Feature Importance Consistency",
       subtitle = paste("Rank Correlation =", round(rank_correlation, 3)),
       x = "Rank Difference Between Datasets",
       y = "Absolute Importance Difference",
       color = "Importance (Test)",
       size = "Importance (Test)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 12))

ggsave("Part4_Results/Model_Stability_Analysis.png", p_stability, width = 12, height = 8, dpi = 300)

# ==================== Save Results ====================
writeData(wb4, "Feature_Importance_Test", feature_importance_test, startRow = 1)
writeData(wb4, "Feature_Importance_External", feature_importance_external, startRow = 1)
writeData(wb4, "Feature_Importance_Comparison", importance_comparison, startRow = 1)

# 保存部分依赖数据
pd_summary_test <- data.frame()
pd_summary_external <- data.frame()

for(feature in top_10_features_combined) {
  if(feature %in% colnames(train_features)) {
    # 测试集部分依赖
    pd_data_test <- create_partial_dependence(final_nb_model, test_features, feature, grid_size = 10)
    pd_data_test$Feature_Name <- feature
    pd_data_test$Dataset <- "Test"
    pd_summary_test <- rbind(pd_summary_test, pd_data_test)
    
    # 外部集部分依赖
    pd_data_external <- create_partial_dependence(final_nb_model, external_features, feature, grid_size = 10)
    pd_data_external$Feature_Name <- feature
    pd_data_external$Dataset <- "External"
    pd_summary_external <- rbind(pd_summary_external, pd_data_external)
  }
}

writeData(wb4, "Partial_Dependence_Test", pd_summary_test, startRow = 1)
writeData(wb4, "Partial_Dependence_External", pd_summary_external, startRow = 1)

# 保存个体预测
individual_predictions_test <- data.frame()
for(i in seq_along(test_sample_indices)) {
  idx <- test_sample_indices[i]
  case_features <- test_features[idx, , drop = FALSE]
  
  pred_prob <- predict_naive_bayes(final_nb_model, case_features)
  
  individual_predictions_test <- rbind(individual_predictions_test, 
                                       data.frame(
                                         Case_ID = idx,
                                         True_Label = as.character(test_labels[idx]),
                                         Predicted_Probability = pred_prob,
                                         Predicted_Label = ifelse(pred_prob > 0.5, "Case", "Control"),
                                         Dataset = "Test"
                                       ))
}

individual_predictions_external <- data.frame()
for(i in seq_along(external_sample_indices)) {
  idx <- external_sample_indices[i]
  case_features <- external_features[idx, , drop = FALSE]
  
  pred_prob <- predict_naive_bayes(final_nb_model, case_features)
  
  individual_predictions_external <- rbind(individual_predictions_external, 
                                           data.frame(
                                             Case_ID = idx,
                                             True_Label = as.character(external_labels[idx]),
                                             Predicted_Probability = pred_prob,
                                             Predicted_Label = ifelse(pred_prob > 0.5, "Case", "Control"),
                                             Dataset = "External"
                                           ))
}

writeData(wb4, "Individual_Predictions_Test", individual_predictions_test, startRow = 1)
writeData(wb4, "Individual_Predictions_External", individual_predictions_external, startRow = 1)

# 保存解释摘要
interpretation_summary <- data.frame(
  Metric = c(
    "Number of Features",
    "Feature Importance Correlation",
    "Rank Correlation (Spearman)",
    "Number of Unstable Features",
    "Average Importance Difference",
    "Maximum Importance Difference"
  ),
  Value = c(
    nrow(importance_comparison),
    round(importance_cor, 4),
    round(rank_correlation, 4),
    nrow(unstable_features),
    round(mean(importance_comparison$Difference), 4),
    round(max(importance_comparison$Difference), 4)
  ),
  Description = c(
    "Total number of features analyzed",
    "Pearson correlation of feature importance scores",
    "Spearman correlation of feature importance ranks",
    "Features with large importance differences between datasets",
    "Mean absolute difference in importance scores",
    "Maximum absolute difference in importance scores"
  )
)

writeData(wb4, "Interpretation_Summary", interpretation_summary, startRow = 1)

# 尝试提取模型系数（如果可能）
tryCatch({
  if(!is.null(final_nb_model$model_type) && final_nb_model$model_type == "e1071") {
    # 对于e1071的naiveBayes，我们可以查看先验概率和条件概率
    model_coefs <- data.frame(
      Component = c("Prior_Probabilities", "Conditional_Probabilities"),
      Description = c("Class prior probabilities", "Feature conditional probabilities"),
      stringsAsFactors = FALSE
    )
    writeData(wb4, "Model_Coefficients", model_coefs, startRow = 1)
    
    # 保存先验概率
    prior_probs <- final_nb_model$apriori
    prior_df <- data.frame(
      Class = names(prior_probs),
      Prior_Probability = as.numeric(prior_probs)
    )
    writeData(wb4, "Model_Coefficients", data.frame(Prior_Probabilities = ""), startRow = 4)
    writeData(wb4, "Model_Coefficients", prior_df, startRow = 5)
  }
}, error = function(e) {
  cat("Cannot extract model coefficients:", e$message, "\n")
})

saveWorkbook(wb4, "Part4_Model_Interpretation_Results.xlsx", overwrite = TRUE)

# Save interpretation objects
saveRDS(feature_importance_test, "feature_importance_test.rds")
saveRDS(feature_importance_external, "feature_importance_external.rds")
saveRDS(importance_comparison, "feature_importance_comparison.rds")
saveRDS(interpretation_summary, "interpretation_summary.rds")

cat("\n=== Part 4 Complete ===\n")
cat("Results saved to:\n")
cat("- Part4_Model_Interpretation_Results.xlsx\n")
cat("- Part4_Results/ directory\n")
cat("- feature_importance_test.rds, feature_importance_external.rds\n")
cat("- feature_importance_comparison.rds, interpretation_summary.rds\n")

cat("\n=== INTERPRETATION SUMMARY ===\n")
cat("\nTop 10 Features by Importance (Test Set):\n")
for(i in 1:min(10, nrow(feature_importance_test))) {
  cat(i, ".", feature_importance_test$Feature[i], "(Importance:", round(feature_importance_test$Importance[i], 4), ")\n")
}

cat("\nTop 10 Features by Importance (External Set):\n")
for(i in 1:min(10, nrow(feature_importance_external))) {
  cat(i, ".", feature_importance_external$Feature[i], "(Importance:", round(feature_importance_external$Importance[i], 4), ")\n")
}

cat("\nModel Stability Assessment:\n")
cat("Feature importance correlation:", round(importance_cor, 4), "\n")
cat("Rank correlation:", round(rank_correlation, 4), "\n")
if(importance_cor > 0.7 && rank_correlation > 0.7) {
  cat("✓ Excellent model stability across datasets\n")
} else if(importance_cor > 0.5 && rank_correlation > 0.5) {
  cat("○ Moderate model stability across datasets\n")
} else {
  cat("⚠ Potential model instability or dataset shift\n")
}

cat("\nDataset Information:\n")
cat("Test set samples:", nrow(test_features), "\n")
cat("External set samples:", nrow(external_features), "\n")
cat("Features analyzed:", length(final_features), "\n")

cat("\nModel Interpretation Analysis includes:\n")
cat("- Permutation-based feature importance for both datasets\n")
cat("- Feature importance comparison and correlation analysis\n")
cat("- Partial dependence plots comparison\n")
cat("- Individual prediction explanations for both datasets\n")
cat("- Feature distribution comparison by class\n")
cat("- Model stability assessment\n")
cat("\nNote: External dataset interpretation provides insights into model generalizability.\n")