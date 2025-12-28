# Part 4: SHAP Analysis for XGBoost Model with External Validation
# Load required packages
library(readxl)
library(xgboost)
library(SHAPforxgboost)
library(ggplot2)
library(openxlsx)
library(viridis)
library(ggforce)
library(dplyr)

# Load objects from previous parts
final_xgb_model <- readRDS("final_xgb_model.rds")
final_features <- readRDS("final_features.rds")
train_data_balanced <- readRDS("train_data_balanced.rds")
test_data <- read_excel("C:/Users/Bo Wang/Desktop/Internal Validation Cohort.xlsx")
external_data <- read_excel("C:/Users/Bo Wang/Desktop/External Validation Cohort.xlsx") # 请修改为您的实际路径

test_data$Diagnosis <- as.factor(test_data$Diagnosis)
external_data$Diagnosis <- as.factor(external_data$Diagnosis)

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
addWorksheet(wb4, "SHAP_Summary")
addWorksheet(wb4, "SHAP_Values_Train")
addWorksheet(wb4, "SHAP_Values_Test")
addWorksheet(wb4, "SHAP_Values_External")
addWorksheet(wb4, "Feature_Interactions")
addWorksheet(wb4, "SHAP_Consistency_Analysis")

# ==================== SHAP Analysis ====================
cat("=== Part 4: SHAP Analysis with External Validation ===\n")

# ========== 修复：确保所有特征都是数值类型 ==========
cat("Preparing and converting features to numeric...\n")

# 准备训练数据
train_features <- train_data_balanced[, final_features]

# 将因子和字符变量转换为数值
for(col in final_features) {
  if(is.factor(train_features[[col]])) {
    cat("Converting factor variable", col, "to numeric\n")
    train_features[[col]] <- as.numeric(train_features[[col]])
  } else if(is.character(train_features[[col]])) {
    cat("Converting character variable", col, "to numeric\n")
    # 先转换为因子，再转换为数值
    train_features[[col]] <- as.numeric(as.factor(train_features[[col]]))
  } else if(!is.numeric(train_features[[col]])) {
    cat("Converting non-numeric variable", col, "to numeric\n")
    train_features[[col]] <- as.numeric(train_features[[col]])
  }
}

# 检查是否有任何非数值数据
if(any(sapply(train_features, function(x) !is.numeric(x)))) {
  cat("Warning: Some training features are still not numeric after conversion\n")
  # 强制转换为数值
  train_features <- as.data.frame(lapply(train_features, function(x) as.numeric(as.factor(x))))
}

# 转换为矩阵
train_features <- as.matrix(train_features)
train_labels <- as.numeric(train_data_balanced$Diagnosis) - 1

# 检查是否有NA值并处理
if(any(is.na(train_features))) {
  cat("Warning: NA values found in training features, imputing with mean...\n")
  for(j in 1:ncol(train_features)) {
    na_indices <- is.na(train_features[, j])
    if(any(na_indices)) {
      train_features[na_indices, j] <- mean(train_features[, j], na.rm = TRUE)
    }
  }
}

# 准备测试数据
test_features_all <- test_data[, !names(test_data) %in% "Diagnosis"]

# Simple imputation for test set for SHAP analysis
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

# 转换测试特征为数值
test_features <- test_features_all[, final_features, drop = FALSE]

for(col in final_features) {
  if(col %in% names(test_features)) {
    if(is.factor(test_features[[col]])) {
      test_features[[col]] <- as.numeric(test_features[[col]])
    } else if(is.character(test_features[[col]])) {
      test_features[[col]] <- as.numeric(as.factor(test_features[[col]]))
    } else if(!is.numeric(test_features[[col]])) {
      test_features[[col]] <- as.numeric(test_features[[col]])
    }
  }
}

if(any(sapply(test_features, function(x) !is.numeric(x)))) {
  test_features <- as.data.frame(lapply(test_features, function(x) as.numeric(as.factor(x))))
}

test_features <- as.matrix(test_features)

# 检查测试特征是否有NA值
if(any(is.na(test_features))) {
  cat("Warning: NA values found in test features, imputing with mean...\n")
  for(j in 1:ncol(test_features)) {
    na_indices <- is.na(test_features[, j])
    if(any(na_indices)) {
      test_features[na_indices, j] <- mean(test_features[, j], na.rm = TRUE)
    }
  }
}

# ========== 准备外部验证集数据 ==========
cat("Preparing external validation data for SHAP analysis...\n")

# 准备外部验证数据
external_features_all <- external_data[, !names(external_data) %in% "Diagnosis"]

# Simple imputation for external set for SHAP analysis
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

# Ensure all features are present
missing_features_external <- setdiff(final_features, names(external_features_all))
if(length(missing_features_external) > 0) {
  for(feature in missing_features_external) {
    if(is.numeric(train_data_balanced[[feature]])) {
      external_features_all[[feature]] <- mean(train_data_balanced[[feature]], na.rm = TRUE)
    } else {
      mode_val <- names(sort(table(train_data_balanced[[feature]]), decreasing = TRUE))[1]
      external_features_all[[feature]] <- mode_val
    }
  }
}

# 转换外部验证特征为数值
external_features <- external_features_all[, final_features, drop = FALSE]

for(col in final_features) {
  if(col %in% names(external_features)) {
    if(is.factor(external_features[[col]])) {
      external_features[[col]] <- as.numeric(external_features[[col]])
    } else if(is.character(external_features[[col]])) {
      external_features[[col]] <- as.numeric(as.factor(external_features[[col]]))
    } else if(!is.numeric(external_features[[col]])) {
      external_features[[col]] <- as.numeric(external_features[[col]])
    }
  }
}

if(any(sapply(external_features, function(x) !is.numeric(x)))) {
  external_features <- as.data.frame(lapply(external_features, function(x) as.numeric(as.factor(x))))
}

external_features <- as.matrix(external_features)

# 检查外部验证特征是否有NA值
if(any(is.na(external_features))) {
  cat("Warning: NA values found in external features, imputing with mean...\n")
  for(j in 1:ncol(external_features)) {
    na_indices <- is.na(external_features[, j])
    if(any(na_indices)) {
      external_features[na_indices, j] <- mean(external_features[, j], na.rm = TRUE)
    }
  }
}

# 调试信息
cat("Training features dimensions:", dim(train_features), "\n")
cat("Training features type:", class(train_features), "\n")
cat("Test features dimensions:", dim(test_features), "\n")
cat("Test features type:", class(test_features), "\n")
cat("External features dimensions:", dim(external_features), "\n")
cat("External features type:", class(external_features), "\n")

# ==================== Calculate SHAP Values ====================
cat("Calculating SHAP values for all datasets...\n")

# For training set
shap_values_train <- shap.values(xgb_model = final_xgb_model, X_train = train_features)
shap_data_train <- shap_values_train$shap_score

# For test set
shap_values_test <- shap.values(xgb_model = final_xgb_model, X_train = test_features)
shap_data_test <- shap_values_test$shap_score

# For external validation set
shap_values_external <- shap.values(xgb_model = final_xgb_model, X_train = external_features)
shap_data_external <- shap_values_external$shap_score

# Extract mean SHAP values for all datasets
if (is.matrix(shap_values_train$mean_shap_score)) {
  shap_summary_train <- data.frame(
    Feature = rownames(shap_values_train$mean_shap_score),
    Mean_SHAP_Train = as.numeric(shap_values_train$mean_shap_score),
    stringsAsFactors = FALSE
  )
} else {
  shap_summary_train <- data.frame(
    Feature = names(shap_values_train$mean_shap_score),
    Mean_SHAP_Train = as.numeric(shap_values_train$mean_shap_score),
    stringsAsFactors = FALSE
  )
}

if (is.matrix(shap_values_test$mean_shap_score)) {
  shap_summary_test <- data.frame(
    Feature = rownames(shap_values_test$mean_shap_score),
    Mean_SHAP_Test = as.numeric(shap_values_test$mean_shap_score),
    stringsAsFactors = FALSE
  )
} else {
  shap_summary_test <- data.frame(
    Feature = names(shap_values_test$mean_shap_score),
    Mean_SHAP_Test = as.numeric(shap_values_test$mean_shap_score),
    stringsAsFactors = FALSE
  )
}

if (is.matrix(shap_values_external$mean_shap_score)) {
  shap_summary_external <- data.frame(
    Feature = rownames(shap_values_external$mean_shap_score),
    Mean_SHAP_External = as.numeric(shap_values_external$mean_shap_score),
    stringsAsFactors = FALSE
  )
} else {
  shap_summary_external <- data.frame(
    Feature = names(shap_values_external$mean_shap_score),
    Mean_SHAP_External = as.numeric(shap_values_external$mean_shap_score),
    stringsAsFactors = FALSE
  )
}

# 合并所有数据集的SHAP摘要
shap_summary_combined <- merge(shap_summary_train, shap_summary_test, by = "Feature", all = TRUE)
shap_summary_combined <- merge(shap_summary_combined, shap_summary_external, by = "Feature", all = TRUE)

# 计算平均SHAP值用于排序
shap_summary_combined$Mean_SHAP_Overall <- rowMeans(shap_summary_combined[, c("Mean_SHAP_Train", "Mean_SHAP_Test", "Mean_SHAP_External")], na.rm = TRUE)
shap_summary_combined <- shap_summary_combined[order(-shap_summary_combined$Mean_SHAP_Overall), ]

# ==================== Generate SHAP Visualizations ====================
cat("Generating SHAP visualizations for all datasets...\n")
if(!dir.exists("Part4_Results")) dir.create("Part4_Results")

# 1. SHAP Feature Importance Comparison (Top 20)
top_20_shap <- head(shap_summary_combined, 20)

# 转换为长格式用于绘图
top_20_long <- reshape2::melt(top_20_shap[, c("Feature", "Mean_SHAP_Train", "Mean_SHAP_Test", "Mean_SHAP_External")], 
                              id.vars = "Feature", 
                              variable.name = "Dataset", 
                              value.name = "Mean_SHAP")

# 清理数据集名称
top_20_long$Dataset <- gsub("Mean_SHAP_", "", top_20_long$Dataset)

p1 <- ggplot(top_20_long, aes(x = reorder(Feature, Mean_SHAP), y = Mean_SHAP, fill = Dataset)) +
  geom_bar(stat = "identity", position = position_dodge(0.8), alpha = 0.8) +
  coord_flip() +
  labs(title = "SHAP Feature Importance Comparison (Top 20)",
       x = "Features",
       y = "Mean |SHAP| Value") +
  scale_fill_manual(values = c("Train" = "blue", "Test" = "red", "External" = "green"),
                    labels = c("Train" = "Training", "Test" = "Test", "External" = "External Validation")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        axis.text = element_text(size = 10),
        axis.title = element_text(size = 12),
        legend.position = "top")

ggsave("Part4_Results/SHAP_Feature_Importance_Comparison.png", p1, width = 14, height = 8, dpi = 300)

# 2. SHAP Summary Plot - 三数据集比较
cat("Creating SHAP summary plot for all datasets...\n")
top_15_features <- head(shap_summary_combined$Feature, 15)

# 准备训练集SHAP数据
shap_long_list_train <- list()
for(feature in top_15_features) {
  if(feature %in% colnames(shap_data_train)) {
    temp_df <- data.frame(
      Feature = rep(feature, nrow(shap_data_train)),
      SHAP_Value = shap_data_train[[feature]],
      Feature_Value = train_features[, feature],
      Dataset = "Training"
    )
    shap_long_list_train[[feature]] <- temp_df
  }
}

# 准备测试集SHAP数据
shap_long_list_test <- list()
for(feature in top_15_features) {
  if(feature %in% colnames(shap_data_test)) {
    temp_df <- data.frame(
      Feature = rep(feature, nrow(shap_data_test)),
      SHAP_Value = shap_data_test[[feature]],
      Feature_Value = test_features[, feature],
      Dataset = "Test"
    )
    shap_long_list_test[[feature]] <- temp_df
  }
}

# 准备外部验证集SHAP数据
shap_long_list_external <- list()
for(feature in top_15_features) {
  if(feature %in% colnames(shap_data_external)) {
    temp_df <- data.frame(
      Feature = rep(feature, nrow(shap_data_external)),
      SHAP_Value = shap_data_external[[feature]],
      Feature_Value = external_features[, feature],
      Dataset = "External"
    )
    shap_long_list_external[[feature]] <- temp_df
  }
}

# 合并所有数据
shap_long_train <- do.call(rbind, shap_long_list_train)
shap_long_test <- do.call(rbind, shap_long_list_test)
shap_long_external <- do.call(rbind, shap_long_list_external)
shap_long_combined <- rbind(shap_long_train, shap_long_test, shap_long_external)

# 计算每个特征的绝对SHAP均值用于排序
feature_order <- shap_long_combined %>%
  group_by(Feature) %>%
  summarise(Mean_Abs_SHAP = mean(abs(SHAP_Value))) %>%
  arrange(Mean_Abs_SHAP) %>%
  pull(Feature)

shap_long_combined$Feature <- factor(shap_long_combined$Feature, levels = feature_order)

# 创建SHAP摘要图
p2 <- ggplot(shap_long_combined, aes(x = SHAP_Value, y = Feature, color = Feature_Value)) +
  geom_point(alpha = 0.6, size = 1) +
  scale_color_viridis_c(name = "Feature Value") +
  facet_wrap(~ Dataset, ncol = 3) +
  labs(title = "SHAP Summary Plot - Training vs Test vs External Validation",
       x = "SHAP Value (Impact on Model Output)",
       y = "Features") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        axis.text = element_text(size = 9),
        axis.title = element_text(size = 11),
        strip.text = element_text(size = 10, face = "bold"))

ggsave("Part4_Results/SHAP_Summary_Plot_Three_Sets.png", p2, width = 16, height = 10, dpi = 300)

# 3. SHAP Dependence Plots for Top 10 Features (三数据集比较)
top_10_features <- head(shap_summary_combined$Feature, 10)

for(feature in top_10_features) {
  if(feature %in% colnames(train_features)) {
    # 为三个数据集创建依赖图数据
    dependence_data_train <- data.frame(
      Feature_Value = train_features[, feature],
      SHAP_Value = shap_data_train[[feature]],
      Dataset = "Training"
    )
    
    dependence_data_test <- data.frame(
      Feature_Value = test_features[, feature],
      SHAP_Value = shap_data_test[[feature]],
      Dataset = "Test"
    )
    
    dependence_data_external <- data.frame(
      Feature_Value = external_features[, feature],
      SHAP_Value = shap_data_external[[feature]],
      Dataset = "External"
    )
    
    # 合并数据
    dependence_combined <- rbind(dependence_data_train, dependence_data_test, dependence_data_external)
    
    # 创建组合依赖图
    p_dep_combined <- ggplot(dependence_combined, aes(x = Feature_Value, y = SHAP_Value, color = Dataset)) +
      geom_point(alpha = 0.3, size = 1) +
      geom_smooth(method = "loess", se = TRUE, aes(fill = Dataset), alpha = 0.2) +
      scale_color_manual(values = c("Training" = "blue", "Test" = "red", "External" = "green")) +
      scale_fill_manual(values = c("Training" = "lightblue", "Test" = "pink", "External" = "lightgreen")) +
      labs(title = paste("SHAP Dependence Plot:", feature),
           x = feature,
           y = "SHAP Value") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
            legend.position = "top")
    
    # 使用安全的文件名
    safe_feature_name <- safe_filename(feature)
    ggsave(paste0("Part4_Results/SHAP_Dependence_Combined_", safe_feature_name, ".png"), 
           p_dep_combined, width = 12, height = 6, dpi = 300)
  }
}

# 4. SHAP Consistency Analysis Across Datasets
cat("Performing SHAP consistency analysis...\n")

# 计算特征排名相关性
calculate_rank_correlation <- function(shap_summary1, shap_summary2, dataset1_name, dataset2_name) {
  merged <- merge(shap_summary1, shap_summary2, by = "Feature")
  correlation <- cor(merged[, 2], merged[, 3], method = "spearman")
  return(data.frame(
    Comparison = paste(dataset1_name, "vs", dataset2_name),
    Spearman_Correlation = correlation,
    Features_Compared = nrow(merged)
  ))
}

# 计算排名相关性
rank_correlations <- rbind(
  calculate_rank_correlation(shap_summary_train, shap_summary_test, "Training", "Test"),
  calculate_rank_correlation(shap_summary_train, shap_summary_external, "Training", "External"),
  calculate_rank_correlation(shap_summary_test, shap_summary_external, "Test", "External")
)

# 可视化排名一致性
p_consistency <- ggplot(rank_correlations, aes(x = Comparison, y = Spearman_Correlation, fill = Comparison)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_text(aes(label = paste0(round(Spearman_Correlation, 3), "\n(n=", Features_Compared, ")")), 
            vjust = -0.5, size = 4) +
  labs(title = "SHAP Feature Ranking Consistency Across Datasets",
       x = "Dataset Comparison",
       y = "Spearman Correlation") +
  ylim(0, 1) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("Part4_Results/SHAP_Consistency_Analysis.png", p_consistency, width = 10, height = 8, dpi = 300)

# 5. SHAP Force Plots for Representative Cases from All Datasets
cat("Generating individual explanation plots for all datasets...\n")

# 从每个数据集选择代表性案例
set.seed(123)
train_sample_indices <- sample(1:nrow(train_features), min(2, nrow(train_features)))
test_sample_indices <- sample(1:nrow(test_features), min(2, nrow(test_features)))
external_sample_indices <- sample(1:nrow(external_features), min(2, nrow(external_features)))

# 为每个选中的案例生成力导向图
generate_force_plot <- function(shap_data, features, dataset_name, indices) {
  for(i in seq_along(indices)) {
    idx <- indices[i]
    
    # 准备力导向图数据
    shap_force_values <- as.numeric(shap_data[idx, ])
    feature_names <- colnames(shap_data)
    
    shap_force_data <- data.frame(
      Feature = feature_names,
      SHAP_Value = shap_force_values,
      Feature_Value = features[idx, ],
      stringsAsFactors = FALSE
    )
    
    # 取前10个特征
    shap_force_data <- shap_force_data[order(-abs(shap_force_data$SHAP_Value)), ]
    shap_force_data_top <- head(shap_force_data, 10)
    
    p_force <- ggplot(shap_force_data_top, aes(x = reorder(Feature, abs(SHAP_Value)), y = SHAP_Value, 
                                               fill = SHAP_Value > 0)) +
      geom_bar(stat = "identity", alpha = 0.8) +
      coord_flip() +
      scale_fill_manual(values = c("TRUE" = "red", "FALSE" = "blue"), 
                        labels = c("Positive Impact", "Negative Impact")) +
      labs(title = paste("SHAP Force Plot -", dataset_name, "Case", idx),
           x = "Features",
           y = "SHAP Value",
           fill = "Impact Direction") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
            legend.position = "top")
    
    ggsave(paste0("Part4_Results/SHAP_Force_Plot_", dataset_name, "_Case_", idx, ".png"), 
           p_force, width = 10, height = 6, dpi = 300)
  }
}

# 为三个数据集生成力导向图
generate_force_plot(shap_data_train, train_features, "Training", train_sample_indices)
generate_force_plot(shap_data_test, test_features, "Test", test_sample_indices)
generate_force_plot(shap_data_external, external_features, "External", external_sample_indices)

# 6. SHAP Waterfall Plot for Top Cases from All Datasets
cat("Generating SHAP waterfall plots for all datasets...\n")

generate_waterfall_plot <- function(shap_data, dataset_name, indices) {
  for(i in 1:min(2, length(indices))) {
    idx <- indices[i]
    
    # 准备瀑布图数据
    shap_waterfall_values <- as.numeric(shap_data[idx, ])
    feature_names <- colnames(shap_data)
    
    waterfall_data <- data.frame(
      Feature = feature_names,
      SHAP_Value = shap_waterfall_values,
      stringsAsFactors = FALSE
    )
    
    # 按绝对SHAP值排序并取前15个
    waterfall_data <- waterfall_data[order(-abs(waterfall_data$SHAP_Value)), ]
    waterfall_data_top <- head(waterfall_data, 15)
    
    # 计算瀑布图的累积和
    waterfall_data_top <- waterfall_data_top[order(waterfall_data_top$SHAP_Value), ]
    waterfall_data_top$Cumulative <- cumsum(waterfall_data_top$SHAP_Value)
    waterfall_data_top$Start <- c(0, head(waterfall_data_top$Cumulative, -1))
    
    p_waterfall <- ggplot(waterfall_data_top) +
      geom_segment(aes(x = reorder(Feature, SHAP_Value), xend = Feature, 
                       y = Start, yend = Cumulative, 
                       color = SHAP_Value > 0), 
                   size = 2) +
      geom_point(aes(x = Feature, y = Cumulative, color = SHAP_Value > 0), size = 3) +
      scale_color_manual(values = c("TRUE" = "red", "FALSE" = "blue"),
                         labels = c("Positive", "Negative")) +
      coord_flip() +
      labs(title = paste("SHAP Waterfall Plot -", dataset_name, "Case", idx),
           x = "Features",
           y = "Cumulative SHAP Value",
           color = "Impact") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"))
    
    ggsave(paste0("Part4_Results/SHAP_Waterfall_", dataset_name, "_Case_", idx, ".png"), 
           p_waterfall, width = 10, height = 8, dpi = 300)
  }
}

# 为三个数据集生成瀑布图
generate_waterfall_plot(shap_data_train, "Training", train_sample_indices)
generate_waterfall_plot(shap_data_test, "Test", test_sample_indices)
generate_waterfall_plot(shap_data_external, "External", external_sample_indices)

# 7. SHAP Value Distribution by Dataset
cat("Analyzing SHAP value distributions across datasets...\n")

# 计算每个数据集的SHAP值总和分布
shap_distribution_train <- data.frame(
  SHAP_Sum = rowSums(abs(shap_data_train)),
  Dataset = "Training"
)

shap_distribution_test <- data.frame(
  SHAP_Sum = rowSums(abs(shap_data_test)),
  Dataset = "Test"
)

shap_distribution_external <- data.frame(
  SHAP_Sum = rowSums(abs(shap_data_external)),
  Dataset = "External"
)

shap_distribution_combined <- rbind(shap_distribution_train, shap_distribution_test, shap_distribution_external)

p_distribution <- ggplot(shap_distribution_combined, aes(x = Dataset, y = SHAP_Sum, fill = Dataset)) +
  geom_violin(alpha = 0.7) +
  geom_boxplot(width = 0.1, alpha = 0.8) +
  labs(title = "SHAP Value Distribution Across Datasets",
       x = "Dataset",
       y = "Sum of Absolute SHAP Values") +
  scale_fill_manual(values = c("Training" = "blue", "Test" = "red", "External" = "green")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

ggsave("Part4_Results/SHAP_Distribution_Across_Datasets.png", p_distribution, width = 10, height = 6, dpi = 300)

# 8. SHAP Interaction Analysis (三数据集比较)
cat("Calculating SHAP interactions for all datasets...\n")
tryCatch({
  # 为计算效率，仅为前5个特征计算交互
  top_5_features <- head(shap_summary_combined$Feature, 5)
  
  if(length(top_5_features) >= 2) {
    # 为三个数据集创建相关性矩阵
    interaction_train <- cor(shap_data_train[, top_5_features])
    interaction_test <- cor(shap_data_test[, top_5_features])
    interaction_external <- cor(shap_data_external[, top_5_features])
    
    # 转换为长格式用于绘图
    interaction_long_train <- reshape2::melt(interaction_train)
    interaction_long_train$Dataset <- "Training"
    
    interaction_long_test <- reshape2::melt(interaction_test)
    interaction_long_test$Dataset <- "Test"
    
    interaction_long_external <- reshape2::melt(interaction_external)
    interaction_long_external$Dataset <- "External"
    
    interaction_combined <- rbind(interaction_long_train, interaction_long_test, interaction_long_external)
    colnames(interaction_combined) <- c("Feature1", "Feature2", "Correlation", "Dataset")
    
    # 创建组合交互图
    p_interaction <- ggplot(interaction_combined, aes(x = Feature1, y = Feature2, fill = Correlation)) +
      geom_tile() +
      geom_text(aes(label = round(Correlation, 2)), color = "white", size = 3) +
      facet_wrap(~ Dataset, ncol = 3) +
      scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                           midpoint = 0, name = "Correlation") +
      labs(title = "SHAP Value Correlation Matrix (Top 5 Features) - All Datasets",
           x = "Feature 1",
           y = "Feature 2") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
            axis.text.x = element_text(angle = 45, hjust = 1))
    
    ggsave("Part4_Results/SHAP_Correlation_Matrix_Three_Sets.png", p_interaction, width = 14, height = 6, dpi = 300)
    
    # 保存交互矩阵
    writeData(wb4, "Feature_Interactions", 
              cbind(Feature = rownames(interaction_train), 
                    as.data.frame(interaction_train),
                    as.data.frame(interaction_test),
                    as.data.frame(interaction_external)), 
              startRow = 1)
  }
}, error = function(e) {
  cat("SHAP interaction calculation failed:", e$message, "\n")
})

# ==================== Save Results ====================
writeData(wb4, "SHAP_Summary", shap_summary_combined, startRow = 1)

# 保存SHAP一致性分析结果
writeData(wb4, "SHAP_Consistency_Analysis", rank_correlations, startRow = 1)

# 保存SHAP值（前1000行以避免文件过大）
if(nrow(shap_data_train) > 0) {
  writeData(wb4, "SHAP_Values_Train", 
            cbind(ID = 1:nrow(shap_data_train), 
                  as.data.frame(shap_data_train))[1:min(1000, nrow(shap_data_train)), ], 
            startRow = 1)
}

if(nrow(shap_data_test) > 0) {
  writeData(wb4, "SHAP_Values_Test", 
            cbind(ID = 1:nrow(shap_data_test), 
                  as.data.frame(shap_data_test))[1:min(1000, nrow(shap_data_test)), ], 
            startRow = 1)
}

if(nrow(shap_data_external) > 0) {
  writeData(wb4, "SHAP_Values_External", 
            cbind(ID = 1:nrow(shap_data_external), 
                  as.data.frame(shap_data_external))[1:min(1000, nrow(shap_data_external)), ], 
            startRow = 1)
}

saveWorkbook(wb4, "Part4_SHAP_Analysis_Results_Three_Sets.xlsx", overwrite = TRUE)

# 保存SHAP对象
saveRDS(shap_values_train, "shap_values_train.rds")
saveRDS(shap_values_test, "shap_values_test.rds")
saveRDS(shap_values_external, "shap_values_external.rds")
saveRDS(shap_summary_combined, "shap_summary_combined.rds")
saveRDS(rank_correlations, "shap_consistency_analysis.rds")

cat("\n=== Part 4 Complete ===\n")
cat("Results saved to:\n")
cat("- Part4_SHAP_Analysis_Results_Three_Sets.xlsx\n")
cat("- Part4_Results/ directory\n")
cat("- shap_values_train.rds, shap_values_test.rds, shap_values_external.rds\n")
cat("- shap_summary_combined.rds, shap_consistency_analysis.rds\n")

cat("\nTop 10 Features by Overall SHAP Importance:\n")
for(i in 1:min(10, nrow(shap_summary_combined))) {
  cat(i, ".", shap_summary_combined$Feature[i], 
      "(Train:", round(shap_summary_combined$Mean_SHAP_Train[i], 4),
      "Test:", round(shap_summary_combined$Mean_SHAP_Test[i], 4),
      "External:", round(shap_summary_combined$Mean_SHAP_External[i], 4), ")\n")
}

cat("\nSHAP Ranking Consistency (Spearman Correlation):\n")
for(i in 1:nrow(rank_correlations)) {
  cat(rank_correlations$Comparison[i], ":", round(rank_correlations$Spearman_Correlation[i], 4), "\n")
}
cat("\nSHAP Analysis includes:\n")
cat("- Feature importance comparison across three datasets\n")
cat("- Summary plots (bee swarm) for all datasets\n")
cat("- Combined dependence plots for top features\n")
cat("- SHAP ranking consistency analysis\n")
cat("- Individual case explanations (force plots) for all datasets\n")
cat("- Waterfall plots for detailed explanations\n")
cat("- SHAP value distribution across datasets\n")
cat("- Feature correlation analysis for all datasets\n")