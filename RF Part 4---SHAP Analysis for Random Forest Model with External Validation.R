# Part 4: SHAP Analysis for Random Forest Model with External Validation
# Load required packages
library(readxl)
library(randomForest)
library(fastshap)  # 用于随机森林的SHAP分析
library(ggplot2)
library(openxlsx)
library(viridis)
library(ggforce)
library(reshape2)  # 添加reshape2包

# Load objects from previous parts
final_rf_model <- readRDS("final_rf_model.rds")
final_features <- readRDS("final_features.rds")
train_data_balanced <- readRDS("train_data_balanced.rds")
test_data <- read_excel("C:/Users/Bo Wang/Desktop/Internal Validation Cohort.xlsx")
test_data$Diagnosis <- as.factor(test_data$Diagnosis)

# ==================== Load External Validation Set ====================
cat("Loading external validation set...\n")
external_data <- read_excel("C:/Users/Bo Wang/Desktop/External Validation Cohort.xlsx")
external_data$Diagnosis <- as.factor(external_data$Diagnosis)
cat("External validation set dimensions:", dim(external_data), "\n")
cat("External validation set Diagnosis distribution:\n")
print(table(external_data$Diagnosis))

# Create workbook for results
wb4 <- createWorkbook()
addWorksheet(wb4, "SHAP_Summary")
addWorksheet(wb4, "SHAP_Values_Train")
addWorksheet(wb4, "SHAP_Values_Test")
addWorksheet(wb4, "SHAP_Values_External")
addWorksheet(wb4, "Feature_Interactions")

# ==================== SHAP Analysis ====================
cat("=== Part 4: SHAP Analysis for Random Forest ===\n")

# 检查特征一致性
cat("Checking feature consistency...\n")
cat("Final features from model:", length(final_features), "\n")
cat("Features in balanced training data:", sum(final_features %in% names(train_data_balanced)), "\n")

# 确保所有最终特征都在训练数据中
available_features <- intersect(final_features, names(train_data_balanced))
if(length(available_features) < length(final_features)) {
  cat("Warning: Some features missing from training data. Using available features only.\n")
  cat("Missing features:", setdiff(final_features, names(train_data_balanced)), "\n")
  final_features <- available_features
}

# Prepare data
train_features <- train_data_balanced[, final_features, drop = FALSE]
train_labels <- as.numeric(train_data_balanced$Diagnosis) - 1

# Prepare test data (using same imputation as Part 3)
test_features_all <- test_data[, !names(test_data) %in% "Diagnosis"]
external_features_all <- external_data[, !names(external_data) %in% "Diagnosis"]

# Simple imputation function
impute_data <- function(data, train_data_balanced) {
  for(feature in names(data)) {
    if(any(is.na(data[[feature]]))) {
      if(is.numeric(data[[feature]])) {
        mean_val <- mean(train_data_balanced[[feature]], na.rm = TRUE)
        data[[feature]][is.na(data[[feature]])] <- mean_val
      } else {
        mode_val <- names(sort(table(train_data_balanced[[feature]]), decreasing = TRUE))[1]
        data[[feature]][is.na(data[[feature]])] <- mode_val
      }
    }
  }
  
  # Ensure all features are present
  missing_features <- setdiff(final_features, names(data))
  if(length(missing_features) > 0) {
    cat("Adding missing features:", paste(missing_features, collapse = ", "), "\n")
    for(feature in missing_features) {
      if(is.numeric(train_data_balanced[[feature]])) {
        data[[feature]] <- mean(train_data_balanced[[feature]], na.rm = TRUE)
      } else {
        mode_val <- names(sort(table(train_data_balanced[[feature]]), decreasing = TRUE))[1]
        data[[feature]] <- mode_val
      }
    }
  }
  
  return(data[, final_features, drop = FALSE])
}

# Apply imputation
test_features <- impute_data(test_features_all, train_data_balanced)
external_features <- impute_data(external_features_all, train_data_balanced)

# 检查数据维度
cat("Training data dimensions:", dim(train_features), "\n")
cat("Test data dimensions:", dim(test_features), "\n")
cat("External validation data dimensions:", dim(external_features), "\n")

# ==================== Calculate SHAP Values ====================
cat("Calculating SHAP values for Random Forest...\n")

# 定义预测函数用于SHAP计算
pfun <- function(object, newdata) {
  predict(object, newdata, type = "prob")[, 2]
}

# 计算SHAP值（使用子集以提高计算效率）
set.seed(123)
# 使用相同样本数量以避免维度不一致问题
sample_size <- min(40, nrow(train_features), nrow(test_features), nrow(external_features))
cat("Using sample size:", sample_size, "for all datasets\n")

train_sample_indices <- sample(1:nrow(train_features), sample_size)
test_sample_indices <- sample(1:nrow(test_features), sample_size)
external_sample_indices <- sample(1:nrow(external_features), sample_size)

train_features_sample <- train_features[train_sample_indices, , drop = FALSE]
test_features_sample <- test_features[test_sample_indices, , drop = FALSE]
external_features_sample <- external_features[external_sample_indices, , drop = FALSE]

# 确保数据格式正确
cat("Sample training data dimensions:", dim(train_features_sample), "\n")
cat("Sample test data dimensions:", dim(test_features_sample), "\n")
cat("Sample external validation data dimensions:", dim(external_features_sample), "\n")

# 计算SHAP值
cat("Computing SHAP values for training set...\n")
shap_values_train <- tryCatch({
  fastshap::explain(
    final_rf_model,
    X = train_features_sample,
    pred_wrapper = pfun,
    nsim = 50  # 减少模拟次数以提高速度
  )
}, error = function(e) {
  cat("Error computing SHAP values for training set:", e$message, "\n")
  return(NULL)
})

cat("Computing SHAP values for test set...\n")
shap_values_test <- tryCatch({
  fastshap::explain(
    final_rf_model,
    X = test_features_sample,
    pred_wrapper = pfun,
    nsim = 50
  )
}, error = function(e) {
  cat("Error computing SHAP values for test set:", e$message, "\n")
  return(NULL)
})

cat("Computing SHAP values for external validation set...\n")
shap_values_external <- tryCatch({
  fastshap::explain(
    final_rf_model,
    X = external_features_sample,
    pred_wrapper = pfun,
    nsim = 50
  )
}, error = function(e) {
  cat("Error computing SHAP values for external validation set:", e$message, "\n")
  return(NULL)
})

# 检查SHAP值计算是否成功
if(is.null(shap_values_train) || is.null(shap_values_test) || is.null(shap_values_external)) {
  stop("SHAP value calculation failed. Cannot proceed with analysis.")
}

# 确保SHAP值是数据框格式
shap_values_train <- as.data.frame(shap_values_train)
shap_values_test <- as.data.frame(shap_values_test)
shap_values_external <- as.data.frame(shap_values_external)

# 设置正确的列名
colnames(shap_values_train) <- final_features
colnames(shap_values_test) <- final_features
colnames(shap_values_external) <- final_features

# 计算平均SHAP值（使用训练集作为基准）
shap_summary <- data.frame(
  Feature = colnames(shap_values_train),
  Mean_SHAP_Train = colMeans(abs(shap_values_train)),
  Mean_SHAP_Test = colMeans(abs(shap_values_test)),
  Mean_SHAP_External = colMeans(abs(shap_values_external)),
  stringsAsFactors = FALSE
)

shap_summary <- shap_summary[order(-shap_summary$Mean_SHAP_Train), ]

# ==================== Generate SHAP Visualizations ====================
cat("Generating SHAP visualizations...\n")
if(!dir.exists("Part4_Results")) dir.create("Part4_Results")

# 辅助函数：创建安全的文件名
create_safe_filename <- function(base_name, extension = "png") {
  # 移除或替换不安全的字符
  safe_name <- gsub("[^a-zA-Z0-9._-]", "_", base_name)
  safe_name <- gsub("_{2,}", "_", safe_name)  # 替换多个下划线为单个
  safe_name <- gsub("^_|_$", "", safe_name)   # 移除开头和结尾的下划线
  paste0(safe_name, ".", extension)
}

# 1. SHAP Feature Importance Comparison (Top 20)
top_20_shap <- head(shap_summary, 20)

# 转换为长格式用于绘图
shap_comparison_long <- reshape2::melt(
  top_20_shap, 
  id.vars = "Feature",
  measure.vars = c("Mean_SHAP_Train", "Mean_SHAP_Test", "Mean_SHAP_External"),
  variable.name = "Dataset",
  value.name = "Mean_SHAP"
)

# 清理数据集名称
shap_comparison_long$Dataset <- gsub("Mean_SHAP_", "", shap_comparison_long$Dataset)
shap_comparison_long$Dataset <- factor(shap_comparison_long$Dataset, 
                                       levels = c("Train", "Test", "External"))

p1 <- ggplot(shap_comparison_long, aes(x = reorder(Feature, Mean_SHAP), y = Mean_SHAP, fill = Dataset)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  coord_flip() +
  labs(title = "SHAP Feature Importance Comparison - Random Forest (Top 20)",
       x = "Features",
       y = "Mean |SHAP| Value") +
  scale_fill_manual(values = c("Train" = "darkred", "Test" = "darkblue", "External" = "darkgreen")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        axis.text = element_text(size = 10),
        axis.title = element_text(size = 12),
        legend.position = "top")

# 使用安全的文件名保存
safe_filename <- create_safe_filename("SHAP_Feature_Importance_Comparison")
ggsave(file.path("Part4_Results", safe_filename), p1, width = 14, height = 8, dpi = 300)

# 2. SHAP Summary Plot for All Datasets - 修复版本
cat("Creating SHAP summary plot for all datasets...\n")
top_15_features <- head(shap_summary$Feature, 15)

# 准备SHAP摘要图数据 - 修复版本
shap_long_combined <- data.frame()

for(feature in top_15_features) {
  # 训练集数据
  if(feature %in% colnames(shap_values_train) && feature %in% colnames(train_features_sample)) {
    feature_values <- train_features_sample[[feature]]
    
    # 确保特征值是数值型
    if(!is.numeric(feature_values)) {
      feature_values <- as.numeric(as.character(feature_values))
      # 如果转换失败，使用随机颜色
      if(any(is.na(feature_values))) {
        feature_values <- runif(length(feature_values))
        cat("Warning: Feature", feature, "contains non-numeric values. Using random values for coloring.\n")
      }
    }
    
    temp_train <- data.frame(
      Feature = feature,
      SHAP_Value = shap_values_train[[feature]],
      Feature_Value = feature_values,
      Dataset = "Training"
    )
    shap_long_combined <- rbind(shap_long_combined, temp_train)
  }
  
  # 测试集数据
  if(feature %in% colnames(shap_values_test) && feature %in% colnames(test_features_sample)) {
    feature_values <- test_features_sample[[feature]]
    
    # 确保特征值是数值型
    if(!is.numeric(feature_values)) {
      feature_values <- as.numeric(as.character(feature_values))
      # 如果转换失败，使用随机颜色
      if(any(is.na(feature_values))) {
        feature_values <- runif(length(feature_values))
        cat("Warning: Feature", feature, "contains non-numeric values. Using random values for coloring.\n")
      }
    }
    
    temp_test <- data.frame(
      Feature = feature,
      SHAP_Value = shap_values_test[[feature]],
      Feature_Value = feature_values,
      Dataset = "Test"
    )
    shap_long_combined <- rbind(shap_long_combined, temp_test)
  }
  
  # 外部验证集数据
  if(feature %in% colnames(shap_values_external) && feature %in% colnames(external_features_sample)) {
    feature_values <- external_features_sample[[feature]]
    
    # 确保特征值是数值型
    if(!is.numeric(feature_values)) {
      feature_values <- as.numeric(as.character(feature_values))
      # 如果转换失败，使用随机颜色
      if(any(is.na(feature_values))) {
        feature_values <- runif(length(feature_values))
        cat("Warning: Feature", feature, "contains non-numeric values. Using random values for coloring.\n")
      }
    }
    
    temp_external <- data.frame(
      Feature = feature,
      SHAP_Value = shap_values_external[[feature]],
      Feature_Value = feature_values,
      Dataset = "External"
    )
    shap_long_combined <- rbind(shap_long_combined, temp_external)
  }
}

# 计算每个特征的绝对SHAP均值用于排序
feature_importance <- aggregate(abs(SHAP_Value) ~ Feature, data = shap_long_combined, FUN = mean)
feature_order <- feature_importance[order(feature_importance$`abs(SHAP_Value)`), "Feature"]

shap_long_combined$Feature <- factor(shap_long_combined$Feature, levels = feature_order)
shap_long_combined$Dataset <- factor(shap_long_combined$Dataset, 
                                     levels = c("Training", "Test", "External"))

# 创建SHAP摘要图 - 修复版本
p2 <- ggplot(shap_long_combined, aes(x = SHAP_Value, y = Feature, color = Feature_Value)) +
  geom_point(alpha = 0.6, size = 1) +
  scale_color_viridis_c(name = "Feature Value", na.value = "grey50") +
  facet_wrap(~ Dataset, ncol = 3) +
  labs(title = "SHAP Summary Plot - Random Forest (Training vs Test vs External Sets)",
       x = "SHAP Value (Impact on Model Output)",
       y = "Features") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        axis.text = element_text(size = 9),
        axis.title = element_text(size = 11),
        strip.text = element_text(size = 10, face = "bold"))

safe_filename <- create_safe_filename("SHAP_Summary_Plot_All")
ggsave(file.path("Part4_Results", safe_filename), p2, width = 16, height = 10, dpi = 300)

# 3. SHAP Dependence Plots for Top 10 Features (All Datasets)
top_10_features <- head(shap_summary$Feature, 10)

for(feature in top_10_features) {
  if(feature %in% colnames(train_features_sample)) {
    # 为所有数据集创建依赖图
    dependence_data_all <- data.frame()
    
    # Training set
    feature_values_train <- train_features_sample[[feature]]
    if(!is.numeric(feature_values_train)) {
      feature_values_train <- as.numeric(as.character(feature_values_train))
    }
    
    dependence_data_train <- data.frame(
      Feature_Value = feature_values_train,
      SHAP_Value = shap_values_train[[feature]],
      Dataset = "Training"
    )
    dependence_data_all <- rbind(dependence_data_all, dependence_data_train)
    
    # Test set
    if(feature %in% colnames(test_features_sample)) {
      feature_values_test <- test_features_sample[[feature]]
      if(!is.numeric(feature_values_test)) {
        feature_values_test <- as.numeric(as.character(feature_values_test))
      }
      
      dependence_data_test <- data.frame(
        Feature_Value = feature_values_test,
        SHAP_Value = shap_values_test[[feature]],
        Dataset = "Test"
      )
      dependence_data_all <- rbind(dependence_data_all, dependence_data_test)
    }
    
    # External set
    if(feature %in% colnames(external_features_sample)) {
      feature_values_external <- external_features_sample[[feature]]
      if(!is.numeric(feature_values_external)) {
        feature_values_external <- as.numeric(as.character(feature_values_external))
      }
      
      dependence_data_external <- data.frame(
        Feature_Value = feature_values_external,
        SHAP_Value = shap_values_external[[feature]],
        Dataset = "External"
      )
      dependence_data_all <- rbind(dependence_data_all, dependence_data_external)
    }
    
    # 创建组合依赖图
    p_dep_combined <- ggplot(dependence_data_all, aes(x = Feature_Value, y = SHAP_Value, color = Dataset)) +
      geom_point(alpha = 0.5) +
      geom_smooth(method = "loess", se = TRUE, aes(fill = Dataset)) +
      scale_color_manual(values = c("Training" = "blue", "Test" = "red", "External" = "green")) +
      scale_fill_manual(values = c("Training" = "lightblue", "Test" = "pink", "External" = "lightgreen")) +
      labs(title = paste("SHAP Dependence Plot - Random Forest:", feature),
           x = feature,
           y = "SHAP Value") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
            legend.position = "top")
    
    safe_feature_name <- create_safe_filename(paste0("SHAP_Dependence_Combined_", feature))
    ggsave(file.path("Part4_Results", safe_feature_name), 
           p_dep_combined, width = 12, height = 6, dpi = 300)
  }
}

# 4. SHAP Force Plots for External Validation Set
cat("Generating individual explanation plots for external validation set...\n")

# Select representative cases from external validation set
set.seed(123)
external_sample_indices_force <- sample(1:nrow(external_features_sample), 
                                        min(6, nrow(external_features_sample)))

for(i in seq_along(external_sample_indices_force)) {
  idx <- external_sample_indices_force[i]
  
  # Prepare data for force plot
  shap_force_values <- as.numeric(shap_values_external[idx, ])
  feature_names <- colnames(shap_values_external)
  
  shap_force_data <- data.frame(
    Feature = feature_names,
    SHAP_Value = shap_force_values,
    Feature_Value = as.numeric(external_features_sample[idx, ]),
    stringsAsFactors = FALSE
  )
  
  # Take top 10 features by absolute SHAP value for clarity
  shap_force_data <- shap_force_data[order(-abs(shap_force_data$SHAP_Value)), ]
  shap_force_data_top <- head(shap_force_data, 10)
  
  p_force <- ggplot(shap_force_data_top, aes(x = reorder(Feature, abs(SHAP_Value)), y = SHAP_Value, 
                                             fill = SHAP_Value > 0)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    coord_flip() +
    scale_fill_manual(values = c("TRUE" = "red", "FALSE" = "blue"), 
                      labels = c("Positive Impact", "Negative Impact")) +
    labs(title = paste("SHAP Force Plot - Random Forest - External Validation Case", idx),
         x = "Features",
         y = "SHAP Value",
         fill = "Impact Direction") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
          legend.position = "top")
  
  safe_filename <- create_safe_filename(paste0("SHAP_Force_Plot_External_Case_", idx))
  ggsave(file.path("Part4_Results", safe_filename), 
         p_force, width = 10, height = 6, dpi = 300)
}

# 5. SHAP Waterfall Plot for External Validation Cases
cat("Generating SHAP waterfall plots for external validation set...\n")
for(i in 1:min(3, length(external_sample_indices_force))) {
  idx <- external_sample_indices_force[i]
  
  # Prepare waterfall data
  shap_waterfall_values <- as.numeric(shap_values_external[idx, ])
  feature_names <- colnames(shap_values_external)
  
  waterfall_data <- data.frame(
    Feature = feature_names,
    SHAP_Value = shap_waterfall_values,
    stringsAsFactors = FALSE
  )
  
  # Order by absolute SHAP value and take top 15
  waterfall_data <- waterfall_data[order(-abs(waterfall_data$SHAP_Value)), ]
  waterfall_data_top <- head(waterfall_data, 15)
  
  # Calculate cumulative sum for waterfall
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
    labs(title = paste("SHAP Waterfall Plot - Random Forest - External Validation Case", idx),
         x = "Features",
         y = "Cumulative SHAP Value",
         color = "Impact") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"))
  
  safe_filename <- create_safe_filename(paste0("SHAP_Waterfall_External_Case_", idx))
  ggsave(file.path("Part4_Results", safe_filename), 
         p_waterfall, width = 10, height = 8, dpi = 300)
}

# 6. SHAP Value Distribution Comparison - 修复版本
cat("Comparing SHAP value distributions across datasets...\n")

# 修复：使用相同样本数量，避免维度不一致问题
shap_distribution_data <- data.frame(
  Dataset = c(rep("Train", nrow(shap_values_train)),
              rep("Test", nrow(shap_values_test)),
              rep("External", nrow(shap_values_external))),
  SHAP_Sum = c(rowSums(abs(shap_values_train)),
               rowSums(abs(shap_values_test)),
               rowSums(abs(shap_values_external)))
)

p_dist <- ggplot(shap_distribution_data, aes(x = Dataset, y = SHAP_Sum, fill = Dataset)) +
  geom_violin(alpha = 0.7) +
  geom_boxplot(width = 0.1, alpha = 0.8) +
  labs(title = "SHAP Value Distribution Comparison - Random Forest",
       x = "Dataset",
       y = "Sum of Absolute SHAP Values") +
  scale_fill_manual(values = c("Train" = "lightblue", "Test" = "lightcoral", "External" = "lightgreen")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

safe_filename <- create_safe_filename("SHAP_Distribution_Comparison")
ggsave(file.path("Part4_Results", safe_filename), p_dist, width = 10, height = 6, dpi = 300)

# 7. SHAP Interaction Analysis (simplified version)
cat("Calculating simplified SHAP interactions...\n")
tryCatch({
  # For computational efficiency, calculate interactions only for top 5 features
  top_5_features <- head(shap_summary$Feature, 5)
  
  if(length(top_5_features) >= 2) {
    # Create correlation-based interaction matrices for all datasets
    interaction_train <- cor(shap_values_train[, top_5_features])
    interaction_test <- cor(shap_values_test[, top_5_features])
    interaction_external <- cor(shap_values_external[, top_5_features])
    
    # Convert to long format for plotting
    interaction_long_train <- reshape2::melt(interaction_train)
    interaction_long_test <- reshape2::melt(interaction_test)
    interaction_long_external <- reshape2::melt(interaction_external)
    
    interaction_long_train$Dataset <- "Training"
    interaction_long_test$Dataset <- "Test"
    interaction_long_external$Dataset <- "External"
    
    interaction_long_combined <- rbind(interaction_long_train, interaction_long_test, interaction_long_external)
    colnames(interaction_long_combined) <- c("Feature1", "Feature2", "Correlation", "Dataset")
    
    p_interaction <- ggplot(interaction_long_combined, aes(x = Feature1, y = Feature2, fill = Correlation)) +
      geom_tile() +
      geom_text(aes(label = round(Correlation, 2)), color = "white", size = 3) +
      scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                           midpoint = 0, name = "Correlation") +
      facet_wrap(~ Dataset, ncol = 3) +
      labs(title = "SHAP Value Correlation Matrix - Random Forest (Top 5 Features)",
           x = "Feature 1",
           y = "Feature 2") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
            axis.text.x = element_text(angle = 45, hjust = 1))
    
    safe_filename <- create_safe_filename("SHAP_Correlation_Matrix_All")
    ggsave(file.path("Part4_Results", safe_filename), p_interaction, width = 14, height = 6, dpi = 300)
    
    # Save interaction matrices
    interaction_results <- list(
      Training = cbind(Feature = rownames(interaction_train), as.data.frame(interaction_train)),
      Test = cbind(Feature = rownames(interaction_test), as.data.frame(interaction_test)),
      External = cbind(Feature = rownames(interaction_external), as.data.frame(interaction_external))
    )
    
    # 写入Excel文件
    start_row <- 1
    for(dataset_name in names(interaction_results)) {
      writeData(wb4, "Feature_Interactions", 
                data.frame(Dataset = dataset_name), 
                startRow = start_row)
      writeData(wb4, "Feature_Interactions", 
                interaction_results[[dataset_name]], 
                startRow = start_row + 1)
      start_row <- start_row + nrow(interaction_results[[dataset_name]]) + 3
    }
  }
}, error = function(e) {
  cat("SHAP interaction calculation failed:", e$message, "\n")
})

# ==================== Save Results ====================
writeData(wb4, "SHAP_Summary", shap_summary, startRow = 1)

# Save SHAP values for all datasets
if(nrow(shap_values_train) > 0) {
  writeData(wb4, "SHAP_Values_Train", 
            cbind(ID = 1:nrow(shap_values_train), 
                  as.data.frame(shap_values_train)), 
            startRow = 1)
}

if(nrow(shap_values_test) > 0) {
  writeData(wb4, "SHAP_Values_Test", 
            cbind(ID = 1:nrow(shap_values_test), 
                  as.data.frame(shap_values_test)), 
            startRow = 1)
}

if(nrow(shap_values_external) > 0) {
  writeData(wb4, "SHAP_Values_External", 
            cbind(ID = 1:nrow(shap_values_external), 
                  as.data.frame(shap_values_external)), 
            startRow = 1)
}

# 使用安全的Excel文件名
excel_filename <- "Part4_SHAP_Analysis_Results.xlsx"
saveWorkbook(wb4, excel_filename, overwrite = TRUE)

# Save SHAP objects
saveRDS(shap_values_train, "shap_values_train.rds")
saveRDS(shap_values_test, "shap_values_test.rds")
saveRDS(shap_values_external, "shap_values_external.rds")
saveRDS(shap_summary, "shap_summary.rds")

cat("\n=== Part 4 Complete ===\n")
cat("Results saved to:\n")
cat("- Part4_SHAP_Analysis_Results.xlsx\n")
cat("- Part4_Results/ directory\n")
cat("- shap_values_train.rds\n")
cat("- shap_values_test.rds\n")
cat("- shap_values_external.rds\n")
cat("- shap_summary.rds\n")
cat("\nTop 10 Features by SHAP Importance (Training Set):\n")
for(i in 1:min(10, nrow(shap_summary))) {
  cat(i, ".", shap_summary$Feature[i], 
      "(Train:", round(shap_summary$Mean_SHAP_Train[i], 4),
      "Test:", round(shap_summary$Mean_SHAP_Test[i], 4),
      "External:", round(shap_summary$Mean_SHAP_External[i], 4), ")\n")
}
cat("\nSHAP Analysis includes:\n")
cat("- Feature importance comparison across all datasets\n")
cat("- Summary plots for training, test, and external validation sets\n")
cat("- Dependence plots for top features (combined view)\n")
cat("- Individual case explanations for external validation set\n")
cat("- Waterfall plots for external validation cases\n")
cat("- Distribution comparison across datasets\n")
cat("- Feature correlation analysis for all datasets\n")
cat("\nNote: SHAP values were computed using fastshap package with 50 simulations per observation.\n")
cat("All datasets were sampled to", sample_size, "observations each for computational efficiency and consistency.\n")