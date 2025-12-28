# Part 1: Feature Selection and Model Training on Training Set
# Load required packages
library(readxl)
library(caret)
library(glmnet)
library(xgboost)
library(mice)
library(pROC)
library(openxlsx)
library(ggplot2)
library(boot)

# Check and install missing packages if needed
required_packages <- c("smotefamily", "randomForest")
for(pkg in required_packages) {
  if(!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# Read training data
cat("Loading training data...\n")
train_data <- read_excel("C:/Users/Bo Wang/Desktop/Training Cohort.xlsx")  # 修正为训练集
train_data$Diagnosis <- as.factor(train_data$Diagnosis)

# Basic data check
cat("Data dimensions:", dim(train_data), "\n")
cat("Diagnosis distribution:\n")
print(table(train_data$Diagnosis))

# Create workbook for results
wb1 <- createWorkbook()
addWorksheet(wb1, "Univariate_Analysis")
addWorksheet(wb1, "Feature_Importance")
addWorksheet(wb1, "Selected_Features")

# ==================== Step 1: Multiple Imputation ====================
cat("=== Part 1: Feature Selection and Model Training ===\n")
cat("Step 1: Multiple Imputation on Training Set...\n")

missing_count_train <- colSums(is.na(train_data))
cat("Missing values per column:\n")
print(missing_count_train[missing_count_train > 0])

if(sum(missing_count_train) > 0) {
  cat("Total missing values in training set:", sum(missing_count_train), "\n")
  
  # 设置mice参数
  mice_args <- list(
    data = train_data,
    m = 5,
    maxit = 10,
    printFlag = FALSE,
    seed = 123
  )
  
  # 自动选择方法
  var_types <- sapply(train_data, class)
  methods <- rep("", length(var_types))
  
  for(i in 1:length(var_types)) {
    if(var_types[i] %in% c("numeric", "integer")) {
      methods[i] <- "pmm"
    } else if(var_types[i] == "factor") {
      if(length(levels(train_data[[i]])) == 2) {
        methods[i] <- "logreg"
      } else {
        methods[i] <- "polyreg"
      }
    }
  }
  
  mice_args$method <- methods
  imp_model <- do.call(mice, mice_args)
  train_data_imputed <- complete(imp_model, 1)
} else {
  train_data_imputed <- train_data
}

# 检查插补后的数据
cat("Data after imputation - dimensions:", dim(train_data_imputed), "\n")

# ==================== Step 2: Univariate Analysis ====================
cat("Step 2: Univariate Analysis...\n")
p_value_threshold <- 0.1
all_features <- names(train_data_imputed)[names(train_data_imputed) != "Diagnosis"]
p_values <- sapply(all_features, function(feature) {
  tryCatch({
    if(is.numeric(train_data_imputed[[feature]])) {
      wilcox.test(train_data_imputed[[feature]] ~ train_data_imputed$Diagnosis)$p.value
    } else {
      chisq.test(train_data_imputed[[feature]], train_data_imputed$Diagnosis)$p.value
    }
  }, error = function(e) {
    cat("Error in feature", feature, ":", e$message, "\n")
    return(1)
  })
})

univariate_df <- data.frame(
  Feature = all_features, 
  P_Value = p_values, 
  Significant = p_values < p_value_threshold
)
univariate_df <- univariate_df[order(univariate_df$P_Value), ]

selected_features_univariate <- as.character(univariate_df$Feature[univariate_df$P_Value < p_value_threshold])
if(length(selected_features_univariate) == 0) {
  cat("Warning: No features selected by univariate analysis, using all features...\n")
  selected_features_univariate <- all_features
}

cat("Features selected by univariate analysis:", length(selected_features_univariate), "\n")

# ==================== Step 3: LASSO Feature Selection ====================
cat("Step 3: LASSO Feature Selection...\n")

# 确保有足够的数据进行LASSO
if(length(selected_features_univariate) < 2) {
  cat("Warning: Too few features for LASSO, using univariate results...\n")
  final_features <- selected_features_univariate
} else {
  x_lasso <- as.matrix(train_data_imputed[, selected_features_univariate])  
  y_lasso <- as.numeric(train_data_imputed$Diagnosis) - 1
  
  # 检查数据有效性
  if(any(is.na(x_lasso)) | any(is.na(y_lasso))) {
    cat("Warning: NA values in LASSO data, using univariate results...\n")
    final_features <- selected_features_univariate
  } else {
    cv_fit <- tryCatch({
      cv.glmnet(
        x = x_lasso, 
        y = y_lasso, 
        alpha = 1, 
        family = "binomial",
        type.measure = "auc"
      )
    }, error = function(e) {
      cat("LASSO CV error:", e$message, "\n")
      return(NULL)
    })
    
    if(!is.null(cv_fit)) {
      lambda_1se <- cv_fit$lambda.1se
      lasso_model_1se <- glmnet(
        x = x_lasso, 
        y = y_lasso, 
        alpha = 1, 
        lambda = lambda_1se,
        family = "binomial"
      )
      
      lasso_coef_matrix <- as.matrix(coef(lasso_model_1se))
      lasso_coef_no_intercept <- lasso_coef_matrix[-1, , drop = FALSE]
      selected_features_lasso <- rownames(lasso_coef_no_intercept)[lasso_coef_no_intercept != 0]
      
      if(length(selected_features_lasso) == 0) {
        cat("Warning: LASSO did not select any features, using univariate analysis results...\n")
        selected_features_lasso <- selected_features_univariate
      }
      final_features <- selected_features_lasso
    } else {
      cat("Using univariate features due to LASSO failure\n")
      final_features <- selected_features_univariate
    }
  }
}

cat("Final selected features:", length(final_features), "\n")
cat("Features:", paste(final_features, collapse = ", "), "\n")

# ==================== Step 4: Handle Class Imbalance ====================
cat("Step 4: Handling Class Imbalance...\n")
class_ratio <- min(table(train_data_imputed$Diagnosis)) / nrow(train_data_imputed)
cat("Class ratio:", round(class_ratio, 3), "\n")

if(class_ratio < 0.3) {
  cat("Class imbalance detected, applying SMOTE...\n")
  
  # 准备SMOTE数据
  smote_data_prep <- train_data_imputed[, c(final_features, "Diagnosis")]
  
  # 确保没有缺失值
  if(any(is.na(smote_data_prep))) {
    cat("Warning: NA values in SMOTE data, using original data...\n")
    train_data_balanced <- train_data_imputed
  } else {
    smote_data <- SMOTE(
      X = smote_data_prep[, final_features],
      target = smote_data_prep$Diagnosis,
      K = 5
    )
    
    if(!is.null(smote_data$data)) {
      train_data_balanced <- smote_data$data
      # 重命名class列为Diagnosis
      if("class" %in% names(train_data_balanced)) {
        names(train_data_balanced)[names(train_data_balanced) == "class"] <- "Diagnosis"
      }
      train_data_balanced$Diagnosis <- as.factor(train_data_balanced$Diagnosis)
    } else {
      cat("SMOTE failed, using original data...\n")
      train_data_balanced <- train_data_imputed
    }
  }
} else {
  train_data_balanced <- train_data_imputed
}

cat("Balanced data dimensions:", dim(train_data_balanced), "\n")

# ==================== Step 5: Train XGBoost Model ====================
cat("Step 5: Training XGBoost Model...\n")

# 确保最终特征在平衡数据中都存在
missing_in_balanced <- setdiff(final_features, names(train_data_balanced))
if(length(missing_in_balanced) > 0) {
  cat("Warning: Missing features in balanced data:", paste(missing_in_balanced, collapse = ", "), "\n")
  final_features <- intersect(final_features, names(train_data_balanced))
}

# 检查并转换特征为数值类型
cat("Converting features to numeric...\n")
train_features <- train_data_balanced[, final_features]

# 将因子变量转换为数值
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
  cat("Warning: Some features are still not numeric after conversion\n")
  # 强制转换为数值
  train_features <- as.data.frame(lapply(train_features, function(x) as.numeric(as.factor(x))))
}

# 转换为矩阵
train_features <- as.matrix(train_features)
train_labels <- as.numeric(train_data_balanced$Diagnosis) - 1

# 检查数据有效性
if(ncol(train_features) == 0) {
  stop("No features available for model training!")
}

cat("Training features dimensions:", dim(train_features), "\n")
cat("Training features type:", class(train_features), "\n")
cat("Training features data type:", typeof(train_features), "\n")

# 检查是否有NA值
if(any(is.na(train_features))) {
  cat("Warning: NA values found in training features, imputing with mean...\n")
  # 用列均值填充NA值
  for(i in 1:ncol(train_features)) {
    na_indices <- is.na(train_features[, i])
    if(any(na_indices)) {
      train_features[na_indices, i] <- mean(train_features[, i], na.rm = TRUE)
    }
  }
}

# 创建DMatrix
dtrain <- xgb.DMatrix(data = train_features, label = train_labels)

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1,
  max_depth = 6,
  colsample_bytree = 0.8,
  subsample = 0.8,
  min_child_weight = 1
)

final_xgb_model <- xgb.train(
  params, 
  dtrain, 
  nrounds = 100,
  print_every_n = 10,
  verbose = 0
)
# ==================== Step 6: Feature Importance ====================
cat("Step 6: Calculating Feature Importance...\n")
importance_matrix <- xgb.importance(model = final_xgb_model, feature_names = final_features)

if(!is.null(importance_matrix)) {
  feature_importance_df <- data.frame(
    Feature = importance_matrix$Feature,
    Importance = importance_matrix$Gain
  )
  feature_importance_df <- feature_importance_df[order(-feature_importance_df$Importance), ]
  
  # Save top 20 features
  top_20_features <- head(feature_importance_df, 20)
} else {
  cat("Warning: Could not calculate feature importance\n")
  feature_importance_df <- data.frame(Feature = final_features, Importance = 1/length(final_features))
  top_20_features <- feature_importance_df
}

# ==================== Step 7: Generate Visualizations ====================
cat("Step 7: Generating Visualizations...\n")
if(!dir.exists("Part1_Results")) dir.create("Part1_Results")

# 1. Top 20 Feature Importance Plot
if(nrow(top_20_features) > 0) {
  p1 <- ggplot(top_20_features, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "steelblue", alpha = 0.8) +
    coord_flip() +
    labs(title = "Top 20 Feature Importance - XGBoost Model",
         x = "Features",
         y = "Importance Score") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          axis.text = element_text(size = 10),
          axis.title = element_text(size = 12))
  
  ggsave("Part1_Results/Top20_Feature_Importance.png", p1, width = 12, height = 8, dpi = 300)
}

# 2. Feature Selection Summary Plot
feature_selection_summary <- data.frame(
  Stage = c("Initial", "Univariate", "LASSO"),
  Count = c(length(all_features), length(selected_features_univariate), length(final_features))
)

p2 <- ggplot(feature_selection_summary, aes(x = Stage, y = Count, fill = Stage)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_text(aes(label = Count), vjust = -0.5, size = 5) +
  labs(title = "Feature Selection Process Summary",
       x = "Selection Stage",
       y = "Number of Features") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        legend.position = "none")

ggsave("Part1_Results/Feature_Selection_Summary.png", p2, width = 8, height = 6, dpi = 300)

# 3. 数据分布图（可选）
# 显示SMOTE前后的类别分布
class_distribution <- data.frame(
  Dataset = rep(c("Original", "Balanced"), each = 2),
  Class = rep(c("Control", "Case"), 2),
  Count = c(
    sum(train_data_imputed$Diagnosis == 1),  # 对照组
    sum(train_data_imputed$Diagnosis == 2),  # 实验组
    sum(train_data_balanced$Diagnosis == 1), # 平衡后对照组
    sum(train_data_balanced$Diagnosis == 2)  # 平衡后实验组
  )
)

p3 <- ggplot(class_distribution, aes(x = Dataset, y = Count, fill = Class)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  labs(title = "Class Distribution Before and After SMOTE",
       x = "Dataset",
       y = "Number of Samples") +
  scale_fill_manual(values = c("Control" = "blue", "Case" = "red")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

ggsave("Part1_Results/Class_Distribution.png", p3, width = 8, height = 6, dpi = 300)

# ==================== Step 8: Save Results ====================
writeData(wb1, "Univariate_Analysis", univariate_df, startRow = 1)
writeData(wb1, "Feature_Importance", feature_importance_df, startRow = 1)
writeData(wb1, "Selected_Features", data.frame(Features = final_features), startRow = 1)

saveWorkbook(wb1, "Part1_Feature_Selection_Results.xlsx", overwrite = TRUE)

# Save model and important objects for next parts
saveRDS(final_xgb_model, "final_xgb_model.rds")
saveRDS(final_features, "final_features.rds")
saveRDS(train_data_balanced, "train_data_balanced.rds")
saveRDS(train_data_imputed, "train_data_imputed.rds")  # 保存原始插补数据供参考

cat("\n=== Part 1 Complete ===\n")
cat("Results saved to:\n")
cat("- Part1_Feature_Selection_Results.xlsx\n")
cat("- Part1_Results/ directory\n")
cat("- final_xgb_model.rds\n")
cat("- final_features.rds\n")
cat("- train_data_balanced.rds\n")
cat("- train_data_imputed.rds\n")

cat("\n=== Feature Selection Summary ===\n")
cat("Initial features:", length(all_features), "\n")
cat("After univariate analysis:", length(selected_features_univariate), "\n")
cat("After LASSO selection:", length(final_features), "\n")

cat("\nTop 5 Features by Importance:\n")
for(i in 1:min(5, nrow(top_20_features))) {
  cat(i, ".", top_20_features$Feature[i], "(Importance:", round(top_20_features$Importance[i], 4), ")\n")
}

cat("\n=== Data Processing Summary ===\n")
cat("Original training samples:", nrow(train_data), "\n")
cat("After imputation:", nrow(train_data_imputed), "\n")
cat("After balancing:", nrow(train_data_balanced), "\n")
cat("Class distribution in balanced data:\n")
print(table(train_data_balanced$Diagnosis))

cat("\n=== Note ===\n")
cat("Model performance will be evaluated in Part 2 (10-fold cross-validation)\n")
cat("and Part 3 (test set validation). Training set performance is not reported\n")
cat("here to avoid data leakage and over-optimistic estimates.\n")