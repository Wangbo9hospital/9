# Part 1: Feature Selection and Model Training on Training Set
# Load required packages
library(readxl)
library(caret)
library(glmnet)
library(e1071)
library(mice)
library(pROC)
library(openxlsx)
library(ggplot2)
library(boot)

# Check and install missing packages if needed
required_packages <- c("smotefamily", "randomForest", "klaR")
for(pkg in required_packages) {
  if(!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# 改进的列名清理函数 - 确保唯一性
clean_column_names <- function(df) {
  original_names <- names(df)
  
  # 第一步：基本清理
  clean_names <- gsub("[^a-zA-Z0-9_]", "_", original_names)  # 将非字母数字字符替换为下划线
  clean_names <- gsub("_{2,}", "_", clean_names)  # 将多个连续下划线替换为单个
  clean_names <- gsub("_$", "", clean_names)  # 移除末尾的下划线
  
  # 第二步：确保名称唯一性
  make_unique_names <- function(names) {
    new_names <- character(length(names))
    name_count <- list()
    
    for(i in seq_along(names)) {
      name <- names[i]
      if(is.null(name_count[[name]])) {
        name_count[[name]] <- 1
        new_names[i] <- name
      } else {
        name_count[[name]] <- name_count[[name]] + 1
        new_names[i] <- paste0(name, "_", name_count[[name]])
      }
    }
    return(new_names)
  }
  
  clean_names <- make_unique_names(clean_names)
  
  # 记录名称映射
  name_mapping <- data.frame(
    Original = original_names,
    Clean = clean_names
  )
  
  # 应用新名称
  names(df) <- clean_names
  
  return(list(data = df, mapping = name_mapping))
}

# Read training data
cat("Loading training data...\n")
train_data <- read_excel("C:/Users/Bo Wang/Desktop/Training Cohort.xlsx")

# 清理列名并获取映射
cleaned_result <- clean_column_names(train_data)
train_data <- cleaned_result$data
name_mapping <- cleaned_result$mapping

# 打印名称映射以便检查
cat("Column name mapping:\n")
print(name_mapping)

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
addWorksheet(wb1, "Name_Mapping")

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
  
  # 准备SMOTE数据 - 确保数据类型正确
  smote_data_prep <- train_data_imputed[, c(final_features, "Diagnosis")]
  
  # 确保所有特征都是数值型
  for(feature in final_features) {
    if(!is.numeric(smote_data_prep[[feature]])) {
      smote_data_prep[[feature]] <- as.numeric(as.factor(smote_data_prep[[feature]]))
    }
  }
  
  # 确保没有缺失值
  if(any(is.na(smote_data_prep))) {
    cat("Warning: NA values in SMOTE data, using original data...\n")
    train_data_balanced <- train_data_imputed
  } else {
    # 使用更安全的SMOTE调用方式
    tryCatch({
      smote_data <- SMOTE(
        X = smote_data_prep[, final_features],
        target = smote_data_prep$Diagnosis,
        K = 5,
        dup_size = 0
      )
      
      if(!is.null(smote_data$data)) {
        train_data_balanced <- smote_data$data
        # 重命名class列为Diagnosis
        if("class" %in% names(train_data_balanced)) {
          names(train_data_balanced)[names(train_data_balanced) == "class"] <- "Diagnosis"
        }
        train_data_balanced$Diagnosis <- as.factor(train_data_balanced$Diagnosis)
        cat("SMOTE applied successfully.\n")
      } else {
        cat("SMOTE returned NULL data, using original data...\n")
        train_data_balanced <- train_data_imputed
      }
    }, error = function(e) {
      cat("SMOTE error:", e$message, "\n")
      cat("Using original data instead of SMOTE...\n")
      train_data_balanced <- train_data_imputed
    })
  }
} else {
  train_data_balanced <- train_data_imputed
}

cat("Balanced data dimensions:", dim(train_data_balanced), "\n")

# ==================== Step 5: Train GLM Model ====================
cat("Step 5: Training GLM Model...\n")

# 确保最终特征在平衡数据中都存在
missing_in_balanced <- setdiff(final_features, names(train_data_balanced))
if(length(missing_in_balanced) > 0) {
  cat("Warning: Missing features in balanced data:", paste(missing_in_balanced, collapse = ", "), "\n")
  final_features <- intersect(final_features, names(train_data_balanced))
}

# 准备训练数据
train_features <- train_data_balanced[, final_features, drop = FALSE]
train_labels <- train_data_balanced$Diagnosis

# 检查数据有效性
if(ncol(train_features) == 0) {
  stop("No features available for model training!")
}

# 准备GLM公式 - 使用清理后的列名
glm_formula <- as.formula(paste("Diagnosis ~", paste(final_features, collapse = " + ")))

# 训练GLM模型
tryCatch({
  final_glm_model <- glm(
    formula = glm_formula,
    data = train_data_balanced,
    family = binomial(link = "logit")  # 二分类逻辑回归
  )
  cat("GLM model trained successfully.\n")
}, error = function(e) {
  stop("Failed to train GLM model: ", e$message)
})

# ==================== Step 6: Feature Importance for GLM ====================
cat("Step 6: Calculating Feature Importance for GLM...\n")

# 对于GLM模型，我们可以使用系数的绝对值作为重要性指标
calculate_feature_importance_glm <- function(glm_model) {
  # 提取系数（排除截距项）
  coefficients <- coef(glm_model)[-1]
  
  # 使用系数的绝对值作为重要性
  importance_scores <- abs(coefficients)
  
  # 归一化重要性分数
  if(max(importance_scores) > 0) {
    importance_scores <- importance_scores / max(importance_scores)
  }
  
  # 创建数据框
  feature_importance_df <- data.frame(
    Feature = names(importance_scores),
    Importance = importance_scores
  )
  
  # 按重要性排序
  feature_importance_df <- feature_importance_df[order(-feature_importance_df$Importance), ]
  
  return(feature_importance_df)
}

feature_importance_df <- calculate_feature_importance_glm(final_glm_model)

# Save top 20 features
top_20_features <- head(feature_importance_df, 20)

# ==================== Step 7: Generate Visualizations ====================
cat("Step 7: Generating Visualizations...\n")
if(!dir.exists("Part1_Results")) dir.create("Part1_Results")

# 1. Top 20 Feature Importance Plot
if(nrow(top_20_features) > 0) {
  p1 <- ggplot(top_20_features, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "steelblue", alpha = 0.8) +
    coord_flip() +
    labs(title = "Top 20 Feature Importance - GLM Model",
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

# 4. GLM系数可视化
if(nrow(feature_importance_df) > 0) {
  # 取前15个特征显示系数
  top_15_coef <- head(feature_importance_df, 15)
  
  # 获取实际系数值（非绝对值）
  coef_values <- coef(final_glm_model)[top_15_coef$Feature]
  top_15_coef$Coefficient <- coef_values
  top_15_coef$Effect <- ifelse(coef_values > 0, "Positive", "Negative")
  
  p4 <- ggplot(top_15_coef, aes(x = reorder(Feature, abs(Coefficient)), y = Coefficient, fill = Effect)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    coord_flip() +
    labs(title = "Top 15 Feature Coefficients - GLM Model",
         x = "Features",
         y = "Coefficient Value") +
    scale_fill_manual(values = c("Positive" = "red", "Negative" = "blue")) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          axis.text = element_text(size = 10),
          axis.title = element_text(size = 12))
  
  ggsave("Part1_Results/GLM_Coefficients.png", p4, width = 12, height = 8, dpi = 300)
}

# ==================== Step 8: Save Results ====================
writeData(wb1, "Univariate_Analysis", univariate_df, startRow = 1)
writeData(wb1, "Feature_Importance", feature_importance_df, startRow = 1)
writeData(wb1, "Selected_Features", data.frame(Features = final_features), startRow = 1)
writeData(wb1, "Name_Mapping", name_mapping, startRow = 1)

saveWorkbook(wb1, "Part1_Feature_Selection_Results.xlsx", overwrite = TRUE)

# Save model and important objects for next parts
saveRDS(final_glm_model, "final_glm_model.rds")
saveRDS(final_features, "final_features.rds")
saveRDS(train_data_balanced, "train_data_balanced.rds")
saveRDS(train_data_imputed, "train_data_imputed.rds")  # 保存原始插补数据供参考
saveRDS(name_mapping, "name_mapping.rds")  # 保存名称映射

cat("\n=== Part 1 Complete ===\n")
cat("Results saved to:\n")
cat("- Part1_Feature_Selection_Results.xlsx\n")
cat("- Part1_Results/ directory\n")
cat("- final_glm_model.rds\n")
cat("- final_features.rds\n")
cat("- train_data_balanced.rds\n")
cat("- train_data_imputed.rds\n")
cat("- name_mapping.rds\n")

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

cat("\n=== Model Information ===\n")
cat("Model type: Generalized Linear Model (Logistic Regression)\n")
cat("Family: binomial\n")
cat("Link function: logit\n")
cat("Number of features used:", length(final_features), "\n")
cat("AIC:", round(AIC(final_glm_model), 2), "\n")

# 计算伪R方
null_deviance <- final_glm_model$null.deviance
residual_deviance <- final_glm_model$deviance
pseudo_r_squared <- 1 - (residual_deviance / null_deviance)
cat("Pseudo R-squared:", round(pseudo_r_squared, 4), "\n")

cat("\n=== Note ===\n")
cat("Model performance will be evaluated in Part 2 (10-fold cross-validation)\n")
cat("and Part 3 (test set validation). Training set performance is not reported\n")
cat("here to avoid data leakage and over-optimistic estimates.\n")