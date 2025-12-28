# Part 1: Feature Selection and Model Training on Training Set
# Load required packages
library(readxl)
library(caret)
library(glmnet)
library(e1071)  # 用于NaiveBayes模型
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

# Read training data
cat("Loading training data...\n")
train_data <- read_excel("C:/Users/Bo Wang/Desktop/Training Cohort.xlsx")
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

# ==================== Step 3: LASSO Feature Selection (筛选9个变量) ====================
cat("Step 3: LASSO Feature Selection (筛选9个变量)...\n")

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
    # 使用交叉验证找到最佳lambda
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
      # 找到选择9个变量的lambda值
      # 首先拟合完整的LASSO路径
      lasso_full <- glmnet(
        x = x_lasso, 
        y = y_lasso, 
        alpha = 1, 
        family = "binomial"
      )
      
      # 获取每个lambda选择的变量数量
      lasso_coefs <- coef(lasso_full)
      num_features_selected <- colSums(lasso_coefs != 0) - 1  # 减去截距
      
      # 找到选择9个变量的lambda索引
      lambda_idx <- which(num_features_selected == 9)[1]
      
      if(!is.na(lambda_idx)) {
        # 找到选择9个变量的lambda
        lambda_9feat <- lasso_full$lambda[lambda_idx]
        lasso_model_9feat <- glmnet(
          x = x_lasso, 
          y = y_lasso, 
          alpha = 1, 
          lambda = lambda_9feat,
          family = "binomial"
        )
        
        lasso_coef_matrix <- as.matrix(coef(lasso_model_9feat))
        lasso_coef_no_intercept <- lasso_coef_matrix[-1, , drop = FALSE]
        selected_features_lasso <- rownames(lasso_coef_no_intercept)[lasso_coef_no_intercept != 0]
        
        cat("LASSO selected 9 features at lambda =", round(lambda_9feat, 6), "\n")
      } else {
        # 如果没有正好选择9个变量的lambda，找到最接近的
        cat("No lambda selects exactly 9 features, finding closest match...\n")
        # 找到选择变量数量最接近9的lambda
        diff_features <- abs(num_features_selected - 9)
        closest_idx <- which.min(diff_features)
        lambda_closest <- lasso_full$lambda[closest_idx]
        num_selected <- num_features_selected[closest_idx]
        
        lasso_model_closest <- glmnet(
          x = x_lasso, 
          y = y_lasso, 
          alpha = 1, 
          lambda = lambda_closest,
          family = "binomial"
        )
        
        lasso_coef_matrix <- as.matrix(coef(lasso_model_closest))
        lasso_coef_no_intercept <- lasso_coef_matrix[-1, , drop = FALSE]
        selected_features_lasso <- rownames(lasso_coef_no_intercept)[lasso_coef_no_intercept != 0]
        
        cat("LASSO selected", num_selected, "features (closest to 9) at lambda =", round(lambda_closest, 6), "\n")
      }
      
      # 如果LASSO没有选择任何特征，使用单变量分析结果
      if(length(selected_features_lasso) == 0) {
        cat("Warning: LASSO did not select any features, using top 9 univariate features...\n")
        # 取单变量分析中P值最小的9个特征
        if(length(selected_features_univariate) >= 9) {
          selected_features_lasso <- head(selected_features_univariate, 9)
        } else {
          selected_features_lasso <- selected_features_univariate
        }
      }
      
      final_features <- selected_features_lasso
    } else {
      cat("Using top 9 univariate features due to LASSO failure\n")
      # 取单变量分析中P值最小的9个特征
      if(length(selected_features_univariate) >= 9) {
        final_features <- head(selected_features_univariate, 9)
      } else {
        final_features <- selected_features_univariate
      }
    }
  }
}

# 确保最终选择9个特征（如果可用）
if(length(final_features) > 9) {
  cat("Selected more than 9 features, keeping top 9 by LASSO coefficients...\n")
  # 重新运行LASSO获取系数绝对值
  x_lasso_final <- as.matrix(train_data_imputed[, final_features])
  lasso_model_final <- glmnet(
    x = x_lasso_final, 
    y = y_lasso, 
    alpha = 1, 
    family = "binomial"
  )
  
  # 使用一个较小的lambda以保留更多特征，然后选择系数绝对值最大的9个
  lambda_small <- lasso_model_final$lambda[length(lasso_model_final$lambda) * 0.3]  # 选择较小的lambda
  lasso_coef_small <- as.matrix(coef(glmnet(
    x = x_lasso_final, 
    y = y_lasso, 
    alpha = 1, 
    lambda = lambda_small,
    family = "binomial"
  )))
  
  lasso_coef_no_intercept <- lasso_coef_small[-1, , drop = FALSE]
  coef_abs <- abs(lasso_coef_no_intercept)
  top9_idx <- order(coef_abs, decreasing = TRUE)[1:min(9, length(coef_abs))]
  final_features <- rownames(lasso_coef_no_intercept)[top9_idx]
} else if(length(final_features) < 9 && length(selected_features_univariate) >= 9) {
  cat("Selected less than 9 features, supplementing with top univariate features...\n")
  # 补充单变量分析中P值最小的特征直到有9个
  top_univariate <- head(selected_features_univariate, 9)
  final_features <- unique(c(final_features, top_univariate))[1:9]
}

cat("Final selected features (target: 9):", length(final_features), "\n")
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

# ==================== Step 5: Train NaiveBayes Model ====================
cat("Step 5: Training NaiveBayes Model...\n")

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

# 训练NaiveBayes模型 - 修正版本
tryCatch({
  # 方法1: 使用e1071包的naiveBayes函数
  final_nb_model <- naiveBayes(
    x = train_features,
    y = train_labels
  )
  model_type <- "e1071"
  cat("NaiveBayes model trained successfully using e1071 package.\n")
}, error = function(e) {
  cat("Error with e1071 naiveBayes:", e$message, "\n")
  cat("Trying klaR package...\n")
  
  # 方法2: 如果e1071失败，尝试klaR包
  tryCatch({
    final_nb_model <- NaiveBayes(
      x = train_features,
      grouping = train_labels,
      usekernel = TRUE  # 使用核密度估计处理连续变量
    )
    model_type <- "klaR"
    cat("NaiveBayes model trained successfully using klaR package.\n")
  }, error = function(e2) {
    stop("Both e1071 and klaR packages failed to train NaiveBayes model: ", e2$message)
  })
})

# 保存模型类型信息
final_nb_model$model_type <- model_type

# ==================== Step 6: Feature Importance for NaiveBayes ====================
cat("Step 6: Calculating Feature Importance for NaiveBayes...\n")

# 对于NaiveBayes模型，我们可以使用基于排列的重要性或基于信息增益的重要性
# 这里我们使用基于信息增益的方法

calculate_feature_importance_nb <- function(features, target) {
  importance_scores <- sapply(names(features), function(feature_name) {
    if(is.numeric(features[[feature_name]])) {
      # 对于数值特征，使用方差分析
      anova_result <- summary(aov(features[[feature_name]] ~ target))
      f_value <- anova_result[[1]]$"F value"[1]
      if(is.na(f_value)) f_value <- 0
      return(f_value)
    } else {
      # 对于分类特征，使用卡方检验
      chi_result <- chisq.test(features[[feature_name]], target)
      return(-log10(chi_result$p.value))  # 使用p值的负对数
    }
  })
  
  # 归一化重要性分数
  if(max(importance_scores) > 0) {
    importance_scores <- importance_scores / max(importance_scores)
  }
  
  return(importance_scores)
}

feature_importance_scores <- calculate_feature_importance_nb(train_features, train_labels)
feature_importance_df <- data.frame(
  Feature = names(feature_importance_scores),
  Importance = feature_importance_scores
)
feature_importance_df <- feature_importance_df[order(-feature_importance_df$Importance), ]

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
    labs(title = "Top 20 Feature Importance - NaiveBayes Model",
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

# 4. LASSO系数路径图
if(length(selected_features_univariate) >= 9) {
  # 重新运行LASSO以绘制系数路径
  x_lasso_plot <- as.matrix(train_data_imputed[, selected_features_univariate])
  lasso_plot_model <- glmnet(
    x = x_lasso_plot, 
    y = y_lasso, 
    alpha = 1, 
    family = "binomial"
  )
  
  # 保存系数路径图
  png("Part1_Results/LASSO_Coefficient_Path.png", width = 10, height = 8, units = "in", res = 300)
  plot(lasso_plot_model, xvar = "lambda", main = "LASSO Coefficient Path")
  if(exists("lambda_9feat")) {
    abline(v = log(lambda_9feat), col = "red", lty = 2, lwd = 2)
    legend("topright", legend = paste("9 features (λ =", round(lambda_9feat, 4), ")"), 
           col = "red", lty = 2, lwd = 2, bty = "n")
  }
  dev.off()
}

# ==================== Step 8: Save Results ====================
writeData(wb1, "Univariate_Analysis", univariate_df, startRow = 1)
writeData(wb1, "Feature_Importance", feature_importance_df, startRow = 1)
writeData(wb1, "Selected_Features", data.frame(Features = final_features), startRow = 1)

saveWorkbook(wb1, "Part1_Feature_Selection_Results.xlsx", overwrite = TRUE)

# Save model and important objects for next parts
saveRDS(final_nb_model, "final_nb_model.rds")
saveRDS(final_features, "final_features.rds")
saveRDS(train_data_balanced, "train_data_balanced.rds")
saveRDS(train_data_imputed, "train_data_imputed.rds")  # 保存原始插补数据供参考

cat("\n=== Part 1 Complete ===\n")
cat("Results saved to:\n")
cat("- Part1_Feature_Selection_Results.xlsx\n")
cat("- Part1_Results/ directory\n")
cat("- final_nb_model.rds\n")
cat("- final_features.rds\n")
cat("- train_data_balanced.rds\n")
cat("- train_data_imputed.rds\n")

cat("\n=== Feature Selection Summary ===\n")
cat("Initial features:", length(all_features), "\n")
cat("After univariate analysis:", length(selected_features_univariate), "\n")
cat("After LASSO selection (target: 9):", length(final_features), "\n")

cat("\nTop 9 Features by Importance:\n")
for(i in 1:min(9, nrow(top_20_features))) {
  cat(i, ".", top_20_features$Feature[i], "(Importance:", round(top_20_features$Importance[i], 4), ")\n")
}

cat("\n=== Data Processing Summary ===\n")
cat("Original training samples:", nrow(train_data), "\n")
cat("After imputation:", nrow(train_data_imputed), "\n")
cat("After balancing:", nrow(train_data_balanced), "\n")
cat("Class distribution in balanced data:\n")
print(table(train_data_balanced$Diagnosis))

cat("\n=== Model Information ===\n")
cat("Model type: NaiveBayes\n")
cat("Package used:", model_type, "\n")
cat("Number of features used:", length(final_features), "\n")

cat("\n=== Note ===\n")
cat("Model performance will be evaluated in Part 2 (10-fold cross-validation)\n")
cat("and Part 3 (test set validation). Training set performance is not reported\n")
cat("here to avoid data leakage and over-optimistic estimates.\n")