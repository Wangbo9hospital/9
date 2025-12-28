# 第二部分：模型训练与验证代码（集成多重插补和结果输出）
# 加载必要的包
library(readxl)
library(caret)
library(glmnet)
library(xgboost)
library(mice)
library(pROC)
library(openxlsx)
library(ggplot2)
library(randomForest)
library(e1071)  # SVM和Naive Bayes
library(adabag)  # Adaboost
library(rpart)   # Decision Tree
library(gbm)     # Gradient Boosting
library(class)   # KNN
library(lightgbm) # LightGBM
library(nnet)    # MLP (神经网络)
library(naivebayes) # Naive Bayes (替代方案)
library(pROC)    # ROC分析
library(boot)    # 置信区间计算

# 新增TabNet包
if (!require("tabnet")) {
  install.packages("tabnet")
  library(tabnet)
}

# 读取训练数据
train_data <- read_excel("C:/Users/Bo Wang/Desktop/DATA.xlsx")
train_data$Diagnosis <- as.factor(train_data$Diagnosis)

# 设置十折交叉验证
set.seed(123)
folds <- createFolds(train_data$Diagnosis, k = 10, returnTrain = TRUE)

# 初始化结果存储
results_all_models <- list()
feature_importance_all <- list()
roc_curves_all <- list()
best_features_per_model <- vector("list", 10)  # 预分配长度为10的列表
p_value_threshold <- 0.1  
lasso_alpha <- 1          
wb <- createWorkbook()

# 创建结果工作表
addWorksheet(wb, "单因素分析结果")
addWorksheet(wb, "特征频率统计")
addWorksheet(wb, "CV性能汇总")
addWorksheet(wb, "13模型比较结果")  # 改为13模型
addWorksheet(wb, "详细性能指标")
addWorksheet(wb, "特征重要性汇总")
addWorksheet(wb, "最佳特征比较")
addWorksheet(wb, "模型特征排序")

# 定义计算所有指标的函数（包括balanced accuracy）
calculate_all_metrics <- function(true_labels, pred_probs, threshold = 0.5) {
  pred_labels <- ifelse(pred_probs > threshold, 1, 0)
  true_labels_numeric <- as.numeric(true_labels) - 1  # 转换为0/1
  
  # 计算混淆矩阵
  cm <- table(Predicted = pred_labels, Actual = true_labels_numeric)
  
  # 处理可能的维度问题
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
  
  # 计算各项指标
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  sensitivity <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
  specificity <- ifelse((TN + FP) > 0, TN / (TN + FP), 0)
  ppv <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
  npv <- ifelse((TN + FN) > 0, TN / (TN + FN), 0)
  f1_score <- ifelse((sensitivity + ppv) > 0, 2 * sensitivity * ppv / (sensitivity + ppv), 0)
  balanced_accuracy <- (sensitivity + specificity) / 2
  
  # 计算AUC和95%置信区间
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

# 定义排列重要性计算函数
calculate_permutation_importance <- function(model, train_features, train_labels, test_features, test_labels, model_name, n_permutations = 10) {
  tryCatch({
    # 计算基准性能
    if(model_name == "KNN") {
      train_scaled <- scale(train_features)
      test_scaled <- scale(test_features, center = attr(train_scaled, "scaled:center"), 
                           scale = attr(train_scaled, "scaled:scale"))
      knn_pred <- knn(train_scaled, test_scaled, as.factor(train_labels), k = 5, prob = TRUE)
      knn_probs <- attr(knn_pred, "prob")
      baseline_probs <- ifelse(knn_pred == "1", knn_probs, 1 - knn_probs)
    } else {
      if(model_name == "SVM") {
        baseline_probs <- attr(predict(model, test_features, probability = TRUE), "probabilities")[, 2]
      } else if(model_name == "MLP") {
        baseline_probs <- predict(model, test_features)[, 2]
      } else if(model_name == "NaiveBayes") {
        baseline_probs <- predict(model, data.frame(test_features), type = "prob")[, 2]
      } else if(model_name == "TabNet") {
        baseline_probs <- predict(model, data.frame(test_features), type = "prob")[, 2]
      }
    }
    
    baseline_auc <- auc(roc(test_labels, baseline_probs))
    
    # 对每个特征计算排列重要性
    importance_scores <- numeric(ncol(test_features))
    names(importance_scores) <- colnames(test_features)
    
    for(feature_idx in 1:ncol(test_features)) {
      feature_aucs <- numeric(n_permutations)
      
      for(perm in 1:n_permutations) {
        # 复制测试数据并打乱当前特征
        test_permuted <- test_features
        test_permuted[, feature_idx] <- sample(test_permuted[, feature_idx])
        
        # 计算打乱后的性能
        if(model_name == "KNN") {
          test_scaled_perm <- scale(test_permuted, center = attr(train_scaled, "scaled:center"), 
                                    scale = attr(train_scaled, "scaled:scale"))
          knn_pred_perm <- knn(train_scaled, test_scaled_perm, as.factor(train_labels), k = 5, prob = TRUE)
          knn_probs_perm <- attr(knn_pred_perm, "prob")
          perm_probs <- ifelse(knn_pred_perm == "1", knn_probs_perm, 1 - knn_probs_perm)
        } else {
          if(model_name == "SVM") {
            perm_probs <- attr(predict(model, test_permuted, probability = TRUE), "probabilities")[, 2]
          } else if(model_name == "MLP") {
            perm_probs <- predict(model, test_permuted)[, 2]
          } else if(model_name == "NaiveBayes") {
            perm_probs <- predict(model, data.frame(test_permuted), type = "prob")[, 2]
          } else if(model_name == "TabNet") {
            perm_probs <- predict(model, data.frame(test_permuted), type = "prob")[, 2]
          }
        }
        
        perm_auc <- auc(roc(test_labels, perm_probs))
        feature_aucs[perm] <- perm_auc
      }
      
      # 计算重要性得分（性能下降）
      importance_scores[feature_idx] <- baseline_auc - mean(feature_aucs)
    }
    
    return(importance_scores)
  }, error = function(e) {
    cat("排列重要性计算失败:", e$message, "\n")
    return(NULL)
  })
}

# 开始十折交叉验证循环
for(i in 1:10) {
  cat("正在处理第", i, "折交叉验证...\n")
  
  # 划分训练集和验证集
  fold_train <- train_data[folds[[i]], ]
  fold_test <- train_data[-folds[[i]], ]
  
  # 1. 多重插补 - 对训练集和测试集都进行处理
  cat("  正在进行多重插补...\n")
  
  # 检查训练集和测试集的缺失值
  missing_count_train <- colSums(is.na(fold_train))
  missing_count_test <- colSums(is.na(fold_test))
  
  # 合并训练集和测试集进行插补，确保相同的插补模型
  combined_data <- rbind(
    cbind(fold_train, DataType = "train"),
    cbind(fold_test, DataType = "test")
  )
  
  if(sum(missing_count_train) > 0 || sum(missing_count_test) > 0) {
    cat("  训练集缺失值数量:", sum(missing_count_train), "\n")
    cat("  测试集缺失值数量:", sum(missing_count_test), "\n")
    
    # 设置mice参数
    mice_args <- list(
      data = combined_data[, !names(combined_data) %in% "DataType"],
      m = 5,
      maxit = 10,
      printFlag = FALSE,
      seed = i * 100
    )
    
    var_types <- sapply(combined_data[, !names(combined_data) %in% "DataType"], class)
    methods <- rep("", length(var_types))
    methods[var_types %in% c("numeric", "integer")] <- "pmm"
    methods[var_types == "factor" & sapply(combined_data[, !names(combined_data) %in% "DataType"], function(x) length(levels(x)) == 2)] <- "logreg"
    methods[var_types == "factor" & sapply(combined_data[, !names(combined_data) %in% "DataType"], function(x) length(levels(x)) > 2)] <- "polyreg"
    
    mice_args$method <- methods
    imp_model <- do.call(mice, mice_args)
    combined_imputed <- complete(imp_model, 1)
    
    # 分离插补后的训练集和测试集
    fold_train_imputed <- combined_imputed[1:nrow(fold_train), ]
    fold_test_imputed <- combined_imputed[(nrow(fold_train) + 1):nrow(combined_imputed), ]
    fold_test_imputed$Diagnosis <- fold_test$Diagnosis  # 恢复测试集的Diagnosis列
  } else {
    fold_train_imputed <- fold_train
    fold_test_imputed <- fold_test
  }
  
  # 2. 单因素分析筛选特征
  cat("  进行单因素分析筛选特征...\n")
  all_features <- names(fold_train_imputed)[names(fold_train_imputed) != "Diagnosis"]
  p_values <- sapply(all_features, function(feature) {
    if(is.numeric(fold_train_imputed[[feature]])) {
      wilcox.test(fold_train_imputed[[feature]] ~ fold_train_imputed$Diagnosis)$p.value
    } else {
      chisq.test(fold_train_imputed[[feature]], fold_train_imputed$Diagnosis)$p.value
    }
  })
  univariate_df <- data.frame(
    Feature = all_features, P_Value = p_values, Significant = p_values < p_value_threshold
  )
  univariate_df <- univariate_df[order(univariate_df$P_Value), ]
  if(i == 1) writeData(wb, "单因素分析结果", univariate_df, startRow = 1)
  
  # 单因素筛选后的特征
  selected_features_univariate <- as.character(univariate_df$Feature[univariate_df$P_Value < p_value_threshold])
  if(length(selected_features_univariate) == 0) {
    cat("  警告: 单因素分析未选择任何特征，使用所有特征继续分析...\n")
    selected_features_univariate <- all_features
  }
  
  # 3. LASSO进一步筛选特征 - 修改为使用lambda.1se
  cat("  进行LASSO筛选（使用lambda.1se）...\n")
  x_lasso <- as.matrix(fold_train_imputed[, selected_features_univariate])  
  y_lasso <- as.numeric(fold_train_imputed$Diagnosis) - 1
  
  cv_fit <- cv.glmnet(
    x = x_lasso, 
    y = y_lasso, 
    alpha = lasso_alpha, 
    family = "binomial",
    type.measure = "auc",
    nfolds = 10
  )
  
  # 使用lambda.1se选择最佳模型（更简化的模型）
  best_lambda <- cv_fit$lambda.1se  # 修改这里：从lambda.min改为lambda.1se
  lasso_model_best <- glmnet(
    x = x_lasso, 
    y = y_lasso, 
    alpha = lasso_alpha, 
    lambda = best_lambda,
    family = "binomial"
  )
  
  lasso_coef_matrix <- as.matrix(coef(lasso_model_best))
  lasso_coef_no_intercept <- lasso_coef_matrix[-1, , drop = FALSE]
  selected_features_lasso <- rownames(lasso_coef_no_intercept)[lasso_coef_no_intercept != 0]
  
  cat("   LASSO选择了", length(selected_features_lasso), "个特征（使用lambda.1se）\n")
  
  if(length(selected_features_lasso) == 0) {
    cat("  警告：LASSO未保留任何特征，使用单因素分析结果替代...\n")
    selected_features_lasso <- selected_features_univariate
  }
  
  selected_features <- selected_features_lasso
  
  # 确保best_features_per_model[[i]]已初始化
  if(is.null(best_features_per_model[[i]])) {
    best_features_per_model[[i]] <- list()
  }
  
  # 4. 处理类别不平衡
  cat("  检查类别平衡...\n")
  class_ratio <- min(table(fold_train_imputed$Diagnosis)) / nrow(fold_train_imputed)
  if(class_ratio < 0.3) {
    cat("  检测到类别不平衡，应用SMOTE...\n")
    if(!require("smotefamily")) install.packages("smotefamily"); library(smotefamily)
    
    smote_data <- SMOTE(
      fold_train_imputed[, c(selected_features, "Diagnosis")], 
      fold_train_imputed$Diagnosis, 
      K = 5
    )
    fold_train_balanced <- smote_data$data
    fold_train_balanced$class <- as.factor(fold_train_balanced$class)
    names(fold_train_balanced)[ncol(fold_train_balanced)] <- "Diagnosis"
  } else {
    fold_train_balanced <- fold_train_imputed
  }
  
  # 准备训练和测试数据 - 确保没有缺失值
  for(feature in selected_features) {
    if(any(is.na(fold_test_imputed[[feature]]))) {
      if(is.numeric(fold_test_imputed[[feature]])) {
        mean_val <- mean(fold_train_balanced[[feature]], na.rm = TRUE)
        fold_test_imputed[[feature]][is.na(fold_test_imputed[[feature]])] <- mean_val
      } else {
        mode_val <- names(sort(table(fold_train_balanced[[feature]]), decreasing = TRUE))[1]
        fold_test_imputed[[feature]][is.na(fold_test_imputed[[feature]])] <- mode_val
      }
    }
  }
  
  # 最终检查确保没有缺失值
  if(any(is.na(fold_train_balanced[, selected_features]))) {
    fold_train_balanced[, selected_features] <- apply(fold_train_balanced[, selected_features], 2, function(x) {
      if(any(is.na(x))) x[is.na(x)] <- mean(x, na.rm = TRUE)
      return(x)
    })
  }
  
  if(any(is.na(fold_test_imputed[, selected_features]))) {
    fold_test_imputed[, selected_features] <- apply(fold_test_imputed[, selected_features], 2, function(x) {
      if(any(is.na(x))) x[is.na(x)] <- mean(x, na.rm = TRUE)
      return(x)
    })
  }
  
  train_features <- as.matrix(fold_train_balanced[, selected_features])
  train_labels <- as.numeric(fold_train_balanced$Diagnosis) - 1
  test_features <- as.matrix(fold_test_imputed[, selected_features])
  test_labels <- as.numeric(fold_test_imputed$Diagnosis) - 1
  true_labels_factor <- fold_test_imputed$Diagnosis
  
  # 5. 训练所有12种机器学习模型（添加TabNet）
  cat("  训练12种机器学习模型（包括TabNet）...\n")
  
  # 初始化当前折的结果
  fold_results <- list()
  fold_importance <- list()
  fold_rocs <- list()
  
  # 模型列表 - 添加TabNet，共12种模型
  model_names <- c("RF", "SVM", "AdaBoost", "DecisionTree", "GradientBoosting", 
                   "KNN", "LightGBM", "GLM", "MLP", "NaiveBayes", "XGBoost", "TabNet")
  
  # 为每个模型训练并记录特征
  for(model_name in model_names) {
    cat("   训练", model_name, "...\n")
    
    # 使用tryCatch确保即使某个模型失败也不会中断整个流程
    model_result <- tryCatch({
      # 根据模型名称选择训练方法
      if(model_name == "RF") {
        # Random Forest
        model <- randomForest(x = train_features, y = as.factor(train_labels), ntree = 100, importance = TRUE)
        probs <- predict(model, test_features, type = "prob")[, 2]
        
        # 存储特征重要性
        imp <- importance(model)
        fold_importance$RF <- data.frame(
          Feature = rownames(imp),
          Importance = imp[, "MeanDecreaseAccuracy"],
          Fold = i,
          Model = "RF"
        )
      } else if(model_name == "SVM") {
        # SVM
        model <- svm(x = train_features, y = as.factor(train_labels), 
                     probability = TRUE, kernel = "radial")
        probs <- attr(predict(model, test_features, probability = TRUE), "probabilities")[, 2]
        
        # SVM特征重要性（使用排列重要性）
        svm_imp <- calculate_permutation_importance(model, train_features, train_labels, test_features, test_labels, model_name = "SVM")
        if(!is.null(svm_imp)) {
          fold_importance$SVM <- data.frame(
            Feature = names(svm_imp),
            Importance = svm_imp,
            Fold = i,
            Model = "SVM"
          )
        }
      } else if(model_name == "AdaBoost") {
        # AdaBoost
        ada_data <- data.frame(train_features, Diagnosis = as.factor(train_labels))
        model <- boosting(Diagnosis ~ ., data = ada_data)
        probs <- predict(model, data.frame(test_features))$prob[, 2]
        
        # AdaBoost特征重要性
        imp <- model$importance
        if(!is.null(imp)) {
          fold_importance$AdaBoost <- data.frame(
            Feature = names(imp),
            Importance = as.numeric(imp),
            Fold = i,
            Model = "AdaBoost"
          )
        }
      } else if(model_name == "DecisionTree") {
        # Decision Tree
        tree_data <- data.frame(train_features, Diagnosis = as.factor(train_labels))
        model <- rpart(Diagnosis ~ ., data = tree_data, method = "class")
        probs <- predict(model, data.frame(test_features))[, 2]
        
        # 存储特征重要性
        imp <- model$variable.importance
        if(!is.null(imp)) {
          fold_importance$DecisionTree <- data.frame(
            Feature = names(imp),
            Importance = as.numeric(imp),
            Fold = i,
            Model = "DecisionTree"
          )
        }
      } else if(model_name == "GradientBoosting") {
        # Gradient Boosting
        model <- gbm.fit(x = train_features, y = train_labels, 
                         distribution = "bernoulli", n.trees = 100, verbose = FALSE)
        probs <- predict(model, test_features, n.trees = 100, type = "response")
        
        # 存储特征重要性
        imp <- summary(model, plotit = FALSE)
        fold_importance$GradientBoosting <- data.frame(
          Feature = imp$var,
          Importance = imp$rel.inf,
          Fold = i,
          Model = "GradientBoosting"
        )
      } else if(model_name == "KNN") {
        # KNN
        # 标准化特征
        train_scaled <- scale(train_features)
        test_scaled <- scale(test_features, center = attr(train_scaled, "scaled:center"), 
                             scale = attr(train_scaled, "scaled:scale"))
        knn_pred <- knn(train_scaled, test_scaled, as.factor(train_labels), k = 5, prob = TRUE)
        knn_probs <- attr(knn_pred, "prob")
        probs <- ifelse(knn_pred == "1", knn_probs, 1 - knn_probs)
        model <- NULL  # KNN没有模型对象
        
        # KNN特征重要性（使用排列重要性）
        knn_imp <- calculate_permutation_importance(NULL, train_features, train_labels, test_features, test_labels, model_name = "KNN")
        if(!is.null(knn_imp)) {
          fold_importance$KNN <- data.frame(
            Feature = names(knn_imp),
            Importance = knn_imp,
            Fold = i,
            Model = "KNN"
          )
        }
      } else if(model_name == "LightGBM") {
        # LightGBM
        lgb_train <- lgb.Dataset(data = train_features, label = train_labels)
        lgb_params <- list(objective = "binary", metric = "auc", learning_rate = 0.1)
        model <- lgb.train(params = lgb_params, data = lgb_train, nrounds = 100, verbose = 0)
        probs <- predict(model, test_features)
        
        # 存储特征重要性
        imp <- lgb.importance(model)
        if(!is.null(imp)) {
          fold_importance$LightGBM <- data.frame(
            Feature = imp$Feature,
            Importance = imp$Gain,
            Fold = i,
            Model = "LightGBM"
          )
        }
      } else if(model_name == "GLM") {
        # GLM (Logistic Regression)
        glm_data <- data.frame(train_features, Diagnosis = train_labels)
        model <- glm(Diagnosis ~ ., data = glm_data, family = binomial)
        probs <- predict(model, data.frame(test_features), type = "response")
        
        # 存储特征重要性（系数绝对值）
        coefs <- coef(model)[-1]  # 去除截距
        fold_importance$GLM <- data.frame(
          Feature = names(coefs),
          Importance = abs(coefs),
          Fold = i,
          Model = "GLM"
        )
      } else if(model_name == "MLP") {
        # MLP (Neural Network)
        model <- nnet(x = train_features, y = class.ind(as.factor(train_labels)), 
                      size = 10, softmax = TRUE, maxit = 200, trace = FALSE)
        probs <- predict(model, test_features)[, 2]
        
        # MLP特征重要性（使用排列重要性）
        mlp_imp <- calculate_permutation_importance(model, train_features, train_labels, test_features, test_labels, model_name = "MLP")
        if(!is.null(mlp_imp)) {
          fold_importance$MLP <- data.frame(
            Feature = names(mlp_imp),
            Importance = mlp_imp,
            Fold = i,
            Model = "MLP"
          )
        }
      } else if(model_name == "NaiveBayes") {
        # Naive Bayes
        nb_data <- data.frame(train_features, Diagnosis = as.factor(train_labels))
        model <- naive_bayes(Diagnosis ~ ., data = nb_data)
        probs <- predict(model, data.frame(test_features), type = "prob")[, 2]
        
        # Naive Bayes特征重要性（使用排列重要性）
        nb_imp <- calculate_permutation_importance(model, train_features, train_labels, test_features, test_labels, model_name = "NaiveBayes")
        if(!is.null(nb_imp)) {
          fold_importance$NaiveBayes <- data.frame(
            Feature = names(nb_imp),
            Importance = nb_imp,
            Fold = i,
            Model = "NaiveBayes"
          )
        }
      } else if(model_name == "XGBoost") {
        # XGBoost
        dtrain <- xgb.DMatrix(data = train_features, label = train_labels)
        dtest <- xgb.DMatrix(data = test_features, label = test_labels)
        
        params <- list(
          objective = "binary:logistic",
          eval_metric = "auc",
          eta = 0.1,
          max_depth = 6,
          colsample_bytree = 0.8,
          subsample = 0.8,
          min_child_weight = 1
        )
        
        model <- xgb.train(
          params, dtrain, nrounds = 100, watchlist = list(test = dtest),
          print_every_n = 10, early_stopping_rounds = 10, verbose = 0
        )
        probs <- predict(model, dtest)
        
        # 存储特征重要性
        imp <- xgb.importance(model = model, feature_names = colnames(train_features))
        if(!is.null(imp)) {
          fold_importance$XGBoost <- data.frame(
            Feature = imp$Feature,
            Importance = imp$Gain,
            Fold = i,
            Model = "XGBoost"
          )
        }
      } else if(model_name == "TabNet") {
        # TabNet算法
        # 准备数据
        tabnet_data_train <- data.frame(train_features)
        tabnet_data_train$Diagnosis <- as.factor(fold_train_balanced$Diagnosis)
        
        tabnet_data_test <- data.frame(test_features)
        tabnet_data_test$Diagnosis <- as.factor(fold_test_imputed$Diagnosis)
        
        # 设置TabNet参数
        tabnet_params <- list(
          epochs = 30,
          batch_size = min(32, nrow(tabnet_data_train)),
          decision_width = 8,
          attention_width = 8,
          num_steps = 3,
          virtual_batch_size = 128,
          num_independent = 2,
          num_shared = 2,
          momentum = 0.02,
          mask_type = "sparsemax"
        )
        
        # 训练TabNet模型
        model <- tabnet_fit(Diagnosis ~ ., 
                            data = tabnet_data_train,
                            epochs = tabnet_params$epochs,
                            verbose = FALSE)
        
        # 预测概率
        probs <- predict(model, tabnet_data_test, type = "prob")$.pred_1
        
        # TabNet特征重要性
        tabnet_imp <- tabnet_explain(model, tabnet_data_train)
        if(!is.null(tabnet_imp$importances)) {
          # 获取特征重要性
          tabnet_imp_df <- as.data.frame(tabnet_imp$importances$importance)
          tabnet_imp_df$Feature <- rownames(tabnet_imp$importances$importance)
          
          fold_importance$TabNet <- data.frame(
            Feature = tabnet_imp_df$Feature,
            Importance = tabnet_imp_df[, 1],  # 第一列是重要性分数
            Fold = i,
            Model = "TabNet"
          )
        } else {
          # 如果无法获取特征重要性，使用默认值
          fold_importance$TabNet <- data.frame(
            Feature = colnames(train_features),
            Importance = 1/length(colnames(train_features)),
            Fold = i,
            Model = "TabNet"
          )
        }
      }
      
      # 计算性能指标
      metrics <- calculate_all_metrics(true_labels_factor, probs)
      
      # 存储ROC对象
      if(!is.null(metrics$ROC_Obj)) {
        fold_rocs[[model_name]] <- metrics$ROC_Obj
      }
      
      # 返回结果
      list(metrics = metrics, model = model)
    }, error = function(e) {
      cat("   ", model_name, "训练失败:", e$message, "\n")
      # 返回默认结果
      list(metrics = list(
        AUC = 0.5, AUC_CI_Lower = 0.5, AUC_CI_Upper = 0.5,
        Sensitivity = 0, Specificity = 0, PPV = 0, NPV = 0, 
        Accuracy = 0, F1 = 0, Balanced_Accuracy = 0
      ), model = NULL)
    })
    
    # 存储结果
    fold_results[[model_name]] <- model_result$metrics
    
    # 记录当前折的特征 - 确保列表已初始化
    if(is.null(best_features_per_model[[i]])) {
      best_features_per_model[[i]] <- list()
    }
    best_features_per_model[[i]][[model_name]] <- selected_features
  }
  
  # 存储当前折的结果
  results_all_models[[i]] <- fold_results
  feature_importance_all[[i]] <- fold_importance
  roc_curves_all[[i]] <- fold_rocs
  
  cat("第", i, "折完成\n")
}

# 6. 汇总所有模型的结果
cat("汇总所有模型的结果...\n")

# 模型列表 - 12种模型
model_names <- c("RF", "SVM", "AdaBoost", "DecisionTree", "GradientBoosting", 
                 "KNN", "LightGBM", "GLM", "MLP", "NaiveBayes", "XGBoost", "TabNet")

# 初始化汇总结果
summary_results <- data.frame(
  Model = model_names,
  AUC = numeric(length(model_names)),
  AUC_CI_Lower = numeric(length(model_names)),
  AUC_CI_Upper = numeric(length(model_names)),
  Sensitivity = numeric(length(model_names)),
  Specificity = numeric(length(model_names)),
  PPV = numeric(length(model_names)),
  NPV = numeric(length(model_names)),
  Accuracy = numeric(length(model_names)),
  F1 = numeric(length(model_names)),
  Balanced_Accuracy = numeric(length(model_names)),
  stringsAsFactors = FALSE
)

# 计算每个指标的平均值
for(model in model_names) {
  model_metrics <- list(
    AUC = numeric(10),
    AUC_CI_Lower = numeric(10),
    AUC_CI_Upper = numeric(10),
    Sensitivity = numeric(10),
    Specificity = numeric(10),
    PPV = numeric(10),
    NPV = numeric(10),
    Accuracy = numeric(10),
    F1 = numeric(10),
    Balanced_Accuracy = numeric(10)
  )
  
  for(fold in 1:10) {
    if(!is.null(results_all_models[[fold]][[model]])) {
      result <- results_all_models[[fold]][[model]]
      model_metrics$AUC[fold] <- result$AUC
      model_metrics$AUC_CI_Lower[fold] <- result$AUC_CI_Lower
      model_metrics$AUC_CI_Upper[fold] <- result$AUC_CI_Upper
      model_metrics$Sensitivity[fold] <- result$Sensitivity
      model_metrics$Specificity[fold] <- result$Specificity
      model_metrics$PPV[fold] <- result$PPV
      model_metrics$NPV[fold] <- result$NPV
      model_metrics$Accuracy[fold] <- result$Accuracy
      model_metrics$F1[fold] <- result$F1
      model_metrics$Balanced_Accuracy[fold] <- result$Balanced_Accuracy
    }
  }
  
  # 计算平均值
  summary_results[summary_results$Model == model, "AUC"] <- mean(model_metrics$AUC, na.rm = TRUE)
  summary_results[summary_results$Model == model, "AUC_CI_Lower"] <- mean(model_metrics$AUC_CI_Lower, na.rm = TRUE)
  summary_results[summary_results$Model == model, "AUC_CI_Upper"] <- mean(model_metrics$AUC_CI_Upper, na.rm = TRUE)
  summary_results[summary_results$Model == model, "Sensitivity"] <- mean(model_metrics$Sensitivity, na.rm = TRUE)
  summary_results[summary_results$Model == model, "Specificity"] <- mean(model_metrics$Specificity, na.rm = TRUE)
  summary_results[summary_results$Model == model, "PPV"] <- mean(model_metrics$PPV, na.rm = TRUE)
  summary_results[summary_results$Model == model, "NPV"] <- mean(model_metrics$NPV, na.rm = TRUE)
  summary_results[summary_results$Model == model, "Accuracy"] <- mean(model_metrics$Accuracy, na.rm = TRUE)
  summary_results[summary_results$Model == model, "F1"] <- mean(model_metrics$F1, na.rm = TRUE)
  summary_results[summary_results$Model == model, "Balanced_Accuracy"] <- mean(model_metrics$Balanced_Accuracy, na.rm = TRUE)
}

# 按AUC排序
summary_results <- summary_results[order(-summary_results$AUC), ]

# 输出结果
cat("\n=== 12种机器学习模型性能比较 ===\n")
print(summary_results)

# 将结果写入Excel
writeData(wb, "13模型比较结果", summary_results, startRow = 1)

# 7. 生成详细的性能指标表格（包含每折的结果）
detailed_results <- data.frame()
for(fold in 1:10) {
  for(model in model_names) {
    if(!is.null(results_all_models[[fold]][[model]])) {
      result <- results_all_models[[fold]][[model]]
      detailed_results <- rbind(detailed_results, data.frame(
        Fold = fold,
        Model = model,
        AUC = result$AUC,
        AUC_CI_Lower = result$AUC_CI_Lower,
        AUC_CI_Upper = result$AUC_CI_Upper,
        Sensitivity = result$Sensitivity,
        Specificity = result$Specificity,
        PPV = result$PPV,
        NPV = result$NPV,
        Accuracy = result$Accuracy,
        F1 = result$F1,
        Balanced_Accuracy = result$Balanced_Accuracy
      ))
    }
  }
}

writeData(wb, "详细性能指标", detailed_results, startRow = 1)

# 8. 汇总特征重要性
importance_summary <- data.frame()
for(fold in 1:10) {
  if(!is.null(feature_importance_all[[fold]])) {
    for(model in names(feature_importance_all[[fold]])) {
      if(!is.null(feature_importance_all[[fold]][[model]])) {
        importance_summary <- rbind(importance_summary, feature_importance_all[[fold]][[model]])
      }
    }
  }
}

# 计算平均特征重要性
if(nrow(importance_summary) > 0) {
  avg_importance <- aggregate(Importance ~ Feature + Model, data = importance_summary, mean)
  writeData(wb, "特征重要性汇总", avg_importance, startRow = 1)
  
  # 生成每个模型的特征排序
  cat("生成各模型特征排序...\n")
  model_feature_ranking <- data.frame()
  
  for(model in unique(avg_importance$Model)) {
    model_imp <- avg_importance[avg_importance$Model == model, ]
    model_imp <- model_imp[order(-model_imp$Importance), ]
    model_imp$Rank <- 1:nrow(model_imp)
    
    model_feature_ranking <- rbind(model_feature_ranking, model_imp)
  }
  
  # 按模型和排名排序
  model_feature_ranking <- model_feature_ranking[order(model_feature_ranking$Model, model_feature_ranking$Rank), ]
  writeData(wb, "模型特征排序", model_feature_ranking, startRow = 1)
}

# 9. 找到最佳AUC对应的共同变量
cat("分析最佳AUC对应的共同变量...\n")

# 找出每个模型在10折中AUC最高的那一折
best_folds <- list()
for(model in model_names) {
  best_auc <- 0
  best_fold <- 1
  for(fold in 1:10) {
    # 添加安全检查
    if(!is.null(results_all_models[[fold]]) && !is.null(results_all_models[[fold]][[model]])) {
      auc_value <- results_all_models[[fold]][[model]]$AUC
      if(auc_value > best_auc) {
        best_auc = auc_value
        best_fold = fold
      }
    }
  }
  best_folds[[model]] <- best_fold
}

# 获取每个模型在最佳AUC时使用的特征
best_features <- list()
for(model in model_names) {
  best_fold <- best_folds[[model]]
  # 添加安全检查
  if(!is.null(best_features_per_model[[best_fold]]) && !is.null(best_features_per_model[[best_fold]][[model]])) {
    best_features[[model]] <- best_features_per_model[[best_fold]][[model]]
  } else {
    cat("警告：无法获取模型", model, "在最佳折", best_fold, "的特征\n")
    best_features[[model]] <- character(0)  # 设置为空字符向量
  }
}

# 找到所有模型共有的特征
common_features <- NULL
if(length(best_features) > 0) {
  # 过滤掉空的特征列表
  non_empty_features <- best_features[sapply(best_features, length) > 0]
  if(length(non_empty_features) > 0) {
    common_features <- non_empty_features[[1]]
    for(i in 2:length(non_empty_features)) {
      common_features <- intersect(common_features, non_empty_features[[i]])
    }
  }
}

# 输出共同变量
cat("所有模型在最佳AUC时共有的特征:\n")
if(length(common_features) > 0) {
  print(common_features)
} else {
  cat("没有找到共同特征\n")
}

# 创建最佳特征比较表格
feature_comparison <- data.frame(Model = character(), Features = character(), stringsAsFactors = FALSE)
for(model in model_names) {
  if(!is.null(best_features[[model]]) && length(best_features[[model]]) > 0) {
    feature_comparison <- rbind(feature_comparison, 
                                data.frame(Model = model, 
                                           Features = paste(best_features[[model]], collapse = ", "),
                                           stringsAsFactors = FALSE))
  }
}

# 添加共同特征行
if(length(common_features) > 0) {
  feature_comparison <- rbind(feature_comparison, 
                              data.frame(Model = "共同特征", 
                                         Features = paste(common_features, collapse = ", "),
                                         stringsAsFactors = FALSE))
}

writeData(wb, "最佳特征比较", feature_comparison, startRow = 1)

# 保存Excel工作簿
saveWorkbook(wb, "12种机器学习模型完整分析结果.xlsx", overwrite = TRUE)

# 10. 生成ROC曲线
cat("生成ROC曲线...\n")
if(!dir.exists("ROC_Curves")) dir.create("ROC_Curves")

# 为每个模型生成平均ROC曲线
roc_plot_data <- data.frame()
colors <- rainbow(length(model_names))

# 创建一个包含所有模型ROC曲线的图
png("ROC_Curves/All_Models_ROC_Curves.png", width = 800, height = 600)
plot(1, type = "n", xlim = c(0, 1), ylim = c(0, 1), 
     xlab = "1 - Specificity", ylab = "Sensitivity",
     main = "12种机器学习模型的ROC曲线")

for(i in 1:length(model_names)) {
  model <- model_names[i]
  
  # 收集所有折的ROC数据
  all_sensitivities <- numeric(0)
  all_specificities <- numeric(0)
  
  for(fold in 1:10) {
    if(!is.null(roc_curves_all[[fold]]) && !is.null(roc_curves_all[[fold]][[model]])) {
      roc_obj <- roc_curves_all[[fold]][[model]]
      all_sensitivities <- c(all_sensitivities, roc_obj$sensitivities)
      all_specificities <- c(all_specificities, roc_obj$specificities)
    }
  }
  
  if(length(all_sensitivities) > 0) {
    # 计算平均ROC曲线
    unique_specificities <- seq(0, 1, length.out = 100)
    sensitivities <- approx(1 - all_specificities, all_sensitivities, 
                            xout = unique_specificities, method = "linear")$y
    sensitivities[is.na(sensitivities)] <- 0
    
    lines(unique_specificities, sensitivities, type = "l", col = colors[i], lwd = 2)
    
    # 存储绘图数据
    roc_plot_data <- rbind(roc_plot_data, 
                           data.frame(
                             Model = model,
                             Specificity = unique_specificities,
                             Sensitivity = sensitivities
                           ))
  }
}

# 添加对角线
abline(0, 1, lty = 2, col = "gray")

# 添加图例
legend("bottomright", legend = model_names, col = colors, lwd = 2, cex = 0.8)
dev.off()

# 11. 生成特征重要性图
cat("生成特征重要性图...\n")
if(nrow(importance_summary) > 0) {
  # 为每个模型生成特征重要性图
  for(model in unique(avg_importance$Model)) {
    model_importance <- avg_importance[avg_importance$Model == model, ]
    model_importance <- model_importance[order(-model_importance$Importance), ]
    
    if(nrow(model_importance) > 10) {
      model_importance <- head(model_importance, 10)  # 只显示前10个最重要的特征
    }
    
    p <- ggplot(model_importance, aes(x = reorder(Feature, Importance), y = Importance)) +
      geom_bar(stat = "identity", fill = "steelblue") +
      coord_flip() +
      labs(title = paste(model, "模型特征重要性"),
           x = "特征", y = "重要性") +
      theme_minimal()
    
    ggsave(paste0("ROC_Curves/", model, "_Feature_Importance.png"), p, width = 10, height = 6)
  }
}

# 12. 创建模型性能比较图（包含所有指标）
performance_metrics <- c("AUC", "Sensitivity", "Specificity", "Accuracy", "F1", "Balanced_Accuracy")
performance_data <- data.frame()

for(metric in performance_metrics) {
  for(i in 1:nrow(summary_results)) {
    performance_data <- rbind(performance_data, data.frame(
      Model = summary_results$Model[i],
      Metric = metric,
      Value = summary_results[[metric]][i]
    ))
  }
}

# 创建雷达图样式的比较图
p <- ggplot(performance_data, aes(x = Metric, y = Value, group = Model, color = Model)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  facet_wrap(~ Model, ncol = 4) +
  labs(title = "12种机器学习模型性能比较",
       x = "性能指标", y = "值") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("ROC_Curves/Model_Performance_Comparison.png", p, width = 12, height = 8)

# 13. 生成共同特征图
if(length(common_features) > 0) {
  cat("生成共同特征图...\n")
  
  # 创建共同特征出现频率图
  feature_freq <- table(unlist(best_features))
  feature_freq_df <- data.frame(Feature = names(feature_freq), Frequency = as.numeric(feature_freq))
  feature_freq_df <- feature_freq_df[order(-feature_freq_df$Frequency), ]
  
  # 只显示前20个特征
  if(nrow(feature_freq_df) > 20) {
    feature_freq_df <- head(feature_freq_df, 20)
  }
  
  p <- ggplot(feature_freq_df, aes(x = reorder(Feature, Frequency), y = Frequency)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "特征在最佳模型中的出现频率",
         x = "特征", y = "出现次数") +
    theme_minimal()
  
  ggsave("ROC_Curves/Feature_Frequency_in_Best_Models.png", p, width = 10, height = 8)
  
  # 标记共同特征
  if(length(common_features) > 0) {
    p <- p + geom_bar(data = feature_freq_df[feature_freq_df$Feature %in% common_features, ], 
                      aes(x = reorder(Feature, Frequency), y = Frequency), 
                      stat = "identity", fill = "red")
    ggsave("ROC_Curves/Feature_Frequency_with_Common_Features.png", p, width = 10, height = 8)
  }
}

cat("\n所有分析完成！结果已保存到以下文件：\n")
cat("- 12种机器学习模型完整分析结果.xlsx (包含所有详细结果)\n")
cat("- ROC_Curves/ 目录 (包含所有ROC曲线和特征重要性图)\n")
cat("- 模型性能比较图.png (综合性能比较)\n")
if(length(common_features) > 0) {
  cat("- 共同特征分析图.png (共同特征可视化)\n")
}
cat("\n各模型特征排序已保存在Excel工作簿的'模型特征排序'工作表中\n")
cat("\n所有模型在最佳AUC时共有的特征:\n")
if(length(common_features) > 0) {
  print(common_features)
} else {
  cat("没有找到共同特征\n")
}