# 安装和加载包
library(e1071)
library(magrittr)

# 示例数据, kqtest, kqtrain为31列的dataframe
# 划分训练集和测试集
train_data <- kqtrain[,c("Diagnosis",selected_vars)]
test_data <- kqtest[,c("Diagnosis",selected_vars)]

train_data$Diagnosis=as.factor(train_data$Diagnosis)
test_data$Diagnosis=as.factor(test_data$Diagnosis)
#train_data$Diagnosis=as.numeric(train_data$Diagnosis)-1
#test_data$Diagnosis=as.numeric(test_data$Diagnosis)-1
# 训练模型
nb_model <- naiveBayes(Diagnosis ~ ., data = train_data)

# 查看模型摘要
print(nb_model)

##########################################################
# 预测类别
predictions_class <- predict(nb_model, test_data)

# 预测概率（用于计算AUC）
predictions_prob <- predict(nb_model, test_data, type = "raw")

# 获取真实标签
true_labels <- test_data[["Diagnosis"]]

predictions=list(
                  class = predictions_class,
                  prob = predictions_prob,
                  true = true_labels
                )

positive_class <- levels(predictions$true)[2]
prob_positive <- predictions$prob[, positive_class]

# 计算ROC曲线
roc_obj <- roc(response = predictions$true, 
               predictor = prob_positive,
               levels = levels(predictions$true))

# 显示AUC值
auc_value <- auc(roc_obj)
cat(sprintf("\nAUC面积: %.4f\n", auc_value))



# 使用Permutation Importance计算变量重要性
calculate_permutation_importance <- function(model, test_data, target_col, 
                                             n_permutations = 30, n_cores = 1) {
  
  library(doParallel)
  library(foreach)
  
  # 基准性能
  predictions <- predict(model, test_data)
  baseline_accuracy <- mean(predictions == test_data[[target_col]])
  
  # 特征列表
  features <- names(test_data)[names(test_data) != target_col]
  
    importance_scores <- data.frame(
      feature = features,
      importance = NA,
      std = NA
    )
    
    for (i in 1:length(features)) {
      feature <- features[i]
      accuracy_drops <- numeric(n_permutations)
      
      for (perm in 1:n_permutations) {
        # 复制测试数据
        perm_data <- test_data
        
        # 打乱当前特征的值
        perm_data[[feature]] <- sample(perm_data[[feature]])
        
        # 计算新准确率
        perm_predictions <- predict(model, perm_data)
        perm_accuracy <- mean(perm_predictions == perm_data[[target_col]])
        
        # 计算准确率下降
        accuracy_drops[perm] <- baseline_accuracy - perm_accuracy
      }
      
      importance_scores$importance[i] <- mean(accuracy_drops)
      importance_scores$std[i] <- sd(accuracy_drops)
    }
  
  
  # 排序
  importance_scores <- importance_scores %>%
  arrange(desc(importance))
  
  return(importance_scores)
}

# 使用排列重要性
perm_importance <- calculate_permutation_importance(
  nb_model,
  test_data,
  "Diagnosis",
  n_permutations = 30,
  n_cores = 1
)

# 获取前30个特征
top_15_perm <- head(perm_importance, 15)
print(top_15_perm)

top15=top_15_perm$feature
train_top15=train_data[,c("Diagnosis",top15)]
test_top15=test_data[,c("Diagnosis",top15)]
auc=data.frame(matrix(nrow=15,ncol=2))
#分别纳入前i-1个特征，计算AUC
auc[,1]=top15
i=2
for (i in 2:16)
{
  train_data <- train_top15[,1:i]
  test_data <- test_top15[,1:i]
  #train_data$Diagnosis=as.numeric(train_data$Diagnosis)-1
  #test_data$Diagnosis=as.numeric(test_data$Diagnosis)-1
  # 训练模型
  nb_model <- naiveBayes(Diagnosis ~ ., data = train_data)
  
  ##########################################################
  # 预测类别
  predictions_class <- predict(nb_model, test_data)
  
  # 预测概率（用于计算AUC）
  predictions_prob <- predict(nb_model, test_data, type = "raw")
  
  # 获取真实标签
  true_labels <- test_data[["Diagnosis"]]
  
  predictions=list(
    class = predictions_class,
    prob = predictions_prob,
    true = true_labels
  )
  
  positive_class <- levels(predictions$true)[2]
  prob_positive <- predictions$prob[, positive_class]
  
  # 计算ROC曲线
  roc_obj <- roc(response = predictions$true, 
                 predictor = prob_positive,
                 levels = levels(predictions$true))
  
  # 显示AUC值
  auc_value <- auc(roc_obj)
  cat(sprintf("\nAUC面积: %.4f\n", auc_value))
  auc[i-1,2]=auc_value
}
write.csv(auc,"auc.csv") # 输出30次预测的全部AUC

colnames(kqtrain)[c(4,29,22,17,
                  26,27,23,7,
                  30,2,21,12,
                  13,28,11,25,
                  8,18,6,20,
                  10,3,24,31,
                  5,16,9,19,
                  14,15)]
