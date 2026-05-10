# 安装和加载包
library(class)
library(caret)
library(vip)

#示例数据, kqtest, kqtrain为31列的dataframe
train_data=kqtrain[,c("Diagnosis",selected_vars)]
test_data=kqtest[,c("Diagnosis",selected_vars)]
train_data$Diagnosis=as.factor(train_data$Diagnosis)
test_data$Diagnosis=as.factor(test_data$Diagnosis)
levels(train_data$Diagnosis) <- make.names(levels(train_data$Diagnosis))
levels(test_data$Diagnosis) <- make.names(levels(test_data$Diagnosis))

ctrl <- trainControl(
  method = "cv",        # 交叉验证
  number = 5,          # 5折交叉验证
  classProbs = TRUE,   # 计算类别概率
  summaryFunction = twoClassSummary
)


# 训练KNN模型
knn_model <- train(
  Diagnosis ~ .,          # 公式
  data = train_data,
  method = "knn",       # 使用KNN
  trControl = ctrl,
  tuneGrid = expand.grid(k = seq(1, 20, 2)),  # 测试不同的k值
  metric = "Accuracy",  # 评估指标
  preProcess = c("center", "scale")  # 数据标准化
)

prob_predictions <- predict(knn_model, test_data, type = "prob")

# 计算AUC
roc_result <- roc(test_data$Diagnosis, prob_predictions[, "X1"])
auc(roc_result)

# 使用ReliefF算法排序变量重要性（专门为KNN设计）
library(FSelector)

relief_importance <- function(data, target, neighbors.count = 5, sample.size = 20) {
  # 使用ReliefF算法
  weights <- relief(
    as.formula(paste(target, "~ .")),
    data,
    neighbours.count = neighbors.count,
    sample.size = sample.size
  )
  
  result <- data.frame(
    Feature = rownames(weights),
    ReliefF_Score = weights$attr_importance
  )
  
  result <- result[order(-result$ReliefF_Score), ]
  return(result)
}

# 计算重要性
relief_imp <- relief_importance(train_data, "Diagnosis", neighbors.count = 5)
print(relief_imp)

top15=relief_imp$Feature[1:15]

train_top15=train_data[,c("Diagnosis",top15)]
test_top15=test_data[,c("Diagnosis",top15)]
auc=data.frame(matrix(nrow = 15,ncol = 2))
#纳入前i-1个变量进行预测
auc[,1]=top15
i=2
for (i in 2:16)
{
  train_data=train_top15[,1:i]
  test_data=test_top15[,1:i]
  
  # 训练KNN模型
  knn_model <- train(
    Diagnosis ~ .,          # 公式
    data = train_data,
    method = "knn",       # 使用KNN
    trControl = ctrl,
    tuneGrid = expand.grid(k = seq(1, 20, 2)),  # 测试不同的k值
    metric = "Accuracy",  # 评估指标
    preProcess = c("center", "scale")  # 数据标准化
  )
  
  prob_predictions <- predict(knn_model, test_data, type = "prob")
  
  # 计算ROC曲线和AUC
  roc_result <- roc(test_data$Diagnosis, prob_predictions[, "X1"])
  auc[i-1,2]=auc(roc_result)
  print(auc(roc_result))
}
write.csv(auc,"auc.csv") #输出30次预测全部的AUC


