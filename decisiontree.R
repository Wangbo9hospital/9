library(rpart)
library(rpart.plot)
library(caret)

# 示例数据, kqtest, kqtrain为31列的dataframe
# 划分训练集和测试集
train_data <- kqtrain[,c("Diagnosis",selected_vars)]
test_data <- kqtest[,c("Diagnosis",selected_vars)]

# 训练决策树模型
# 使用所有特征预测Species
tree_model <- rpart(Diagnosis ~ ., data = train_data, method = "class")

# 查看模型摘要
print(tree_model)

# 排列变量重要性
var_importance <- data.frame(tree_model$variable.importance)

# 未参与排序的变量手动添加在最后
top15=c(row.names(var_importance),
        setdiff(colnames(train_data)[2:16],row.names(var_importance)))

prob_predictions <- predict(tree_model, test_data, type = "prob")


# 计算ROC曲线和AUC
roc_result <- roc(test_data$Diagnosis, prob_predictions[, "1"])
auc(roc_result)

train_top15=kqtrain[,c("Diagnosis",top15)]
test_top15=kqtest[,c("Diagnosis",top15)]
auc=data.frame(matrix(nrow = 15,ncol = 2))
auc[,1]=top15
# 纳入前i-1个变量进行预测
i=2
for (i in 2:16)
{
  train_data=train_top15[,1:i]
  test_data=test_top15[,1:i]
  
  tree_model <- rpart(Diagnosis ~ ., data = train_data, method = "class")
  prob_predictions <- predict(tree_model, test_data, type = "prob")
  
  # 计算ROC曲线和AUC
  roc_result <- roc(test_data$Diagnosis, prob_predictions[, "1"])
  auc[i-1,2]=auc(roc_result)
  print(auc(roc_result))
}
write.csv(auc,"auc.csv") # 输出30次预测全部的AUC
