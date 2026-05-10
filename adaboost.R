# 加载包
library(ada)
library(caret)

#示例数据, kqtest, kqtrain为31列的dataframe
train_data <- kqtrain[,c("Diagnosis",selected_vars)]
test_data <- kqtest[,c("Diagnosis",selected_vars)]
train_data$Diagnosis=as.factor(train_data$Diagnosis)
test_data$Diagnosis=as.factor(test_data$Diagnosis)
# 训练AdaBoost模型
ada_model <- ada(Diagnosis ~ ., data = train_data, iter = 50)


imp_details <- varplot(ada_model, plot.it = FALSE, type = "scores")

# 预测概率
pred_prob <- predict(ada_model, test_data, type = "prob")

print(head(pred_prob))

# 假设正类是第二列（需要根据实际情况调整）
# 通常ada返回两列：负类和正类的概率
if(is.matrix(pred_prob)) {
  # 如果有矩阵形式的概率
  if(ncol(pred_prob) == 2) {
    positive_prob <- pred_prob[, 2]  # 取正类的概率
  } else {
    # 如果只有一列，可能已经是正类概率
    positive_prob <- pred_prob
  }
} else {
  # 如果是向量
  positive_prob <- pred_prob
}

# 创建ROC曲线对象
# 需要将真实标签转换为数值（通常1代表正类，0代表负类）
if(is.factor(test_data$Diagnosis)) {
  true_labels <- as.numeric(test_data$Diagnosis) - 1  # 转换为0/1
} else {
  true_labels <- test_data$Diagnosis
}

# 计算ROC
roc_obj <- roc(true_labels, positive_prob)

# 输出AUC
auc_value <- auc(roc_obj)
print(paste("AUC面积:", round(auc_value, 4)))

# 转换为数据框并排序
imp_df <- as.data.frame(imp_details)
imp_df$Feature <- rownames(imp_df)
colnames(imp_df) <- c("Importance", "Feature")
imp_df <- imp_df[order(-imp_df$Importance), ]

top15=imp_df$Feature
top15

train_top15=train_data[,c("Diagnosis",top15)]
test_top15=test_data[,c("Diagnosis",top15)]
auc=data.frame(matrix(nrow = 15,ncol = 2))
auc[,1]=top15
#纳入前i-1个变量进行预测
i=2
for (i in 2:16)
{
  train_data=train_top15[,1:i]
  test_data=test_top15[,1:i]
  ada_model <- ada(Diagnosis ~ ., data = train_data, iter = 50)
  
  pred_prob <- predict(ada_model, test_data, type = "prob")
  
  if(is.matrix(pred_prob)) {
    # 如果有矩阵形式的概率
    if(ncol(pred_prob) == 2) {
      positive_prob <- pred_prob[, 2]  # 取正类的概率
    } else {
      # 如果只有一列，可能已经是正类概率
      positive_prob <- pred_prob
    }
  } else {
    # 如果是向量
    positive_prob <- pred_prob
  }
  
  # 创建ROC曲线对象
  # 需要将真实标签转换为数值（通常1代表正类，0代表负类）
  if(is.factor(test_data$Diagnosis)) {
    true_labels <- as.numeric(test_data$Diagnosis) - 1  # 转换为0/1
  } else {
    true_labels <- test_data$Diagnosis
  }
  
  # 计算ROC
  roc_obj <- roc(true_labels, positive_prob)
  
  # 输出AUC
  auc_value <- auc(roc_obj)
  auc[i-1,2]=auc_value
  print(paste("AUC面积:", round(auc_value, 4)))
  
}
write.csv(auc,"auc.csv") #输出30次预测全部的AUC
