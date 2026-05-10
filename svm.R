install.packages("e1071")
library(e1071)

#示例数据, kqtest, kqtrain为31列的dataframe
train_svm=kqtrain[,c("Diagnosis",selected_vars)]
test_svm=kqtest[,c("Diagnosis",selected_vars)]
svm_model <- svm(Diagnosis ~ .,scale=TRUE, data = train_svm, kernel = "linear")
train_svm$Diagnosis=as.factor(train_svm$Diagnosis)
test_svm$Diagnosis=as.factor(test_svm$Diagnosis)

#计算变量重要性
w <- t(svm_model$coefs) %*% svm_model$SV
print(w)

# 计算每个特征的重要性（权重的绝对值）
importance <- abs(w)
colnames(importance) <- colnames(svm_model$SV)
print(importance)

# 排序
importance=data.frame(importance)
importance=t(importance)
df_sorted <- data.frame(importance[order(-importance[,]),])


top15=rownames(df_sorted)[1:15]
train_top15=kqtrain[,c("Diagnosis",top15)]
test_top15=kqtest[,c("Diagnosis",top15)]

#纳入前i-1个变量进行预测，计算AUC
auc=data.frame(matrix(nrow = 15,ncol = 2))
auc[,1]=top15
i=2
for (i in 2:15)
{
  train_svm=train_top30[,1:i]
  test_svm=test_top30[,1:i]
  train_svm$Diagnosis=as.factor(train_svm$Diagnosis)
  test_svm$Diagnosis=as.factor(test_svm$Diagnosis)
  svm_model <- svm(Diagnosis ~ .,
                   data = train_svm,
                   kernel = "linear",
                   scale = TRUE,
                   probability = TRUE,  # 必须设置为TRUE
                   )
  # 获取测试集预测概率
  svm_probs <- predict(svm_model,
                       test_svm,
                       probability = TRUE)
  prob_matrix <- attr(svm_probs, "probabilities")
  
  # 通常第一列是负类，第二列是正类
  # 这里假设"setosa"是正类（可根据实际情况调整）
  positive_class <- "1"
  if (positive_class %in% colnames(prob_matrix)) {
    predictions <- prob_matrix[, positive_class]
  } else {
    # 如果列名不是类别名，可能需要尝试不同的列
    predictions <- prob_matrix[, 2]  # 通常第二列是第二个类别的概率
  }
  
  # 实际标签（转换为数值：0/1）
  actual <- ifelse(test_svm$Diagnosis == positive_class, 1, 0)
  roc_obj <- roc(actual, predictions)
 
  # 获取详细的AUC值
  auc_value <- auc(roc_obj)
  auc[i-1,2]=auc_value
  
}
write.csv(auc,"auc.csv") #输出30次预测所有的AUC数据
