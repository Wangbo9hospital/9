install.packages("randomForest")
library(randomForest)
library(pROC)
library(randomForest)
library(caret)
library(ROCR)

#示例数据, kqtest, kqtrain为31列的dataframe

# 划分训练集和测试集
train_data=kqtrain[,c("Diagnosis",selected_vars)]
test_data=kqtest[,c("Diagnosis",selected_vars)]

train_data$Diagnosis=as.factor(train_data$Diagnosis)
test_data$Diagnosis=as.factor(test_data$Diagnosis)
#test_data <- iris[-train_index, ]
# 训练随机森林模型（分类问题）
rf_model <- randomForest(Diagnosis ~ ., 
                         data = train_data,
                         ntree = 500,      # 树的数量
                         mtry = 4,         # 每棵树使用的特征数，分类问题默认sqrt(p)，回归默认p/3
                         importance = TRUE, # 计算变量重要性
                         proximity = TRUE)  # 计算样本接近度
#预测

pred_prob <- predict(rf_model, 
                     test_data, 
                     type="prob")

positive_prob <- pred_prob[, "1"]  # 假设"pos"是正类

# 使用pROC包计算AUC
roc_obj <- roc(response = test_data$Diagnosis,  # 真实标签
               predictor = positive_prob,      # 预测概率
               levels = c(0, 1),       # 因子水平顺序
               direction = "<")                # 概率越大越可能是正类

# 获取AUC值
auc_value <- auc(roc_obj)
print(auc_value)


#获取变量重要性，并按重要性重新排列变量
df=data.frame(importance(rf_model))
df_sorted <- df[order(-df$MeanDecreaseAccuracy), ]
top15=rownames(df_sorted)[1:15]
train_top15=kqtrain[,c("Diagnosis",top15)]
test_top15=kqtest[,c("Diagnosis",top15)]

#纳入前i-1个变量进行预测，计算AUC
auc=data.frame(matrix(nrow = 15,ncol = 2))
auc[,1]=top15
i=2
for (i in 2:16)
{
  train_rf=train_top15[,1:i]
  test_rf=test_top15[,1:i]
  train_rf$Diagnosis=as.factor(train_rf$Diagnosis)
  test_rf$Diagnosis=as.factor(test_rf$Diagnosis)
  rf_model <- randomForest(Diagnosis ~ ., 
                           data = train_rf,
                           ntree = 500,      # 树的数量
                           mtry = floor(sqrt(i)),         # 每棵树使用的特征数，分类问题默认sqrt(p)，回归默认p/3
                           importance = TRUE, # 计算变量重要性
                           proximity = TRUE)

  # 获取测试集预测概率
  pred_prob <- predict(rf_model, test_rf, type="prob")
  pred_prob_positive <- pred_prob[, "1"]  # 正类概率（M为阳性）
  
  # 计算AUC
  roc_obj <- roc(response = test_rf$Diagnosis, 
                 predictor = pred_prob_positive,
                 levels = c("0", "1"))  # R为阴性，M为阳性
  auc[i-1,2]=auc(roc_obj)
}
write.csv(auc,"auc.csv") #输出30次预测所有的AUC数据


rf_pred <- predict(rf_model, train_data)

# 构建混淆矩阵（注意：要指定正类别，否则指标计算会错位）
# 这里将"Setosa"设为正类别（Positive Class）
conf_mat <- confusionMatrix(
  rf_pred,               # 预测值
  train_data$Diagnosis,     # 真实值
  positive = "1"    # 明确正类别（关键！）
)

# 提取并输出目标指标
# 1. 敏感度（Sensitivity）：真阳性率 TP/(TP+FN)
sensitivity <- conf_mat$byClass["Sensitivity"]
# 2. 特异性（Specificity）：真阴性率 TN/(TN+FP)
specificity <- conf_mat$byClass["Specificity"]
# 3. PPV（阳性预测值/精确率）：TP/(TP+FP)
ppv <- conf_mat$byClass["Precision"]
# 4. NPV（阴性预测值）：TN/(TN+FN)
npv <- conf_mat$byClass["Negative Predictive Value"]
# 5. F1分数：2*(PPV*敏感度)/(PPV+敏感度)
f1 <- conf_mat$byClass["F1"]

accuracy <- conf_mat$overall["Accuracy"]
# 整理并输出所有指标
metrics <- data.frame(
  指标 = c("敏感度(Sensitivity)", "特异性(Specificity)", "PPV","NPV", "准确度","F1分数"),
  数值 = c(sensitivity, specificity, ppv, npv, accuracy,f1)
)
print(metrics, row.names = FALSE)