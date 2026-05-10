library(gbm)
library(ROCR)
library(pROC)

#示例数据, kqtest, kqtrain为31列的dataframe
train_data <- kqtrain[,c("Diagnosis",selected_vars)]
test_data <- kqtest[,c("Diagnosis",selected_vars)]
train_dat=kqtrain
test_data=kqtest

# 转换目标变量为0/1数值型
train_data$Diagnosis <- as.numeric(as.character(train_data$Diagnosis))
test_data$Diagnosis <- as.numeric(as.character(test_data$Diagnosis))

# 训练模型
gbm_model <- gbm(
  formula = Diagnosis ~ .,  # 公式
  data = train_data[, -5],       # 移除原始的因子型target列
  distribution = "bernoulli",    # 二分类分布
  n.trees = 100,                 # 树的数量
  interaction.depth = 3,         # 树深度
  shrinkage = 0.1,               # 学习率
  cv.folds = 5,                  # 交叉验证折数
  n.minobsinnode = 10,           # 节点最小观测数
  verbose = FALSE
)

# 选择最优树数量
best_iter <- gbm.perf(gbm_model, method = "cv")

# 预测
predictions <- predict(gbm_model, 
                       newdata = test_data,
                       n.trees = best_iter,
                       type = "response")

#输出AUC
roc_curve <- roc(test_data$Diagnosis, predictions)
auc(roc_curve)


# 获取变量重要性
importance_gbm <- summary(gbm_model, plotit = FALSE)

top15=importance_gbm$var[1:15] 
top15[15]="TP" #查看排序结果，未参与排序的变量手动添加在最后
train_top15=kqtrain[,c("Diagnosis",top15)]
test_top15=kqtest[,c("Diagnosis",top15)]
auc=data.frame(matrix(nrow = 15,ncol = 2))
auc[,1]=top15
#纳入前i-1个变量进行预测
i=2
for (i in 2:16)
{
  train_data=train_top15[,1:i]
  test_data=test_top15[,1:i]
  # 转换目标变量为0/1数值型
  train_data$Diagnosis <- as.numeric(as.character(train_data$Diagnosis))
  test_data$Diagnosis <- as.numeric(as.character(test_data$Diagnosis))
  
  # 训练模型
  gbm_model <- gbm(
    formula = Diagnosis ~ .,  # 公式
    data = train_data[, -5],       # 移除原始的因子型target列
    distribution = "bernoulli",    # 二分类分布
    n.trees = 100,                 # 树的数量
    interaction.depth = 3,         # 树深度
    shrinkage = 0.1,               # 学习率
    cv.folds = 5,                  # 交叉验证折数
    n.minobsinnode = 10,           # 节点最小观测数
    verbose = FALSE
  )
  
  # 选择最优树数量
  best_iter <- gbm.perf(gbm_model, method = "cv")
  
  # 预测
  predictions <- predict(gbm_model, 
                         newdata = test_data,
                         n.trees = best_iter,
                         type = "response")
  
  # 评估
  roc_curve <- roc(test_data$Diagnosis, predictions)
  auc[i-1,2]=auc(roc_curve)
  print(auc(roc_curve))
}
write.csv(auc,"auc.csv") #输出30次预测的全部AUC
