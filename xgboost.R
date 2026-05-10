# 加载包
library(xgboost)
library(caret)
library(dplyr)
library(Matrix)

#示例数据, kqtest, kqtrain为31列的dataframe
train_data=kqtrain[,c("Diagnosis",selected_vars)]
test_data=kqtest[,c("Diagnosis",selected_vars)]
train_x <- as.matrix(train_data[,2:16])
train_y <- as.numeric(train_data$Diagnosis)
test_x <- as.matrix(test_data[,2:16])
test_y <- as.numeric(test_data$Diagnosis)

# 创建DMatrix格式（XGBoost专用格式）
dtrain <- xgb.DMatrix(data = train_x, label = train_y)
dtest <- xgb.DMatrix(data = test_x, label = test_y)

# 定义参数
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",  # 二分类逻辑回归
  eval_metric = "logloss",        # 评估指标：对数损失
  # eval_metric = "auc",          # 或者使用AUC
  # eval_metric = "error",        # 或者使用错误率
  
  # 树参数
  max_depth = 6,                  # 树的最大深度
  eta = 0.3,                      # 学习率
  gamma = 0,                      # 最小损失减少值
  min_child_weight = 1,           # 子节点最小权重和
  
  # 正则化参数
  subsample = 0.8,                # 每棵树随机采样的比例
  colsample_bytree = 0.8,         # 每棵树随机采样的列比例
  
  # 其他参数
  nthread = 4,                     # 并行线程数
  base_score = 0.5
)

# 设置早停规则
watchlist <- list(train = dtrain, test = dtest)


# 计算训练集中正类的比例作为合理的 base_score
unique(train_y)

# 训练XGBoost模型
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,                   # 迭代次数
  evals = watchlist,
  early_stopping_rounds = 10,     # 早停轮数
  verbose = 1                     # 显示训练过程
)


val_pred_prob <- predict(xgb_model, dtest)
# 2.2 计算AUC
roc_obj <- roc(test_data$Diagnosis, val_pred_prob)
auc(roc_obj)


#排列变量重要性
importance_matrix <- xgb.importance(
  model = xgb_model,
  feature_names = colnames(train_x)  # 传入特征名称
)

# 查看全部特征重要性
print("特征重要性矩阵：")
print(importance_matrix)

# 查看前30个特征
print("前15个重要特征：")
top15_features <- head(importance_matrix, 15)
print(top15_features)

top15=top15_features$Feature[1:15]
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
  train_x <- as.matrix(train_data[,2:i])
  train_y <- as.numeric(train_data$Diagnosis)
  test_x <- as.matrix(test_data[,2:i])
  test_y <- as.numeric(test_data$Diagnosis)
  
  # 创建DMatrix格式（XGBoost专用格式）
  dtrain <- xgb.DMatrix(data = train_x, label = train_y)
  dtest <- xgb.DMatrix(data = test_x, label = test_y)
  
  # 定义参数
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",  # 二分类逻辑回归
    eval_metric = "logloss",        # 评估指标：对数损失
    # eval_metric = "auc",          # 或者使用AUC
    # eval_metric = "error",        # 或者使用错误率
    
    # 树参数
    max_depth = 6,                  # 树的最大深度
    eta = 0.3,                      # 学习率
    gamma = 0,                      # 最小损失减少值
    min_child_weight = 1,           # 子节点最小权重和
    
    # 正则化参数
    subsample = 0.8,                # 每棵树随机采样的比例
    colsample_bytree = 0.8,         # 每棵树随机采样的列比例
    
    # 其他参数
    nthread = 4,                     # 并行线程数
    base_score = 0.5
  )
  
  # 设置早停规则
  watchlist <- list(train = dtrain, test = dtest)
  
  
  # 计算训练集中正类的比例作为合理的 base_score
  
  
  # 训练XGBoost模型
  xgb_model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 100,                   # 迭代次数
    evals = watchlist,
    early_stopping_rounds = 10     # 早停轮数
    #verbose = 1                     # 显示训练过程
  )
  
  test_data$Diagnosis=as.numeric(test_data$Diagnosis)
  val_pred_prob <- predict(xgb_model, dtest)
  # 2.2 计算AUC
  roc_obj <- roc(test_data$Diagnosis, val_pred_prob)
  auc[i-1,2]=auc(roc_obj)
}
write.csv(auc,"auc.csv") #输出30次预测的全部AUC
