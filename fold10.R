#fold20=kqtrain[1:2040,c("Diagnosis","TT","NEUT","PDW","A1","RBC","PT")]

fold20=kqtrain[1:2040,c("Diagnosis",selected_vars)]
fold20=kqtrain[1:2040,c("Diagnosis","TT","PT","NEUT","RBC","PDW","A1")]
fold20=kqtrain[1:2040,]

# 创建20折交叉验证索引, 示例数据fold10，将kqtrain数据去掉两行使得每折的样本量相同
n_folds <- 20
folds <- cut(seq(1, nrow(fold20)), breaks = n_folds, labels = FALSE)

# 存储每折的结果
res=data.frame(matrix(nrow=102)) #2040个样本划分20折，每折的样本量为102

# 进行交叉验证
i=1
for(i in 1:n_folds) {
  # 划分训练集和测试集
  test_indices <- which(folds == i, arr.ind = TRUE)
  train_data <- fold20[-test_indices, ]
  test_data <- fold20[test_indices, ]
  #注释内的部分分别带入后面的每种机器学习模型代码
  #############################################################################
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
  positive_prob=predictions 
  #############################################################################
  #每轮模型的预测结果应当包含一个数组，表示一列样本预测为正类的概率
  #将每列的结果存入auc中，得到20折交叉验证每折的auc
  true_labels=test_data$Diagnosis
  res=cbind(res,positive_prob,true_labels)
  
}

write.csv(res,"D:/R/fig/20fold_gbm_15vars.csv")

###############################################################################
train_data$Diagnosis=as.factor(train_data$Diagnosis)
test_data$Diagnosis=as.factor(test_data$Diagnosis)
# 训练SVM模型
svm_model <- svm(
  Diagnosis ~ .,
  data = train_data,
  kernel = "linear",
  scale = TRUE,
  probability = TRUE,
)

svm_probs <- predict(svm_model,
                     test_data,
                     probability = TRUE)

prob_matrix <- attr(svm_probs, "probabilities")

positive_class <- "1"
if (positive_class %in% colnames(prob_matrix)) {
  predictions <- prob_matrix[, positive_class]
} else {
  # 如果列名不是类别名，可能需要尝试不同的列
  predictions <- prob_matrix[, 2]  # 通常第二列是第二个类别的概率
}
positive_prob=predictions
################################################################################
train_data$Diagnosis=as.factor(train_data$Diagnosis)
test_data$Diagnosis=as.factor(test_data$Diagnosis)
# 训练随机森林模型（分类问题）
rf_model <- randomForest(Diagnosis ~ ., 
                         data = train_data,
                         ntree = 500,      # 树的数量
                         mtry = 6,         # 每棵树使用的特征数，分类问题默认sqrt(p)，回归默认p/3
                         importance = TRUE, # 计算变量重要性
                         proximity = TRUE)  # 计算样本接近度
#预测

pred_prob <- predict(rf_model, 
                     test_data, 
                     type="prob")

positive_prob <- pred_prob[, "1"]  # 假设"pos"是正类

################################################################################
train_data$Diagnosis=as.numeric(train_data$Diagnosis)
test_data$Diagnosis=as.numeric(test_data$Diagnosis)
train_x <- as.matrix(train_data[,2:31])
train_y <- as.numeric(train_data$Diagnosis)
test_x <- as.matrix(test_data[,2:31])
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
positive_prob=val_pred_prob
################################################################################
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
positive_prob=predictions[,2]
################################################################################
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
positive_prob=prob_predictions[,2]
################################################################################
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
################################################################################
# 训练决策树模型
# 使用所有特征预测Species
tree_model <- rpart(Diagnosis ~ ., data = train_data, method = "class")
prob_predictions <- predict(tree_model, test_data, type = "prob")
################################################################################

train_data[,2:31]=scale(train_data[,2:31])
test_data[,2:31]=scale(test_data[,2:31])
train_data$Diagnosis=as.factor(train_data$Diagnosis)
test_data$Diagnosis=as.factor(test_data$Diagnosis)

rec <- recipe(Diagnosis ~ ., data = train_data)

# 3. 配置模型超参数
config <- tabnet_config(
  epochs = 30,                  # 训练轮数，小数据集可先设小值测试
  batch_size = 64,              # 批大小
  decision_width = 24,          # 决策层宽度，控制模型容量
  attention_width = 24,         # 注意力层宽度，常与decision_width相同[citation:3]
  num_steps = 5,                # 网络步数（注意力步骤）
  penalty = 0.001,              # 稀疏性损失系数，值越大特征选择越稀疏[citation:3]
  virtual_batch_size = 128,     # 幽灵批归一化大小[citation:1]
  valid_split = 0             # 从训练集划分20%作为验证集[citation:3]
)

# 4. 训练模型
fit <- tabnet_fit(
  rec,                          # 使用配方
  data = train_data,
  config = config,
  verbose = TRUE                # 打印训练过程
)

# 在测试集上进行预测
test_predictions <- predict(fit, test_data, type="prob")

# 将预测概率与真实标签合并
results_df <- bind_cols(
  test_predictions,                       # 预测的概率
  truth = test_data$Diagnosis               # 真实的标签
)
################################################################################
train_data$Diagnosis=as.factor(train_data$Diagnosis)
test_data$Diagnosis=as.factor(test_data$Diagnosis)
#train_data$Diagnosis=as.numeric(train_data$Diagnosis)-1
#test_data$Diagnosis=as.numeric(test_data$Diagnosis)-1
# 训练模型
nb_model <- naiveBayes(Diagnosis ~ ., data = train_data)

# 查看模型摘要
print(nb_model)

# 预测类别
predictions_class <- predict(nb_model, test_data)

# 预测概率（用于计算AUC）
predictions_prob <- predict(nb_model, test_data, type = "raw")
positive_prob=predictions_prob[,2]