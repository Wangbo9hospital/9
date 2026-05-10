# 加载包
library(torch)
library(tabnet)
library(tidymodels) # 用于数据预处理和评估，可选但推荐
library(recipes)    # 用于数据预处理
library(caret)

# 示例数据, kqtest, kqtrain为31列的dataframe
# 划分训练集和测试集
train_data <- kqtrain[,c("Diagnosis",selected_vars)]
test_data <- kqtest[,c("Diagnosis",selected_vars)]

train_data[,2:16]=scale(kqtrain[,2:16])
test_data[,2:16]=scale(kqtest[,2:16])
train_data$Diagnosis=as.factor(kqtrain$Diagnosis)
test_data$Diagnosis=as.factor(kqtest$Diagnosis)

rec <- recipe(Diagnosis ~ ., data = train_data)

# 3. 配置模型超参数
config <- tabnet_config(
  epochs = 10,                  # 训练轮数，小数据集可先设小值测试
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



# 计算AUC
# 注意：roc_auc()需要指定概率列（此处是".pred_setosa"）和真实标签列
auc_value <- results_df %>%
  roc_auc(truth = truth, 
          .pred_1,         # 这里使用"setosa"类别的概率作为正例概率
          event_level = "second"
          )           # 指定哪个类别作为“正例”

print(auc_value)
# 输出示例：# A tibble: 1 × 3
#   .metric .estimator .estimate
#   <chr>   <chr>          <dbl>
# 1 roc_auc binary         0.975


explain <- tabnet_explain(fit, test_data)
# 2. 提取全局特征重要性
#    $importance 是一个数据框，包含两列：feature（特征名）和 importance（重要性值）

importance <- data.frame(colMeans(explain$masks[[1]]))

importance <- importance[order(importance$colMeans.explain.masks..1..., decreasing = TRUE), , drop = FALSE]

top15=rownames(importance)[1:15]
train_top15=kqtrain[,c("Diagnosis",top15)]
test_top15=kqtest[,c("Diagnosis",top15)]
train_top15[,2:16]=as.numeric(as.matrix(train_top15[,2:16]))


auc=data.frame(matrix(nrow = 15,ncol = 2))
# 纳入前i-1个变量进行预测
auc[,1]=top15
i=2
for (i in 2:16)
{
    train_data=train_top15[,1:i]
    test_data=test_top15[,1:i]
    train_data$Diagnosis=as.factor(kqtrain$Diagnosis)
    test_data$Diagnosis=as.factor(kqtest$Diagnosis)
    rec <- recipe(Diagnosis ~ ., data = train_data)
    config <- tabnet_config(
                            epochs = 10,                  # 训练轮数，小数据集可先设小值测试
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
    verbose = TRUE               # 打印训练过程
  )
  
  # 在测试集上进行预测
  test_predictions <- predict(fit, test_data, type="prob")
  
  # 将预测概率与真实标签合并
  results_df <- bind_cols(
    test_predictions,                       # 预测的概率
    truth = test_data$Diagnosis               # 真实的标签
  )
  
  
  
  # 计算AUC

  # 注意：roc_auc()需要指定概率列（此处是".pred_setosa"）和真实标签列
  auc_value <- results_df %>%
    roc_auc(truth = truth, 
            .pred_1,         # 这里使用"setosa"类别的概率作为正例概率
            event_level = "second"
    )           # 指定哪个类别作为“正例”
  auc[i-1,2]=auc_value$.estimate
  print(auc_value)
}
write.csv(auc,"auc.csv") # 输出30次预测全部的AUC

