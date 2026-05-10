autogluon=read.csv("D:/R/文章全部数据/test_autogluon.csv",header = T)
pred_prob=autogluon$pred_tabpfn
true_label=autogluon$Diagnosis
#pred_prob=pred_probs[,i]
#true_label=true_labels[,i]
pred_label <- ifelse(pred_prob > 0.5, 1, 0)
# 2. 计算混淆矩阵
TP <- sum(pred_label == 1 & true_label == 1)
TN <- sum(pred_label == 0 & true_label == 0)
FP <- sum(pred_label == 1 & true_label == 0)
FN <- sum(pred_label == 0 & true_label == 1)

# 3. 计算各项指标
# 敏感度（召回率）
sensitivity <- TP / (TP + FN)

# 特异性
specificity <- TN / (TN + FP)

# 阳性预测值（精确率）
ppv <- ifelse((TP + FP) > 0, TP / (TP + FP), NA)

# 阴性预测值
npv <- ifelse((TN + FN) > 0, TN / (TN + FN), NA)

# 准确度
accuracy <- (TP + TN) / (TP + TN + FP + FN)

f1 <- ifelse((TP + FP) > 0, 
             2 * TP / (2 * TP + FP + FN), 
             NA)

auc_value <- as.numeric(pROC::roc(true_label, pred_prob)$auc)
print(c(auc_value,sensitivity,specificity,ppv,npv,accuracy,f1))

roc1=roc(autogluon$Diagnosis,autogluon$pred)
roc2=roc(autogluon$Diagnosis,autogluon$pred_tabpfn)
delong_test <- roc.test(roc1, roc2, method = "delong")
cat(delong_test$p.value)
