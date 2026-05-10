#计算敏感度、特异性、阳性预测值、阴性预测值、准确度、F1分数
#需要的数据：pred_prob是一个数值型数组，第n个元素表示第n个样本被预测为阳性的概率
#            true_label是一个数值型数组，第n个元素表示第n个样本的真实值（阳性或阴性）

pred_probs=read.csv("D:/R/fig/2.20fold/20fold_tabpfn_15vars_predprobs.csv",header = FALSE)
true_labels=read.csv("D:/R/fig/2.20fold/20fold_tabpfn_15vars_truelabels.csv",header = FALSE)
#也可从其他数据中提取pred_prob和true_label
fold20perf=data.frame(matrix(nrow = 1, ncol = 8))

res=read.csv("D:/R/fig/2.20fold/20fold_rf_6vars.csv")
res=read.csv("D:/R/fig/20fold_gbm_15vars.csv")
res=read.csv("D:/R/3.2/20fold_tabpfn_15vars.csv")

auc=data.frame(matrix(nrow = 20,ncol = 10))


res=read.csv("D:/R/kq/20fold/res_dt.csv")


i=1
tauc=tsens=tspec=tppv=tnpv=tacc=tf1=0
auc
for (i in 1:20)
{
  
  pred_prob=res[,i*2-1]
  true_label=res[,i*2] 
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
  auc[i,10]=auc_value
  tsens=tsens+sensitivity*0.05
  tspec=tspec+specificity*0.05
  tauc=tauc+auc_value*0.05
  tppv=tppv+ppv*0.05
  tnpv=tnpv+npv*0.05
  tacc=tacc+accuracy*0.05
  tf1=tf1+f1*0.05 #统计20fold结果的平均值
}
print(c(tauc,tsens,tspec,tppv,tnpv,tacc,tf1)) #输出计算结果
fold20perf=rbind(fold20perf,c("knn",tauc,tsens,tspec,tppv,tnpv,tacc,tf1))

write.csv(fold20perf,"D:/R/fig/2.20fold/fold20perf.csv")
wilcox.test(auc$X1,auc$X2)
auc=read.csv("D:/R/文章全部数据/7.20折交叉验证（15个变量的），所有的机器学习加TabPFN, COX非参数比了每个模型20个AUC数值（只用了训练集）/20auc_10models.csv")
auc=read.csv("D:/R/3.2/20auc_10models.csv")
wilcox.test(auc$tabpfn,auc$knn)
for (i in 2:31)
  cat(wilcox.test(kqtrain[,i],kqtest[,i])$p.value,"\n")