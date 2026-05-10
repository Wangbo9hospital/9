library(pROC)
pred=read.csv("D:/R/文章全部数据/test_autogluon.csv",header = TRUE)
roc1=roc(pred[,2],pred[,1])
roc2=roc(pred[,2],pred[,3])
delong_test <- roc.test(roc1, roc2, method = "delong")
cat(delong_test$p.value)

pred=read.csv("D:/R/文章全部数据/pred_15vars.csv",header = FALSE)
for (i in 1:15)
{
  roc1=roc(pred[,16],pred[,15])
  roc2=roc(pred[,16],pred[,i])
  delong_test <- roc.test(roc1, roc2, method = "delong")
  cat("15 vars vs",i,"vars delong test p=",delong_test$p.value,"\n")
  
}

pred=read.csv("D:/R/文章全部数据/fold20_15vars.csv",header = F)
for (i in 1:15)
{
  cat(shapiro.test(pred[,i])$p.value,"\n")
  cat("15 vars vs",i,"vars wilcox test p=",t.test(pred[,i],pred[,15])$p.value,"\n")
}
shapiro.test()

pred=read.csv("D:/R/fig/2.20fold/20fold_rf_15vars.csv",header = TRUE)
pred2=pred[,1]
label2=label[,2]
for (i in 2:20)
{
  pred2=c(pred2,pred[,2*i-1])
  label2=c(label2,pred[,2*i])
}

pred=read.csv("D:/R/fig/2.20fold/20fold_xgboost_15vars.csv",header = TRUE)
pred3=pred[,1]
label3=label[,2]
for (i in 2:20)
{
  pred3=c(pred3,pred[,2*i-1])
  label3=c(label3,pred[,2*i])
}

pred=read.csv("D:/R/fig/2.20fold/20fold_ada_15vars.csv",header = TRUE)
pred4=pred[,1]
label4=label[,2]
for (i in 2:20)
{
  pred4=c(pred4,pred[,2*i-1])
  label4=c(label4,pred[,2*i])
}

pred=read.csv("D:/R/fig/2.20fold/20fold_knn_15vars.csv",header = TRUE)
pred5=pred[,1]
label5=label[,2]
for (i in 2:20)
{
  pred5=c(pred5,pred[,2*i-1])
  label5=c(label5,pred[,2*i])
}

roc1=roc(label1,pred1)
roc2=roc(label2,pred2)
roc3=roc(label3,pred3)
roc4=roc(label4,pred4)
roc5=roc(label5,pred5)

delong_test <- roc.test(roc1, roc3, method = "delong")
print(delong_test)

remotes::install_github("cardiomoon/multipleROC")
remotes::install_git("https://gitee.com/swcyo/multipleROC/")
library(multipleROC)
df=data.frame(cbind(label1,pred1,label2,pred2,label3,pred3,label4,pred4,label5,pred5))
p1=multipleROC(label1~pred1,data=df)
p2=multipleROC(label2~pred2,data=df)
p3=multipleROC(label3~pred3,data=df)
p4=multipleROC(label4~pred4,data=df)
p5=multipleROC(label5~pred5,data=df)
plot_ROC(list(p1,p2,p3,p4,p5),
         show.points = T, 
         show.eta = F, 
         show.sens = F, 
         show.AUC = F, 
         facet = F )

pred=read.csv("D:/R/fig/3.5models/fold15_auc.csv",header = FALSE)
pred=pred[,1:15]
for (i in 1:14)
{
  cat("15 vars vs",i,"vars wilcox test p=",wilcox.test(pred[,15],pred[,i])$p.value,"\n")
}