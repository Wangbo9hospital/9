install.packages("missForest")
library(missForest)
#输入数据为一个dataframe，行为样本，列为变量，其中包含NA值，对其进行插补
data_missing=read.csv("D:/R/kq/kq_all2.csv")

table(data_missing$Diagnosis)
# 查看缺失情况
summary(data_missing)
colSums(is.na(data_missing))
data_missing <- data_missing %>%
  select(where(~ mean(is.na(.)) <= 0.05))
data_missing <- as.data.frame(sapply(data_missing, as.numeric))

# 进行随机森林插补
imputed_data <- missForest(
  data_missing,
  maxiter = 10,           # 最大迭代次数
  ntree = 100,           # 每棵树的数量
  variablewise = FALSE,  # 计算每个变量的OOB误差
  verbose = TRUE        # 显示迭代过程
)

# 提取插补后的数据
data_complete <- imputed_data$ximp
data=data_complete
# 查看插补效果
imputed_data$OOBerror  # OOB误差（越小越好）


###
set.seed(123)
n <- nrow(data)
sample_size <- round(0.4 * n)
sample_indices <- sample(1:n, size = sample_size)
kqtrain=data[-sample_indices,]
kqtest=data[sample_indices,]
kqval=kqtest[681:1362,]
kqtest=kqtest[1:680,]
table(kqval$Diagnosis)
write.csv(kqtest,"kqtest.csv")

write.csv(kqval,"kqval.csv")


kqtrain[,32]=kqtrain$PLT*kqtrain$NEUT/kqtrain$LYMPH
kqtrain[,33]=kqtrain$NEUT*kqtrain$MONO/kqtrain$LYMPH
kqtrain[,34]=kqtrain$NEUT/kqtrain$LYMPH
kqtrain[,35]=kqtrain$NEUT/(kqtrain$NEUT-kqtrain$LYMPH)
kqtrain[,36]=kqtrain$PLT/kqtrain$LYMPH
kqtrain[,37]=kqtrain$MONO/kqtrain$LYMPH
kqtrain[,38]=kqtrain$NEUT/kqtrain$PLT
kqtrain[,39]=(kqtrain$MONO+kqtrain$NEUT)/kqtrain$LYMPH
kqtrain[,40]=kqtrain$NEUT*kqtrain$MONO*kqtrain$PLT/kqtrain$LYMPH
kqtrain[,41]=(kqtrain$NEUT+kqtrain$LYMPH+kqtrain$MONO+kqtrain$EO)/(kqtrain$RBC*kqtrain$MCH)
kqtrain[,42]=kqtrain$ALB*(kqtrain$RBC*kqtrain$MCH)*kqtrain$LYMPH/kqtrain$PLT
kqtrain[,43]=kqtrain$ALB+0.005*kqtrain$LYMPH
colnames(kqtrain)[32:43]=c("SII","SIRI","NLR","dNLR","PLR","MLR","NPR","NMLR","AISI","WBC_Hb","HALP","PNI")

kqtest[,32]=kqtest$PLT*kqtest$NEUT/kqtest$LYMPH
kqtest[,33]=kqtest$NEUT*kqtest$MONO/kqtest$LYMPH
kqtest[,34]=kqtest$NEUT/kqtest$LYMPH
kqtest[,35]=kqtest$NEUT/(kqtest$NEUT-kqtest$LYMPH)
kqtest[,36]=kqtest$PLT/kqtest$LYMPH
kqtest[,37]=kqtest$MONO/kqtest$LYMPH
kqtest[,38]=kqtest$NEUT/kqtest$PLT
kqtest[,39]=(kqtest$MONO+kqtest$NEUT)/kqtest$LYMPH
kqtest[,40]=kqtest$NEUT*kqtest$MONO*kqtest$PLT/kqtest$LYMPH
kqtest[,41]=(kqtest$NEUT+kqtest$LYMPH+kqtest$MONO+kqtest$EO)/(kqtest$RBC*kqtest$MCH)
kqtest[,42]=kqtest$ALB*(kqtest$RBC*kqtest$MCH)*kqtest$LYMPH/kqtest$PLT
kqtest[,43]=kqtest$ALB+0.005*kqtest$LYMPH
colnames(kqtest)[32:43]=c("SII","SIRI","NLR","dNLR","PLR","MLR","NPR","NMLR","AISI","WBC_Hb","HALP","PNI")

sxtest=data_complete

sxtrain$Diagnosis=as.factor(sxtrain$Diagnosis)
sxtest$Diagnosis=as.factor(sxtest$Diagnosis)
sxtest=sxtest[,c(51,1:50)]

write.csv(sxtrain,"sxtrain_completed.csv")
write.csv(sxtest,"sxtest_completed.csv")
names(train)
data=train[,c("Diagnosis","WBC","NEUT","LYMPH","MONO",
               "EO","RBC","Hb","HCT","RDW.CV",
               "PLT","MPV","PDW","PT",
               "APTT","TT","FIB","TBIL","DBIL",
               "TP","ALB","ALT","AST","GGT",
               "ALP","TBA","CG","PA","UREA","UA","GLU")]
train=data
test=test[,colnames(train)]
