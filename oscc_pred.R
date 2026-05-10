oscc=read.csv("D:/R/带病理.csv",header = T)
cox(oscc$SIZE,oscc$PRED)

# 假设heightweight是包含身高和体重数据的数据框
fit=model <- lm(PRED ~ SIZE, data=oscc)

summary(model)
values=summary(model)

cor(oscc$SIZE,oscc$PRED,method="spearman")
plot(oscc$SIZE,oscc$PRED)
wilcox.test(oscc[which(oscc[,8]==1),9],oscc[oscc[,8]==3,9])
