#下面是LASSO回归的包：没有的需要下载
library(glmnet)
data=data_complete[,colnames(kqtrain)]
data=data_complete
#1数据预处理
##导入数据
data=read.csv("D:/R/kq/kq_all.csv")
#查看变量类型，如果字符型需要转化成因子
str(data)
#查看变量名
names(data)
#本例使用的变量``
#自变量：分类变量：sex(5列)  ph.ecog(6列)   
#       连续变量：age ph.karno pat.karno meal.cal wt.loss
#因变量：status
#示例数据：kqtrain是31列的dataframe，第一列是因变量
data=kqtrain
x<-as.matrix(data[,2:30])#指定自变量为矩阵
y<-as.matrix(data$Diagnosis)#指定因变量为矩阵
model<-glmnet(x,y,family = "binomial")
cv_model=cv.glmnet(x,y,family="binomial")
plot(model,xvar="lambda",label = TRUE) #显示系数路径图

coef_1se <- coef(cv_model, s = "lambda.1se") #可以提取lambda.min或lambda.1se
print(coef_1se)


cv_fit <- cv.glmnet(x, y, alpha = 1, nfolds = 10)
plot(cv_fit)

# 2. 查看lambda值
cat("最佳lambda值：\n")
cat("lambda.min:", cv_fit$lambda.min, "\n")
cat("lambda.1se:", cv_fit$lambda.1se, "\n\n")

# 3. 提取lambda.1se处的系数
coef_1se <- coef(cv_fit, s = "lambda.1se")
print(coef_1se)


selected_vars <- coef_1se@Dimnames[[1]][which(coef_1se != 0)]
selected_vars <- selected_vars[selected_vars != "(Intercept)"]

cat("在lambda.1se下保留的变量（共", length(selected_vars), "个）：\n")
print(selected_vars) #显示在lambda.1se下保留的变量

x=x[,selected_vars]

