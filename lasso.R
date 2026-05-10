library(glmnet)
library(caret)

x=scale(train[,2:52])
y=as.numeric(train$Diagnosis)-1

# 3. 创建训练集和测试集
set.seed(123)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
x_train <- x[train_index, ]
y_train <- y[train_index]
x_test <- x[-train_index, ]
y_test <- y[-train_index]

# 4. 交叉验证选择lambda
cv_model <- cv.glmnet(x, y, 
                      alpha = 1,
                      nfolds = 10,
                      standardize = TRUE)

# 5. 使用最优lambda
best_lambda <- cv_model$lambda.min
final_model <- glmnet(x, y, 
                      alpha = 1, 
                      lambda = best_lambda,
                      standardize = TRUE)

# 6. 模型评估
predictions <- predict(final_model, newx = x_test)
mse <- mean((predictions - y_test)^2)
rmse <- sqrt(mse)

cat("测试集RMSE:", rmse, "\n")

# 7. 查看系数
coef_matrix <- as.matrix(coef(final_model))
selected_vars <- coef_matrix[coef_matrix != 0, ]
print("选择的变量及系数:")
print(selected_vars)

# 8. 可视化
plot(cv_model, main = "交叉验证误差")
plot(final_model, xvar = "lambda", label = TRUE, lwd=2)
plot(model,xvar="lambda",label = TRUE)
abline(v = log(best_lambda), col = "red", lty = 2)
