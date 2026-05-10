#示例数据：kqtrain是31列的dataframe, 第一列是因变量
#以训练集为例
data=kqtrain 
data$Diagnosis=as.numeric(data$Diagnosis)

model1 <- lm(Diagnosis ~ .  
             , data = data)
vif=data.frame(car::vif(model1))
vif #显示每个变量的方差膨胀因子，＞10表示该变量与其他的变量可能存在共线性

