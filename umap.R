data=read.csv("D:/R/文章全部数据/kqtrain_6vars.csv",header = TRUE)
#data=read.csv("D:/R/model/kqtrain_15vars.csv",header = TRUE)
data[which(data[,1]==1), 1]="1"
data[which(data[,1]==0), 1]="0"

library(umap)
features <- data[, 2:7]
scaled_features <- scale(features)

# 使用 umap 默认配置
umap_config <- umap.defaults
umap_result <- umap(scaled_features, config = umap_config)

# 提取结果
umap_df <- as.data.frame(umap_result$layout)
colnames(umap_df) <- c("UMAP1", "UMAP2")
umap_df$Diagnosis <- data$Diagnosis

ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = Diagnosis)) +
  geom_point(size = 2, alpha = 0.7) +
  labs(title = "UMAP 降维结果", x = "UMAP 维度1", y = "UMAP 维度2") +
  theme_classic()

library(cluster)
# 计算轮廓系数（需要距离矩阵或原始嵌入坐标）
dist_matrix <- dist(umap_df[, 1:2])   # 使用 UMAP 二维坐标
sil <- silhouette(as.numeric(umap_df[,3]), dist_matrix)
mean_sil <- mean(sil[, 3])            # 平均轮廓系数
print(mean_sil)


