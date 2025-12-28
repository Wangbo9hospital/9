# Comprehensive Model Performance Comparison Across All Datasets - Random Forest
# Load required packages
library(readxl)
library(caret)
library(e1071)
library(pROC)
library(openxlsx)
library(ggplot2)
library(dplyr)
library(reshape2)
library(gridExtra)
library(patchwork)
library(randomForest)

# ==================== Configuration ====================
cat("=== Comprehensive RF Model Performance Comparison ===\n")

# File paths
model_path <- "final_rf_model.rds"
features_path <- "final_features.rds"
train_data_path <- "C:/Users/Bo Wang/Desktop/Training Cohort.xlsx"
test_data_path <- "C:/Users/Bo Wang/Desktop/Internal Validation Cohort.xlsx"  # 测试集
external_data_path <- "C:/Users/Bo Wang/Desktop/External Validation Cohort.xlsx"  # 外部验证集
prospective_data_path <- "C:/Users/Bo Wang/Desktop/Prospective Cohort.xlsx"  # 前瞻性数据集

# Output directories
output_dir <- "Comprehensive_RF_Model_Performance"
plots_dir <- file.path(output_dir, "Plots")
data_dir <- file.path(output_dir, "Data")

# Create directories
if(!dir.exists(output_dir)) dir.create(output_dir)
if(!dir.exists(plots_dir)) dir.create(plots_dir)
if(!dir.exists(data_dir)) dir.create(data_dir)

# ==================== Load Model and Features ====================
cat("Step 1: Loading RF Model and Features...\n")

# Load trained model and features
final_rf_model <- readRDS(model_path)
final_features <- readRDS(features_path)

cat("Random Forest Model loaded successfully\n")
cat("Number of features:", length(final_features), "\n")
cat("Model type: Random Forest\n")

# ==================== Unified Prediction Function for RF ====================
predict_random_forest <- function(model, newdata) {
  # Check model type
  model_class <- class(model)[1]
  
  if(model_class == "randomForest") {
    # Standard randomForest package
    pred_probs <- predict(model, newdata = newdata, type = "prob")
    pred_class <- predict(model, newdata = newdata, type = "response")
  } else if(model_class == "train") {
    # caret-trained model
    pred_probs <- predict(model, newdata = newdata, type = "prob")
    pred_class <- predict(model, newdata = newdata)
  } else {
    # Try generic predict
    pred_probs <- predict(model, newdata = newdata, type = "prob")
    pred_class <- predict(model, newdata = newdata)
    
    # If probabilities not available, try alternative
    if(is.null(pred_probs)) {
      pred_class <- predict(model, newdata = newdata)
      # Create dummy probabilities
      classes <- levels(pred_class)
      pred_probs <- matrix(0, nrow = length(pred_class), ncol = length(classes))
      colnames(pred_probs) <- classes
      for(i in 1:length(pred_class)) {
        pred_probs[i, as.character(pred_class[i])] <- 1
      }
    }
  }
  
  return(list(probabilities = pred_probs, predictions = pred_class))
}

# ==================== Performance Metrics Function ====================
calculate_performance_metrics <- function(true_labels, pred_probs, pred_class = NULL, threshold = 0.5) {
  if(is.null(pred_class)) {
    pred_class <- ifelse(pred_probs > threshold, 1, 0)
  }
  
  true_numeric <- as.numeric(true_labels) - 1
  pred_numeric <- as.numeric(pred_class) - 1
  
  cm <- table(Predicted = pred_numeric, Actual = true_numeric)
  
  # Ensure 2x2 matrix for binary classification
  if(nrow(cm) == 1) {
    if(rownames(cm) == "0") {
      cm <- rbind(cm, c(0, 0))
    } else {
      cm <- rbind(c(0, 0), cm)
    }
    rownames(cm) <- c("0", "1")
  }
  if(ncol(cm) == 1) {
    if(colnames(cm) == "0") {
      cm <- cbind(cm, c(0, 0))
    } else {
      cm <- cbind(c(0, 0), cm)
    }
    colnames(cm) <- c("0", "1")
  }
  
  TP <- cm[2, 2]
  TN <- cm[1, 1]
  FP <- cm[2, 1]
  FN <- cm[1, 2]
  
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  sensitivity <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
  specificity <- ifelse((TN + FP) > 0, TN / (TN + FP), 0)
  ppv <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
  npv <- ifelse((TN + FN) > 0, TN / (TN + FN), 0)
  f1 <- ifelse((sensitivity + ppv) > 0, 2 * sensitivity * ppv / (sensitivity + ppv), 0)
  balanced_accuracy <- (sensitivity + specificity) / 2
  
  # AUC calculation
  auc_value <- tryCatch({
    if(length(unique(true_numeric)) == 2) {
      roc_obj <- roc(true_numeric, pred_probs)
      auc(roc_obj)
    } else {
      0.5
    }
  }, error = function(e) 0.5)
  
  # ROC object
  roc_obj <- NULL
  if(length(unique(true_numeric)) == 2) {
    roc_obj <- tryCatch({
      roc(true_numeric, pred_probs)
    }, error = function(e) NULL)
  }
  
  return(list(
    Accuracy = accuracy,
    Sensitivity = sensitivity,
    Specificity = specificity,
    PPV = ppv,
    NPV = npv,
    F1_Score = f1,
    Balanced_Accuracy = balanced_accuracy,
    AUC = auc_value,
    Confusion_Matrix = cm,
    ROC_Obj = roc_obj
  ))
}

# ==================== Data Processing Function ====================
process_dataset <- function(data_path, dataset_name, train_data_ref) {
  cat("Processing", dataset_name, "...\n")
  
  # Load data
  data <- read_excel(data_path)
  cat("  Data dimensions:", dim(data), "\n")
  
  # Check for Diagnosis column
  has_labels <- "Diagnosis" %in% names(data)
  if(has_labels) {
    data$Diagnosis <- as.factor(data$Diagnosis)
    cat("  Diagnosis distribution:\n")
    print(table(data$Diagnosis))
  } else {
    cat("  No Diagnosis column found\n")
    data$Diagnosis <- NA
  }
  
  # Handle missing values using training data statistics
  missing_count <- colSums(is.na(data))
  if(sum(missing_count) > 0) {
    cat("  Imputing missing values...\n")
    for(feature in names(data)) {
      if(feature %in% names(train_data_ref) && feature != "Diagnosis") {
        na_indices <- is.na(data[[feature]])
        if(any(na_indices)) {
          if(is.numeric(train_data_ref[[feature]])) {
            impute_value <- median(train_data_ref[[feature]], na.rm = TRUE)
            data[na_indices, feature] <- impute_value
          } else {
            freq_table <- table(train_data_ref[[feature]])
            impute_value <- names(freq_table)[which.max(freq_table)]
            data[na_indices, feature] <- impute_value
          }
        }
      }
    }
  }
  
  # Check and align features
  missing_features <- setdiff(final_features, names(data))
  if(length(missing_features) > 0) {
    cat("  Adding missing features:", paste(missing_features, collapse = ", "), "\n")
    for(feature in missing_features) {
      if(feature %in% names(train_data_ref)) {
        if(is.numeric(train_data_ref[[feature]])) {
          data[[feature]] <- median(train_data_ref[[feature]], na.rm = TRUE)
        } else {
          freq_table <- table(train_data_ref[[feature]])
          data[[feature]] <- names(freq_table)[which.max(freq_table)]
        }
      } else {
        data[[feature]] <- 0
      }
    }
  }
  
  # Prepare features for prediction
  features <- data[, final_features, drop = FALSE]
  
  # Ensure factor levels match training data
  for(feature in final_features) {
    if(is.factor(train_data_ref[[feature]]) && is.character(features[[feature]])) {
      features[[feature]] <- factor(features[[feature]], 
                                    levels = levels(train_data_ref[[feature]]))
    }
  }
  
  cat("  Feature dimensions:", dim(features), "\n")
  
  # Make predictions using RF
  pred_results <- predict_random_forest(final_rf_model, features)
  predictions <- pred_results$predictions
  probabilities <- pred_results$probabilities
  
  # Extract probability for positive class (assuming binary classification)
  if(is.matrix(probabilities) && ncol(probabilities) == 2) {
    pred_prob <- probabilities[, 2]
  } else if(is.matrix(probabilities)) {
    pred_prob <- apply(probabilities, 1, max)
  } else {
    pred_prob <- as.numeric(probabilities)
  }
  
  # Create results dataframe
  results <- data.frame(
    Dataset = dataset_name,
    SampleID = if("SampleID" %in% names(data)) data$SampleID else 1:nrow(data),
    Actual_Diagnosis = if(has_labels) data$Diagnosis else NA,
    Predicted_Diagnosis = predictions,
    Predicted_Probability = pred_prob
  )
  
  # Calculate performance metrics if labels available
  metrics <- NULL
  if(has_labels && !all(is.na(data$Diagnosis))) {
    metrics <- calculate_performance_metrics(
      data$Diagnosis, 
      results$Predicted_Probability,
      results$Predicted_Diagnosis
    )
  }
  
  return(list(
    name = dataset_name,
    data = data,
    features = features,
    results = results,
    metrics = metrics,
    has_labels = has_labels
  ))
}

# ==================== Process All Datasets ====================
cat("Step 2: Processing All Datasets with RF Model...\n")

# Load training data for reference
train_data_ref <- read_excel(train_data_path)
train_data_ref$Diagnosis <- as.factor(train_data_ref$Diagnosis)

# Process all datasets
datasets <- list()

# Training set
datasets$train <- process_dataset(train_data_path, "Training Set", train_data_ref)

# Test set (if available)
if(file.exists(test_data_path)) {
  datasets$test <- process_dataset(test_data_path, "Test Set", train_data_ref)
} else {
  cat("Test set not found at:", test_data_path, "\n")
}

# External validation set (if available)
if(file.exists(external_data_path)) {
  datasets$external <- process_dataset(external_data_path, "External Validation Set", train_data_ref)
} else {
  cat("External validation set not found at:", external_data_path, "\n")
}

# Prospective dataset
datasets$prospective <- process_dataset(prospective_data_path, "Prospective Set", train_data_ref)

# ==================== Performance Comparison ====================
cat("Step 3: RF Performance Comparison...\n")

# Collect performance metrics from all datasets
performance_comparison <- data.frame()
all_results <- data.frame()

for(dataset_name in names(datasets)) {
  dataset <- datasets[[dataset_name]]
  all_results <- rbind(all_results, dataset$results)
  
  if(!is.null(dataset$metrics)) {
    metrics_df <- data.frame(
      Dataset = dataset$name,
      Accuracy = dataset$metrics$Accuracy,
      Sensitivity = dataset$metrics$Sensitivity,
      Specificity = dataset$metrics$Specificity,
      PPV = dataset$metrics$PPV,
      NPV = dataset$metrics$NPV,
      F1_Score = dataset$metrics$F1_Score,
      Balanced_Accuracy = dataset$metrics$Balanced_Accuracy,
      AUC = dataset$metrics$AUC,
      Samples = nrow(dataset$results)
    )
    performance_comparison <- rbind(performance_comparison, metrics_df)
  }
}

# Print performance comparison
if(nrow(performance_comparison) > 0) {
  cat("\n=== RF PERFORMANCE COMPARISON ===\n")
  print(performance_comparison)
}

# ==================== Comprehensive Visualization ====================
cat("Step 4: Generating Comprehensive RF Visualizations...\n")

# 1. Performance Metrics Comparison (Bar Plot)
if(nrow(performance_comparison) > 0) {
  perf_long <- melt(performance_comparison, id.vars = c("Dataset", "Samples"), 
                    variable.name = "Metric", value.name = "Value")
  
  p1 <- ggplot(perf_long, aes(x = Dataset, y = Value, fill = Metric)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    labs(title = "RF Performance Metrics Comparison Across Datasets",
         x = "Dataset", y = "Value") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "right") +
    facet_wrap(~Metric, scales = "free_y", ncol = 3) +
    scale_fill_brewer(palette = "Set1")
  
  ggsave(file.path(plots_dir, "01_RF_Performance_Comparison.png"), p1, width = 14, height = 10, dpi = 300)
}

# 2. ROC Curves Comparison (if binary classification)
roc_objects <- list()
for(dataset_name in names(datasets)) {
  dataset <- datasets[[dataset_name]]
  if(!is.null(dataset$metrics) && !is.null(dataset$metrics$ROC_Obj)) {
    roc_objects[[dataset$name]] <- dataset$metrics$ROC_Obj
  }
}

if(length(roc_objects) > 0) {
  p2 <- ggplot() +
    labs(title = "RF ROC Curves Comparison Across Datasets",
         x = "False Positive Rate (1 - Specificity)",
         y = "True Positive Rate (Sensitivity)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
  
  colors <- c("Training Set" = "blue", "Test Set" = "red", 
              "External Validation Set" = "green", "Prospective Set" = "purple")
  
  for(dataset_name in names(roc_objects)) {
    roc_obj <- roc_objects[[dataset_name]]
    roc_data <- data.frame(
      FPR = 1 - roc_obj$specificities,
      TPR = roc_obj$sensitivities,
      Dataset = dataset_name
    )
    p2 <- p2 + 
      geom_line(data = roc_data, aes(x = FPR, y = TPR, color = Dataset), size = 1.2) +
      annotate("text", x = 0.7, y = 0.3 - 0.05 * which(names(roc_objects) == dataset_name), 
               label = paste(dataset_name, "AUC =", round(auc(roc_obj), 3)), 
               color = colors[dataset_name], size = 4)
  }
  
  p2 <- p2 + 
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
    scale_color_manual(values = colors)
  
  ggsave(file.path(plots_dir, "02_RF_ROC_Comparison.png"), p2, width = 10, height = 8, dpi = 300)
}

# 3. Prediction Distribution Across Datasets
p3 <- ggplot(all_results, aes(x = Predicted_Diagnosis, fill = Dataset)) +
  geom_bar(position = "dodge", alpha = 0.7) +
  labs(title = "RF Prediction Distribution Across Datasets",
       x = "Predicted Diagnosis", y = "Count") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        legend.position = "top")

ggsave(file.path(plots_dir, "03_RF_Prediction_Distribution.png"), p3, width = 12, height = 8, dpi = 300)

# 4. Probability Distribution Comparison
p4 <- ggplot(all_results, aes(x = Predicted_Probability, fill = Dataset)) +
  geom_density(alpha = 0.5) +
  labs(title = "RF Prediction Probability Distribution Across Datasets",
       x = "Predicted Probability", y = "Density") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        legend.position = "top")

ggsave(file.path(plots_dir, "04_RF_Probability_Distribution.png"), p4, width = 12, height = 8, dpi = 300)

# 5. Confusion Matrix Heatmaps (for datasets with labels)
confusion_matrices <- list()
for(dataset_name in names(datasets)) {
  dataset <- datasets[[dataset_name]]
  if(!is.null(dataset$metrics)) {
    cm_data <- as.data.frame(dataset$metrics$Confusion_Matrix)
    cm_data$Dataset <- dataset$name
    confusion_matrices[[dataset_name]] <- cm_data
  }
}

if(length(confusion_matrices) > 0) {
  all_cm <- do.call(rbind, confusion_matrices)
  
  p5 <- ggplot(all_cm, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "white", size = 4, fontface = "bold") +
    scale_fill_gradient(low = "lightblue", high = "darkblue") +
    labs(title = "RF Confusion Matrices Across Datasets",
         x = "Actual Class", y = "Predicted Class") +
    facet_wrap(~Dataset, ncol = 2) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          legend.position = "none")
  
  ggsave(file.path(plots_dir, "05_RF_Confusion_Matrices.png"), p5, width = 12, height = 10, dpi = 300)
}

# 6. Calibration Plots Comparison (for datasets with labels)
calibration_data <- data.frame()
for(dataset_name in names(datasets)) {
  dataset <- datasets[[dataset_name]]
  if(!is.null(dataset$metrics) && dataset$has_labels) {
    temp_data <- data.frame(
      Probability = dataset$results$Predicted_Probability,
      True_Label = as.numeric(dataset$results$Actual_Diagnosis) - 1,
      Dataset = dataset$name
    )
    calibration_data <- rbind(calibration_data, temp_data)
  }
}

if(nrow(calibration_data) > 0) {
  # Create bins and calculate calibration statistics
  calibration_data$Bin <- cut(calibration_data$Probability, 
                              breaks = seq(0, 1, by = 0.1), 
                              include.lowest = TRUE)
  
  calibration_summary <- calibration_data %>%
    group_by(Dataset, Bin) %>%
    summarise(
      Mean_Predicted = mean(Probability),
      Mean_Actual = mean(True_Label),
      Count = n(),
      .groups = 'drop'
    ) %>%
    filter(Count > 0)
  
  p6 <- ggplot(calibration_summary, aes(x = Mean_Predicted, y = Mean_Actual, color = Dataset)) +
    geom_point(aes(size = Count), alpha = 0.7) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    geom_smooth(method = "loess", se = FALSE) +
    labs(title = "RF Calibration Plots Comparison Across Datasets",
         x = "Mean Predicted Probability",
         y = "Observed Fraction Positive") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          legend.position = "top")
  
  ggsave(file.path(plots_dir, "06_RF_Calibration_Comparison.png"), p6, width = 10, height = 8, dpi = 300)
}

# 7. Performance Trend Across Datasets (if multiple validation sets)
if(nrow(performance_comparison) >= 3) {
  perf_trend <- melt(performance_comparison, id.vars = "Dataset", 
                     measure.vars = c("Accuracy", "Sensitivity", "Specificity", "AUC"),
                     variable.name = "Metric", value.name = "Value")
  
  # Order datasets logically
  dataset_order <- c("Training Set", "Test Set", "External Validation Set", "Prospective Set")
  perf_trend$Dataset <- factor(perf_trend$Dataset, levels = dataset_order)
  perf_trend <- perf_trend[order(perf_trend$Dataset), ]
  
  p7 <- ggplot(perf_trend, aes(x = Dataset, y = Value, color = Metric, group = Metric)) +
    geom_line(size = 1.2) +
    geom_point(size = 3) +
    labs(title = "RF Performance Trend Across Datasets",
         x = "Dataset", y = "Performance Value") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "top") +
    scale_color_brewer(palette = "Set1")
  
  ggsave(file.path(plots_dir, "07_RF_Performance_Trend.png"), p7, width = 12, height = 8, dpi = 300)
}

# 8. Sample Size and Class Distribution
dataset_summary <- data.frame()
for(dataset_name in names(datasets)) {
  dataset <- datasets[[dataset_name]]
  if(dataset$has_labels) {
    class_dist <- table(dataset$results$Actual_Diagnosis)
    summary_row <- data.frame(
      Dataset = dataset$name,
      Total_Samples = nrow(dataset$results),
      Class_0 = ifelse(length(class_dist) >= 1, class_dist[1], 0),
      Class_1 = ifelse(length(class_dist) >= 2, class_dist[2], 0)
    )
    dataset_summary <- rbind(dataset_summary, summary_row)
  } else {
    summary_row <- data.frame(
      Dataset = dataset$name,
      Total_Samples = nrow(dataset$results),
      Class_0 = NA,
      Class_1 = NA
    )
    dataset_summary <- rbind(dataset_summary, summary_row)
  }
}

p8 <- ggplot(dataset_summary, aes(x = Dataset, y = Total_Samples, fill = Dataset)) +
  geom_bar(stat = "identity", alpha = 0.7) +
  geom_text(aes(label = Total_Samples), vjust = -0.5, size = 4) +
  labs(title = "Sample Size Distribution Across Datasets",
       x = "Dataset", y = "Number of Samples") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

ggsave(file.path(plots_dir, "08_Sample_Distribution.png"), p8, width = 10, height = 8, dpi = 300)

# 9. RF Feature Importance Plot
if(!is.null(final_rf_model$importance) && length(final_features) > 0) {
  # Extract feature importance from RF model
  if(!is.null(final_rf_model$importance)) {
    feature_importance <- final_rf_model$importance
    if(is.matrix(feature_importance)) {
      # For classification, use MeanDecreaseGini or first column
      if("MeanDecreaseGini" %in% colnames(feature_importance)) {
        imp_scores <- feature_importance[, "MeanDecreaseGini"]
      } else {
        imp_scores <- feature_importance[, 1]
      }
    } else {
      imp_scores <- feature_importance
    }
    
    feature_imp_df <- data.frame(
      Feature = names(imp_scores),
      Importance = as.numeric(imp_scores)
    ) %>%
      arrange(desc(Importance)) %>%
      head(15)  # Top 15 features
    
    p9 <- ggplot(feature_imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
      geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
      coord_flip() +
      labs(title = "RF - Top 15 Feature Importance",
           x = "Features",
           y = "Importance Score (MeanDecreaseGini)") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
    
    ggsave(file.path(plots_dir, "09_RF_Feature_Importance.png"), p9, width = 12, height = 8, dpi = 300)
  }
}

# ==================== Generate Comprehensive Report ====================
cat("Step 5: Generating Comprehensive RF Report...\n")

# Create Excel workbook
wb <- createWorkbook()

# 1. Performance Comparison Sheet
addWorksheet(wb, "RF_Performance_Comparison")
if(nrow(performance_comparison) > 0) {
  writeData(wb, "RF_Performance_Comparison", performance_comparison, startRow = 1)
}

# 2. All Predictions Sheet
addWorksheet(wb, "RF_All_Predictions")
writeData(wb, "RF_All_Predictions", all_results, startRow = 1)

# 3. Dataset Summary Sheet
addWorksheet(wb, "RF_Dataset_Summary")
writeData(wb, "RF_Dataset_Summary", dataset_summary, startRow = 1)

# 4. Model Information Sheet
addWorksheet(wb, "RF_Model_Information")
model_info <- data.frame(
  Item = c("Model Type", "Number of Features", "Features Used", "Training Date", "Analysis Date"),
  Value = c(
    "Random Forest",
    length(final_features),
    paste(final_features, collapse = ", "),
    if(!is.null(final_rf_model$training_date)) as.character(final_rf_model$training_date) else "Unknown",
    as.character(Sys.Date())
  )
)
writeData(wb, "RF_Model_Information", model_info, startRow = 1)

# 5. Confusion Matrices Sheet
if(length(confusion_matrices) > 0) {
  addWorksheet(wb, "RF_Confusion_Matrices")
  all_confusion <- do.call(rbind, confusion_matrices)
  writeData(wb, "RF_Confusion_Matrices", all_confusion, startRow = 1)
}

# 6. Feature Importance Sheet
if(exists("feature_imp_df")) {
  addWorksheet(wb, "RF_Feature_Importance")
  writeData(wb, "RF_Feature_Importance", feature_imp_df, startRow = 1)
}

# Save workbook
saveWorkbook(wb, file.path(output_dir, "Comprehensive_RF_Performance_Report.xlsx"), overwrite = TRUE)

# Save R objects
saveRDS(all_results, file.path(data_dir, "rf_all_predictions.rds"))
if(nrow(performance_comparison) > 0) {
  saveRDS(performance_comparison, file.path(data_dir, "rf_performance_comparison.rds"))
}
saveRDS(dataset_summary, file.path(data_dir, "rf_dataset_summary.rds"))

# ==================== Final Summary ====================
cat("\n=== COMPREHENSIVE RF PERFORMANCE ANALYSIS COMPLETE ===\n")
cat("Results saved to:", output_dir, "\n")
cat("Plots saved to:", plots_dir, "\n")
cat("Data saved to:", data_dir, "\n")

cat("\n=== DATASET SUMMARY ===\n")
for(dataset_name in names(datasets)) {
  dataset <- datasets[[dataset_name]]
  cat(dataset$name, ": ", nrow(dataset$results), " samples\n")
  if(dataset$has_labels && !is.null(dataset$metrics)) {
    cat("  AUC: ", round(dataset$metrics$AUC, 3), 
        ", Accuracy: ", round(dataset$metrics$Accuracy, 3), "\n")
  }
}

if(nrow(performance_comparison) > 0) {
  cat("\n=== RF KEY FINDINGS ===\n")
  
  # Find best performing dataset
  best_auc <- performance_comparison[which.max(performance_comparison$AUC), ]
  cat("Best AUC: ", best_auc$Dataset, " (", round(best_auc$AUC, 3), ")\n")
  
  best_accuracy <- performance_comparison[which.max(performance_comparison$Accuracy), ]
  cat("Best Accuracy: ", best_accuracy$Dataset, " (", round(best_accuracy$Accuracy, 3), ")\n")
  
  # Check for performance degradation
  if("Training Set" %in% performance_comparison$Dataset && "Prospective Set" %in% performance_comparison$Dataset) {
    train_auc <- performance_comparison$AUC[performance_comparison$Dataset == "Training Set"]
    prospect_auc <- performance_comparison$AUC[performance_comparison$Dataset == "Prospective Set"]
    auc_degradation <- train_auc - prospect_auc
    
    if(auc_degradation > 0.1) {
      cat("WARNING: Significant performance degradation detected (AUC drop: ", round(auc_degradation, 3), ")\n")
    } else if(auc_degradation > 0.05) {
      cat("NOTE: Moderate performance degradation detected (AUC drop: ", round(auc_degradation, 3), ")\n")
    } else {
      cat("GOOD: RF Model shows stable performance across datasets\n")
    }
  }
}

cat("\n=== FILES GENERATED ===\n")
cat("- Comprehensive_RF_Performance_Report.xlsx (Excel report)\n")
cat("- 9 comprehensive RF visualization plots\n")
cat("- R data files for further analysis\n")

cat("\n=== RF ANALYSIS COMPLETE ===\n")