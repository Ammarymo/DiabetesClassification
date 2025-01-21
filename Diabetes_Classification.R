# Install and load required packages
required_pkgs <- c("tidyverse", "caret", "xgboost", "randomForest", "plotROC", 
                   "corrplot", "tidyplots", "DataExplorer", "gt")
new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[,"Package"])]
if(length(new_pkgs)) install.packages(new_pkgs)

library(tidyverse)
library(caret)
library(xgboost)
library(randomForest)
library(corrplot)
library(DataExplorer)
library(gt)

#Importing Data
#Source:https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

#How to Download: Terminal Code================
#`$ pip install kaggle`
#`$ mkdir data`
#`$ cd data`
#`$ kaggle datasets download uciml/pima-indians-diabetes-database`
#`$ unzip pima-indians-diabetes-database.zip`

diabetes <- read_csv("data/diabetes.csv")

#Quick EDA
head(diabetes)
summary(diabetes)
glimpse(diabetes)
plot_histogram(diabetes)
plot_bar(diabetes)

#Some variables have meaningless zeros, replaced with median
diabetes <- diabetes %>%
  mutate(across(c(Glucose, BloodPressure, SkinThickness, Insulin, BMI), ~ replace(., . == 0, NA))) %>%
  mutate(across(everything(), ~ replace_na(., median(., na.rm = TRUE)))) %>%
  mutate(Outcome = factor(Outcome))


plot_histogram(diabetes)

#Checking Correlations using Spearman correlation
cor_matrix <- cor(diabetes[,-9],method = "spearman")
corrplot(cor_matrix, method = "color",
         addCoef.col = "black", tl.col = "black", type="lower", diag = F,
         cl.ratio = 0.2, tl.srt = 45, tl.cex =0.7, cl.cex = 0.7, number.cex = 0.7)


#Machine Learning: Classification of Diabetic Observations
set.seed(123)
train_indices <- createDataPartition(diabetes$Outcome, p = 0.8, list = FALSE)
train_set <- diabetes[train_indices, ]
test_set <- diabetes[-train_indices, ]

# Logistic Regression
set.seed(123)
lr_model <- train(
  Outcome ~ ., data = train_set, method = "glm",
  family = binomial(link = "logit"),
  trControl = trainControl(method = "cv", number = 10)
)
lr_model_predictions <- predict(lr_model, test_set)

# Random Forest
set.seed(123)
rf_model <- train(Outcome ~ .,
                  data = train_set,
                  method = "rf",
                  trControl = trainControl(method = "cv", number = 10),
                  tuneGrid = expand.grid(mtry = 1:5))

rf_model_predictions <- predict(rf_model, test_set)

# XGBoost
set.seed(123)
xg.grid <- expand.grid(
  nrounds = c(50, 100),
  eta = seq(0.1, 0.3, length.out = 2),
  max_depth = 6:8,
  gamma = seq(0, 5, length.out = 3),
  colsample_bytree = seq(0.6, 1, length.out = 3),
  min_child_weight = 1:5,
  subsample = seq(0.6, 1, length.out = 3)
)


xg_model <- train(
  Outcome ~ ., data = train_set, method = "xgbTree",
  trControl = trainControl(method = "cv", number = 3, search = "random"),
  tuneGrid = xg.grid,
  tuneLength = 10
)
xg_model_predictions <- predict(xg_model, test_set)

# Updated Logistic Regression (with PCA)
set.seed(123)
pca_preprocess <- preProcess(train_set %>% select(-Outcome), method = c("center", "scale", "pca"))
train_set_pca <- predict(pca_preprocess, train_set %>% select(-Outcome))
test_set_pca <- predict(pca_preprocess, test_set %>% select(-Outcome))

lr_model_updated <- train(
  x = train_set_pca, y = train_set$Outcome, method = "glm",
  family = binomial(link = "logit"),
  trControl = trainControl(method = "cv", number = 10)
)
lr_model_updated_predictions <- predict(lr_model_updated, test_set_pca)

# Ensemble Model
ensemble_model <- tibble(
  lr = as.numeric(as.character(lr_model_predictions)),
  rf = as.numeric(as.character(rf_model_predictions)),
  xg = as.numeric(as.character(xg_model_predictions)),
  lr_u = as.numeric(as.character(lr_model_updated_predictions))
) %>%
  mutate(Final = rowMeans(select(., lr, rf, xg, lr_u), na.rm = TRUE)) %>%
  mutate(Final = factor(if_else(Final >= 0.5, 1, 0)))


# Evaluation Metrics

results <- tibble(
  Model = c("Logistic Regression", "Random Forest", "XGBoost", "Updated Logistic Regression", "Ensemble"),
  Accuracy = c(
    confusionMatrix(lr_model_predictions, test_set$Outcome)$overall[["Accuracy"]],
    confusionMatrix(rf_model_predictions, test_set$Outcome)$overall[["Accuracy"]],
    confusionMatrix(xg_model_predictions, test_set$Outcome)$overall[["Accuracy"]],
    confusionMatrix(lr_model_updated_predictions, test_set$Outcome)$overall[["Accuracy"]],
    confusionMatrix(ensemble_model$Final, test_set$Outcome)$overall[["Accuracy"]]
  ),
  F1_Score = c(
    F_meas(lr_model_predictions, reference = test_set$Outcome, relevant = "1"),
    F_meas(rf_model_predictions, reference = test_set$Outcome, relevant = "1"),
    F_meas(xg_model_predictions, reference = test_set$Outcome, relevant = "1"),
    F_meas(lr_model_updated_predictions, reference = test_set$Outcome, relevant = "1"),
    F_meas(ensemble_model$Final, reference = test_set$Outcome, relevant = "1")
  )
)

results_table <- results %>%
  gt() %>%
  tab_header(
    title = "Model Evaluation Results"
  ) %>%
  cols_label(
    Model = "Model",
    Accuracy = "Accuracy",
    F1_Score = "F1 Score"
  ) %>%
  tab_spanner(
    label = "Performance Metrics",
    columns = c("Accuracy", "F1_Score")
  ) %>%
  fmt_number(
    columns = c("Accuracy", "F1_Score"),
    decimals = 3
  ) %>%
  tab_spanner_delim(" ")

# Display the table
print(results_table)


