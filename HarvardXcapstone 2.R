if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")
if(!require(gam)) install.packages("gam", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")

###Import dataset###

urlfile <- "https://raw.githubusercontent.com/alecbanner/HarvardX-Capstone-2-Biomechanics/master/column_2C_weka.csv"
column_2C_weka <- read_csv(url(urlfile))
head(column_2C_weka)
#rename the pelvic tilt column
column_2C_weka <- column_2C_weka %>% 
  rename('pelvic_tilt' = 'pelvic_tilt numeric')
head(column_2C_weka)

###Exploring the data###

#dimensions of the dataset
dim(column_2C_weka)

#class of data
str(column_2C_weka)

#class column as factor (when y in createdatapartition is a factor the relative amounts are conserved)
column_2C_weka$class <- as.factor(column_2C_weka$class)
str(column_2C_weka)

#check for any NAs in the dataset
apply(column_2C_weka, 2, function(x) any(is.na(x)))

#visualise relative ratios of each disease
column_2C_weka %>%
  group_by(class) %>%
  ggplot(aes(class)) +
  geom_bar() +
  geom_label(stat='count',aes(label=..count..))

#make histograms of each factor
h1 <- ggplot(column_2C_weka, aes(pelvic_incidence))+geom_histogram(binwidth = 5) 
h2 <- ggplot(column_2C_weka, aes(pelvic_tilt))+geom_histogram(binwidth = 5) 
h3 <- ggplot(column_2C_weka, aes(lumbar_lordosis_angle))+geom_histogram(binwidth = 5) 
h4 <- ggplot(column_2C_weka, aes(sacral_slope))+geom_histogram(binwidth = 5) 
h5 <- ggplot(column_2C_weka, aes(pelvic_radius))+geom_histogram(binwidth = 5) 
h6 <- ggplot(column_2C_weka, aes(degree_spondylolisthesis))+geom_histogram(binwidth = 5)
grid.arrange(h1,h2,h3,h4,h5,h6)

#make boxplots of each factor
b1 <- ggplot(column_2C_weka, aes(class, pelvic_incidence)) + geom_boxplot()
b2 <- ggplot(column_2C_weka, aes(class, pelvic_tilt)) + geom_boxplot()
b3 <- ggplot(column_2C_weka, aes(class, lumbar_lordosis_angle)) + geom_boxplot()
b4 <- ggplot(column_2C_weka, aes(class, sacral_slope)) + geom_boxplot()
b5 <- ggplot(column_2C_weka, aes(class, pelvic_radius)) + geom_boxplot()
b6 <- ggplot(column_2C_weka, aes(class, degree_spondylolisthesis)) + geom_boxplot()
grid.arrange(b1, b2, b3, b4, b5, b6)

###create training and test sets###

#createdatapartitionbased on class(factor) to maintain ratios. Take 20% for test set as small dataset so need slightly more data to make sure representative of entire dataset
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = column_2C_weka$class, times = 1, p =  0.2, list = FALSE)
test_set <- column_2C_weka[test_index,]
data <- column_2C_weka[-test_index,]
#check the realative ratios are maintained
mean(test_set$class == "Normal")
mean(data$class == "Normal")
#save the partitioned data
save(test_set, file = "test_set.rdata")
save(data, file = "data.rdata")


###Building models###

#split data into training and validation sets
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = data$class, times = 1, p = 0.2, list = FALSE)
validation <- data[test_index,]
training <- data[-test_index,]
#check the ratios are maintained
mean(validation$class == "Normal")
mean(training$class == "Normal")
#save the trainging and validation datasets
save(validation, file = "validation.rdata")
save(training, file = "training.rdata")

#define cross validation method for 'train'
control <- trainControl(method = "boot", number = 50, p = 0.5)

#Regression tree
fit <- rpart(class ~ ., data = training)
plot(fit, margin = 0.1)
text(fit, cex = 0.75)

#Generalised linear model (logistic regression)
train_glm <- train(class ~ ., data = training, method = "glm", trControl = control)
glm_yhat <- predict(train_glm, validation)
glm_con_mat <- confusionMatrix(glm_yhat, validation$class)
glm_con_mat

#K Nearest Neighbours
set.seed(1, sample.kind = "Rounding")
train_knn <- train(class ~ . , data = training, method = "knn", tuneGrid = data.frame(k = seq(3, 51, 2)), trControl = control)
ggplot(train_knn, highlight = TRUE)
train_knn$bestTune
knn_yhat <- predict(train_knn, validation)
knn_con_mat <- confusionMatrix(knn_yhat, validation$class)
knn_con_mat

#gamLoess
grid <- expand.grid(span = seq(0.1, 0.9, 0.05), degree = 1)
set.seed(1, sample.kind = "Rounding")
train_gamloess <- train(class~ ., data = training, method = "gamLoess", tuneGrid = grid, trControl = control)
ggplot(train_gamloess, highlight = TRUE)
train_gamloess$bestTune
gamloess_yhat <- predict(train_gamloess, validation)
gamloess_con_mat <- confusionMatrix(gamloess_yhat, validation$class)
gamloess_con_mat

#Random Forest
set.seed(1, sample.kind = "Rounding")
train_rf <- train(class~ ., data = training, method = "rf", tuneGrid = data.frame(mtry = c(1:10)), trControl = control)
ggplot(train_rf, highlight = TRUE)
#look at the importance of each variable
plot(varImp(train_rf))

rf_yhat <- predict(train_rf, validation)
rf_con_mat <- confusionMatrix(rf_yhat, validation$class)
rf_con_mat

#Gradient boosted model 
grid = expand.grid(.n.trees=seq(100,500, by=100), .interaction.depth=seq(4,7, by=1), .shrinkage = 0.01, .n.minobsinnode = c(3,5,10))
set.seed(1, sample.kind = "Rounding")
train_gbm <- train(class~., data = training, method = "gbm", tuneGrid = grid, trControl = control)
ggplot(train_gbm, highlight = TRUE)
gbm_yhat <- predict(train_gbm, validation)
gbm_con_mat <- confusionMatrix(gbm_yhat, validation$class)
gbm_con_mat

#Ensemble Model
fits <- list(train_glm, train_knn, train_gamloess, train_rf, train_gbm)
pred <- sapply(fits, function(object){
  predict(object, validation)
})
colMeans(pred == validation$class)
x <- rowMeans(pred == "Abnormal")
ensemble_vote <- ifelse(x>0.5, "Abnormal", "Normal")
ensemble_acc <- mean(ensemble_vote == validation$class)

#Ensemble model 2
fits <- list(train_gamloess, train_rf, train_gbm)
pred <- sapply(fits, function(object){
  predict(object, validation)
})
x <- rowMeans(pred == "Abnormal")
ensemble_vote <- ifelse(x>0.5, "Abnormal", "Normal")
ensemble_2_acc <- mean(ensemble_vote == validation$class)

#Final Model

#gamLoess
grid <- expand.grid(span = 0.85, degree = 1)
set.seed(1, sample.kind = "Rounding")
train_gamloess_final <- train(class~ ., data = data, method = "gamLoess", tuneGrid = grid, trControl = control)
#Random Forest
set.seed(1, sample.kind = "Rounding")
train_rf_final <- train(class~ ., data = data, method = "rf", tuneGrid = data.frame(mtry = 3), trControl = control)
#Gradient boosted model 
grid = expand.grid(.n.trees=100, .interaction.depth=6, .shrinkage = 0.01, .n.minobsinnode = 10)
set.seed(1, sample.kind = "Rounding")
train_gbm_final <- train(class~., data = data, method = "gbm", tuneGrid = grid, trControl = control)
#Ensemble
fits <- list(train_gamloess_final, train_rf_final, train_gbm_final)
pred <- sapply(fits, function(object){
  predict(object, test_set)
})
x <- rowMeans(pred == "Abnormal")
ensemble_vote <- ifelse(x>0.5, "Abnormal", "Normal")
ensemble_final_acc <- mean(ensemble_vote == test_set$class)
ensemble_final_acc

