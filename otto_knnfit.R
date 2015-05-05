library(dplyr)
library(readr)
library(ggplot2)
library(caret)


#Reading data into r
#for some reason using read_csv throws the error
#"Error in knn(train, test, cl = classes, k = 3, prob = TRUE) : 
#'train' and 'class' have different lengths"
#No idea why
#Reading data into r
train <- read.csv("train.csv")
test <- read.csv("test.csv")

##############################################################################
#EDA
str(train)
#So it appears we have integer data that is generally very sparse, consisting
#of mainly 0 values

#List of frequency tables for easy access
tables <- lapply(train[, 2:94], table)
#List of histogram plots to easily see skew of all variables.
plots <- lapply(train[, 2:94], qplot)

#Target variable probably needs to be a factor
train$target <- as.factor(train$target)


#Splitting training set to allow for tuning and perfomance metrics
set.seed(123)
knnTraining <- createDataPartition(y = train$target, p = .75, list = FALSE)
training <- train[knnTraining,]
cv <- train[-knnTraining,]


#removing id column
training <- training[, -1]
cv <- cv[, -1]

#creating ctrl parameters
set.seed(100)
indx <- createFolds(training$target, returnTrain = TRUE)
ctrl <- trainControl(method = "LGOCV",  
                     classProbs = TRUE,
                     index = indx,
                     savePredictions = TRUE)

#Probably should use parallel computing or this could take forever
library(doParallel)
cl <- makeCluster(3)
registerDoParallel(cl)

#Running the knn over a range of k values 
knn.time1 <- system.time(otto.knn.fit1 <- train(x = training[,-94],
                       y = training[,94],
                       method = "knn",
                       metric = "ROC",
                       preProc = c("center", "scale"),
                       tuneGrid = data.frame(k =c(4*(0:5)+1)),
                       trControl = ctrl))
#ROC metric did not run so accuracy was defaulted to. Turns out ROC is only
#good for two class classification problems

otto.knn.fit1$pred <- merge(otto.knn.fit1$pred,  otto.knn.fit1$bestTune)
otto.knnCM <- confusionMatrix(otto.knn.fit1, norm = "none")
otto.knnCM
plot(otto.knn.fit1, metric="Accuracy")


knn.pred <- predict(otto.knn.fit1, newdata = cv[,-94])
knn.pred <- data.frame(knn.pred)

#Calculating model performance
postResample(pred = knn.pred, obs = cv[,94])

#Applying to test set and testing out submission
knn.submission <- predict(otto.knn.fit1, newdata = test[,-1], type = "prob")
knn.final <- data.frame(id = test$id, knn.submission)

#writing csv file to submit
write.csv(knn.final,"knn_final.csv", row.names=FALSE, quote=FALSE )

