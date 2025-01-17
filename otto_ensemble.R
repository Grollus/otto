#Setting working directory to otto folder-------------------------------------------
setwd("D:/RProgram/otto")

#Loading required packages----------------------------------------------------------
packages <- c("caret", "doParallel", "h2o", "xgboost")
lapply(packages, library, character.only = TRUE)

#setting up parallel backend--------------------------------------------------------
cl <- makeCluster(7)
registerDoParallel(cl)

#Registering h2o cluster on local machine
localH2O <- h2o.init(nthread = 8, max_mem_size = "24g")

#Loading data NEVER MODIFY THESE BASE FRAMES----------------------------------------
train.full <- read.csv("train.csv")
test.full <- read.csv("test.csv")

#Need data formatted in different ways for h2o and xgboost. xgboost needs a matrix
#form and h2o uses h2o specific frames. Need to set up a framework where I can easily 
#work with the two------------------------------------------------------------------

#First I will create a separate predictions frame and CV structure since they will 
#be used for both (and should be able to be used with other models as well)---------
predictions <- train.full[, ncol(train.full)]

#Matrix for storing predicted probablities; needs to be dimensions of train.only
pred.prob <- model.matrix(~ predictions - 1, data = predictions) 
#Using caret to split the data for CV-----------------------------------------------
set.seed(1234)
cv.index <- createDataPartition(predictions, p = 0.25, list = FALSE)

#Removing class label column and IDs; setting IDs aside for use later---------------
train <- train.full[-ncol(train.full)]
ids <- train[, 1]
train <- train[-1]
test.ids <- test.full[, 1]
test <- test.full[, -1]

#Partitioning to training and cv sets-----------------------------------------------
train.only <- train[-cv.index,]
cv <- train[cv.index, ]

#Setting up train/cv formats for xgboost model--------------------------------------
y <- predictions[-cv.index] #Response variable from train.only
y <- gsub("Class_", "", y)  #Just class number
y <- as.integer(y) - 1

x <- rbind(train.only, cv)#Predictors from train.only and cv sets
x <- as.matrix(x)         #Converting to matrix
x <- matrix(as.numeric(x), nrow(x), ncol(x)) #Converting chr to num 

trind <- 1:length(y)      #index to identify training data
cvind <- (length(y)+1):nrow(x)

#Test set as matrix-----------------------------------------------------------------
x.test <- as.matrix(test.full)
x.test <- matrix(as.numeric(x.test), nrow(x.test), ncol(x.test))

#Setting up train/cv formats for h2o.deeplearning-----------------------------------
#Creating fresh train.only and cv frames for h2o model------------------------------
h2o.train.only <- train.full[-cv.index,]
h2o.cv <- train.full[cv.index,]
h2o.test.full <- test.full


for(i in 2:94){
  h2o.train.only[, i] <- as.numeric(h2o.train.only[,i]) #Converting everything to num
  h2o.train.only[, i] <- sqrt(h2o.train.only[,i] + (3/8)) #sqrt + 3/8 trans instead of log
}

for(i in 2:94){
  h2o.cv[, i] <- as.numeric(h2o.cv[, i])
  h2o.cv[, i] <- sqrt(h2o.cv[, i] + (3/8))
}

for(i in 2:94){
  h2o.test.full[, i] <- as.numeric(h2o.test.full[, i])
  h2o.test.full[, i] <- sqrt(h2o.test.full[, i] + (3/8))
}
#Creating h2o objects---------------------------------------------------------------
train.hex <- as.h2o(localH2O, h2o.train.only)
cv.hex <- as.h2o(localH2O, h2o.cv[, 2:94])
test.hex <- as.h2o(localH2O, h2o.test.full)

#h2o requires identification of predictors and responses----------------------------
predictors <- 2:(ncol(train.hex)-1)
responses <- ncol(train.hex)

#Creating data frame to store CV model results--------------------------------------
submission <- read.csv("sampleSubmission.csv")
submission[, 2:10] <- 0
submission <- submission[1:nrow(cv),]

###############################Xgboost Modeling#####################################
#Random search function used for tuning parameters----------------------------------
random_search <- function(n_set){
  #param is a list of parameters
  
  # Set necessary parameter
  param <- list("objective" = "multi:softprob",
                "max_depth"=6,
                "eta"=0.1,
                "subsample"=0.7,
                "colsample_bytree"= 1,
                "gamma"=2,
                "min_child_weight"=4,
                "eval_metric" = "mlogloss",
                "silent"=1,
                "num_class" = 9,
                "nthread" = 8)
  
  param_list <- list()
  
  for (i in seq(n_set)){
    
    ## n_par <- length(param)
    param$max_depth <- sample(3:10,1, replace=T)
    param$eta <- runif(1,0.01,0.6)
    param$subsample <- runif(1,0.1,1)
    param$colsample_bytree <- runif(1,0.1,1)
    param$min_child_weight <- sample(1:17,1, replace=T)
    param$gamma <- runif(1,0.1,10)
    param$min_child_weight <- sample(1:15,1, replace=T)
    param_list[[i]] <- param
    
  }
  
  return(param_list)
}
#Set of parameters to test----------------------------------------------------------
xgb.param <- random_search(10)
#Running CV before doing ensemble CV------------------------------------------------
cv.nround = 1000
TrainRes <- matrix(, nrow=cv.nround, ncol=length(xgb.param))
TestRes <- matrix(, nrow= cv.nround, ncol=length(xgb.param))

for(i in 1:length(param)){
  print(paste0("CV Round", i))
  bst.cv <- xgb.cv(param = xgb.param[[i]], data = x[trind,], label = y, 
                   nfold = 3, nrounds=cv.nround)
  TrainRes[,i] <- as.numeric(bst.cv[,train.mlogloss.mean])
  TestRes[,i]  <- as.numeric(bst.cv[,test.mlogloss.mean])
  
}

#Already found my best parameters in xgboost_script.R, so just inputting here-------
final.xgb.params <- list("objective" = "multi:softprob",
               "max_depth"=9,
               "eta"=0.05431259,
               "subsample"=0.7851139 ,
               "colsample_bytree"= 0.3923619,
               "gamma"=0.5,
               "min_child_weight"=5,
               "eval_metric" = "mlogloss",
               "silent"=1,
               "num_class" = 9,
               "nthread" = 8) 
#Training solo xgboost model--------------------------------------------------------
nround = 712
xgb.bst <- xgboost(param = final.xgb.params, data = x[trind,], label = y, 
                   nrounds = nround)

#################################h2odeeplearn training##############################
for( i in 1:20){
  print(i)
  h2o.model <- h2o.deeplearning(x = predictors,
                            y = responses,
                            data = train.hex,
                            classification = TRUE,
                            activation = "RectifierWithDropout",
                            hidden = c(1024, 512, 256),
                            hidden_dropout_ratio = c(0.5, 0.5, 0.5),
                            input_dropout_ratio = 0.05,
                            epochs = 50,
                            l1 = 1e-5,
                            l2 = 1e-5,
                            rho = 0.99,
                            epsilon= 1e-8,
                            train_samples_per_iteration = 2000,
                            max_w2 = 10,
                            seed = 1)
  submission[,2:10] <- submission[,2:10] + as.data.frame(h2o.predict(h2o.model, cv.hex))[,2:10]
  print(i)
  
}
h2o.saveAll(object = localH2O, dir = "D:/RProgram/otto", save_cv = TRUE)
#Dividing by the number of runs to have probabilities on 0 to 1 scale---------------
submission.avg <- submission[,-1]/20

##################################CV before Ensembling##############################
xgb.model.prob <- predict(xgb.bst, newdata = x[cvind,])
xgb.model.prob <- t(matrix(xgb.model.prob,9,length(xgb.model.prob)/9))

#Dividing by number of h2o runs to get 0 to 1 scale---------------------------------
#This was done in the for loop above for the h2o model------------------------------
h2o.deeplearn.prob.avg <- submission[, -1]/20

#Calculating logloss----------------------------------------------------------------
#logloss function; unsure of how accurately this reflects LB calculations-----------
ll <- function(predicted, actual, eps = 1e-15){
  predicted[predicted < eps] <- eps
  predicted[predicted > 1 - eps] <- 1 - eps
  score <- -1/nrow(actual)*(sum(actual*log(predicted)))
  score
}

#Calculating ll for individual models-----------------------------------------------
xgb.model.ll <- ll(xgb.model.prob, pred.prob[cv.index,])
h2o.deeplearn.ll <- ll(as.data.frame(h2o.deeplearn.prob.avg), pred.prob[cv.index,])

#################################Ensembling#########################################
#Building grid to weight model probabilities----------------------------------------
weight.grid <- data.frame(expand.grid(w1 = seq(from = 0.01, to = 1, by = 0.05),
                                      w2 = seq(from = 0.01, to = 1, by = 0.05)),
                          ll = NA)

#searching through weights to find optimum------------------------------------------
for(x in 1:nrow(weight.grid)){
  cv.prob.test <- 
    ((xgb.model.prob * weight.grid$w1[x]) +
       (h2o.deeplearn.prob.avg * weight.grid$w2[x]))/
    sum(weight.grid[x, c("w1", "w2")])
  weight.grid$ll[x] <- ll(cv.prob.test, pred.prob[cv.index, ])
  print(x)
}

weight.best <- which.min(weight.grid$ll)
weight.grid[weight.best, ]

################################Building Test Probabilities#########################
xgb.bst.prob.test <- predict(xgb.bst, newdata = x.test[, -1])
h2o.deeplearn.prob.test <- h2o.predict(h2o.model, test.hex[, -1])


test.prob <-
  ((xgb.bst.prob.test * weight.grid[weight.best, "w1"]) +
     (as.data.frame(h2o.deeplearn.prob.test[, -1]) * weight.grid[weight.best, "w2"]))/
  sum(weight.grid[weight.best, c("w1", "w2")])

test.simple.avg <- ((xgb.bst.prob.test) + (as.data.frame(h2o.deeplearn.prob.test[, -1])))/
  2
#Save-------------------------------------------------------------------------------
ensemble_submission1 <- read.csv("sampleSubmission.csv")
h2o_deeplearn_submission2 <- read.csv("sampleSubmission.csv")

ensemble_submission1[,2:10] <- ensemble_submission1[,2:10] + as.data.frame(test.prob)
write.csv(ensemble_submission1, file = "ensemble_submission1.csv", row.names = FALSE)

h2o_deeplearn_submission2[,2:10] <- ensemble_submission1[,2:10] + as.data.frame(h2o.deeplearn.prob.test[, -1])
write.csv(h2o_deeplearn_submission2, file = "h2o_deeplearn_benchmark2.csv", row.names = FALSE)

simple_avg_ensemble_sub <- read.csv("sampleSubmission.csv")
simple_avg_ensemble_sub[, 2:10] <- simple_avg_ensemble_sub[, 2:10] + as.data.frame(test.simple.avg)
write.csv(simple_avg_ensemble_sub, file = "simple_avg_ensemble_sub.csv", row.names = FALSE)



save.image(file = "otto_ensemble_workspace.Rdata")

temp1 <- read.csv("D:/RProgram/submission7.csv")
temp2 <- read.csv("submission_h2o_deeplearn_bench.csv")
temp2[, -1] <- temp2[, -1]/20

temp3 <- (temp1[, -1] + temp2[, -1])/2
temp3 <- cbind(id = temp1[, 1], temp3)

write.csv(temp3, file = "simple_avg_ensemble_sub.csv", row.names = FALSE)




