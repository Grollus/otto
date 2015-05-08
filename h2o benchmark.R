library(h2o)

#Launch h2o on localhost, using all cores
h2oServer = h2o.init(nthreads = -1)

#Pointing to kaggle data
dir <- paste0(path.expand("~"), "/otto")

#############Importing data and creating Train/Validation Splits###################
train.hex <- h2o.importFile(paste0(dir, "train.csv"), destination_frame = "train.hex")
test.hex <- h2o.importFile(paste0(dir, "test.csv"), destination_frame = "test.hex")

dim(train.hex)
summary(train.hex)

predictors <- 2:(ncol(train.hex)-1) #Ignores first column 'id'
response <- ncol(train.hex)

#Splitting into 80/20 Train/Validation
rnd <- h2o.runif(train.hex, 1234)
train_holdout.hex <- h2o.assign(train.hex[rnd<0.8,], "train_holdout.hex")
valid_holdout.hex <- h2o.assign(train.hex[rnd>=0.8,], "valid_holdout.hex")

###################################################################################
#Using h2o Flow to inspect the data and build some models on 
#train_holdout.hex/valid_holdout.hex to get a feel for the problem

##Connect browser to http://localhost:54321

###################################################################################
#GBM Hyper-Parameter Tuning with Random Search
models <- c()
for(i in 1:10){
  rand_numtrees <- sample(1:50, 1) ## 1 to 50 trees
  rand_max_depth <- sample(5:15, 1) ## 5 to 15 max depth
  rand_min_rows <- sample(1:10, 1) ## 1 to 10 min rows
  rand_learn_rate <- 0.025*sample(1:10, 1) ## 0.025 to 0.25 learning rate
  model_name <- paste0("GBMModel_", i,
                       "_ntrees", rand_numtrees,
                       "_maxdepth", rand_max_depth,
                       "_minrows", rand_min_rows,
                       "_learnrate", rand_learn_rate)
  model <- h2o.gbm(x = predictors,
                   y = response,
                   training_frame = train_holdout.hex,
                   validation_frame = valid_holdout.hex,
                   model_id = model_name,
                   distribution = "multinomial",
                   ntrees = rand_numtrees,
                   max_depth = rand_max_depth,
                   min_rows = rand_min_rows,
                   learn_rate = rand_learn_rate)
  models <= c(models, model)
}

##Find the best model (lowest logloss on the validation holdout set)
best_err <- 1e3
for(i in 1:length(models)){
  err <- h2o.logloss(h2o.performance(models[[i]], valid_holdout.hex))
  if(err < best_err){
    best_err <- err
    best_model <- models[[i]]
  }
}

##Show the "winning" parameters
parms <- best_model@allparameters
parms$ntrees
parms$max_depth
parms$min_rows
parms$learn_rate

##Training set performance metrics
train_perf <- h2o.performance(best_model, train_holdout.hex)
h2o.confusionMatrix(train_perf)
h2o.logloss(train_perf)


#######################Build Final Model using Full Training Data##################
model <- h2o.gbm(x = predictors,
                 y = response,
                 model_id = "final_model",
                 training_frame = train.hex,
                 distribution = "multinomial",
                 ntrees = 42,
                 max_depth = 10,
                 min_rows = 10,
                 learn_rate = 0.175)


####################Make Final Test Set Predictions for Submission##################

##Predictions: label + 9 per-class probabilities
pred <- predict(model, test.hex)
head(pred)

##Remove label
pred <- pred[, -1]
head(pred)

##Paste the ids (first col of test set) together with the predictions
submission <- h2o.cbind(test.hex[,1], pred)
head(submission)

##Save submission to disk
h2o.exportFile(submission, paste0(dir, "submission8.csv"))