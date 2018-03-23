loadLibraries <- function(){
  library(caret)
  library(plyr)
  library(recipes)
  library(dplyr)
  library(elasticnet)
  library(doSNOW)
  library(doParallel)
  library(pROC)
  library(party)
  library(DMwR)
  library(naivebayes)
  library(nnet)
  library(mgcv)
}

readData <- function(){
  #Must set working directory prior to calling.
  train <- read.csv("cs-training.csv", header=TRUE)
  test <- read.csv("cs-test.csv", header=TRUE)
  mylist <- list(train, test)
  return(mylist)
}

NAtoMedian <- function(variable) {
  variable <- as.numeric(as.character(variable))
  variable[is.na(variable)] <- median(variable, na.rm=TRUE)
  return(variable)
}

medianReplacement <- function(medianData) {
  medianData$MonthlyIncome <- NAtoMedian(medianData$MonthlyIncome)
  medianData$NumberOfDependents <- NAtoMedian(medianData$NumberOfDependents)
  medianData$NumberOfTimes90DaysLate <- NAtoMedian(medianData$NumberOfTimes90DaysLate)
  medianData$NumberOfTime60.89DaysPastDueNotWorse <- NAtoMedian(medianData$NumberOfTime60.89DaysPastDueNotWorse)
  medianData$NumberOfTime30.59DaysPastDueNotWorse <- NAtoMedian(medianData$NumberOfTime30.59DaysPastDueNotWorse)
  medianData$age <- NAtoMedian(medianData$age)
  return(medianData)
}

imputeReplacement <- function(imputeData) {
  #transform all features to dummy variables
  dummy.vars <- dummyVars(~ ., data=imputeData[, -1])
  train.dummy <- predict(dummy.vars, imputeData[, -1])
  #leverage bagged desicion trees to impute missing values
  pre.process <- preProcess(train.dummy, method="bagImpute")
  imputed.data <- predict(pre.process, train.dummy)
  #add imputed values to data set
  imputeData$NumberOfTime30.59DaysPastDueNotWorse <- imputed.data[, 3]
  imputeData$NumberOfTime60.89DaysPastDueNotWorse <- imputed.data[, 9]
  imputeData$NumberOfTimes90DaysLate <- imputed.data[, 7]
  imputeData$MonthlyIncome <- imputed.data[, 5]
  imputeData$NumberOfDependents <- imputed.data[, 10]
  imputeData$age <- imputed.data[, 2]
  return(imputeData)
}

medianOrImpute <- function(medianData, imputeData) {
  set.seed(12345)
  indexes <- createDataPartition(medianData$SeriousDlqin2yrs, p = 0.67, list = FALSE)
  medianData.train <- medianData[indexes, ]
  medianData.test <- medianData[-indexes, ]
  imputeData.train <- imputeData[indexes, ]
  imputeData.test <- imputeData[-indexes, ]
  
  train.control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid", 
                                classProbs=TRUE, summaryFunction=twoClassSummary)
  
  cl <- makeCluster(3, type = "SOCK")
  registerDoSNOW(cl)
  set.seed(12345)
  median.logitModel <- train(SeriousDlqin2yrs~., data=medianData.train, method="glm", 
                             trControl=train.control, metric="ROC", family=binomial(link="logit"), 
                             preProcess=c("center", "scale"))
  set.seed(12345)
  impute.logitModel <- train(SeriousDlqin2yrs~., data=imputeData.train, method="glm", 
                             trControl=train.control, metric="ROC", family=binomial(link="logit"), 
                             preProcess=c("center", "scale"))
  stopCluster(cl)
  
  median.probs <- predict(median.logitModel, medianData.test[,-1], type="prob")[,2]
  impute.probs <- predict(impute.logitModel, imputeData.test[,-1], type="prob")[,2]
  median.auc <- auc(medianData.test[,1], median.probs)
  impute.auc <- auc(imputeData.test[,1], impute.probs)
  
  if (median.auc >= impute.auc) {
    return(medianData)
  } else {
    imputed <- TRUE
    return(imputeData)
  }
}

cleanData <- function(data, trainingSet) {
  #trainingSet - boolean indicator for training set being passed
  
  #remove unneccesary id column
  data <- data[ ,-1]
  
  #########################
  ## Indicator variables ##
  #########################
  data$MissingIncome <- ifelse(is.na(data$MonthlyIncome), 1, 0)
  data$ZeroIncome <- ifelse(data$MonthlyIncome==0, 1, 0) #unemployed...?
  data$ZeroIncome[is.na(data$ZeroIncome)] <- 0 
  data$MissingDependents <- ifelse(is.na(data$NumberOfDependents), 1, 0)
  #96 and 98 val across row indicators (0.9999 RevolvingUtilizationOfUnsecuredLines)
  data$Indicator_96 <- ifelse(data$NumberOfTimes90DaysLate==96, 1, 0)
  data$Indicator_98 <- ifelse(data$NumberOfTimes90DaysLate==98, 1, 0)
  
  ####################
  ## Set up factors ##
  ####################
  data$SeriousDlqin2yrs <- as.factor(make.names(data$SeriousDlqin2yrs))
  data$ZeroIncome <- as.factor(data$ZeroIncome)  
  data$MissingIncome <- as.factor(data$MissingIncome)
  data$MissingDependents <- as.factor(data$MissingDependents)
  data$Indicator_96 <- as.factor(data$Indicator_96)
  data$Indicator_98 <- as.factor(data$Indicator_98)
  
  ####################
  ## Modifly values ##
  ####################
  #change MonthlyIncome = 1 values to 0 (no different - indicating the same thing)
  data$MonthlyIncome <- ifelse(data$MonthlyIncome==1, 0, data$MonthlyIncome)
  #change 96/98 values to NA for imputation
  data$NumberOfTime30.59DaysPastDueNotWorse <- ifelse(data$NumberOfTime30.59DaysPastDueNotWorse==96,
                                                      NA, data$NumberOfTime30.59DaysPastDueNotWorse)
  data$NumberOfTime30.59DaysPastDueNotWorse <- ifelse(data$NumberOfTime30.59DaysPastDueNotWorse==98,
                                                      NA, data$NumberOfTime30.59DaysPastDueNotWorse)
  data$NumberOfTime60.89DaysPastDueNotWorse <- ifelse(data$NumberOfTime60.89DaysPastDueNotWorse==96,
                                                      NA, data$NumberOfTime60.89DaysPastDueNotWorse)
  data$NumberOfTime60.89DaysPastDueNotWorse <- ifelse(data$NumberOfTime60.89DaysPastDueNotWorse==98,
                                                      NA, data$NumberOfTime60.89DaysPastDueNotWorse)
  data$NumberOfTimes90DaysLate <- ifelse(data$NumberOfTimes90DaysLate==96,
                                         NA, data$NumberOfTimes90DaysLate)
  data$NumberOfTimes90DaysLate <- ifelse(data$NumberOfTimes90DaysLate==98,
                                         NA, data$NumberOfTimes90DaysLate)
  #change age=0 value to NA for imputation
  data$age <- ifelse(data$age==0, NA, data$age)
  
  if(trainingSet) {
    #working with training set
    medianData <- medianReplacement(data)
    imputeData <- imputeReplacement(data)
    cleanData <- medianOrImpute(medianData, imputeData)
  } else {
    #working with test set
    if(imputed) {
      #if imputation worked best with training set
      cleanData <- imputeReplacement(data)
    } else {
      #else median replacement worked best with training set
      cleanData <- medianReplacement(data)
    }
  }
  return(cleanData)
}

modelAUC <- function(myModels, newData){
  #Returns auc on test set for models passed in.
  myProbs <- lapply(myModels, function(m){predict(m, newData[,-1], type="prob")[,2]})
  myAUC <- lapply(myProbs, function(p){auc(newData[,1], p)})
  
  return(myAUC)
}

findWinners <- function(myModels, newData, num){
  #Reutrns n models with best auc on test set.
  myAUCs <- as.numeric(modelAUC(myModels, newData))
  winners <- list()
  
  for(i in 1:num){
    k <- which.max(myAUCs)
    winners[[i]] <- myModels[[k]]
    myAUCs[k] <- -1
  }
  
  return(winners)
}

makeSubmission <- function(model.final, submodels, fileName){
  #Predict defaults on test set and then write results to .csv file.
  id <- test[, 1]
  test <- cleanData(test, FALSE)
  test <- test[,-1]
  
  subProbs <- subPredictions(submodels, test)
  
  prob<- predict(model.final, subProbs, type="prob")[, 2]
  
  pred <- cbind.data.frame(id, prob)
  colnames(pred) <- c("Id", "Probability")
  write.csv(pred, file = fileName, row.names = FALSE)
}

buildGLM <- function(train1, fitCtrl){
  #Fit generalized linear models on training data.
  cl <- makeCluster(4, type="SOCK")
  registerDoSNOW(cl)
  
  set.seed(12345)
  model.logit <- train(SeriousDlqin2yrs~., 
                       data=train1, 
                       method="glm",  
                       trControl=fitCtrl,
                       metric="ROC", 
                       family=binomial(link="logit"), 
                       preProcess=c("center", "scale"))
  
  set.seed(12345)
  model.probit <- train(SeriousDlqin2yrs~., 
                        data=train1, 
                        method="glm",  
                        trControl=fitCtrl, 
                        metric="ROC", 
                        family=binomial(link="probit"), 
                        preProcess=c("center", "scale"))
  
  set.seed(12345)
  model.cloglog <- train(SeriousDlqin2yrs~., 
                         data=train1, 
                         method="glm",  
                         trControl=fitCtrl,
                         metric="ROC", 
                         family=binomial(link="cloglog"), 
                         preProcess=c("center", "scale"))
  
  stopCluster(cl)
  registerDoSEQ()
  
  models.GLM <- list(logit = model.logit,
                     probit = model.probit, 
                     cloglog = model.cloglog)
  
  save(models.GLM, file = "models_GLM.rda")
  return(models.GLM)
}

buildPGLM <- function(train1, fitCtrl){
  #Fit penealized regression models on training data.
  cl <- makeCluster(4, type="SOCK")
  registerDoSNOW(cl)
  
  set.seed(12345)
  model.elnet <- train(SeriousDlqin2yrs~., 
                       data=train1, 
                       method="glmnet", 
                       trControl=fitCtrl, 
                       metric="ROC", 
                       family="binomial", 
                       tuneLength=100, 
                       preProcess=c("center", "scale"))
  
  
  grid.lasso <- expand.grid(alpha = 1, lambda = unique(model.elnet$results$lambda))
  grid.ridge <- expand.grid(alpha = 0, lambda = unique(model.elnet$results$lambda))
  
  
  set.seed(12345)
  model.LASSO <- train(SeriousDlqin2yrs~., 
                       data=train1, method="glmnet", 
                       trControl=fitCtrl, metric="ROC", 
                       family="binomial", 
                       tuneGrid=grid.lasso, 
                       preProcess=c("center", "scale"))
  
  
  set.seed(12345)
  model.ridge <- train(SeriousDlqin2yrs~., 
                       data=train1, 
                       method="glmnet", 
                       trControl=fitCtrl, 
                       metric="ROC", 
                       family="binomial", 
                       tuneGrid=grid.ridge, 
                       preProcess=c("center", "scale"))
  
  stopCluster(cl)
  registerDoSEQ()
  
  models.PGLM <- list(elnet = model.elnet, 
                      LASSO = model.LASSO, 
                      ridge = model.ridge)
  
  save(models.PGLM, file = "models_PGLM.rda")
  return(models.PGLM)
}

buildMisc <- function(train1, fitCtrl){
  #Fit other models on trainig data.
  cl <- makeCluster(4, type="SOCK")
  registerDoSNOW(cl)
  
  
  set.seed(12345)
  #Naive Bayes classifier
  model.naiveb <- train(SeriousDlqin2yrs~., 
                        data=train1, 
                        method="naive_bayes", 
                        trControl=fitCtrl,
                        metric="ROC",  
                        preProcess=c("center", "scale"))
  
  
  #Neural net
  grid.nnet <- expand.grid(size=10, decay=0)
  
  set.seed(12345)
  model.nnet <- train(SeriousDlqin2yrs~., 
                      data=train1, 
                      method="nnet", 
                      trControl=fitCtrl, 
                      tuneGrid=grid.nnet, 
                      metric="ROC", 
                      preProcess=c("center", "scale"), 
                      trace=FALSE, 
                      reltol=1e-3)
  
  
  set.seed(12345)
  #Generalized additive model
  model.gam <- train(SeriousDlqin2yrs~., 
                     data=train1, 
                     method="gamSpline", 
                     trControl=fitCtrl,
                     metric="ROC", 
                     famil=binomial(link="logit"),
                     preProcess=c("center", "scale"))
  
  set.seed(12345)
  #Conditional random forest
  model.cforest <- train(SeriousDlqin2yrs~., 
                         data=train1[sample(1:nrow(train1), 1000),], #Fit on subset due to computational&storage limites
                         method="cforest", 
                         trControl=fitCtrl,
                         metric="ROC", 
                         preProcess=c("center", "scale"))
  
  
  
  
  stopCluster(cl)
  registerDoSEQ()
  
  models.Misc <- list(cforest = model.cforest, 
                      neuralnet = model.nnet, 
                      naiveBayes = model.naiveb,
                      gam = model.gam)
  
  save(models.Misc, file = "models_Misc.rda")
  return(models.Misc)
}

buildEnsemble <- function(myModels, train2, fitCtrl){
  #Fit Bayesian glm using predicted probabilities from selected models.
  
  myProbs <- subPredictions(myModels, train2)
  myProbs$Y <- train2$SeriousDlqin2yrs
  
  model.blogit <- train(Y ~., 
                        data=myProbs, 
                        method="bayesglm", 
                        trControl=fitCtrl, 
                        metric="ROC", 
                        family=binomial(link="logit"))
  
  save(model.blogit, file = "model_FINAL.rda")
  
  return(model.blogit)
}

subPredictions <- function(submodels, train2){
  probs <- lapply(submodels, function(m){predict(m, train2[,-1], type="prob")[,2]})
  names(probs)<-paste("m", 1:length(probs), sep = "")
  probs<-as.data.frame(probs)
  return(probs)
}

runItAll <- function(){
  setwd("~/Desktop/STAT458/Project")
  loadLibraries()
  
  compData <- readData()
  
  imputed <- FALSE
  train <- cleanData(compData[[1]], TRUE)

  set.seed(12345)
  index <- createDataPartition(train$SeriousDlqin2yrs, p=0.67, list=FALSE)
  train1 <- train[index,]
  train2 <- train[-index,]
  
  d1 <- SMOTE(SeriousDlqin2yrs~., data = train1, k=10, perc.over = 1000, perc.under = 0)
  train1 <- rbind(train1, d1)
  
  fitCtrl <- trainControl(method="repeatedcv", 
                          number=10, 
                          repeats=3, 
                          classProbs=TRUE, 
                          summaryFunction=twoClassSummary, 
                          selectionFunction ="oneSE")
  
  
  
  myGLM <- buildGLM(train1, fitCtrl)
  myPGLM <- buildPGLM(train1, fitCtrl)
  myMisc <- buildMisc(train1, fitCtrl)
  
  submodels <- c(findWinners(myMisc, train2, 2), 
                 findWinners(myPGLM, train2, 1),
                 findWinners(myGLM, train2, 1))
  
  
  #---------------IF LOADING MODELS IN---------------
  
  load("models_GLM.rda")
  load("models_PGLM.rda")
  load("models_Misc.rda")
  models.Misc<-models.Misc[-4] #GAM throws error when predicting is loaded in
  
  submodels <- c(findWinners(models.Misc, train2, 2), 
                 findWinners(models.GLM, train2, 1),
                 findWinners(models.PGLM, train2, 1))
  
  #---------------------------------------------------- 
  
  
  #submodels <- c(findWinners(myMisc, train2, 2), 
  #                findWinners(myPGLM, train2, 1),
  #                findWinners(myGLM, train2, 1))
  
  model.FINAL <- buildEnsemble(submodels, train2, fitCtrl)
  makeSubmission(model.FINAL, submodels, "testtesttest6.csv")
  
}
