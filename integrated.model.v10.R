loadLibraries <- function(){
  library(mgcv)
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
}

imputeReplacement <- function(imputeData) {
  # transform all features to dummy variables -------------------------------
  dummy.vars <- dummyVars(~ ., data=imputeData[, -1])
  train.dummy <- predict(dummy.vars, imputeData[, -1])
  # leverage bagged desicion trees to impute missing values -----------------
  pre.process <- preProcess(train.dummy, method="bagImpute")
  imputed.data <- predict(pre.process, train.dummy)
  # Integrate imputed vals with data ----------------------------------------
  imputeData$MonthlyIncome <- imputed.data[, 5]
  imputeData$NumberOfDependents <- imputed.data[, 10]
  
  return(imputeData)
}

cleanData <- function(data) {
  data <- data[ ,-1] #remove unneccesary id column
  # Set up factors ----------------------------------------------------------
  data$SeriousDlqin2yrs <- as.factor(make.names(data$SeriousDlqin2yrs))
  # Deal with outliers ------------------------------------------------------
  data$MonthlyIncome <- ifelse(data$MonthlyIncome > 300000,
                               300000,
                               data$MonthlyIncome)
  data$DebtRatio <- ifelse(data$DebtRatio > 500,
                           500,
                           data$DebtRatio)
  data$RevolvingUtilizationOfUnsecuredLines <- ifelse(data$RevolvingUtilizationOfUnsecuredLines > 2,
                                                      2,
                                                      data$RevolvingUtilizationOfUnsecuredLines)
  data$NumberRealEstateLoansOrLines <- ifelse(data$NumberRealEstateLoansOrLines > 17,
                                              17,
                                              data$NumberRealEstateLoansOrLines)
  data$NumberOfDependents <- ifelse(data$NumberOfDependents > 10,
                                    10,
                                    data$NumberOfDependents)
  # Median replacement ------------------------------------------------------
  data$MonthlyIncome <- ifelse(data$MonthlyIncome==1, 0, data$MonthlyIncome) #MonthlyIncome = 1 values to 0
  data$NumberOfTime30.59DaysPastDueNotWorse <- ifelse(data$NumberOfTime30.59DaysPastDueNotWorse==96,
                                                      median(data$NumberOfTime30.59DaysPastDueNotWorse, na.rm=TRUE), 
                                                      data$NumberOfTime30.59DaysPastDueNotWorse)
  data$NumberOfTime30.59DaysPastDueNotWorse <- ifelse(data$NumberOfTime30.59DaysPastDueNotWorse==98,
                                                      median(data$NumberOfTime30.59DaysPastDueNotWorse, na.rm=TRUE), 
                                                      data$NumberOfTime30.59DaysPastDueNotWorse)
  data$NumberOfTime60.89DaysPastDueNotWorse <- ifelse(data$NumberOfTime60.89DaysPastDueNotWorse==96,
                                                      median(data$NumberOfTime60.89DaysPastDueNotWorse, na.rm=TRUE), 
                                                      data$NumberOfTime60.89DaysPastDueNotWorse)
  data$NumberOfTime60.89DaysPastDueNotWorse <- ifelse(data$NumberOfTime60.89DaysPastDueNotWorse==98,
                                                      median(data$NumberOfTime60.89DaysPastDueNotWorse, na.rm=TRUE), 
                                                      data$NumberOfTime60.89DaysPastDueNotWorse)
  data$NumberOfTimes90DaysLate <- ifelse(data$NumberOfTimes90DaysLate==96,
                                         median(data$NumberOfTimes90DaysLate, na.rm=TRUE), 
                                         data$NumberOfTimes90DaysLate)
  data$NumberOfTimes90DaysLate <- ifelse(data$NumberOfTimes90DaysLate==98,
                                         median(data$NumberOfTimes90DaysLate, na.rm=TRUE), 
                                         data$NumberOfTimes90DaysLate)
  data$age <- ifelse(data$age==0, 
                     median(data$age, na.rm=TRUE), 
                     data$age)
  # Impute NAs: MonthlyIncome & NumberOfDependents --------------------------
  data <- imputeReplacement(data)
  # Return cleaned data -----------------------------------------------------
  
  return(data)
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

makeSubmission <- function(prob, id, fileName){
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
  
  save(models.GLM, file = "models_GLMvm.rda")
  
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
  
  save(models.PGLM, file = "models_PGLMvm.rda")
  
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
  
  save(models.Misc, file = "models_Miscvm.rda")
  
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
  
  save(model.blogit, file = "model_FINALvm.rda")
  
  return(model.blogit)
}

plotROC <- function(response, modelPreds, modelNames) {
  #pass function response variable, model predictions, and model name(s) 
  colors <- c("coral", "royalblue1", "mediumseagreen", "yellow")
  auc <- c(rep(0,length(modelNames)))
  if (length(modelNames) > 1) {
    for (i in 1:length(modelNames)) { 
      auc[i] <- roc(response, modelPreds[ ,i])$auc 
    }
    plot(roc(response, modelPreds[ ,1]), 
         main="ROC Curve(s)", 
         col=colors[1],
         legacy.axes=TRUE,
         ylab="Sensitivity (TPR)",
         xlab="1 - Specificity (FPR)",
         ylim=c(0,1),
         xlim=c(1,0))
    for (i in 1:length(modelNames)-1) {
      plot(roc(response, modelPreds[ ,i+1]), 
           add=TRUE, 
           col=colors[i+1])
    }
  } else {
    auc <- roc(response, modelPreds)$auc
    plot(roc(response, modelPreds), 
         main="ROC Curve(s)", 
         col=colors[1],
         legacy.axes=TRUE,
         ylab="Sensitivity (TPR)",
         xlab="1 - Specificity (FPR)",
         ylim=c(0,1),
         xlim=c(1,0))
  }
  grid()
  legend(0.6, 0.3, legend=paste(modelNames, round(auc, 4), sep=": AUC = "),
         col=colors, lty=1, cex=0.6, bg="lightgrey", text.font=4)
}

subPredictions <- function(submodels, train2){
  probs <- lapply(submodels, function(m){predict(m, train2[,-1], type="prob")[,2]})
  names(probs)<-paste("m", 1:length(probs), sep = "")
  probs<-as.data.frame(probs)
  
  return(probs)
}

ensemblePredictions <- function(model.final, submodels, test){
  subProbs <- subPredictions(submodels, test)
  prob<- predict(model.final, subProbs, type="prob")[, 2]
  
  return(prob)
}

savePredictions <- function(myModels, newData, fileName){
  myProbs <- as.data.frame(lapply(myModels, function(m){predict(m, newData[,-1], type="prob")[,2]}))
  y <- newData$SeriousDlqin2yrs
  myPreds <- cbind.data.frame(y, myProbs)
  
  save(myPreds, file=fileName)
}

runItAll <- function(){
  loadLibraries()
  
  train <- read.csv("cs-training.csv", header=TRUE)
  train <- cleanData(train)
  
  set.seed(12345)
  index <- createDataPartition(train$SeriousDlqin2yrs, p=0.67, list=FALSE)
  
  train1 <- train[index,]
  train2 <- train[-index,]
 
  d1 <- SMOTE(SeriousDlqin2yrs~., data = train1, k=10, perc.over = 800, perc.under = 0)
  train1 <- rbind(train1, d1)
  
  fitCtrl <- trainControl(method="repeatedcv", 
                          number=10, 
                          repeats=3, 
                          classProbs=TRUE, 
                          summaryFunction=twoClassSummary, 
                          selectionFunction ="oneSE")

  
  #Fit and save models
  myGLM <- buildGLM(train1, fitCtrl)
  myPGLM <- buildPGLM(train1, fitCtrl)
  myMisc <- buildMisc(train1, fitCtrl)
  
  
  #Save predictions for report
  savePredictions(myGLM, train2, "GLM_predvm.rda")
  savePredictions(myPGLM, train2, "PGLM_predvm.rda")
  savePredictions(myMisc, train2, "Misc_predvm.rda")
  
  
  #Find n best of each model type
  submodels <- c(findWinners(myMisc, train2, 2), 
                 findWinners(myPGLM, train2, 1),
                 findWinners(myGLM, train2, 1))
  
  save(submodels, file="submodelsvm.rda")
  
  #Remove models to save space
  rm(myGLM, myPGLM, myMisc)
  gc(verbose=FALSE)
  
 
  #Fit ensemble model
  model.FINAL <- buildEnsemble(submodels, train2, fitCtrl)
  
  #Prep test data and make submission
  test <- read.csv("cs-test.csv", header=TRUE)
  id <- test[,1]
  test <- cleanData(test)
  
  pred <- ensemblePredictions(model.FINAL, submodels, test)
  makeSubmission(pred, id, "Submission_vm.csv")
}
