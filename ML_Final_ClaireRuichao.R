#-------------------------------------------------------#
# Machine Learning in Public Health                     #
# Final Project Predicting Heart Failure Related Death  #
# Model building                                        #
# Ruichao Shi, Xiangying(Claire) Chu                    #
#-------------------------------------------------------#

# load packages
library(boot)
library(pROC)
library(tree)
library(gbm)
library(randomForest)
library(tidyverse)
library(caret)
# import data set
hf.dt <- read.csv("~/Downloads/heart_failure_clinical_records_dataset.csv")


####### Data cleaning and Descriptive statistics ######

#check missing
na <- is.na(hf.dt)
table(na)
#summary statistic to check outliers
sapply(hf.dt, sd, na.rm=TRUE)
summary(hf.dt) # no outlier
#multicollinearity(checked in each model below)



##########################logistic regression#################################
accuracy = c()
auc = c()
mcc = c()
hf.dt$DEATH_EVENT <- as.factor(hf.dt$DEATH_EVENT)
# cross validation----set training and testing
# loop 10 times to get the average performance
for (i in 1:10){
set.seed(i)
train.num <- sample(1:299, 300*0.8, replace=FALSE)
Train_set <- hf.dt[train.num,]
Test_set <- hf.dt[-train.num,]
# build logistic model
M1 <- glm(DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + diabetes + ejection_fraction + 
            high_blood_pressure + platelets + serum_creatinine +serum_sodium + 
            sex + smoking, data = Train_set, family=binomial)
prob1 = predict(M1, Test_set, type = "response")
pred1=rep ("0" ,59)
pred1[prob1 >.5]="1"
# metric calculation
accuracy = c(accuracy,mean(pred1==Test_set$DEATH_EVENT))
auc[i] <- roc(Test_set$DEATH_EVENT,prob1)$auc
mcc=c(mcc,mccr(Test_set$DEATH_EVENT,pred1))
}
#check multicollinearity
car::vif(M1) # no problem
# model assessment metric
mean(accuracy)  #0.740678
mean(auc)       #0.7531225
mean(mcc)       #0.3286184

################################# Bagging #################################

accuracy = c()
auc = c()
mcc = c()
# cross validation----set training and testing
# loop 10 times to have the average performance
for (i in 1:10){
  set.seed(i)
  train.num <- sample(1:299, 300*0.8, replace=FALSE)
  Train_set <- hf.dt[train.num,]
  Test_set <- hf.dt[-train.num,]
  # tree building
  M2 = randomForest(DEATH_EVENT~., data = Train_set, mtry=11, importance=TRUE)
  # assessment-- metric calculation
  yhat.bag = predict(M2, newdata = Test_set)
  accuracy = c(accuracy,mean(yhat.bag==Test_set$DEATH_EVENT))
  auc[i] <- roc(Test_set$DEATH_EVENT,as.numeric(yhat.bag))$auc
  mcc=c(mcc,mccr(Test_set$DEATH_EVENT,yhat.bag))
} 

mean(accuracy)  # 0.8152542
mean(auc) # 0.7784277
mean(mcc) # 0.5618025

########################## Random Forest #################################
accuracy = c()
auc = c()
mcc = c()
# cross validation----set training and testing
# loop 10 times to have the average performance
for (i in 1:10) {
  set.seed(i)
  train.num <- sample(1:299, 300*0.8, replace=FALSE)
  Train_set <- hf.dt[train.num,]
  Test_set <- hf.dt[-train.num,]
  M3 <- randomForest(DEATH_EVENT~., data = Train_set, mtry = 3, importance=TRUE)
  # prediction
  pred3 <- predict(M3, Test_set)
  # metric calculation
  accuracy = c(accuracy,mean(pred3==Test_set$DEATH_EVENT))
  auc[i] <- roc(Test_set$DEATH_EVENT,as.numeric(pred3))$auc
  mcc=c(mcc,mccr(Test_set$DEATH_EVENT,pred3))
}
mean(accuracy) #0.8474576
mean(auc) # 0.8011567
mean(mcc) # 0.6307342

################################# Boosting #################################
accuracy=c()
auc=c()
mcc=c()
hf.dt$DEATH_EVENT = as.numeric(hf.dt$DEATH_EVENT)
for (i in 1:10){
  set.seed(i)
  train.num <- sample(1:299, 300*0.8, replace=FALSE)
  Train_set <- hf.dt[train.num,]
  Test_set <- hf.dt[-train.num,]
  M4 = gbm(DEATH_EVENT~.,data=Train_set, distribution=
             "bernoulli",n.trees =5000 , interaction.depth =4)
  prob4 = predict(M4, newdata = Test_set, n.trees =5000,type="response")
  pred4=rep ("0" ,59)
  pred4[prob4 >.5]="1"
  # metric calculation
  accuracy = c(accuracy,mean(pred4==Test_set$DEATH_EVENT))
  auc[i] <- roc(Test_set$DEATH_EVENT,prob4)$auc
  mcc=c(mcc,mccr(Test_set$DEATH_EVENT,as.numeric(pred4)))
}
mean(accuracy)  # 0.8355932
mean(auc) # 0.8858226
mean(mcc) # 0.5949253


################################# Decision Tree #################################
accuracy=c()
auc=c()
mcc=c()
hf.dt$DEATH_EVENT = as.factor(hf.dt$DEATH_EVENT)
for (i in 1:10){
  set.seed(i)
  train.num <- sample(1:299, 300*0.8, replace=FALSE)
  Train_set <- hf.dt[train.num,]
  Test_set <- hf.dt[-train.num,]
  M5 = tree(DEATH_EVENT~., data = Train_set)
  prob5 = predict(M5, newdata = Test_set)[,2]
  pred5=rep ("0" ,59)
  pred5[prob5 >.5]="1"
  # metric calculation
  accuracy = c(accuracy,mean(pred5==Test_set$DEATH_EVENT))
  auc[i] <- roc(Test_set$DEATH_EVENT,as.numeric(prob5))$auc
  mcc=c(mcc,mccr(Test_set$DEATH_EVENT,as.numeric(pred5)))
}
mean(accuracy)  # 0.7949153
mean(auc) # 0.8315364
mean(mcc) # 0.5123956

##################################################################
library(readr)
library(tidyverse)
library(ROCR)
library(mccr)
hf.dt <- read_csv("heart_failure_clinical_records_dataset.csv")
hf.dt$DEATH_EVENT=factor(hf.dt$DEATH_EVENT)

# KNN
library(class)
n=nrow(hf.dt)
accuracy=c()
auc=c()
mcc=c()
for (i in 1:10) {
  set.seed(i)
  rand_ind = sample(1:n)
  tr_ind = rand_ind[1:floor(0.8*n)]
  train_ind <- tr_ind[1:179]
  val_ind <- tr_ind[180:239]
  train=hf.dt[train_ind,]
  train_death=train$DEATH_EVENT
  train=train %>%
    select(-DEATH_EVENT)
  validation=hf.dt[val_ind,]
  val_death=validation$DEATH_EVENT
  validation=validation %>%
    select(-DEATH_EVENT)
  K_seq <- seq(from = 1, to = 99, by = 1)
  len <- length(K_seq)
  val_err_seq <- rep(0,len)
  for(j in 1:len){
    K <- K_seq[j]
    knn.test.pred = knn(train,
                        validation, 
                        train_death, k = K,prob = T)
    val_err_seq[j] <- mean(knn.test.pred != val_death) 
  }
  opt_ind <- max(which(val_err_seq == min(val_err_seq)))
  opt_K <- K_seq[opt_ind]
  tr=hf.dt[tr_ind,]
  tr_death=tr$DEATH_EVENT
  tr=tr %>%
    select(-DEATH_EVENT)
  test=hf.dt[-tr_ind,]
  test_death=test$DEATH_EVENT
  test=test %>%
    select(-DEATH_EVENT)
  knn.test.pred = knn(tr,
                      test, 
                      tr_death, k = opt_K,prob = T)
  accuracy=c(accuracy,mean(knn.test.pred == test_death))
  auc[i] <- roc(test_death,as.numeric(knn.test.pred))$auc
  mcc=c(mcc,mccr(test_death,knn.test.pred))
}
mean(accuracy) #0.6733333
mean(auc) #0.4881077
mean(mcc) #-0.05965175

# svm linear
library(e1071)
accuracy=c()
roc_auc=c()
auc=c()
mcc=c()
for (i in 1:10) {
  set.seed(i)
  rand_ind = sample(1:299)
  tr_ind = rand_ind[1:floor(0.8*299)]
  train=hf.dt[tr_ind,]
  test=hf.dt[-tr_ind,]
  tune.out=tune(svm,DEATH_EVENT~.,data=train,kernel="linear",ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))
  bestmod=tune.out$best.model
  svmfit=svm(DEATH_EVENT~., data=train, kernel="linear",cost=bestmod$cost,probability=T)
  test.pred0=predict(svmfit,test,probability = T)
  accuracy = c(accuracy,mean(test.pred0==test$DEATH_EVENT))
  auc[i] <- roc(test$DEATH_EVENT,as.numeric(test.pred0))$auc
  mcc=c(mcc,mccr(test$DEATH_EVENT,test.pred0))
}
mean(accuracy) #0.825
mean(auc) #0.7696144
mean(mcc) #0.5682533

# svm radial
accuracy=c()
auc=c()
mcc=c()
for (i in 1:10) {
  set.seed(i)
  rand_ind = sample(1:299)
  tr_ind = rand_ind[1:floor(0.8*299)]
  train=hf.dt[tr_ind,]
  test=hf.dt[-tr_ind,]
  tune.out=tune(svm,DEATH_EVENT~., data=train,kernel="radial",ranges=list(cost=c(0.1,1,10,100,1000),gamma=c(0.5,1,2,3,4)))
  bestmod=tune.out$best.model
  svmfit=svm(DEATH_EVENT~., data=train, kernel="radial",gamma=bestmod$gamma,cost=bestmod$cost,probability=T)
  test.pred0=predict(svmfit,test,probability = T)
  accuracy = c(accuracy,mean(test.pred0==test$DEATH_EVENT))
  auc[i] <- roc(test$DEATH_EVENT,as.numeric(test.pred0))$auc
  mcc=c(mcc,mccr(test$DEATH_EVENT,test.pred0))
}
mean(accuracy) #0.69
mean(auc) #0.5334201
mean(mcc) #0.1091203