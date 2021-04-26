# --> Log the wage variables
# --> Convert back then use the median
# --> Drop the nationality

### ST4248 Projects ###
set.seed(10)
# First, read in the dataset
goalkeeper_train = read.csv("C:/Users/Kiat Kai/Desktop/NOTES/Y4S2/ST4248/Group Project/Data/Goalkeeper_train.csv")
goalkeeper_test = read.csv("C:/Users/Kiat Kai/Desktop/NOTES/Y4S2/ST4248/Group Project/Data/Goalkeeper_test.csv")
outfield_train = read.csv("C:/Users/Kiat Kai/Desktop/NOTES/Y4S2/ST4248/Group Project/Data/Outfield_train.csv")
outfield_test = read.csv("C:/Users/Kiat Kai/Desktop/NOTES/Y4S2/ST4248/Group Project/Data/Outfield_test.csv")

# Remove variables Nationality and ID
goalkeeper_train = subset(goalkeeper_train, select = -c(Position, Nationality, ID))
goalkeeper_test = subset(goalkeeper_test, select = -c(Position, Nationality, ID))
outfield_train = subset(outfield_train, select = -c(Nationality, ID))
outfield_test = subset(outfield_test, select = -c(Nationality, ID))

dim(goalkeeper_train)
dim(goalkeeper_test)
dim(outfield_train)
dim(outfield_test)

# Log the variables wage
goalkeeper_train$Wage = log(goalkeeper_train$Wage)
goalkeeper_test$Wage = log(goalkeeper_test$Wage)
outfield_train$Wage = log(outfield_train$Wage)
outfield_test$Wage = log(outfield_test$Wage)

# test set for goalkeeper_test
ytest_gk = goalkeeper_test$Wage
Xtest_gk = model.matrix(Wage ~., goalkeeper_test)
Xtest1_gk = Xtest_gk[,-1] # remove intercept
dim(Xtest1_gk)

# training set for goalkeeper_train
ytrain_gk = goalkeeper_train$Wage
Xtrain_gk = model.matrix(Wage ~., goalkeeper_train)
Xtrain1_gk = Xtrain_gk[,-1]
dim(Xtrain1_gk)

# test set for outfield_test
ytest_out = outfield_test$Wage
Xtest_out = model.matrix(Wage ~., outfield_test)
Xtest1_out = Xtest_out[,-1] # remove intercept
dim(Xtest1_out)

# training set for goalkeeper_train
ytrain_out = outfield_train$Wage
Xtrain_out = model.matrix(Wage ~., outfield_train)
Xtrain1_out = Xtrain_out[,-1]
dim(Xtrain1_out)


##################################################
# lasso and SCAD for training set for goalkeeper #
##################################################
library(glmnet)        # ridge regression and lasso
library(ncvreg)        # SCAD
library(plotmo)        # plot glmnet model

# apply ridge regression and lasso to standardized predictors #
lasso <- glmnet(Xtrain1_gk, ytrain_gk, alpha=1)       # alpha=1:lasso
scad <- ncvreg(Xtrain1_gk, ytrain_gk, penalty="SCAD")

# plot coefficient paths
par(mar=c(3.7,3.7,0.9,1), mgp=c(5,1,0), mfrow=c(1,2))
plot_glmnet(lasso, label=TRUE, grid.col="lightgray",
            col=c("black", "red", "blue", "purple"), main="Lasso")
plot(scad, log.l=TRUE, main="Path of coefficient estimates"
     , main="SCAD")

# select lambda using 10-fold cross-validation
set.seed(10)
lasso_cv <- cv.glmnet(Xtrain1_gk, ytrain_gk, alpha=1, nfolds=10)
scad_cv <- cv.ncvreg(Xtrain1_gk, ytrain_gk, penalty="SCAD", nfolds=10, seed=10)

# CV plot
par(mar=c(5,3.7,2.5,1), mgp=c(2.2,1,0), mfrow=c(1,2))
plot(lasso_cv, sub="Lasso") 
plot(scad_cv, sub="SCAD", log.l=TRUE, las=3)

# minimum CV error
lasso_cv$lambda.min  
scad_cv$lambda.min  

# one sd rule
(lambdaL <- lasso_cv$lambda.1se)  # optimal lambda = 0.02415556 for lasso

id <- which.min(scad_cv$cve) # cve is the error for each value of lambda, averaged across the cross-validation folds
upper <- scad_cv$cve + scad_cv$cvse  # cvse is the estimated standard error associated with each value of for cve
# select largest lambda whose CV error is within 1 standard error of the minimum
id <- which(scad_cv$cve < upper[id])[1]  # which gives the TRUE indices of a logical object
(lambdaS <- scad_cv$lambda[id])   # optimal lambda = 0.02909546 for SCAD

# get coef of model at selected lambda
CL <- as.matrix(coef(lasso, s=lambdaL)) # coefficient for lasso
CL[CL!=0,]

CS <- coef(scad_cv, lambda=lambdaS) # coefficient for scad
CS[CS!=0]  

# compute predictions at selected lambda
lasso_pred <- predict(lasso, s=lambdaL, newx=Xtrain1_gk) 
scad_pred <- predict(scad, lambda=lambdaS, X=Xtrain1_gk) 

# compute RSS/MSE
(lasso_RSS <- mean((ytrain_gk-lasso_pred)^2))
(scad_RSS <- mean((ytrain_gk-scad_pred)^2))

# compute RMSLE
(lasso_RMSE <- sqrt(mean((ytrain_gk-lasso_pred)^2)))
(scad_RMSE <- sqrt(mean((ytrain_gk-scad_pred)^2)))

# Converting back to normal scale, then get RMSE
sqrt(mean((exp(ytrain_gk) - exp(lasso_pred))^2))
sqrt(mean((exp(ytrain_gk) - exp(scad_pred))^2))

# Median absolute error
median(abs(exp(ytrain_gk) - exp(lasso_pred)))
median(abs(exp(ytrain_gk) - exp(scad_pred)))

##########################################
# lasso and SCAD for test set goalkeeper #
##########################################
# apply lasso to standardized predictors #
lasso_test <- glmnet(Xtest1_gk, ytest_gk, alpha=1)       # alpha=1:lasso
scad_test <- ncvreg(Xtest1_gk, ytest_gk, penalty="SCAD")

# plot coefficient paths
par(mar=c(3.7,3.7,0.9,1), mgp=c(2.2,1,0), mfrow=c(1,2))
plot_glmnet(lasso_test, label=TRUE, grid.col="lightgray",
            col=c("black", "red", "blue", "purple"), main="Lasso")
plot(scad_test, log.l=TRUE, main="Path of coefficient estimates"
     , main="SCAD")

# select lambda using 10-fold cross-validation
set.seed(10)
lasso_cv_test <- cv.glmnet(Xtest1_gk, ytest_gk, alpha=1, nfolds=10)
scad_cv_test <- cv.ncvreg(Xtest1_gk, ytest_gk, penalty="SCAD", nfolds=10, seed=10)

# CV plot
par(mar=c(5,3.7,2.5,1), mgp=c(2.2,1,0), mfrow=c(1,2))
plot(lasso_cv_test, sub="Lasso") 
plot(scad_cv_test, sub="SCAD", log.l=TRUE, las=3)

# minimum CV error
lasso_cv_test$lambda.min  
scad_cv_test$lambda.min  

# one sd rule
(lambdaL_test <- lasso_cv_test$lambda.1se)  # min lambda = 0.001538865 for lasso

id <- which.min(scad_cv_test$cve)
upper <- scad_cv_test$cve + scad_cv_test$cvse
id <- which(scad_cv_test$cve < upper[id])[1]
(lambdaS_test <- scad_cv_test$lambda[id])   # min lambda = 91.21312 for SCAD

# get coef of model at selected lambda
CL <- as.matrix(coef(lasso_test, s=lambdaL_test)) # coefficient for lasso
CL[CL!=0,]

CS <- coef(scad_cv_test, lambda=lambdaS_test) # coefficient for scad
CS[CS!=0]  

# compute predictions at selected lambda
lasso_pred_test <- predict(lasso_test, s=lambdaL_test, newx=Xtest1_gk) 
scad_pred_test <- predict(scad_test, lambda=lambdaS_test, X=Xtest1_gk) 

# compute RSS
(lasso_RSS <- mean((ytest_gk-lasso_pred_test)^2))
(scad_RSS <- mean((ytest_gk-scad_pred_test)^2))

# compute RMSLE
(lasso_RMSE <- sqrt(mean((ytest_gk-lasso_pred_test)^2)))
(scad_RMSE <- sqrt(mean((ytest_gk-scad_pred_test)^2)))

# Converting back to normal scale, then get RMSE
sqrt(mean((exp(ytest_gk) - exp(lasso_pred_test))^2))
sqrt(mean((exp(ytest_gk) - exp(scad_pred_test))^2))

# Median absolute error
median(abs(exp(ytest_gk) - exp(lasso_pred_test)))
median(abs(exp(ytest_gk) - exp(scad_pred_test)))

# Mean absolute error
mean(abs(exp(ytest_gk) - exp(lasso_pred_test)))
mean(abs(exp(ytest_gk) - exp(scad_pred_test)))

######################################################################
########################## END #######################################
######################################################################

################################################
# lasso and SCAD for training set for outfield #
################################################
library(glmnet)        # ridge regression and lasso
library(ncvreg)        # SCAD
library(plotmo)        # plot glmnet model

# apply ridge regression and lasso to standardized predictors #
lasso <- glmnet(Xtrain1_out, ytrain_out, alpha=1)       # alpha=1:lasso
scad <- ncvreg(Xtrain1_out, ytrain_out, penalty="SCAD")

# plot coefficient paths
par(mar=c(3.7,3.7,0.9,1), mgp=c(5,1,0), mfrow=c(1,2))
plot_glmnet(lasso, label=TRUE, grid.col="lightgray",
            col=c("black", "red", "blue", "purple"), main="Lasso")
plot(scad, log.l=TRUE, main="Path of coefficient estimates"
     , main="SCAD")

# select lambda using 10-fold cross-validation
set.seed(10)
lasso_cv <- cv.glmnet(Xtrain1_out, ytrain_out, alpha=1, nfolds=10)
scad_cv <- cv.ncvreg(Xtrain1_out, ytrain_out, penalty="SCAD", nfolds=10, seed=10)

# CV plot
par(mar=c(5,3.7,2.5,1), mgp=c(2.2,1,0), mfrow=c(1,2))
plot(lasso_cv, sub="Lasso") 
plot(scad_cv, sub="SCAD", log.l=TRUE, las=3)

# minimum CV error
lasso_cv$lambda.min  
scad_cv$lambda.min  

# one sd rule
(lambdaL <- lasso_cv$lambda.1se)  # optimal lambda = 0.02415556 for lasso

id <- which.min(scad_cv$cve) # cve is the error for each value of lambda, averaged across the cross-validation folds
upper <- scad_cv$cve + scad_cv$cvse  # cvse is the estimated standard error associated with each value of for cve
# select largest lambda whose CV error is within 1 standard error of the minimum
id <- which(scad_cv$cve < upper[id])[1]  # which gives the TRUE indices of a logical object
(lambdaS <- scad_cv$lambda[id])   # optimal lambda = 0.02909546 for SCAD

# get coef of model at selected lambda
CL <- as.matrix(coef(lasso, s=lambdaL)) # coefficient for lasso
CL[CL!=0,]

CS <- coef(scad_cv, lambda=lambdaS) # coefficient for scad
CS[CS!=0]  

# compute predictions at selected lambda
lasso_pred <- predict(lasso, s=lambdaL, newx=Xtrain1_out) 
scad_pred <- predict(scad, lambda=lambdaS, X=Xtrain1_out) 

# compute RSS
(lasso_RSS <- mean((ytrain_out-lasso_pred)^2))
(scad_RSS <- mean((ytrain_out-scad_pred)^2))

# compute RMSLE
(lasso_RMSE <- sqrt(mean((ytrain_out-lasso_pred)^2)))
(scad_RMSE <- sqrt(mean((ytrain_out-scad_pred)^2)))

# Converting back to normal scale, then get RMSE
sqrt(mean((exp(ytrain_out) - exp(lasso_pred))^2))
sqrt(mean((exp(ytrain_out) - exp(scad_pred))^2))

# Median absolute error
median(abs(exp(ytrain_out) - exp(lasso_pred)))
median(abs(exp(ytrain_out) - exp(scad_pred)))

# Mean absolute error
mean(abs(exp(ytrain_out) - exp(lasso_pred)))
mean(abs(exp(ytrain_out) - exp(scad_pred)))

########################################
# lasso and SCAD for outfield test set #
########################################
# apply lasso to standardized predictors #
lasso_test <- glmnet(Xtest1_out, ytest_out, alpha=1)       # alpha=1:lasso
scad_test <- ncvreg(Xtest1_out, ytest_out, penalty="SCAD")

# plot coefficient paths
par(mar=c(3.7,3.7,0.9,1), mgp=c(2.2,1,0), mfrow=c(1,2))
plot_glmnet(lasso_test, label=TRUE, grid.col="lightgray",
            col=c("black", "red", "blue", "purple"), main="Lasso")
plot(scad_test, log.l=TRUE, main="Path of coefficient estimates"
     , main="SCAD")

# select lambda using 10-fold cross-validation
set.seed(10)
lasso_cv_test <- cv.glmnet(Xtest1_out, ytest_out, alpha=1, nfolds=10)
scad_cv_test <- cv.ncvreg(Xtest1_out, ytest_out, penalty="SCAD", nfolds=10, seed=10)

# CV plot
par(mar=c(5,3.7,1.75,1), mgp=c(2.2,1,0), mfrow=c(1,2))
plot(lasso_cv_test, sub="Lasso") 
plot(scad_cv_test, sub="SCAD", log.l=TRUE, las=3)

# minimum CV error
lasso_cv_test$lambda.min  
scad_cv_test$lambda.min  

# one sd rule
(lambdaL_test <- lasso_cv_test$lambda.1se)  # min lambda = 0.001538865 for lasso

id <- which.min(scad_cv_test$cve)
upper <- scad_cv_test$cve + scad_cv_test$cvse
id <- which(scad_cv_test$cve < upper[id])[1]
(lambdaS_test <- scad_cv_test$lambda[id])   # min lambda = 91.21312 for SCAD

# get coef of model at selected lambda
CL <- as.matrix(coef(lasso_test, s=lambdaL_test)) # coefficient for lasso
CL[CL!=0,]

CS <- coef(scad_cv_test, lambda=lambdaS_test) # coefficient for scad
CS[CS!=0]  

# compute predictions at selected lambda
lasso_pred_test <- predict(lasso_test, s=lambdaL_test, newx=Xtest1_out) 
scad_pred_test <- predict(scad_test, lambda=lambdaS_test, X=Xtest1_out) 

# compute RSS
(lasso_RSS <- mean((ytest_out-lasso_pred_test)^2))
(scad_RSS <- mean((ytest_out-scad_pred_test)^2))

# compute RMSLE
(lasso_RMSE <- sqrt(mean((ytest_out-lasso_pred_test)^2)))
(scad_RMSE <- sqrt(mean((ytest_out-scad_pred_test)^2)))

# Converting back to normal scale, then get RMSE
sqrt(mean((exp(ytest_out) - exp(lasso_pred_test))^2))
sqrt(mean((exp(ytest_out) - exp(scad_pred_test))^2))

# Median absolute error
median(abs(exp(ytest_out) - exp(lasso_pred_test)))
median(abs(exp(ytest_out) - exp(scad_pred_test)))

# Mean absolute error
mean(abs(exp(ytest_out) - exp(lasso_pred_test)))
mean(abs(exp(ytest_out) - exp(scad_pred_test)))

######################################################################
########################## END #######################################
######################################################################



###################################################################
#################### Support Vector Machine #######################
###################################################################


############################################
############ For GoalKeeper ################
############################################
library(e1071)
set.seed(1)  # c=1, ep=0.5, gamma=0.0001
svmfit = svm(Wage~., data = goalkeeper_train, 
             kernel="radial", cost = 1, epsilon=0.5, scale=FALSE)
newfit = predict(svmfit, goalkeeper_test)

# compute RMSLE
RMSLE =  sqrt(mean((newfit-goalkeeper_test$Wage)^2))
RMSLE
# Converting back to normal scale, then get RMSE
sqrt(mean((exp(newfit) - exp(goalkeeper_test$Wage))^2))
# Median absolute error
median(abs(exp(newfit) - exp(goalkeeper_test$Wage)))
# Mean absolute error
mean(abs(exp(newfit) - exp(goalkeeper_test$Wage)))

###################################################################
set.seed(1)  # c=20, ep=0.5, gamma=0.0001
svmfit = svm(Wage~., data = goalkeeper_train, 
             kernel="radial", cost = 20, gamma=0.0001, scale=FALSE)
newfit = predict(svmfit, goalkeeper_test)

# compute RMSLE
RMSLE =  sqrt(mean((newfit-goalkeeper_test$Wage)^2))
RMSLE
# Converting back to normal scale, then get RMSE
sqrt(mean((exp(newfit) - exp(goalkeeper_test$Wage))^2))
# Median absolute error
median(abs(exp(newfit) - exp(goalkeeper_test$Wage)))
# Mean absolute error
mean(abs(exp(newfit) - exp(goalkeeper_test$Wage)))

############################################
############# For outfield #################
############################################
set.seed(1)  # c=100, epsilon=0.5
svmfit = svm(Wage~., data = outfield_train, 
             kernel="radial", cost = 100, epsilon = 0.5, scale=FALSE)
newfit = predict(svmfit, outfield_test)

# compute RMSLE
RMSLE =  sqrt(mean((newfit-outfield_test$Wage)^2))
RMSLE
# Converting back to normal scale, then get RMSE
sqrt(mean((exp(newfit) - exp(outfield_test$Wage))^2))
# Median absolute error
median(abs(exp(newfit) - exp(outfield_test$Wage)))
# Mean absolute error
mean(abs(exp(newfit) - exp(outfield_test$Wage)))

##################################################################
plot(svmfit)
summary(svmfit)
svmfit$index

tune.out = tune(svm, Wage~., data=outfield_train, kernel="linear",
                ranges = list(cost=c(0.01,0.1,1,10,100)))
summary(tune.out)
bestmod = tune.out$best.model # Get the best model
summary(bestmod)

# Predict for test data

################
#### Linear ####
################
# Goalkeeper #

library(e1071)
set.seed(1)  
svmfit = svm(Wage~., data = goalkeeper_train, 
             kernel="linear", cost = 100, scale=FALSE)
newfit = predict(svmfit, goalkeeper_test)

# compute RMSLE
RMSLE =  sqrt(mean((newfit-goalkeeper_test$Wage)^2))
RMSLE
# Converting back to normal scale, then get RMSE
sqrt(mean((exp(newfit) - exp(goalkeeper_test$Wage))^2))
# Median absolute error
median(abs(exp(newfit) - exp(goalkeeper_test$Wage)))
# Mean absolute error
mean(abs(exp(newfit) - exp(goalkeeper_test$Wage)))

###################################################################
# Outfield #

set.seed(1)  
svmfit = svm(Wage~., data = outfield_train, 
             kernel="linear", cost = 100, scale=FALSE)
newfit = predict(svmfit, outfield_test)

# compute RMSLE
RMSLE =  sqrt(mean((newfit-outfield_test$Wage)^2))
RMSLE
# Converting back to normal scale, then get RMSE
sqrt(mean((exp(newfit) - exp(outfield_test$Wage))^2))
# Median absolute error
median(abs(exp(newfit) - exp(outfield_test$Wage)))
# Mean absolute error
mean(abs(exp(newfit) - exp(outfield_test$Wage)))


###################################################################
#################### Support Vector Machine #######################
###################################################################
set.seed(1)
library(caret)
library(e1071)    # Choose tuning parameter via 10-fold cross-validation
tune.out = tune(svm, Wage~., data=goalkeeper_train, kernel="linear",
                ranges=list(cost=c(0.001,0.01,1,10)))
summary(tune.out)
bestmod=tune.out$best.model
summary(bestmod)

svmfit = svm(Wage~., data=goalkeeper_train, kernel="linear", cost=0.01)
plot(svmfit, data=goalkeeper_train, formula=Wage~International.Reputation, fill=TRUE)

ypred = predict(bestmod, newdata = goalkeeper_test)
RMSLE =  sqrt(mean((ypred-goalkeeper_test$Wage)^2))
RMSLE

###################################################################
################### Random Forest with Bagging ####################
###################################################################

library(randomForest)
set.seed(1)

               # Bagging helps to reduce the variance
Bval = seq(100, 1000, 100)  # This is for bagging, to genereate B bootstrap training sets
mval = c(10,11,12,13) # This is the number of predictors used for random forest
T = length(Bval)

RMSLE_rf = matrix(0, length(mval), T)
MAE_rf = matrix(0, length(mval), T)
for (i in 1:length(mval)){
  m = mval[i]
  for (t in 1:T){
    B = Bval[t]
    Bag.rf = randomForest(Wage~., data=goalkeeper_train, mtry=m
                          , ntree=B, importance=TRUE) 
    yhat.rf = predict(Bag.rf, newdata = goalkeeper_test)
    
    # Store errors
    RMSLE_rf[i,t] = sqrt(mean((yhat.rf-goalkeeper_test$Wage)^2))
    MAE_rf[i,t] = median(abs(exp(yhat.rf) - exp(goalkeeper_test$Wage)))
  }
}
plot(Bval, RMSLE_rf[1,], type="l", ylim=c(0.80, 0.85),
     xlab="B", ylab="test error", main="Random forests")
points(Bval, RMSLE_rf[2,], type="l", col="red", lty=2)
points(Bval, RMSLE_rf[3,], type="l", col="blue", lty=2)
points(Bval, RMSLE_rf[4,], type="l", col="green", lty=2)
legend("topright", legend=c("m=10", "m=11", "m=12", "m=13"),
       col=c("seagreen", "black", "red", "blue"), lty=c(1,1,2,2))

# Best model with the lowest error from above
set.seed(1)
which(RMSLE_rf == min(RMSLE_rf),arr.in=TRUE)
rf.Wage_best = randomForest(Wage~., data=goalkeeper_train, mtry=12
                       , ntree=100, importance=TRUE) 
yhat.rf = predict(rf.Wage_best, newdata = goalkeeper_test)
rmsle_best = sqrt(mean((yhat.rf-goalkeeper_test$Wage)^2))
rmsle_best


######################################################################
########################## END #######################################
######################################################################




