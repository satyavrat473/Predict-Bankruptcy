rm(list = ls(all=TRUE))

knnimputation = function(data){
  imputedData <- knnImputation(data = data, k = ceiling(sqrt(nrow(data))))
  sum(is.na(imputedData))
  return(imputedData)
}

#ceiling(sqrt(nrow(train_data)))

CalScores_class = function(model,data){
  Prediction <- data$target
  Reference = predict(model,newdata=data, type="class")
  
  conf_matrix <- table(Prediction,Reference)
  print(conf_matrix)
  recall = conf_matrix[1, 1]/sum(conf_matrix[,1])
  precision = conf_matrix[1, 1]/sum(conf_matrix[1,])
  specificity = conf_matrix[2, 2]/sum(conf_matrix[,2])
  accuracy = sum(diag(conf_matrix))/sum(conf_matrix)
  F1 = (2 * precision * recall) / (precision + recall)
  
  cat('\nRecall      : ',+recall)
  cat('\nPrecision   : ',+precision)
  cat('\nSpecificity : ',+specificity)
  cat('\nAccuracy    : ',+accuracy)
  cat('\nF1 Score    : ',+F1)
}
CalScores_response = function(model,data){
  Prediction <- data$target
  Reference = predict(model,newdata=data, type="response")
  
  conf_matrix <- t(table(Prediction,Reference))
  print(conf_matrix)
  recall = conf_matrix[1, 1]/sum(conf_matrix[,1])
  precision = conf_matrix[1, 1]/sum(conf_matrix[1,])
  specificity = conf_matrix[2, 2]/sum(conf_matrix[,2])
  accuracy = sum(diag(conf_matrix))/sum(conf_matrix)
  F1 = (2 * precision * recall) / (precision + recall)
  
  cat('\nRecall      : ',+recall)
  cat('\nPrecision   : ',+precision)
  cat('\nSpecificity : ',+specificity)
  cat('\nAccuracy    : ',+accuracy)
  cat('\nF1 Score    : ',+F1)
}


setwd("D:\\D\\INSOFE\\Week12 (09-10.02.2019) - CUTe3\\09.02.19(CUTe3)\\SatyaCUTe3")
getwd()
bank_data = read.csv("train.csv",header = T, na.strings = c("?","#",""," ","NA"))
head(bank_data)

sum(is.na(bank_data))
prop.table(table(bank_data$target))

colSums(is.na(bank_data))

str(bank_data)

install.packages("caret") #for createDataPartition 
install.packages("DMwR") #for smote
install.packages("C50")
install.packages("e1071")
install.packages("rpart")
install.packages("rpart.plot")

library(caret)
library(DMwR)
library(C50)
library(e1071)
library(rpart)
library(rpart.plot)
library(randomForest)
library(glmnet)

library(missForest)
install.packages("doParallel")
library(doParallel)
registerDoParallel(cores=8)

bank_data$target = as.factor(bank_data$target)
rownames(bank_data) <- bank_data$ID
#Drop the index column
bank_data$ID = NULL

set.seed(4470)
train_rows = createDataPartition(bank_data$target, p = 0.7, list = F)
train_data = bank_data[train_rows,] #training data
test_data = bank_data[-train_rows,] #test data
dim(bank_data)
dim(train_data)
dim(test_data)
table(train_data$target)
table(test_data$target)


#train_data = missForest(train_data,parallelize = 'forests')

set.seed(2232)
trainsmote = SMOTE(target~.,data=train_data,perc.over = 700,perc.under = 100 )
table(trainsmote$target)
prop.table(table(trainsmote$target))
sum(is.na(trainsmote))
trainsmote = knnimputation(trainsmote)

set.seed(2232)
testsmote = SMOTE(target~.,data=test_data,perc.over = 700,perc.under = 100 )
testsmote = knnimputation(testsmote)

##########C50 Model Code #################
#Build model on smoteddata trainsmote

DT_C50 <- C5.0(target~.,data=trainsmote)
plot(DT_C50)

#Check variable importance
C5imp(DT_C50, pct=TRUE)

##predict on train and validation
#pred_Train = predict(DT_C50,newdata=trainsmote, type="class")
#pred_Test = predict(DT_C50, newdata=testsmote, type="class")
#Error Metrics on train and test
#confusionMatrix(trainsmote$target,pred_Train)
#confusionMatrix(testsmote$target,pred_Test)
CalScores_class(DT_C50,trainsmote)
#Recall      :  0.9646948
#Precision   :  0.9666487
#Specificity :  0.9707658
#Accuracy    :  0.9679269
#F1 Score    :  0.9656707
CalScores_class(DT_C50,testsmote)
#Recall      :  0.6965552
#Precision   :  0.8672343
#Specificity :  0.852125
#Accuracy    :  0.7617366
#F1 Score    :  0.7725804


#Build classification model using RPART
DT_rpart <- train(
  target ~., data = trainsmote, method = "rpart",
  trControl = trainControl("cv", number = 10), #trControl, to set up 10-fold cross validation
  tuneLength = 10, #tuneLength, to specify the number of possible cp values to evaluate. Default value is 3, here we'll use 10.
  na.action = na.pass 
)

plot(DT_rpart) #cp 0.00405077
DT_rpart$bestTune
CalScores_class(DT_rpart$finalModel,trainsmote)
#Recall      :  0.8279957
#Precision   :  0.8378342
#Specificity :  0.8566141
#Accuracy    :  0.8431002
#F1 Score    :  0.8328859
CalScores_class(DT_rpart$finalModel,testsmote)
#Recall      :  0.6369427
#Precision   :  0.8199306
#Specificity :  0.7895319
#Accuracy    :  0.6978661
#F1 Score    :  0.7169447

DT_rpart_Reg <- rpart(target~., 
                      data=trainsmote, 
                      method="class", 
                      control = rpart.control(cp = 0.001)#0.001
)

printcp(DT_rpart_Reg)
plotcp(DT_rpart_Reg)
CP <- DT_rpart_Reg$cptable[which.min(DT_rpart_Reg$cptable[,"xerror"]), "CP" ]
CP
DT_rpart_Reg <- rpart(target~., 
                      data=trainsmote,method="class", 
                      control = rpart.control(cp = 0.001012692)
)
CalScores_class(DT_rpart_Reg,trainsmote)
#Recall      :  0.8943347
#Precision   :  0.9165541
#Specificity :  0.9253623
#Accuracy    :  0.910523
#F1 Score    :  0.9053081
CalScores_class(DT_rpart_Reg,testsmote)
#Recall      :  0.6555609
#Precision   :  0.8625039
#Specificity :  0.8337781
#Accuracy    :  0.7243561
#F1 Score    :  0.7449271

###Random Forest
###Pre Processing
prop.table(table(trainsmote$target))
prop.table(table(testsmote$target))

set.seed(123)
DT_RF = randomForest(target ~ ., 
                     data=trainsmote, 
                     keep.forest=TRUE, 
                     ntree=200,
                     set.seed(123)
) 
CalScores_response(DT_RF,testsmote)
#Recall      :  0.9397666
#Precision   :  0.702996
#Specificity :  0.6525938
#Accuracy    :  0.7866078
#F1 Score    :  0.8043185

DT_RF$importance
rf_Imp_Attr = data.frame(DT_RF$importance)
rf_Imp_Attr = data.frame(Attributes = row.names(rf_Imp_Attr), Importance = rf_Imp_Attr[,1])
rf_Imp_Attr = rf_Imp_Attr[order(rf_Imp_Attr$Importance, decreasing = TRUE),]
rf_Imp_Attr
varImpPlot(DT_RF)

top_Imp_Attr = as.character(rf_Imp_Attr$Attributes[1:35])
DT_RF_Imp = randomForest(target~.,
                         data=trainsmote[,c(top_Imp_Attr,"target")], 
                         keep.forest=TRUE,
                         ntree=100, set.seed(015)
) 
CalScores_response(DT_RF_Imp,testsmote)


# Fine tuning parameters of Random Forest model
# Using For loop to identify the right mtry for model
a=c()
i=11
for (i in 10:20) {
  model3 <- randomForest(target ~ ., data = trainsmote, ntree = 500, mtry = i, importance = TRUE)
  predValid <- predict(model3, testsmote, type = "class")
  a[i-2] = mean(predValid == testsmote$target)
}
a
plot(10:20,a)
RF_tune  <- randomForest(target~., data = trainsmote, ntree = 500, mtry = 19, importance = TRUE, set.seed(2345))
CalScores_response(RF_tune,testsmote)
#Recall      :  0.9397666
#Precision   :  0.7203287
#Specificity :  0.6807395
#Accuracy    :  0.8016188
#F1 Score    :  0.8155446
RF_tune

RF_tune$importance
rf_Imp_Attr = data.frame(RF_tune$importance)
rf_Imp_Attr = data.frame(Attributes = row.names(rf_Imp_Attr), Importance = rf_Imp_Attr[,1])
rf_Imp_Attr = rf_Imp_Attr[order(rf_Imp_Attr$Importance, decreasing = TRUE),]
rf_Imp_Attr
varImpPlot(RF_tune)

top_Imp_Attr = as.character(rf_Imp_Attr$Attributes[1:40])
DT_RF_Imp = randomForest(target~.,
                         data=trainsmote[,c(top_Imp_Attr,"target")], 
                         keep.forest=TRUE,
                         ntree=500,mtry=19, set.seed(1115)
) 
CalScores_response(DT_RF_Imp,testsmote)
#Recall      :  0.9394513
#Precision   :  0.7414136
#Specificity :  0.7133002
#Accuracy    :  0.8188374
#F1 Score    :  0.8287662


###################################################
bank_testdata = read.csv("test.csv",header = T)
rownames(bank_testdata) <- bank_testdata$ID
#Drop the index column
bank_testdata$ID = NULL
sum(is.na(bank_testdata))
bank_testdata = knnimputation(bank_testdata)

ApproxPredict = predict(RF_tune,newdata=bank_testdata, type="class") #0.43
write.csv(x = ApproxPredict, file = "RF_tune_submission1.csv",row.names = T)

test_data_fet = Feature_Zscore(bank_testdata)
ApproxPredict = predict(RF_tune_featureng,newdata=test_data_fet, type="class") #0.11
write.csv(x = ApproxPredict, file = "RF_tune_featureng_submission2.csv",row.names = T)


ApproxPredict = predict(DT_RF_Imp,newdata=bank_testdata, type="class") #0.30
write.csv(x = ApproxPredict, file = "DT_RF_Imp_submission3.csv",row.names = T)

ApproxPredict = predict(DT_RF_Imp,newdata=bank_testdata, type="class") #0.44
write.csv(x = ApproxPredict, file = "DT_RF_Imp_submission4.csv",row.names = T)
tail(ApproxPredict)

######################################################

##Feature Engineering
Feature_Zscore = function(data){
  train_featureEngg = data
  train_featureEngg$Z_Score = NA
  
  train_featureEngg$Z_Score = 1.2*train_featureEngg$Attr3 + 
    1.4*train_featureEngg$Attr6 + 3.3*train_featureEngg$Attr7 + 
    0.6*train_featureEngg$Attr8 + 1.0*train_featureEngg$Attr9
  
  #sum(is.na(train_featureEngg))
  #summary(train_featureEngg)
  std_obj = preProcess(train_featureEngg[, !colnames(train_featureEngg) %in% ("target")],  
                       method = c("center", "scale"))
  std_data = predict(std_obj, train_featureEngg)
  #subset(std_data[,c("Attr3","Attr6","Attr7","Attr8","Z_Score")], rownames(std_data)==10049)
  
  std_data$Z_Score <- ifelse( std_data$Z_Score>2.6, "safe", 
                              ifelse(std_data$Z_Score<=2.6 & 
                                       std_data$Z_Score>=1.1, "grey_area",
                                     "distress_zone"))
  std_data$Z_Score <- as.factor(std_data$Z_Score)
  return(std_data)
  #table(std_data$Z_Score)
}
subset(std_data[,c("Attr3","Attr6","Attr7","Attr8","target","Z_Score")], std_data$target ==0 & std_data$Z_Score == "headed_for_bankruptcy")

train_data_fet = Feature_Zscore(trainsmote)
test_data_fet = Feature_Zscore(testsmote)

RF_tune_featureng  <- randomForest(target~., data = train_data_fet, ntree = 500, mtry = 19, importance = TRUE, set.seed(2345))
CalScores_response(RF_tune_featureng,test_data_fet)
#Recall      :  0.8189845
#Precision   :  0.5811143
#Specificity :  0.4834437
#Accuracy    :  0.6400294
#F1 Score    :  0.6798429
###################
SVM_basic <- svm(target ~ . , trainsmote, 
                 kernel = "linear", 
                 probability = TRUE
)
CalScores_response(SVM_linear,testsmote)


