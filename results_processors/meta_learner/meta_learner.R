install.packages("party")
install.packages("caret")

library(party)
library(caret)  

setwd("C:/Users/joseph.giovanelli2/Documents/AutoML/effective_preprocessing_pipeline_evaluation/results/summary/evaluation3/extension/operator_meta_learning")

# Read the experiments results
# The columns are: DatasetID, meta-features_1, ..., meta-features_n, algorithm, transformation_1, ..., transformation_n
# - dataset ID: the Id of the dataset
# - meta-features_1, ..., meta-features_n: the value of the dataset meta-features
# - algorithm: the used mining algorithm (NB, KNN, RF)
# - transformation_1, ..., transformation_n: the operator (chosen by SMBO) for each of the transformations in the prototype
union <- read.csv(paste("input/data.csv", sep = ","), header = TRUE)
# Drop the dataset IDs
union <- union[,-1]

# For each of the transformations we create an ad-hoc training-set, dropping all the columns that regard the other transformations
encode <- union[ , -which(names(union) %in% c("features", "impute", "normalize", "discretize", "rebalance"))]
features <- union[ , -which(names(union) %in% c("encode", "impute", "normalize", "discretize", "rebalance"))]
impute <- union[ , -which(names(union) %in% c("encode", "features", "normalize", "discretize", "rebalance"))]
normalize <- union[ , -which(names(union) %in% c("encode", "features", "impute", "discretize", "rebalance"))]
discretize <- union[ , -which(names(union) %in% c("encode", "features", "impute", "normalize", "rebalance"))]
rebalance <- union[ , -which(names(union) %in% c("encode", "features", "impute", "normalize", "discretize"))]
rebalance <- rebalance[ , -which(names(rebalance) %in% c("MinSkewnessOfNumericAtts", "NumberOfSymbolicFeatures"))]

# For each of the training-set we train a conditional tree, and we depict the results

ctreeFitEncode <- train(encode ~ ., 
                        data = encode, 
                        method = "ctree", 
                        na.action = na.pass, 
                        trControl = trainControl(method = "cv"),
                        controls=ctree_control(mincriterion=0.005))
plot(ctreeFitEncode$finalModel)

ctreeFitFeatures <- train(features ~ ., 
                          data = features, 
                          method = "ctree2", 
                          na.action = na.pass, 
                          trControl = trainControl(method = "cv"))
plot(ctreeFitFeatures$finalModel)


ctreeFitImpute <- train(impute ~ ., 
                        data = impute, 
                        method = "ctree2", 
                        na.action = na.pass, 
                        trControl = trainControl(method = "cv"),
                        controls=ctree_control(mincriterion=0.01))
plot(ctreeFitImpute$finalModel)


ctreeFitNormalize <- train(normalize ~ ., 
                           data = normalize, 
                           method = "ctree", 
                           na.action = na.pass, 
                           trControl = trainControl(method = "LOOCV"))
plot(ctreeFitNormalize$finalModel)


ctreeFitDiscretize <- train(discretize ~ ., 
                            data = discretize, 
                            method = "ctree", 
                            na.action = na.pass, 
                            trControl = trainControl(method = "cv"))
plot(ctreeFitDiscretize$finalModel)


ctreeFitRebalance <- train(rebalance ~ ., 
                           data = rebalance, 
                           method = "ctree2", 
                           na.action = na.pass, 
                           trControl = trainControl(method = "cv"))
plot(ctreeFitRebalance$finalModel)
