# Titanic Exercises, part 2

library(titanic)    # loads titanic_train data frame
library(caret)
library(tidyverse)
library(rpart)

# 3 significant digits
options(digits = 3)

# clean the data - `titanic_train` is loaded 
# with the titanic package
titanic_clean <- titanic_train %>%
  mutate(Survived = factor(Survived),
         Embarked = factor(Embarked),
         Age = ifelse(is.na(Age), 
                      median(Age, na.rm = TRUE), Age), # NA age to median age
         FamilySize = SibSp + Parch + 1) %>%    # count family members
  select(Survived,  Sex, Pclass, Age, Fare, 
         SibSp, Parch, FamilySize, Embarked)

set.seed(42, sample.kind = "Rounding")

idx <- createDataPartition(titanic_clean$Survived,
                           times = 1, 
                           p = .2, 
                           list = F)
test_set <- titanic_clean[idx,]
train_set <- titanic_clean[-idx,]

# Question 7: Survival by fare - LDA and QDA
# Set the seed to 1. Train a model using 
# linear discriminant analysis (LDA) with the caret lda method 
# using fare as the only predictor.

set.seed(1, sample.kind = "Rounding")
train_lda <- 
  train(Survived ~ Fare, method = "lda", data = train_set)
y_hat <- predict(train_lda, test_set)
confusionMatrix(data = y_hat, 
                reference = test_set$Survived)$overall["Accuracy"]


# What is the accuracy on the test set for the LDA model?
# Accuracy 
# 0.693   
# 
# Set the seed to 1. Train a model 
# using quadratic discriminant analysis (QDA) 
# with the caret qda method using fare as the only predictor.
# 
set.seed(1, sample.kind = "Rounding")
train_qda <- 
  train(Survived ~ Fare, method = "qda", data = train_set)
y_hat <- predict(train_qda, test_set)
confusionMatrix(data = y_hat, 
                reference = test_set$Survived)$overall["Accuracy"]

# What is the accuracy on the test set for the QDA model?
# Accuracy 
# 0.693  


# Question 8: Logistic regression models
# Set the seed to 1. Train a logistic regression model 
# using caret train() with the glm method using age as 
# the only predictor.

set.seed(1, sample.kind = "Rounding")
train_glm <- 
  train(Survived ~ Age, method = "glm", data = train_set)
y_hat <- predict(train_glm, test_set)
confusionMatrix(data = y_hat, 
                reference = test_set$Survived)$overall["Accuracy"]

# What is the accuracy on the test set using age 
# as the only predictor?
# Accuracy 
# 0.615

# Set the seed to 1. Train a logistic regression model 
# using caret train() with the glm method 
# using four predictors: sex, class, fare, and age.

set.seed(1, sample.kind = "Rounding")
train_glm <- 
  train(Survived ~ Sex + Pclass + Fare + Age, 
        method = "glm", data = train_set)
y_hat <- predict(train_glm, test_set)
confusionMatrix(data = y_hat, 
                reference = test_set$Survived)$overall["Accuracy"]

# What is the accuracy on the test set 
# using these four predictors?
# Accuracy 
# 0.849  


# Set the seed to 1. Train a logistic regression model 
# using caret train() with the glm method using all predictors. 
# Ignore warnings about rank-deficient fit.

set.seed(1, sample.kind = "Rounding")
train_glm <- 
  train(Survived ~ ., method = "glm", data = train_set)
y_hat <- predict(train_glm, test_set)
confusionMatrix(data = y_hat, 
                reference = test_set$Survived)$overall["Accuracy"]

# What is the accuracy on the test set using all predictors?
# Accuracy 
#  0.849   


# Question 9a: kNN model
# Set the seed to 6. Train a kNN model 
# on the training set using the caret train function. 
# Try tuning with k = seq(3, 51, 2).

set.seed(6, sample.kind = "Rounding")
train_knn <- 
  train(Survived ~ ., 
        method = "knn", 
        data = train_set, 
        tuneGrid = data.frame(k = seq(3, 51, 2)))
train_knn
train_knn$bestTune # <<< answer
train_knn$finalModel

y_hat <- predict(train_knn, test_set)
confusionMatrix(data = y_hat, 
                reference = test_set$Survived)$overall["Accuracy"]


# What is the optimal value of the number of neighbors k?
# > train_knn$bestTune
#   k
# 5 11

# Question 9b: kNN model
# Plot the kNN model to investigate the relationship 
# between the number of neighbors and accuracy 
# on the training set.
# 

train_knn$results %>%
  ggplot(aes(x = k, y = Accuracy)) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(
    x = k,
    ymin = Accuracy - AccuracySD,
    ymax = Accuracy + AccuracySD
  ))


max(train_knn$results$Accuracy)

plot(train_knn$results$k, train_knn$results$Accuracy)

# Of these values of  k , which yields the highest accuracy?

# Question 9c: kNN model
# What is the accuracy of the kNN model on the test set?

set.seed(6, sample.kind = "Rounding")
train_knn <- 
  train(Survived ~ ., 
        method = "knn", 
        data = train_set, 
        tuneGrid = data.frame(k = seq(3, 51, 2)))
y_hat <- predict(train_glm, test_set)
confusionMatrix(data = y_hat, 
                reference = test_set$Survived)$overall["Accuracy"]

# Accuracy 
# 0.709


# Question 10: Cross-validation
# Set the seed to 8 and train a new kNN model. 
# Instead of the default training control, 
# use 10-fold cross-validation where each partition 
# consists of 10% of the total.

set.seed(8, sample.kind = "Rounding")
train_knn <- 
  train(Survived ~ ., 
        method = "knn", 
        data = train_set, 
        tuneGrid = data.frame(k = seq(3, 51, 2)))
y_hat <- predict(train_glm, test_set)
confusionMatrix(data = y_hat, 
                reference = test_set$Survived)$overall["Accuracy"]


# Try tuning with k = seq(3, 51, 2). What is the optimal value 
# of k using cross-validation?

train_knn$bestTune 












































































