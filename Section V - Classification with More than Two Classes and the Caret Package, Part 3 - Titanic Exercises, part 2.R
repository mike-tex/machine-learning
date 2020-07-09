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

getModelInfo("knn")
modelLookup("knn")

control <- trainControl(method = "cv", number = 10, p = .9)

set.seed(8, sample.kind = "Rounding")
train_knn <-
  train(
    Survived ~ .,
    method = "knn",
    data = train_set,
    tuneGrid = data.frame(k = seq(3, 51, 2)),
    trControl = control
  )

# Try tuning with k = seq(3, 51, 2). What is the optimal value 
# of k using cross-validation?

train_knn$bestTune 
#   k
# 3 5

# Explanation from the course web site:
# The optimal value of k can be found using the following code:
  
#set.seed(8)
set.seed(8, sample.kind = "Rounding")    # simulate R 3.5
train_knn_cv <- train(Survived ~ .,
                      method = "knn",
                      data = train_set,
                      tuneGrid = data.frame(k = seq(3, 51, 2)),
                      trControl = trainControl(method = "cv", number = 10, p = 0.9))
train_knn_cv$bestTune


# What is the accuracy on the test set 
# using the cross-validated kNN model?


y_hat <- predict(train_knn, test_set)
confusionMatrix(data = y_hat, 
                reference = test_set$Survived)$overall["Accuracy"]
# Accuracy 
# 0.648  #  the correct answer is: 0.737 or 0.648

# Explanation from the course web site
# The accuracy can be calculated using the following code:
  
knn_cv_preds <- predict(train_knn_cv, test_set)
mean(knn_cv_preds == test_set$Survived)


# Question 11a: Classification tree model
# Set the seed to 10. Use caret to train a decision tree 
# with the rpart method. Tune the complexity parameter 
# with cp = seq(0, 0.05, 0.002).

set.seed(10, sample.kind = "Rounding")    # simulate R 3.5
train_rpart <- train(Survived ~ .,
                      method = "rpart",
                      data = train_set,
                      tuneGrid = data.frame(cp = seq(0, 0.05, 0.002))
                     )

# fit_rpart <- with(tissue_gene_expression, 
#           train(x, y, method = "rpart",
#                 control = rpart.control(minsplit = 0),
#                 tuneGrid = data.frame(cp = seq(0, 0.1, 0.01))))
# 


# 
# What is the optimal value of the complexity parameter (cp)?
train_rpart$bestTune
#     cp
# 9 0.016

# Explanation
# The optimal value of cp can be found using the following code:
  
#set.seed(10)
set.seed(10, sample.kind = "Rounding")    # simulate R 3.5
train_rpart <- train(Survived ~ ., 
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.05, 0.002)),
                     data = train_set)
train_rpart$bestTune


# What is the accuracy of the decision tree model 
# on the test set?
y_hat <- predict(train_rpart, test_set)
confusionMatrix(data = y_hat, 
                reference = test_set$Survived)$overall["Accuracy"]
# Accuracy 
# 0.838

# Explanation
# The accuracy can be calculated using the following code:
  
rpart_preds <- predict(train_rpart, test_set)
mean(rpart_preds == test_set$Survived)


# Question 11b: Classification tree model
# Inspect the final model and plot the decision tree.

train_rpart$finalModel

plot(train_rpart$finalModel)
text(train_rpart$finalModel)

# Which variables are used in the decision tree?
#   Select ALL that apply.
mean(test_set$Survived[test_set$Sex == "male"] == 1)
mean(test_set$Survived[test_set$Sex == "male" &
                         test_set$Age < 3.5] == 1)
mean(test_set$Survived[test_set$Sex == "female"] == 1)


# A 28-year-old male
mean(test_set$Survived[test_set$Sex == "male" & 
       test_set$Age >= 3.5] == 1)
# [1] 0.145

# A female in the second passenger class
mean(test_set$Survived[test_set$Sex == "female" & 
                         test_set$Pclass <= 2.5] == 1)
# [1] 0.974

# A third-class female who paid a fare of $8
mean(test_set$Survived[test_set$Sex == "female" & 
                         test_set$Pclass >= 2.5 &
                         test_set$Fare <= 23.35] == 1)
# [1] 0.565
# per tree, yes

# A 5-year-old male with 4 siblings
mean(test_set$Survived[test_set$Sex == "male" & 
                         test_set$SibSp == 4 &
                         test_set$Age >= 3.5] == 1)
# [1] 0

# A third-class female who paid a fare of $25
mean(test_set$Survived[test_set$Sex == "female" & 
                         test_set$Fare >= 23.35] == 1)
# [1] 0.886
# per tree, would not survive

# A first-class 17-year-old female with 2 siblings
mean(test_set$Survived[test_set$Sex == "female" & 
                         test_set$Pclass <= 2.5 &
                         test_set$SibSp == 2 &
                         test_set$Age >= 3.5] == 1)

# A first-class 17-year-old male with 2 siblings
mean(test_set$Survived[test_set$Sex == "male" & 
                         test_set$Age >= 3.5 & 
                         test_set$Pclass <= 2.5 &
                         test_set$SibSp == 2] == 1)
# [1] 0


# Answers from course web site:
# A 28-year-old male
# correct  would NOT survive
# A female in the second passenger class
# correct  would survive
# A third-class female who paid a fare of $8
# correct  would survive
# A 5-year-old male with 4 siblings
# correct  would NOT survive
# A third-class female who paid a fare of $25
# correct  would NOT survive
# A first-class 17-year-old female with 2 siblings
# correct  would survive
# A first-class 17-year-old male with 2 siblings
# correct  would NOT survive
# Explanation
# For each case, follow the decision tree to determine 
# whether it results in survived=0 (didn't survive) 
# or survived=1 (did survive).


# Question 12: Random forest model
# Set the seed to 14. Use the caret train() function 
# with the rf method to train a random forest. 
# Test values of mtry ranging from 1 to 7. Set ntree to 100.
# 
set.seed(14, sample.kind = "Rounding")    # simulate R 3.5
train_rf <- train(
  Survived ~ .,
  method = "rf",
  data = train_set,
  tuneGrid =
    data.frame(mtry = seq(1, 7, 1)),
  ntree = 100
)


# What mtry value maximizes accuracy?
train_rf$bestTune
#     mtry
# 2    2

# What is the accuracy of the random forest model 
# on the test set?
y_hat <- predict(train_rf, test_set)
confusionMatrix(data = y_hat, 
                reference = test_set$Survived)$overall["Accuracy"]
# Accuracy 
# 0.844 

# Use varImp() on the random forest model object 
# to determine the importance of various predictors 
# to the random forest model.
# 
# What is the most important variable?
varImp(train_rf)$importance

# Answers / explanation from the course web site:
# What mtry value maximizes accuracy?
#   
#   2
# correct  3 or 2
# 2 
# Explanation
# 
# The mtry value can be calculated using the following code:
#   
#   #set.seed(14)
#   set.seed(14, sample.kind = "Rounding")    # simulate R 3.5
# train_rf <- train(Survived ~ .,
#                   data = train_set,
#                   method = "rf",
#                   ntree = 100,
#                   tuneGrid = data.frame(mtry = seq(1:7)))
# train_rf$bestTune
# What is the accuracy of the random forest model on the test set?
#   
#   0.844
# correct  0.877 or 0.844
# 0.844 
# Explanation
# 
# The accuracy can be calculated using the following code:
#   
#   rf_preds <- predict(train_rf, test_set)
# mean(rf_preds == test_set$Survived)
# Use varImp() on the random forest model object to determine the importance of various predictors to the random forest model.
# 
# What is the most important variable?
#   Be sure to report the variable name exactly as it appears in the code.
# 
# 
# Sexmale
# correct  Sexmale
# Explanation
# 
# The most important variable can be found using the following code:
#   
#   varImp(train_rf)    # first row






























