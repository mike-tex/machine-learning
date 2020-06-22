## Section 5: Classification with More than Two Classes
## and the Caret Package
##   5.3: Titanic Exercises
##   Titanic Exercises Part 1

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

# Question 1: Training and test sets
# Split titanic_clean into test and training sets 
# - after running the setup code, it should have 891 rows 
#   and 9 variables.

# Set the seed to 42, then use the caret package 
# to create a 20% data partition based on the Survived column. 
# Assign the 20% partition to test_set 
# and the remaining 80% partition to train_set.
set.seed(42, sample.kind = "Rounding")

idx <- createDataPartition(titanic_clean$Survived,
                           times = 1, 
                           p = .2, 
                           list = F)
test_set <- titanic_clean[idx,]
train_set <- titanic_clean[-idx,]
# 
# How many observations are in the training set?
nrow(train_set)
# [1] 712
# 
# How many observations are in the test set?
nrow(test_set)
# [1] 179 
# 
# What proportion of individuals in the training set survived?
mean(train_set$Survived == 1)  
# [1] 0.383

# explantion from the site:
# 712
# correct  712
# 712 
# Explanation
# 
# The following code will give the number of observations in the training set:
  
#set.seed(42)
set.seed(42, sample.kind = "Rounding")    # simulate R 3.5
test_index <- createDataPartition(titanic_clean$Survived, times = 1, p = 0.2, list = FALSE)    # create a 20% test set
test_set <- titanic_clean[test_index,]
train_set <- titanic_clean[-test_index,]

nrow(train_set)

# How many observations are in the test set?
#   
#   179 
# correct  179
# 179 
# Explanation
# 
# The following code will give the number of observations in the test set:
  
  nrow(test_set)

# What proportion of individuals in the training set survived?
#   
#   0.383
# correct  0.383
# 0.383 
# Explanation
# 
# The following code will give the survival proportion:
  
  mean(train_set$Survived == 1)


## Question 2: Baseline prediction by guessing the outcome
# The simplest prediction method is randomly guessing 
# the outcome without using additional predictors. 
# These methods will help us determine whether 
# our machine learning algorithm performs better than chance. 
# How accurate are two methods of guessing Titanic passenger 
# survival?
#   
# Set the seed to 3. 
# For each individual in the test set, randomly guess 
# whether that person survived or not by sampling 
# from the vector c(0,1) (Note: use the default argument setting 
# of prob from the sample function). Assume that each person 
# has an equal chance of surviving or not surviving.
# 
  
set.seed(3, sample.kind = "Rounding")
guess <- sample(c(0,1), 179, replace = T)
test_set %>% 
  mutate(accuracy = guess == Survived) %>% 
  summarize(mean(accuracy))

# What is the accuracy of this guessing method?
#   mean(accuracy)
# 1          0.475  

# 0.475 # correct Answer: 0.542 or 0.475

# Explanation from web site
# 
# The accuracy can be calculated using the following code:
  
#set.seed(3)
set.seed(3, sample.kind = "Rounding")
# guess with equal probability of survival
guess <- sample(c(0,1), nrow(test_set), replace = TRUE)
mean(guess == test_set$Survived)

# Question 3a: Predicting survival by sex
# Use the training set to determine whether members of a given sex were more likely to survive or die. Apply this insight to generate survival predictions on the test set.
# 
# What proportion of training set females survived?
mean(train_set$Survived[train_set$Sex == "female"] == 1)
# 0.731
# 
# What proportion of training set males survived?
mean(train_set$Survived[train_set$Sex == "male"] == 1)
# [1] 0.197


# Explanation from the course web site
# What proportion of training set females survived?

# correct answer:  0.733 or 0.731
#  
# Explanation
# 
# The proportion can be calculated using the following code:
#   
train_set %>%
group_by(Sex) %>%
summarize(Survived = mean(Survived == 1)) %>%
filter(Sex == "female") %>%
pull(Survived)

# What proportion of training set males survived?
#   
# correct answer:  0.193 or 0.197
#  
# Explanation
# 
# The proportion can be calculated using the following code:
#   
train_set %>%
group_by(Sex) %>%
summarize(Survived = mean(Survived == 1)) %>%
filter(Sex == "male") %>%
pull(Survived)


# Question 3b: Predicting survival by sex
# Predict survival using sex on the test set: 
# if the survival rate for a sex is over 0.5, 
# predict survival for all individuals of that sex, 
# and predict death if the survival rate for a sex 
# is under 0.5.

s_sex <- ifelse(test_set$Sex == "female", 1, 0)
mean(s_sex == test_set$Survived)

# What is the accuracy of this sex-based prediction method 
# on the test set?
# [1] 0.821

# explanation from the course on the web site
# correct answer: 0.81 or 0.821
# Explanation
# The accuracy can be calculated using the following code:
#   
sex_model <- ifelse(test_set$Sex == "female", 1, 0)    # predict Survived=1 if female, 0 if male
mean(sex_model == test_set$Survived)    # calculate accuracy


## Question 4a: Predicting survival by passenger class
# In the training set, which class(es) (Pclass) of passengers 
# were more likely to survive than die? (1, 2, and/or 3?)

train_set %>% group_by(Pclass) 
  
s1 <- as.numeric(as.character(
  train_set$Survived[train_set$Pclass == 1]))
mean(s1)

train_set %>% 
  group_by(Pclass) %>% 
  summarize(ps = mean(as.numeric(as.character(Survived))))

survival_class <- titanic_clean %>%
  group_by(Pclass) %>%
  summarize(PredictingSurvival = ifelse(mean(Survived == 1) >=0.5, 1, 0))
survival_class

# Explanation
# The survival rates by class can be found using the following code:
train_set %>%
  group_by(Pclass) %>%
  summarize(Survived = mean(Survived == 1))


# Question 4b: Predicting survival by passenger class
# Predict survival using passenger class on the test set: 
# predict survival if the survival rate for a class is over 0.5, 
# otherwise predict death.

class_model <- 
  ifelse(test_set$Pclass == 1, 1, 0)    # predict Survived=1 if pclass = 1, 0 otherwise
mean(class_model == test_set$Survived)    # calculate accuracy

# What is the accuracy of this class-based prediction method 
# on the test set?
# [1] 0.704  
  
# Explanation from the course web site
# The accuracy can be found using the following code:
  
class_model <- ifelse(test_set$Pclass == 1, 1, 0)    # predict survival only if first class
mean(class_model == test_set$Survived)    # calculate accuracy


# Question 4c: Predicting survival by passenger class
# Use the training set to group passengers 
# by both sex and passenger class.

train_set %>%
  group_by(Sex, Pclass) %>%
  summarize(Survived = mean(Survived == 1))

survival_class <- titanic_clean %>%
  group_by(Sex, Pclass) %>%
  summarize(PredictingSurvival = ifelse(mean(Survived == 1) > 0.5, 1, 0))
survival_class

# Which sex and class combinations were more likely 
# to survive than die?
#
# female 1st class
# female 2nd class


# Explanation from the course web site
# The combinations can be found using the following code:
  
train_set %>%
  group_by(Sex, Pclass) %>%
  summarize(Survived = mean(Survived == 1)) %>%
  filter(Survived > 0.5)


# Question 4d: Predicting survival by passenger class
# Predict survival using both sex and passenger class 
# on the test set. Predict survival if the survival rate 
# for a sex/class combination is over 0.5, otherwise 
# predict death.

sex_class_model <- 
  ifelse(test_set$Sex == "female" &
    (test_set$Pclass == 1 | test_set$Pclass == 2), 
    1, 0)    # predict Survived=1 if pclass = 1, 0 otherwise
mean(sex_class_model == test_set$Survived)    # calculate accuracy

# What is the accuracy of this sex- and class-based prediction 
# method on the test set?
# [1] 0.821

# Explanation
# The accuracy can be found using the following code:
  
sex_class_model <- ifelse(test_set$Sex == "female" & 
                            test_set$Pclass != 3, 1, 0)
mean(sex_class_model == test_set$Survived)


# Question 5a: Confusion matrix
# Use the confusionMatrix() function 
# to create confusion matrices for the 
# sex model, class model, and combined sex and class model. 
# You will need to convert predictions and survival status 
# to factors to use this function.
# 

sex_model_CM <- 
  confusionMatrix(factor(sex_model),
                  # predict(sex_model, test_set$Survived, type = "raw"), 
                  factor(test_set$Survived))
class_model_CM <- 
  confusionMatrix(factor(class_model),
                  # predict(sex_model, test_set$Survived, type = "raw"), 
                  factor(test_set$Survived))
sex_class_model_CM <- 
  confusionMatrix(factor(sex_class_model),
                  # predict(sex_model, test_set$Survived, type = "raw"), 
                  factor(test_set$Survived))



# What is the "positive" class used to calculate 
# confusion matrix metrics? (0 or 1?)
#   Positive' Class : 0 

# Which model has the highest sensitivity?
#   sex and class combined

# Which model has the highest specificity?
#   sex only

# Which model has the highest balanced accuracy?
#   sex only



library(broom)

# Confusion Matrix: sex model
sex_model <- train_set %>%
  group_by(Sex) %>%
  summarize(Survived_predict = ifelse(mean(Survived == 1) > 0.5, 1, 0))
test_set1 <- test_set %>%
  inner_join(sex_model, by = 'Sex')
cm1 <- confusionMatrix(data = factor(test_set1$Survived_predict), reference = factor(test_set1$Survived))
cm1 %>%
  tidy() %>%
  filter(term == 'sensitivity') %>%
  .$estimate
cm1 %>%
  tidy() %>%
  filter(term == 'specificity') %>%
  .$estimate
cm1 %>%
  tidy() %>%
  filter(term == 'balanced_accuracy') %>%
  .$estimate
# Confusion Matrix: class model
class_model <- train_set %>%
  group_by(Pclass) %>%
  summarize(Survived_predict = ifelse(mean(Survived == 1) > 0.5, 1, 0))
test_set2 <- test_set %>%
  inner_join(class_model, by = 'Pclass')
cm2 <- confusionMatrix(data = factor(test_set2$Survived_predict), reference = factor(test_set2$Survived))
cm2 %>%
  tidy() %>%
  filter(term == 'sensitivity') %>%
  .$estimate
cm2 %>%
  tidy() %>%
  filter(term == 'specificity') %>%
  .$estimate
cm2 %>%
  tidy() %>%
  filter(term == 'balanced_accuracy') %>%
  .$estimate
# Confusion Matrix: sex and class model
sex_class_model <- train_set %>%
  group_by(Sex, Pclass) %>%
  summarize(Survived_predict = ifelse(mean(Survived == 1) > 0.5, 1, 0))
test_set3 <- test_set %>%
  inner_join(sex_class_model, by=c('Sex', 'Pclass'))
cm3 <- confusionMatrix(data = factor(test_set3$Survived_predict), reference = factor(test_set3$Survived))
cm3 %>%
  tidy() %>%
  filter(term == 'sensitivity') %>%
  .$estimate
cm3 %>%
  tidy() %>%
  filter(term == 'specificity') %>%
  .$estimate
cm3 %>%
  tidy() %>%
  filter(term == 'balanced_accuracy') %>%
  .$estimate


# Question 5b: Confusion matrix
# What is the maximum value of balanced accuracy?
# 0.806

# Explanation from the course web site
# The confusion matrix for each model can be calculated 
# using the following code:
  
confusionMatrix(data = factor(sex_model), 
                reference = factor(test_set$Survived))
confusionMatrix(data = factor(class_model), 
                reference = factor(test_set$Survived))
confusionMatrix(data = factor(sex_class_model), 
                reference = factor(test_set$Survived))


# Question 6: F1 scores
# Use the F_meas() function to calculate F1 scores for the 
# sex model, class model, and combined sex and class model. 
# You will need to convert predictions to factors to use 
# this function.

F_meas(factor(sex_model), test_set$Survived)
F_meas(factor(class_model), test_set$Survived)
F_meas(factor(sex_class_model), test_set$Survived)

# Which model has the highest F1 score?
# answer ->   sex and class combined
#   
# What is the maximum value of the  F1  score?
# [1] 0.872  


# Explanation from the course web site
# The  F1  score for each model can be calculated using 
# the following code:
  
F_meas(data = factor(sex_model), reference = test_set$Survived)
F_meas(data = factor(class_model), reference = test_set$Survived)
F_meas(data = factor(sex_class_model), reference = test_set$Survived)




