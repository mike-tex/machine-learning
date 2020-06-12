## Section 2: Machine Learning Basics
##   2.1: Basics of Evaluating Machine Learning Algorithms
##   Caret package, training and test sets, and overall accuracy

rm(list = ls())

library(tidyverse)
library(caret)
library(dslabs)
data(heights)

# define the outcome and predictors
y <- heights$sex
x <- heights$height

# generate training and test sets
set.seed(2007, sample.kind = "Rounding")
test_index <- 
  createDataPartition(y, times = 1, p = 0.5, list = FALSE)
test_set <- heights[test_index, ]
train_set <- heights[-test_index, ]

# guess the outcome
y_hat <- sample(c("Male", "Female"), 
                length(test_index), 
                replace = TRUE)
y_hat <- sample(c("Male", "Female"), 
                length(test_index), 
                replace = TRUE) %>% 
  factor(levels = levels(test_set$sex))

# compute accuracy
mean(y_hat == test_set$sex)
heights %>% group_by(sex) %>% 
  summarize(mean(height), sd(height))
y_hat <- ifelse(x > 62, "Male", "Female") %>% 
  factor(levels = levels(test_set$sex))
mean(y == y_hat)

# examine the accuracy of 10 cutoffs
cutoff <- seq(61, 70)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(train_set$height > x, "Male", "Female") %>% 
    factor(levels = levels(test_set$sex))
  mean(y_hat == train_set$sex)
})
max(accuracy)
best_cutoff <- cutoff[which.max(accuracy)]
best_cutoff
y_hat <- ifelse(test_set$height > best_cutoff, "Male", "Female") %>% 
  factor(levels = levels(test_set$sex))
y_hat <- factor(y_hat)
mean(y_hat == test_set$sex)


## Comprehension Check:
## Basics of Evaluating Machine Learning Algorithms

rm(list = ls())

# Q2
# How many features are available to us for prediction 
# in the mnist digits dataset?
# You can download the mnist dataset using 
# the read_mnist() function from the dslabs package.

mnist <- read_mnist()
784

## Confusion matrix

# tabulate each combination of prediction and actual value
table(predicted = y_hat, actual = test_set$sex)
test_set %>% 
  mutate(y_hat = y_hat) %>%
  group_by(sex) %>% 
  summarize(accuracy = mean(y_hat == sex))
prev <- mean(y == "Male")

confusionMatrix(data = y_hat, reference = test_set$sex)


# Balanced accuracy and F1 score

# maximize F-score
cutoff <- seq(61, 70)
F_1 <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(train_set$height > x, "Male", "Female") %>% 
    factor(levels = levels(test_set$sex))
  F_meas(data = y_hat, reference = factor(train_set$sex))
})
max(F_1)

best_cutoff <- cutoff[which.max(F_1)]
y_hat <- ifelse(test_set$height > best_cutoff, "Male", "Female") %>% 
  factor(levels = levels(test_set$sex))
sensitivity(data = y_hat, reference = test_set$sex)
specificity(data = y_hat, reference = test_set$sex)


## Prevalence matters in practice

## ROC and precision-recall curves

p <- 0.9
n <- length(test_index)
y_hat <- sample(c("Male", "Female"), n, replace = TRUE, prob=c(p, 1-p)) %>% 
  factor(levels = levels(test_set$sex))
mean(y_hat == test_set$sex)

# ROC curve
probs <- seq(0, 1, length.out = 10)
guessing <- map_df(probs, function(p){
  y_hat <- 
    sample(c("Male", "Female"), n, replace = TRUE, prob=c(p, 1-p)) %>% 
    factor(levels = c("Female", "Male"))
  list(method = "Guessing",
       FPR = 1 - specificity(y_hat, test_set$sex),
       TPR = sensitivity(y_hat, test_set$sex))
})
guessing %>% qplot(FPR, TPR, data =., xlab = "1 - Specificity", ylab = "Sensitivity")

cutoffs <- c(50, seq(60, 75), 80)
height_cutoff <- map_df(cutoffs, function(x){
  y_hat <- ifelse(test_set$height > x, "Male", "Female") %>% 
    factor(levels = c("Female", "Male"))
  list(method = "Height cutoff",
       FPR = 1-specificity(y_hat, test_set$sex),
       TPR = sensitivity(y_hat, test_set$sex))
})

# plot both curves together
bind_rows(guessing, height_cutoff) %>%
  ggplot(aes(FPR, TPR, color = method)) +
  geom_line() +
  geom_point() +
  xlab("1 - Specificity") +
  ylab("Sensitivity")

library(ggrepel)
map_df(cutoffs, function(x){
  y_hat <- ifelse(test_set$height > x, "Male", "Female") %>% 
    factor(levels = c("Female", "Male"))
  list(method = "Height cutoff",
       cutoff = x, 
       FPR = 1-specificity(y_hat, test_set$sex),
       TPR = sensitivity(y_hat, test_set$sex))
}) %>%
  ggplot(aes(FPR, TPR, label = cutoff)) +
  geom_line() +
  geom_point() +
  geom_text_repel(nudge_x = 0.01, nudge_y = -0.01)

# plot precision against recall
guessing <- map_df(probs, function(p){
  y_hat <- sample(c("Male", "Female"), length(test_index), 
                  replace = TRUE, prob=c(p, 1-p)) %>% 
    factor(levels = c("Female", "Male"))
  list(method = "Guess",
       recall = sensitivity(y_hat, test_set$sex),
       precision = precision(y_hat, test_set$sex))
})

height_cutoff <- map_df(cutoffs, function(x){
  y_hat <- ifelse(test_set$height > x, "Male", "Female") %>% 
    factor(levels = c("Female", "Male"))
  list(method = "Height cutoff",
       recall = sensitivity(y_hat, test_set$sex),
       precision = precision(y_hat, test_set$sex))
})

bind_rows(guessing, height_cutoff) %>%
  ggplot(aes(recall, precision, color = method)) +
  geom_line() +
  geom_point()
guessing <- map_df(probs, function(p){
  y_hat <- sample(c("Male", "Female"), length(test_index), replace = TRUE, 
                  prob=c(p, 1-p)) %>% 
    factor(levels = c("Male", "Female"))
  list(method = "Guess",
       recall = sensitivity(y_hat, relevel(test_set$sex, "Male", "Female")),
       precision = precision(y_hat, relevel(test_set$sex, "Male", "Female")))
})

height_cutoff <- map_df(cutoffs, function(x){
  y_hat <- ifelse(test_set$height > x, "Male", "Female") %>% 
    factor(levels = c("Male", "Female"))
  list(method = "Height cutoff",
       recall = sensitivity(y_hat, relevel(test_set$sex, "Male", "Female")),
       precision = precision(y_hat, relevel(test_set$sex, "Male", "Female")))
})
bind_rows(guessing, height_cutoff) %>%
  ggplot(aes(recall, precision, color = method)) +
  geom_line() +
  geom_point()



## Comprehension Check: Practice with Machine Learning, Part 1
rm(list = ls())
library(dslabs)
library(dplyr)
library(lubridate)
data(reported_heights)

dat <- mutate(reported_heights, date_time = ymd_hms(time_stamp)) %>%
  filter(date_time >= make_date(2016, 01, 25) & date_time < make_date(2016, 02, 1)) %>%
  mutate(type = ifelse(day(date_time) == 25 & hour(date_time) == 8 & between(minute(date_time), 15, 30), "inclass","online")) %>%
  select(sex, type)

y <- factor(dat$sex, c("Female", "Male"))
x <- dat$type

# Q1
# The type column of dat indicates whether students 
# took classes in person ("inclass") or online ("online"). 
# What proportion of the inclass group is female? 
# What proportion of the online group is female?
# 

# Enter your answer as a percentage or decimal 
# (eg "50%" or "0.50") to at least the hundredths place.
# 
# In class
sum(dat$type == "inclass" & dat$sex == "Female") /
  sum(dat$type == "inclass")

dat %>% filter(type == "inclass") %>% 
  summarize(mean(sex == "Female"))

# Online
dat %>% filter(type == "online") %>% 
  summarize(mean(sex == "Female"))


# Q2
# In the course videos, height cutoffs were used 
# to predict sex. Instead of using height, use 
# the type variable to predict sex. Use what you learned 
# in Q1 to make an informed guess about sex based on 
# the most prevalent sex for each type. Report the accuracy 
# of your prediction of sex. You do not need to split 
# the data into training and test sets.
# 
# Enter your accuracy as a percentage or decimal 
# (eg "50%" or "0.50") to at least the hundredths place.

library(dslabs)
library(dplyr)
library(lubridate)

data(reported_heights)

dat <- mutate(reported_heights, date_time = ymd_hms(time_stamp)) %>%
  filter(date_time >= make_date(2016, 01, 25) & date_time < make_date(2016, 02, 1)) %>%
  mutate(type = ifelse(day(date_time) == 25 & hour(date_time) == 8 & between(minute(date_time), 15, 30), "inclass","online")) %>%
  select(sex, type)

library(tidyverse)
library(dslabs)
library(caret)



cutoff <- c("online", "inclass")
dat$sex <- factor(dat$sex)
levels(dat$sex)
x <- cutoff[1]
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(dat$type == x, "Male", "Female") %>% 
    factor(levels = levels(dat$sex))
  mean(y_hat == dat$sex)
})
max(accuracy)
best_cutoff <- cutoff[which.max(accuracy)]
best_cutoff
y_hat <- 
  ifelse(dat$type == best_cutoff, "Male", "Female") %>% 
  factor(levels = levels(dat$sex))
y_hat <- factor(y_hat)
mean(y_hat == dat$sex)
# [1] 0.6333333

# Q3
# Write a line of code using the table() function 
# to show the confusion matrix between y_hat and y. 
# Use the exact format function(a, b) (note the spacing!) 
# for your answer and do not name the columns and rows.
# 
# Type the line of code below:

y <- dat$sex
table(y_hat, y)
# y_hat    Female Male
# Female     26   13
# Male       42   69


# Q4
# What is the sensitivity of this prediction? 
# You can use the sensitivity() function from 
# the caret package. Enter your answer as a percentage 
# or decimal (eg "50%" or "0.50") to at least 
# the hundredths place.

# sensitivity(data = y_hat, reference = test_set$sex)
# specificity(data = y_hat, reference = test_set$sex)

y <- dat$sex
sensitivity(data = y_hat, reference = y)


# Q5
# What is the specificity of this prediction? 

y <- dat$sex
specificity(data = y_hat, reference = y)
# 0.8414634


# Q6
# What is the prevalence (% of females) 
# in the dat dataset defined above?

mean(dat$sex == "Female")
# 0.4533333


## Comprehension Check:
## Practice with Machine Learning, Part 2

rm(list = ls())

library(caret)
data(iris)
iris <- iris[-which(iris$Species=='setosa'),]
y <- iris$Species


# Q7
# First let us create an even split of the data 
# into train and test partitions using createDataPartition() 
# from the caret package. The code with a missing line 
# is given below:

# if using R 3.6 or later, 
# use set.seed(2, sample.kind="Rounding")  
set.seed(2, sample.kind = "Rounding")    
# missing line of code
test_index <- createDataPartition(y,times=1,p=0.5,list=FALSE)
test <- iris[test_index,]
train <- iris[-test_index,]


## Q8
# Next we will figure out the singular feature 
# in the dataset that yields the greatest overall accuracy 
# when predicting species. You can use the code 
# from the introduction and from Q7 to start your analysis.

# Using only the train iris dataset, 
# for each feature, perform a simple search to find 
# the cutoff that produces the highest accuracy, 
# predicting virginica if greater than the cutoff 
# and versicolor otherwise. 
# Use the seq function over the range of each feature 
# by intervals of 0.1 for this search.
# 
# Which feature produces the highest accuracy?

levels(iris$Species)
# [1] "setosa"     "versicolor" "virginica" 

feature <- iris %>% select(!"Species") %>% names()

z = 3
feature[z]
cutoff <- seq(min(train[,feature[z]]), 
              max(train[,feature[z]]), .1)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(train[,feature[z]] > x, 
                  "virginica", "versicolor") %>% 
    factor(levels = levels(train$Species))
  mean(y_hat == train$Species)
})
feature[z]
max(accuracy)


# Q9
# For the feature selected in Q8, use the smart cutoff value 
# from the training data to calculate overall accuracy 
# in the test data. What is the overall accuracy?


best_cutoff <- cutoff[which.max(accuracy)]
best_cutoff
# 4.7

train %>% 
  summarize(mean(Petal.Length > 4.7 & Species == "virginica"))
test %>% 
  summarize(mean(Petal.Length > 4.7 & Species == "virginica"))

accuracy <- map_dbl(4.7, function(x){
  y_hat <- ifelse(test[,feature[z]] > x, 
                  "virginica", "versicolor") %>% 
    factor(levels = levels(test$Species))
  mean(y_hat == test$Species)
})
accuracy

# Q10
# Notice that we had an overall accuracy greater 
# than 96% in the training data, but the overall accuracy 
# was lower in the test data. This can happen often 
# if we overtrain. In fact, it could be the case that 
# a single feature is not the best choice. For example, 
# a combination of features might be optimal. Using 
# a single feature and optimizing the cutoff as we did 
# on our training data can lead to overfitting.
# 
# Given that we know the test data, we can treat it like 
# we did our training data to see if the same feature with 
# a different cutoff will optimize our predictions.
# 
# Which feature best optimizes our overall accuracy?

levels(iris$Species)
# [1] "setosa"     "versicolor" "virginica" 

feature <- iris %>% select(!"Species") %>% names()
z = 4
feature[z]
cutoff <- seq(min(iris[,feature[z]]), 
              max(iris[,feature[z]]), .1)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(iris[,feature[z]] > x, 
                  "virginica", "versicolor") %>% 
    factor(levels = levels(iris$Species))
  mean(y_hat == iris$Species)
})
feature[z]
max(accuracy)


# Q11
# Now we will perform some exploratory data analysis 
# on the data.

plot(iris,pch=21,bg=iris$Species)

# Notice that Petal.Length and Petal.Width in combination 
# could potentially be more information 
# than either feature alone.
# 
# Optimize the the cutoffs for Petal.Length and Petal.Width 
# separately in the train dataset by using the seq function 
# with increments of 0.1. Then, report the overall accuracy 
# when applied to the test dataset by creating a rule that 
# predicts virginica if Petal.Length is greater than 
# the length cutoff OR Petal.Width is greater than 
# the width cutoff, and versicolor otherwise.
# 
# What is the overall accuracy for the test data now?


levels(iris$Species)
# [1] "setosa"     "versicolor" "virginica" 

feature <- 
  iris %>% select("Petal.Length", "Petal.Width") %>% 
  names()
ds <- train
z = 2
feature[z]
cutoff <- seq(min(ds[,feature[z]]), 
              max(ds[,feature[z]]), .1)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(ds[,feature[z]] > x, 
                  "virginica", "versicolor") %>% 
    factor(levels = levels(ds$Species))
  mean(y_hat == ds$Species)
})
feature[z]
max(accuracy)
cutoff[which.max(accuracy)]
# [1] "Petal.Width"
# [1] 1.5
# [1] "Petal.Length"
# [1] 4.7


ds <- test
z = 1
cutoff <- seq(min(ds[,feature[z]]), 
              max(ds[,feature[z]]), .1)
feature[z]
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(ds$Petal.Length > 4.7 | 
                    ds$Petal.Width > 1.5, 
                  "virginica", "versicolor") %>% 
    factor(levels = levels(ds$Species))
  acc <- mean(y_hat == ds$Species)
  print(paste0(x, acc))
  return(acc)
})
feature[z]
max(accuracy)
which.max(accuracy)
cutoff[which.max(accuracy)]
# 0.88

mean(y_hat == test$Species)










