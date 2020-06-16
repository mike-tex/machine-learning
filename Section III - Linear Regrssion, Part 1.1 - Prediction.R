## Section 3:
## Linear Regression for Prediction, Smoothing, 
## and Working with Matrices
##   3.1: Linear Regression for Prediction
##   Linear Regression for Prediction

library(tidyverse)
library(caret)
library(HistData)
set.seed(1983, sample.kind = "Rounding")
galton_heights <- GaltonFamilies %>%
  filter(gender == "male") %>%
  group_by(family) %>%
  sample_n(1) %>%
  ungroup() %>%
  select(father, childHeight) %>%
  rename(son = childHeight)

y <- galton_heights$son
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
train_set <- galton_heights %>% slice(-test_index)
test_set <- galton_heights %>% slice(test_index)

m <- mean(train_set$son)
# squared loss
mean((m - test_set$son)^2)

# fit linear regression model
fit <- lm(son ~ father, data = train_set)
fit$coef
y_hat <- fit$coef[1] + fit$coef[2]*test_set$father
mean((y_hat - test_set$son)^2)


## Predict Function

y_hat <- predict(fit, test_set)

?predict.lm    # or ?predict.glm

y_hat <- predict(fit, test_set)
mean((y_hat - test_set$son)^2)

# read help files
?predict.lm
?predict.glm



## Comprehension Check: Linear Regression

rm(list = ls())
options(digits=6)

# Q1
# Create a data set using the following code:
  
# set.seed(1) # 
set.seed(1) # If using R version 4.0.0 then use set.seed(1)
# set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
n <- 100
Sigma <- 9*matrix(c(1.0, 0.5, 0.5, 1.0), 2, 2)
dat <- MASS::mvrnorm(n = 100, c(69, 69), Sigma) %>%
  data.frame() %>% setNames(c("x", "y"))

# set.seed(1) # 
set.seed(1) # If using R version 4.0.0 then use set.seed(1)
# set.seed(1, sample.kind="Rounding") # if using R 3.6 or later

lm100 <- 
  replicate(100, {
    # (1) partition the dataset into test and training sets 
    #     with p=0.5 and using dat$y to generate your indices, 
    y <- dat$y
    test_index <- 
      createDataPartition(y, times = 1, p = 0.5, list = FALSE)
    train_set <- dat %>% slice(-test_index)
    test_set <- dat %>% slice(test_index)
    # (2) train a linear model predicting y from x, 
    fit <- lm(y ~ x, data = train_set)
    # (3) generate predictions on the test set, and 
    y_hat <- fit$coef[1] + fit$coef[2]*test_set$x
    # (4) calculate the RMSE of that model. 
    #sqrt(mean((y_hat - test_set$y)^2))    
    sqrt(mean((y_hat - test_set$y)^2))
  })


# We will build 100 linear models using the data above 
# and calculate the mean and standard deviation of 
# the combined models. First, set the seed to 1 again 
# (make sure to use sample.kind="Rounding" if your R 
#   is version 3.6 or later). 
# Then, within a replicate() loop, 
# (1) partition the dataset into test and training sets 
#     with p=0.5 and using dat$y to generate your indices, 
# (2) train a linear model predicting y from x, 
# (3) generate predictions on the test set, and 
# (4) calculate the RMSE of that model. 
#     Then, report the mean and standard deviation (SD) 
#     of the RMSEs from all 100 models.
# 
# Things to remember for the above questions:
# Calculate RMSE and not MSE. Do that either by RMSE function 
# or by sqrt(your_MSE). Set seed twice for all the exercises. 
# One before the creation of the data and second before running 
# the replicate function. In questions where replicate 
# is not required use it before creating the partition. 
# Also report 3 digits after the decimal, only 2 does not qualify. (0.000 is right and not 0.00)
#
# Report all answers to at least 3 decimal places.
# 
# Mean:
#   
mean(lm100) 
# [1] 2.499521 # NO
# [1] 6.2027 # NO
# [1] 6.25258 # NO
# [1] 2.48866 # YES! used RMSE, not MSE as in example and If using R version 4.0.0 then use set.seed(1)

# 
# Standard deviation (SD):
#   
sd(lm100)  
# [1] 0.118618 # NO
# [1] 0.602371 # NO
# [1] 0.59053 # NO
# [1] 0.124395 # YES!


# Q2
# Now we will repeat the exercise above 
# but using larger datasets. Write a function 
# that takes a size n, then 
# (1) builds a dataset using the code provided at 
# the top of Q1 but with n observations instead of 100 
# and without the set.seed(1), 
# (2) runs the replicate() loop that you wrote to answer Q1, which 
# builds 100 linear models and returns a vector of RMSEs, and 
# (3) calculates the mean and standard deviation of the 100 RMSEs.
# 
# Set the seed to 1 (if using R 3.6 or later, 
# use the argument sample.kind="Rounding", 
# If using R version 4.0.0 then use set.seed(1)) 
# and then use sapply() or map() to apply your new function 
# to n <- c(100, 500, 1000, 5000, 10000).
# 
# You only need to set the seed once before running your function; 
# do not set a seed within your function. Also be sure 
# to use sapply() or map() as you will get different answers 
# running the simulations individually due to setting the seed.
# 

rm(list = ls())

# set.seed(1) # 
set.seed(1) # If using R version 4.0.0 then use set.seed(1)
# set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
n <- 100
# Write a function that takes a size n
fnfit <- function(n) {
  # (2.1) builds a dataset using the code provided at the top 
  #     of Q1 but with n observations instead of 100 
  Sigma <- 9*matrix(c(1.0, 0.5, 0.5, 1.0), 2, 2)
  dat <- MASS::mvrnorm(n = n, c(69, 69), Sigma) %>%
    data.frame() %>% setNames(c("x", "y"))
  # (2.2) runs the replicate() loop that you wrote to answer Q1, 
  # which builds 100 linear models and returns a vector 
  # of RMSEs, and 
  rmse <- replicate(100, {
      # (1.1) partition the dataset into test and training sets 
      #     with p=0.5 and using dat$y to generate your indices, 
      y <- dat$y
      test_index <- 
        createDataPartition(y, times = 1, p = 0.5, list = FALSE)
      train_set <- dat %>% slice(-test_index)
      test_set <- dat %>% slice(test_index)
      # (1.2) train a linear model predicting y from x, 
      fit <- lm(y ~ x, data = train_set)
      # (1.3) generate predictions on the test set, and 
      y_hat <- fit$coef[1] + fit$coef[2]*test_set$x
      # (1.4) calculate the RMSE of that model. 
      sqrt(mean((y_hat - test_set$y)^2))
    })  
  c(n = n, avg = mean(rmse), s = sd(rmse))
  }

n <- 100
n <- c(100, 500, 1000, 5000, 10000)
set.seed(1)
sapply(n, fnfit)

#           [,1]        [,2]        [,3]        [,4]        [,5]
# n   100.000000 500.0000000 1.00000e+03 5.00000e+03 1.00000e+04
# avg   2.497754   2.7209512 2.55554e+00 2.62483e+00 2.61844e+00
# s     0.118082   0.0800211 4.56026e-02 2.30967e-02 1.68920e-02

# Answere Explanation from edX course
# 
# The code below can be used to do this calculation:
#   
#   set.seed(1)    # if R 3.6 or later, set.seed(1, sample.kind="Rounding")
# n <- c(100, 500, 1000, 5000, 10000)
# res <- sapply(n, function(n){
#   Sigma <- 9*matrix(c(1.0, 0.5, 0.5, 1.0), 2, 2)
#   dat <- MASS::mvrnorm(n, c(69, 69), Sigma) %>%
#     data.frame() %>% setNames(c("x", "y"))
#   rmse <- replicate(100, {
#     test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
#     train_set <- dat %>% slice(-test_index)
#     test_set <- dat %>% slice(test_index)
#     fit <- lm(y ~ x, data = train_set)
#     y_hat <- predict(fit, newdata = test_set)
#     sqrt(mean((y_hat-test_set$y)^2))
#   })
#   c(avg = mean(rmse), sd = sd(rmse))
# })
# 
# res

## Q3
# What happens to the RMSE as the size 
# of the dataset becomes larger?
# On average, the RMSE does not change much as n gets larger, 
# but the variability of the RMSE decreases.


## Q4
# Now repeat the exercise from Q1, this time making 
# the correlation between x and y larger, as in 
# the following code:

library(tidyverse)
library(dslabs)
library(caret)
library(HistData)
  
set.seed(1)
n <- 100
Sigma <- 9*matrix(c(1.0, 0.95, 0.95, 1.0), 2, 2)
dat <- MASS::mvrnorm(n = 100, c(69, 69), Sigma) %>%
  data.frame() %>% setNames(c("x", "y"))

## Code from Q1 online
set.seed(1)    # if R 3.6 or later, set.seed(1, sample.kind="Rounding")
rmse <- replicate(100, {
  test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
  train_set <- dat %>% slice(-test_index)
  test_set <- dat %>% slice(test_index)
  fit <- lm(y ~ x, data = train_set)
  y_hat <- predict(fit, newdata = test_set)
  sqrt(mean((y_hat-test_set$y)^2))
})

## my Q1 code
set.seed(1)
set.seed(1, sample.kind = "Rounding")
rmse <- replicate(100, {
  # (1) partition the dataset into test and training sets 
  #     with p=0.5 and using dat$y to generate your indices, 
  y <- dat$y
  test_index <- 
    createDataPartition(y, times = 1, p = 0.5, list = FALSE)
  train_set <- dat %>% slice(-test_index)
  test_set <- dat %>% slice(test_index)
  # (2) train a linear model predicting y from x, 
  fit <- lm(y ~ x, data = train_set)
  # (3) generate predictions on the test set, and 
  y_hat <- fit$coef[1] + fit$coef[2]*test_set$x
  # (4) calculate the RMSE of that model. 
  #sqrt(mean((y_hat - test_set$y)^2))    
  sqrt(mean((y_hat - test_set$y)^2))
})



mean(rmse)
# [1] 0.9078124 # right, used set.seed(1)
sd(rmse)
# [1] 0.05821304 # wrong
# [1] 0.06244347 # right, used set.seed(1, sample.kind = "Rounding")

# Note what happens to RMSE - set the seed to 1 as before.

# code from site:
set.seed(1)
rmse <- replicate(100, {
  test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
  train_set <- dat %>% slice(-test_index)
  test_set <- dat %>% slice(test_index)
  fit <- lm(y ~ x, data = train_set)
  y_hat <- predict(fit, newdata = test_set)
  sqrt(mean((y_hat-test_set$y)^2))
})

mean(rmse)
sd(rmse)


## Q5
# Which of the following best explains why the RMSE 
# in question 4 is so much lower than the RMSE in question 1?
#
# When we increase the correlation between x and y, x has 
# more predictive power and thus provides a better estimate of y.



## Q6
# Create a data set using the following code.


library(tidyverse)
library(dslabs)
library(caret)
library(HistData)

rm(list = ls())

set.seed(1)
# set.seed(1, sample.kind = "Rounding")
Sigma <- matrix(c(1.0, 0.75, 0.75, 0.75, 
                  1.0, 0.25, 0.75, 0.25, 1.0), 3, 3)
dat <- MASS::mvrnorm(n = 100, c(0, 0, 0), Sigma) %>%
  data.frame() %>% setNames(c("y", "x_1", "x_2"))

# Note that y is correlated with both x_1 and x_2 
# but the two predictors are independent of each other, 
# as seen by cor(dat).
# 
# Set the seed to 1, then use the caret package to partition 
# into a test and training set of equal size. Compare the RMSE 
# when using just x_1, just x_2 and both x_1 and x_2. Train 
# a single linear model for each (not 100 like in 
# the previous questions).

cor(dat)
set.seed(1)
test_index <- 
  createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
train_set <- dat %>% slice(-test_index)
test_set <- dat %>% slice(test_index)

fit1 <- lm(y ~ x_1, data = train_set)
y_hat <- predict(fit1, newdata = test_set)
sqrt(mean((y_hat-test_set$y)^2))
# [1] 0.6708231
# [1] 0.6515668 # after restarting session
# [1] 0.600666 # added another set.seed(1) before creating partition

fit2 <- lm(y ~ x_2, data = train_set)
y_hat <- predict(fit2, newdata = test_set)
sqrt(mean((y_hat-test_set$y)^2))
# [1] 0.6775359
# [1] 0.6822152 # after restarting session
# [1] 0.630699 # added set.seed before creating partition

fit12 <- lm(y ~ x_1 + x_2, data = train_set)
y_hat <- predict(fit12, newdata = test_set)
sqrt(mean((y_hat-test_set$y)^2))
RMSE(y_hat, test_set$y)
# [1] 0.3552544
# [1] 0.345693 # after restarting session
# [1] 0.3070962  # added set.seed before creating partition

# 
# Which of the three models performs the best 
# (has the lowest RMSE)?
# x_1 and x_2


## Q7
# Report the lowest RMSE of the three models tested in Q6.

# [1] 0.3552544 # NO
# [1] 0.345693 # NO, after restarting session
# [1] 0.3070962 # YES!  added set.seed(1) before partitioning


## Q8
# Repeat the exercise from Q6 but now create an example 
# in which x_1 and x_2 are highly correlated.


library(tidyverse)
library(dslabs)
library(caret)
library(HistData)

set.seed(1)
Sigma <- matrix(c(1.0, 0.75, 0.75, 0.75, 1.0, 0.95, 0.75, 0.95, 1.0), 3, 3)
dat <- MASS::mvrnorm(n = 100, c(0, 0, 0), Sigma) %>%
  data.frame() %>% setNames(c("y", "x_1", "x_2"))

# Set the seed to 1, then use the caret package 
# to partition into a test and training set of equal size. 
# Compare the RMSE when using just x_1, just x_2, 
# and both x_1 and x_2.

cor(dat)
set.seed(1)
test_index <- 
  createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
train_set <- dat %>% slice(-test_index)
test_set <- dat %>% slice(test_index)

fit1 <- lm(y ~ x_1, data = train_set)
y_hat <- predict(fit1, newdata = test_set)
sqrt(mean((y_hat-test_set$y)^2))
# [1] 0.6467152

fit2 <- lm(y ~ x_2, data = train_set)
y_hat <- predict(fit2, newdata = test_set)
sqrt(mean((y_hat-test_set$y)^2))
# [1] 0.6296398

fit12 <- lm(y ~ x_1 + x_2, data = train_set)
y_hat <- predict(fit12, newdata = test_set)
sqrt(mean((y_hat-test_set$y)^2))
RMSE(y_hat, test_set$y)
# [1] 0.6485341

# 
# Compare the results from Q6 and Q8. What can you conclude?

# Adding extra predictors can improve RMSE substantially, 
# but not when the added predictors are highly correlated 
# with other predictors.

















































































