## Section 5: Classification with More than
## Two Classes and the Caret Package
##   5.1: Classification with More than Two Classes
##   Trees Motivation

## Classification and Regression Trees (CART)
# Load data
cat("\014") ## clear the console
rm(list = ls())

library(tidyverse)
library(dslabs)
data("olive")
olive %>% as_tibble()
table(olive$region)
olive <- select(olive,-area)

# Predict region using KNN
library(caret)
fit <- train(
  region ~ .,
  method = "knn",
  tuneGrid = data.frame(k = seq(1, 15, 2)),
  data = olive
)
ggplot(fit)

# Plot distribution of each predictor stratified by region
olive %>% gather(fatty_acid, percentage,-region) %>%
  ggplot(aes(region, percentage, fill = region)) +
  geom_boxplot() +
  facet_wrap( ~ fatty_acid, scales = "free") +
  theme(axis.text.x = element_blank())

# plot values for eicosenoic and linoleic
p <- olive %>%
  ggplot(aes(eicosenoic, linoleic, color = region)) +
  geom_point()
p + geom_vline(xintercept = 0.065, lty = 2) +
  geom_segment(
    x = -0.2,
    y = 10.54,
    xend = 0.065,
    yend = 10.54,
    color = "black",
    lty = 2
  )

# load data for regression tree
data("polls_2008")
qplot(day, margin, data = polls_2008)

library(rpart)
fit <- rpart(margin ~ ., data = polls_2008)

# visualize the splits
plot(fit, margin = 0.1)
text(fit, cex = 0.75)
polls_2008 %>%
  mutate(y_hat = predict(fit)) %>%
  ggplot() +
  geom_point(aes(day, margin)) +
  geom_step(aes(day, y_hat), col = "red")

# change parameters
fit <- rpart(margin ~ .,
             data = polls_2008,
             control = rpart.control(cp = 0, minsplit = 2))
polls_2008 %>%
  mutate(y_hat = predict(fit)) %>%
  ggplot() +
  geom_point(aes(day, margin)) +
  geom_step(aes(day, y_hat), col = "red")

# use cross validation to choose cp
library(caret)
train_rpart <-
  train(
    margin ~ .,
    method = "rpart",
    tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)),
    data = polls_2008
  )
ggplot(train_rpart)

# access the final model and plot it
plot(train_rpart$finalModel, margin = 0.1)
text(train_rpart$finalModel, cex = 0.75)
polls_2008 %>%
  mutate(y_hat = predict(train_rpart)) %>%
  ggplot() +
  geom_point(aes(day, margin)) +
  geom_step(aes(day, y_hat), col = "red")

# prune the tree
pruned_fit <- prune(fit, cp = 0.01)


## Classification (Decision) Trees

# fit a classification tree and plot it
train_rpart <- train(
  y ~ .,
  method = "rpart",
  tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
  data = mnist_27$train
)
plot(train_rpart)

# compute accuracy
confusionMatrix(predict(train_rpart,
                        mnist_27$test),
                mnist_27$test$y)$overall["Accuracy"]


## Random Forests

library(randomForest)
fit <- randomForest(margin ~ ., data = polls_2008)
plot(fit)

polls_2008 %>%
  mutate(y_hat = predict(fit, newdata = polls_2008)) %>%
  ggplot() +
  geom_point(aes(day, margin)) +
  geom_line(aes(day, y_hat), col = "red")

library(randomForest)
# install.packages("Rborist")
library(Rborist)
library(dslabs)
library(caret)
library(tidyverse)

train_rf <- randomForest(y ~ ., data = mnist_27$train)
confusionMatrix(predict(train_rf, mnist_27$test),
                mnist_27$test$y)$overall["Accuracy"]

# use cross validation to choose parameter
train_rf_2 <- train(
  y ~ .,
  method = "Rborist",
  tuneGrid = data.frame(predFixed = 2,
                        minNode = c(3, 50)),
  data = mnist_27$train
)
confusionMatrix(predict(train_rf_2, mnist_27$test),
                mnist_27$test$y)$overall["Accuracy"]



## Comprehension Check: Trees and Random Forests

# Q1
# Create a simple dataset where the outcome 
# grows 0.75 units on average for every increase 
# in a predictor, using this code:
  
library(rpart)
n <- 1000
sigma <- 0.25
set.seed(1) #set.seed(1, sample.kind = "Rounding") if using R 3.6 or later
x <- rnorm(n, 0, 1)
y <- 0.75 * x + rnorm(n, 0, sigma)
dat <- data.frame(x = x, y = y)

# Which code correctly uses rpart() to fit 
# a regression tree and saves the result to fit?

fit <- rpart(y ~ ., data = dat) 
fit

# Q2
# Which of the following plots has the same tree shape 
# obtained in Q1?

plot(fit, margin = 0.1)
text(fit, cex = 0.75)

# 4 has the tree SHAPE, values different


## Q3
# Below is most of the code to make a scatter plot 
# of y versus x along with the predicted values based 
# on the fit.
library(dplyr)
# install.packages("quantreg")
library(quantreg)

dat %>% 
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(x, y)) +
  #BLANK
  

# Which line of code should be used to replace #BLANK 
# in the code above?

geom_step(aes(x, y_hat), col=2) ## YES, correct answer!!!


## Q4
# Now run Random Forests instead of a regression tree 
# using randomForest() from the randomForest package, 
# and remake the scatterplot with the prediction line. 
# Part of the code is provided for you below.

library(tidyverse)
library(randomForest)
library(rpart)
n <- 1000
sigma <- 0.25
# set.seed(1) #
set.seed(1, sample.kind = "Rounding") # if using R 3.6 or later
x <- rnorm(n, 0, 1)
y <- 0.75 * x + rnorm(n, 0, sigma)
dat <- data.frame(x = x, y = y)

fit <- #BLANK 
  randomForest(y ~ x, data = dat) ## option 1

dat %>% 
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(x, y)) +
  geom_step(aes(x, y_hat), col = "red")

# What code should replace #BLANK in the provided code?

randomForest(y ~ x, data = dat) ## yes, correct


## Q5
# Use the plot() function to see if the Random Forest 
# from Q4 has converged or if we need more trees.

# Which of these graphs is produced by plotting 
# the random forest?

plot(fit)

## option 3 graph


## Q6
# It seems that the default values for the Random Forest 
# result in an estimate that is too flexible (unsmooth). 
# Re-run the Random Forest but this time with a node size of 
# 50 and a maximum of 25 nodes. Remake the plot.

# Part of the code is provided for you below.

library(randomForest)
fit <- #BLANK
  randomForest(y ~ x, data = dat,
               nodesize = 50, maxnodes = 25) # option 4

dat %>% 
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(x, y)) +
  geom_step(aes(x, y_hat), col = "red")

# What code should replace #BLANK in the provided code?

randomForest(y ~ x, data = dat, nodesize = 50, maxnodes = 25)





























































