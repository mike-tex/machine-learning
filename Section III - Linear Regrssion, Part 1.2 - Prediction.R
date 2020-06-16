## Section 3:
## Linear Regression for Prediction, Smoothing,
## and Working with Matrices
##   3.1: Linear Regression for Prediction
##   Regression for a Categorical Outcome

library(dslabs)
library(tidyverse)
library(caret)

data("heights")
y <- heights$height

set.seed(2) #if you are using R 3.5 or earlier
set.seed(2, sample.kind = "Rounding") #if you are using R 3.6 or later

test_index <- 
  createDataPartition(y, times = 1, p = 0.5, list = FALSE)
train_set <- heights %>% slice(-test_index)
test_set <- heights %>% slice(test_index)

train_set %>% 
  filter(round(height)==66) %>%
  summarize(y_hat = mean(sex=="Female"))

heights %>% 
  mutate(x = round(height)) %>%
  group_by(x) %>%
  filter(n() >= 10) %>%
  summarize(prop = mean(sex == "Female")) %>%
  ggplot(aes(x, prop)) +
  geom_point()
lm_fit <- mutate(train_set, 
                 y = as.numeric(sex == "Female")) %>% 
  lm(y ~ height, data = .)
p_hat <- predict(lm_fit, test_set)
y_hat <- ifelse(p_hat > 0.5, "Female", "Male") %>% factor()
confusionMatrix(y_hat, test_set$sex)$overall["Accuracy"]


## Logistic Regression

heights %>% 
  mutate(x = round(height)) %>%
  group_by(x) %>%
  filter(n() >= 10) %>%
  summarize(prop = mean(sex == "Female")) %>%
  ggplot(aes(x, prop)) +
  geom_point() + 
  geom_abline(intercept = lm_fit$coef[1], 
              slope = lm_fit$coef[2])
range(p_hat)

# fit logistic regression model
glm_fit <- train_set %>% 
  mutate(y = as.numeric(sex == "Female")) %>%
  glm(y ~ height, data=., family = "binomial")
p_hat_logit <- 
  predict(glm_fit, newdata = test_set, type = "response")
y_hat_logit <- 
  ifelse(p_hat_logit > 0.5, "Female", "Male") %>% factor
confusionMatrix(y_hat_logit, 
                test_set$sex)$overall[["Accuracy"]]


## Case Study: 2 or 7

.rs.restartR()  ## restart session
rm(list = ls())  ## clear environment

library(tidyverse)
library(dslabs)
library(caret)

mnist <- read_mnist()
is <- mnist_27$index_train[c(which.min(mnist_27$train$x_1), 
                             which.max(mnist_27$train$x_1))]
titles <- c("smallest","largest")
tmp <- lapply(1:2, function(i){
  expand.grid(Row=1:28, Column=1:28) %>%
    mutate(label=titles[i],
           value = mnist$train$images[is[i],])
})
tmp <- Reduce(rbind, tmp)
tmp %>% ggplot(aes(Row, Column, fill=value)) +
  geom_raster() +
  scale_y_reverse() +
  scale_fill_gradient(low="white", high="black") +
  facet_grid(.~label) +
  geom_vline(xintercept = 14.5) +
  geom_hline(yintercept = 14.5)

data("mnist_27")
mnist_27$train %>% 
  ggplot(aes(x_1, x_2, color = y)) + geom_point()

is <- mnist_27$index_train[c(which.min(mnist_27$train$x_2), 
                             which.max(mnist_27$train$x_2))]
titles <- c("smallest","largest")
tmp <- lapply(1:2, function(i){
  expand.grid(Row=1:28, Column=1:28) %>%
    mutate(label=titles[i],
           value = mnist$train$images[is[i],])
})
tmp <- Reduce(rbind, tmp)
tmp %>% ggplot(aes(Row, Column, fill=value)) +
  geom_raster() +
  scale_y_reverse() +
  scale_fill_gradient(low="white", high="black") +
  facet_grid(.~label) +
  geom_vline(xintercept = 14.5) +
  geom_hline(yintercept = 14.5)

fit_glm <- glm(y ~ x_1 + x_2, data=mnist_27$train, 
               family = "binomial")
p_hat_glm <- predict(fit_glm, mnist_27$test)
y_hat_glm <- factor(ifelse(p_hat_glm > 0.5, 7, 2))
confusionMatrix(data = y_hat_glm, 
                reference = mnist_27$test$y)$overall["Accuracy"]

mnist_27$true_p %>% ggplot(aes(x_1, x_2, fill=p)) +
  geom_raster()

mnist_27$true_p %>% ggplot(aes(x_1, x_2, z=p, fill=p)) +
  geom_raster() +
  scale_fill_gradientn(colors=c("#F8766D","white","#00BFC4")) +
  stat_contour(breaks=c(0.5), color="black") 

p_hat <- predict(fit_glm, newdata = mnist_27$true_p)
mnist_27$true_p %>%
  mutate(p_hat = p_hat) %>%
  ggplot(aes(x_1, x_2,  z=p_hat, fill=p_hat)) +
  geom_raster() +
  scale_fill_gradientn(colors=c("#F8766D","white","#00BFC4")) +
  stat_contour(breaks=c(0.5),color="black") 

p_hat <- predict(fit_glm, newdata = mnist_27$true_p)
mnist_27$true_p %>%
  mutate(p_hat = p_hat) %>%
  ggplot() +
  stat_contour(aes(x_1, x_2, z=p_hat), 
               breaks=c(0.5), color="black") +
  geom_point(mapping = aes(x_1, x_2, color=y), 
             data = mnist_27$test)


## Comprehension Check: Logistic Regression

# Q1
# Define a dataset using the following code:

rm(list = ls())
library(tidyverse)
  
set.seed(2) #if you are using R 3.5 or earlier
# set.seed(2, sample.kind="Rounding") #if you are using R 3.6 or later
make_data <- function(n = 1000, p = 0.5, 
                      mu_0 = 0, mu_1 = 2, 
                      sigma_0 = 1,  sigma_1 = 1){
  
  y <- rbinom(n, 1, p)
  f_0 <- rnorm(n, mu_0, sigma_0)
  f_1 <- rnorm(n, mu_1, sigma_1)
  x <- ifelse(y == 1, f_1, f_0)
  
  test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
  
  list(train = data.frame(x = x, y = as.factor(y)) %>% slice(-test_index),
       test = data.frame(x = x, y = as.factor(y)) %>% slice(test_index))
}
dat <- make_data()

# Note that we have defined a variable x that 
# is predictive of a binary outcome y: 
dat$train %>% ggplot(aes(x, color = y)) + geom_density()
# 
# Set the seed to 1, then use the make_data() function 
# defined above to generate 25 different datasets 
# with mu_1 <- seq(0, 3, len=25). Perform logistic regression 
# on each of the 25 different datasets (predict 1 if p>0.5) 
# and plot accuracy (res in the figures) vs mu_1 (delta in 
# the figures).”

set.seed(1)
mu_1 <- seq(0, 3, len=25)
# i <- 1
res <- function(z) {
  x <- make_data(mu_1 = z)
  glm_fit <- x$train %>% 
    glm(y ~ x, data = ., family = "binomial")
  p_hat_logit <- 
    predict(glm_fit, newdata = x$test, type = "response")
  y_hat_logit <- 
    ifelse(p_hat_logit > 0.5, 1, 0) %>% factor
  res.df <<- 
    rbind(res.df, c(z, 
          CM = confusionMatrix(y_hat_logit, 
                          x$test$y)$overall[["Accuracy"]]))
}

res.df <- data.frame(mu_1 = NULL, CM = NULL)

lapply(mu_1, res)

plot(res.df[,1], res.df[,2])
# Which is the correct plot?

# Explanation

# The correct plot can be generated using the following code:
  
set.seed(1) #if you are using R 3.5 or earlier
# set.seed(1, sample.kind="Rounding") #if you are using R 3.6 or later
delta <- seq(0, 3, len = 25)
res <- sapply(delta, function(d) {
  dat <- make_data(mu_1 = d)
  fit_glm <- dat$train %>%
    glm(y ~ x, family = "binomial", data = .)
  y_hat_glm <-
    ifelse(predict(fit_glm, dat$test) > 0.5, 1, 0) %>%
    factor(levels = c(0, 1))
  mean(y_hat_glm == dat$test$y)
})
qplot(delta, res)











































































































