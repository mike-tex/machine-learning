## Section 4: Distance, Knn, Cross-validation,
## and Generative Models
##   4.1: Nearest Neighbors
##   Distance

.rs.restartR()
rm(list = ls())


library(tidyverse)
library(dslabs)
if(!exists("mnist")) mnist <- read_mnist()
# set.seed(1995) # if using R 3.5 or earlier
set.seed(1995) # if using R 3.6 or later
ind <- which(mnist$train$labels %in% c(2,7)) %>% sample(500)

#the predictors are in x and the labels in y
x <- mnist$train$images[ind,]
y <- mnist$train$labels[ind]
y[1:3]
x_1 <- x[1,]
x_2 <- x[2,]
x_3 <- x[3,]

#distance between two numbers
sqrt(sum((x_1 - x_2)^2))
sqrt(sum((x_1 - x_3)^2))
sqrt(sum((x_2 - x_3)^2))

#compute distance using matrix algebra
sqrt(crossprod(x_1 - x_2))
sqrt(crossprod(x_1 - x_3))
sqrt(crossprod(x_2 - x_3))

#compute distance between each row
d <- dist(x)
class(d)
as.matrix(d)[1:3,1:3]

#visualize these distances
image(as.matrix(d))

#order the distance by labels
image(as.matrix(d)[order(y), order(y)])

#compute distance between predictors
d <- dist(t(x))
dim(as.matrix(d))
d_492 <- as.matrix(d)[492,]
image(1:28, 1:28, matrix(d_492, 28, 28))


## Comprehension Check: Distance

# Q1
# Load the following dataset:
  
rm(list = ls())

library(dslabs)
data(tissue_gene_expression)

# This dataset includes a matrix x:
  
dim(tissue_gene_expression$x)

# This matrix has the gene expression levels of 500 genes 
# from 189 biological samples representing seven different 
# tissues. The tissue type is stored in y:
  
table(tissue_gene_expression$y)

# Which of the following lines of code computes 
# the Euclidean distance between each observation 
# and stores it in the object d?

d <- dist(tissue_gene_expression$x)

# Q2
# Using the dataset from Q1, compare the distances 
# between observations 1 and 2 (both cerebellum), 
# observations 39 and 40 (both colon), 
# and observations 73 and 74 (both endometrium).

# Distance-wise, are samples from tissues of 
# the same type closer to each other?

dim(as.matrix(d))

drows <- c(1,2, 39, 40, 73, 74)
as.matrix(d)[drows, drows]

# 
# # Yes, the samples from the same tissue type are closest 
# to each other.
# 

# Explanation

# You can calculate the distances using the following code:
  
ind <- c(1, 2, 39, 40, 73, 74)
as.matrix(d)[ind,ind]


# Q3
# Make a plot of all the distances using 
# the image() function to see if the pattern you observed 
# in Q2 is general.

# Which code would correctly make the desired plot?

image(as.matrix(d))


## Knn

rm(list = ls())

library(tidyverse)
library(dslabs)
if(!exists("mnist")) mnist <- read_mnist()

library(caret)

x <- as.matrix(mnist_27$train[,2:3])
y <- mnist_27$train$y
knn_fit <- knn3(x,y)

#logistic regression
library(caret)
fit_glm <- 
  glm(y~x_1+x_2, data=mnist_27$train, family="binomial")
p_hat_logistic <- predict(fit_glm, mnist_27$test)
y_hat_logistic <- factor(ifelse(p_hat_logistic > 0.5, 7, 2))
confusionMatrix(data = y_hat_logistic, 
                reference = mnist_27$test$y)$overall[1]

#fit knn model
knn_fit <- knn3(y ~ ., data = mnist_27$train)
x <- as.matrix(mnist_27$train[,2:3])
y <- mnist_27$train$y
knn_fit <- knn3(x, y)
knn_fit <- knn3(y ~ ., data = mnist_27$train, k=5)
y_hat_knn <- predict(knn_fit, mnist_27$test, type = "class")
confusionMatrix(data = y_hat_knn, 
                reference = mnist_27$test$y)$overall["Accuracy"]


## Overtraining and Oversmoothing

y_hat_knn <- predict(knn_fit, mnist_27$train, type = "class") 
confusionMatrix(data = y_hat_knn, reference = mnist_27$train$y)$overall["Accuracy"]
y_hat_knn <- predict(knn_fit, mnist_27$test, type = "class")  
confusionMatrix(data = y_hat_knn, reference = mnist_27$test$y)$overall["Accuracy"]

#fit knn with k=1
knn_fit_1 <- knn3(y ~ ., data = mnist_27$train, k = 1)
y_hat_knn_1 <- predict(knn_fit_1, mnist_27$train, type = "class")
confusionMatrix(data=y_hat_knn_1, reference=mnist_27$train$y)$overall[["Accuracy"]]

#fit knn with k=401
knn_fit_401 <- knn3(y ~ ., data = mnist_27$train, k = 401)
y_hat_knn_401 <- predict(knn_fit_401, mnist_27$test, type = "class")
confusionMatrix(data=y_hat_knn_401, reference=mnist_27$test$y)$overall["Accuracy"]

#pick the k in knn
ks <- seq(3, 251, 2)
library(purrr)
accuracy <- map_df(ks, function(k){
  fit <- knn3(y ~ ., data = mnist_27$train, k = k)
  y_hat <- predict(fit, mnist_27$train, type = "class")
  cm_train <- confusionMatrix(data = y_hat, 
                              reference = mnist_27$train$y)
  train_error <- cm_train$overall["Accuracy"]
  y_hat <- predict(fit, mnist_27$test, type = "class")
  cm_test <- confusionMatrix(data = y_hat, 
                             reference = mnist_27$test$y)
  test_error <- cm_test$overall["Accuracy"]
  
  tibble(train = train_error, test = test_error)
})

# pick the k that maximizes accuracy using the estimates 
# built on the test data
ks[which.max(accuracy$test)]
max(accuracy$test)


## Comprehension Check: Nearest Neighbors

# Q1
# Previously, we used logistic regression 
# to predict sex based on height. Now we are going 
# to use knn to do the same. Set the seed to 1, 
# then use the caret package to partition 
# the dslabs heights data into a training 
# and test set of equal size. Use the sapply() 
# or map function to perform knn with k values 
# of seq(1, 101, 3) and calculate F1 scores 
# with the F_meas() function using the default value 
# of the relevant argument.

.rs.restartR(afterRestartCommand = rm(list = ls()))
# rm(list = ls())

library(tidyverse)
library(caret)
library(dslabs)
data(heights)
# set.seed(1)
set.seed(1, sample.kind = "Rounding")
idx <- 
  createDataPartition(heights$sex, times = 1, p = 0.5, list = F)
test_set <- heights[idx,]
train_set <- heights[-idx,]
ks <- seq(1, 101, 3)
F_1 <- map_dbl(ks, function(k){
  fit <- knn3(sex ~ height, data = train_set, k = k)
  y_hat <- predict(fit, test_set, type = "class") 
  F_meas(data = y_hat, reference = factor(test_set$sex)) 
})

plot(ks, F_1)

# What is the max value of F_1?
max(F_1)  
# [1] 0.6 ### YES!!!!!!
# [1] 0.6019417 ## is the correct answer, lucky/unlucky it took .6

# At what value of k does the max occur?
# If there are multiple values of k 
# with the maximum value, report the smallest such k.

ks[which.max(F_1)]
# [1] 46 ### YES!!!!!!!!!!!!!!


# Explanation from the site

## This exercise can be accomplished using the following code:
  
library(dslabs)
library(tidyverse)
library(caret)
data("heights")

# set.seed(1)
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(heights$sex, times = 1, p = 0.5, list = FALSE)
test_set <- heights[test_index, ]
train_set <- heights[-test_index, ]     

ks <- seq(1, 101, 3)
F_1 <- sapply(ks, function(k){
  fit <- knn3(sex ~ height, data = train_set, k = k)
  y_hat <- predict(fit, test_set, type = "class") %>% 
    factor(levels = levels(train_set$sex))
  F_meas(data = y_hat, reference = test_set$sex)
})
plot(ks, F_1)
max(F_1)
ks[which.max(F_1)]

# using set.seed(1), get the WRONG answer:
# > max(F_1)
# [1] 0.619469
# > ks[which.max(F_1)]
# [1] 40
# > 

# using set.seed(1, sample.kind = "Rounding"), get RIGHT answer:
# > max(F_1)
# [1] 0.6019417
# > ks[which.max(F_1)]
# [1] 46
# > 


## Q2
# Next we will use the same gene expression example 
# used in the Comprehension Check: Distance exercises. 
# You can load it like this:

rm(list = ls())

library(tidyverse)
library(dslabs)
library(caret)
data("tissue_gene_expression")

# First, set the seed to 1 and split the data 
# into training and test sets. Then, report the accuracy 
# you obtain from predicting tissue type using KNN 
# with k = 1, 3, 5, 7, 9, 11 using sapply() or map_df(). 
# Note: use the createDataPartition() function outside of 
# sapply() or map_df().

set.seed(1, sample.kind = "Rounding")
idx <- createDataPartition(
  tissue_gene_expression$y, times = 1, p = 0.5, list = F)
test_set <- list(x = tissue_gene_expression$x[idx,], 
                 y = tissue_gene_expression$y[idx])
train_set <- list(x = tissue_gene_expression$x[-idx,], 
                 y = tissue_gene_expression$y[-idx])
ks <- c(1, 3, 5, 7, 9, 11)

accuracy <- map_df(ks, function(k) {
  fit <- knn3(y ~ x, data = train_set, k = k)
  y_hat <- predict(fit, test_set, type = "class")
  cm <- confusionMatrix(data = y_hat,
                        reference = test_set$y)
  tibble(k = k, cm = cm$overall["Accuracy"])
})

accuracy

# # A tibble: 6 x 2
#       k    cm
#     <dbl> <dbl>
# 1     1   0.990
# 2     3   0.969
# 3     5   0.948
# 4     7   0.917
# 5     9   0.917
# 6    11   0.906
# 


# Explanation from web site

# This exercise can be accomplished using the following code:
  
set.seed(1)
library(caret)
y <- tissue_gene_expression$y
x <- tissue_gene_expression$x
test_index <- createDataPartition(y, list = FALSE)
sapply(seq(1, 11, 2), function(k){
  fit <- knn3(x[-test_index,], y[-test_index], k = k)
  y_hat <- predict(fit, newdata = data.frame(x=x[test_index,]),
                   type = "class")
  mean(y_hat == y[test_index])
})

















































