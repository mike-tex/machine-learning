## Section 6: Model Fitting and Recommendation Systems
##   6.1: Case Study: MNIST
##   Case Study: MNIST

library(dslabs)
mnist <- read_mnist()

names(mnist)
dim(mnist$train$images)

class(mnist$train$labels)
table(mnist$train$labels)

# sample 10k rows from training set, 1k rows from test set
set.seed(123, sample.kind = "Rounding")
index <- sample(nrow(mnist$train$images), 10000)
x <- mnist$train$images[index,]
y <- factor(mnist$train$labels[index])

index <- sample(nrow(mnist$test$images), 1000)
#note that the line above is the corrected code 
# - code in video at 0:52 is incorrect

x_test <- mnist$test$images[index,]
y_test <- factor(mnist$test$labels[index])

##
## Preprocessing MNIST Data
## 
# standardizing or transforming predictors and
# removing predictors that are not useful, 
# are highly correlated with others, 
# have very few non-unique values, 
# or have close to zero variation. 

# Code

library(matrixStats)
library(ggplot2)
sds <- colSds(x)
qplot(sds, bins = 256, color = I("black"))

library(caret)
nzv <- nearZeroVar(x)
image(matrix(1:784 %in% nzv, 28, 28))

col_index <- setdiff(1:ncol(x), nzv)
length(col_index)


## 
## Model Fitting for MNIST Data
##

# Key points:
#
# * The caret package requires that we add column names 
#   to the feature matrices.
#
# * In general, it is a good idea to test out a small subset 
#   of the data first to get an idea of how long your code 
#   will take to run.
# 
# Code

colnames(x) <- 1:ncol(mnist$train$images)
colnames(x_test) <- colnames(x)

# If we want to change how we perform cross validation, 
# we can use the trainControl function.  We can make 
# the code above go a bit faster by using, 
# for example, 10-fold cross validation.
# This means we have 10 samples using 10% 
# of the observations each. We accomplish this by
# using the following code:

control <- trainControl(method = "cv", number = 10, p = .9)
train_knn <- train(x[,col_index], y,
                   method = "knn", 
                   tuneGrid = data.frame(k = c(1,3,5,7)),
                   trControl = control)
ggplot(train_knn)

n <- 1000
b <- 2
index <- sample(nrow(x), n)
control <- trainControl(method = "cv", number = b, p = .9)
train_knn <- train(x[index ,col_index], y[index],
                   method = "knn",
                   tuneGrid = data.frame(k = c(3,5,7)),
                   trControl = control)
fit_knn <- knn3(x[ ,col_index], y,  k = 3)

y_hat_knn <- predict(fit_knn,
                     x_test[, col_index],
                     type="class")
cm <- confusionMatrix(y_hat_knn, factor(y_test))
cm$overall["Accuracy"]

cm$byClass[,1:2]

library(Rborist)
control <- trainControl(method="cv", number = 5, p = 0.8)
grid <- expand.grid(minNode = c(1,5) , predFixed = c(10, 15, 25, 35, 50))
# this takes a loooooong time to run!
train_rf <-  train(x[, col_index], y,
                   method = "Rborist",
                   nTree = 50,
                   trControl = control,
                   tuneGrid = grid,
                   nSamp = 5000)
ggplot(train_rf)
train_rf$bestTune

fit_rf <- Rborist(x[, col_index], y,
                  nTree = 1000,
                  minNode = train_rf$bestTune$minNode,
                  predFixed = train_rf$bestTune$predFixed)

y_hat_rf <- factor(levels(y)[predict(fit_rf, x_test[ ,col_index])$yPred])
cm <- confusionMatrix(y_hat_rf, y_test)
cm$overall["Accuracy"]

rafalib::mypar(3,4)
for(i in 1:12){
  image(matrix(x_test[i,], 28, 28)[, 28:1], 
        main = paste("Our prediction:", y_hat_rf[i]),
        xaxt="n", yaxt="n")
}


##
## Variable Importance
##

# Key points: 
# * The Rborist package does not currently support 
#   variable importance calculations, but 
#   the randomForest package does.
# * An important part of data science is visualizing results 
#   to determine why we are failing.
# 
# Code

library(randomForest)
x <- mnist$train$images[index,]
y <- factor(mnist$train$labels[index])
rf <- randomForest(x, y,  ntree = 50)
imp <- importance(rf)
imp

image(matrix(imp, 28, 28))

p_max <- predict(fit_knn, x_test[,col_index])
p_max <- apply(p_max, 1, max)
ind  <- which(y_hat_knn != y_test)
ind <- ind[order(p_max[ind], decreasing = TRUE)]
rafalib::mypar(3,4)
for(i in ind[1:12]){
  image(matrix(x_test[i,], 28, 28)[, 28:1],
        main = paste0("Pr(",y_hat_knn[i],")=",round(p_max[i], 2),
                      " but is a ",y_test[i]),
        xaxt="n", yaxt="n")
}

p_max <- predict(fit_rf, x_test[,col_index])$census  
p_max <- p_max / rowSums(p_max)
p_max <- apply(p_max, 1, max)
ind  <- which(y_hat_rf != y_test)
ind <- ind[order(p_max[ind], decreasing = TRUE)]
rafalib::mypar(3,4)
for(i in ind[1:12]){
  image(matrix(x_test[i,], 28, 28)[, 28:1], 
        main = paste0("Pr(",y_hat_rf[i],")=",round(p_max[i], 2),
                      " but is a ",y_test[i]),
        xaxt="n", yaxt="n")
}


##
## Ensembles
## 

# Key points
# * Ensembles combine multiple machine learning algorithms 
#   into one model to improve predictions.
# 
# Code

p_rf <- predict(fit_rf, x_test[,col_index])$census
p_rf <- p_rf / rowSums(p_rf)
p_knn <- predict(fit_knn, x_test[,col_index])
p <- (p_rf + p_knn)/2
y_pred <- factor(apply(p, 1, which.max)-1)
confusionMatrix(y_pred, y_test)


####################################
##                                ##
## Comprehension Check: Ensembles ##
##                                ##
####################################

# Q1
# Use the training set to build a model with several 
# of the models available from the caret package. 
# We will test out 10 of the most common 
# machine learning models in this exercise:
  
models <- c("glm", "lda", "naive_bayes", 
            "svmLinear", "knn", "gamLoess", 
            "multinom", "qda", "rf", "adaboost")

#Apply all of these models using train() 
# with all the default parameters. You may need 
# to install some packages. Keep in mind that you 
# will probably get some warnings. Also, it will 
# probably take a while to train all of the models - be patient!
  
#  Run the following code to train the various models:
  
library(caret)
library(dslabs)
# set.seed(1) # use `
set.seed(1, sample.kind = "Rounding") #` in R 3.6 or later
data("mnist_27")

fits <- lapply(models, function(model){ 
  print(model)
  train(y ~ ., method = model, data = mnist_27$train)
}) 

names(fits) <- models

# Did you train all of the models? (Yes / No)
# YES


# Q2
# Now that you have all the trained models in a list, 
# use sapply() or map() to create a matrix of predictions 
# for the test set. You should end up with a matrix 
# with length(mnist_27$test$y) rows and length(models) columns.
 
library(purrr)
library(tibble)

pred <- sapply(fits, function(x){
  y_hat <- predict(x, mnist_27$test)
  y_hat = y_hat
})

# What are the dimensions of the matrix of predictions?
#   
#   Number of rows:
#   200  
# 
# Number of columns:
#   10  

# Explanation from the course web site
# You can generate the matrix of predictions for the test set and get its dimensions using the following code:
  
pred <- sapply(fits, function(object)
  predict(object, newdata = mnist_27$test))
dim(pred)


# Q3
# Now compute accuracy for each model on the test set.

fx_acc <- function(x){
  cm <- confusionMatrix(data = factor(x), 
                        factor(mnist_27$test$y))
  cm$overall[["Accuracy"]]
}

accuracy <- rep(NULL, 10)

for (i in 1:10) {
  acc <- fx_acc(pred[,i])
  accuracy[i] <- acc
}

# Report the mean accuracy across all models.
mean(accuracy)
# [1] 0.787

# Explanation from the course web site
# Accuracy for each model in the test set 
# and the mean accuracy across all models 
# can be computed using the following code:
  
acc <- colMeans(pred == mnist_27$test$y)
acc
mean(acc)

# Q4
# Next, build an ensemble prediction by majority vote 
# and compute the accuracy of the ensemble. 
# Vote 7 if more than 50% of the models 
# are predicting a 7, and 2 otherwise.

y_hat <- ifelse(rowMeans(pred == 7) > 0.5, 7, 2)
mean(y_hat == mnist_27$test$y)
 
# What is the accuracy of the ensemble?
mean(y_hat == mnist_27$test$y)
# [1] 0.81

# Explanation from the course web site
# The ensemble prediction can be built using the following code:
  
votes <- rowMeans(pred == "7")
y_hat <- ifelse(votes > 0.5, "7", "2")
mean(y_hat == mnist_27$test$y)

# Q5
# In Q3, we computed the accuracy of each method 
# on the test set and noticed that 
# the individual accuracies varied.

ensemble <- mean(y_hat == mnist_27$test$y)

# How many of the individual methods do better than the ensemble?
sum(acc > ensemble)
# [1] 3

# Which individual methods perform better than the ensemble?
models[acc > ensemble]
# [1] "knn"      "gamLoess" "qda"


# Explanation from the course web site
# The comparison of the individual methods 
# to the ensemble can be done using the following code:
  
ind <- acc > mean(y_hat == mnist_27$test$y)
sum(ind)
models[ind]

# Q6
# It is tempting to remove the methods 
# that do not perform well and re-do the ensemble. 
# The problem with this approach is that we are using 
# the test data to make a decision. However, we could use 
# the minimum accuracy estimates obtained 
# from cross validation with the training data for each model. 
# Obtain these estimates and save them in an object. 
# Report the mean of these training set accuracy estimates.

mins <- rep(NULL, 10)
for (i in models) {
  print(i)
  mins[i] <- min(fits[[i]][[4]]$Accuracy)
}

# What is the mean of these training set accuracy estimates?

mean(mins)
# [1] 0.8085677

# Explanation from the course web site
# You can calculate the mean accuracy of 
# the new estimates using the following code:
  
acc_hat <- sapply(fits, function(fit) min(fit$results$Accuracy))
mean(acc_hat)

# from the discussion forum:
# You don't have to compare anything for Q6. 
# The values you extract from the fits (fit$results) 
# are accuracies. Please calculate and submit the mean 
# of these accuracies.
# ...
# to get the answer right, by pulling the values 
# 



# Q7
# Now let's only consider the methods 
# with an estimated accuracy of greater than or equal to 0.8 
# when constructing the ensemble.

models[acc >= 0.8]

mean(colMeans(pred[, models[acc >= 0.8]]  == mnist_27$test$y))

# What is the accuracy of the ensemble now?


# .835  # correct answer: 0.825

# Explanation from the course web site
# The new ensemble prediction can be built using 
# the following code:
    
ind <- acc_hat >= 0.8
votes <- rowMeans(pred[, ind] == "7")
y_hat <- ifelse(votes >= 0.5, 7, 2)
mean(y_hat == mnist_27$test$y)
# [1] 0.825


##
## Comprehension Check: Dimension Reduction
## 

# The first principal component (PC) of a matrix X is 
# the linear orthogonal transformation of X 
# that maximizes this variability. The function prcomp 
# provides this info:
pca <- prcomp(x)
pca$rotation
#> PC1 PC2
#> [1,] -0.702 0.712
#> [2,] -0.712 -0.702


# Q1
# We want to explore the tissue_gene_expression predictors 
# by plotting them.

library(dslabs)
data("tissue_gene_expression")
dim(tissue_gene_expression$x)
length(tissue_gene_expression$y)


# We want to get an idea of which observations are close 
# to each other, but, as you can see from the dimensions, 
# the predictors are 500-dimensional, making plotting difficult. 
# Plot the first two principal components with color 
# representing tissue type.

library(ggplot2)
library(dplyr)
data.frame(tissue = tissue_gene_expression$y, 
              MAML1 = tissue_gene_expression$x[, "MAML1"], 
              LHPP = tissue_gene_expression$x[, "LHPP"]) %>% 
  ggplot(aes(MAML1, LHPP, color = tissue)) +
  geom_point()

x <- tissue_gene_expression$x
pc <- prcomp(x)
data.frame(pc_1 = pc$x[,1], pc_2 = pc$x[,2], 
           tissue = tissue_gene_expression$y) %>%
  ggplot(aes(pc_1, pc_2, color = tissue)) +
  geom_point()

# 
# Which tissue is in a cluster by itself?
# hippocampus ## NO
# cerebellum ## no
# colon #### NO
# liver

# Explanation from the course web site
# The plot can be made using the following code:
  
pc <- prcomp(tissue_gene_expression$x)
data.frame(pc_1 = pc$x[, 1],
           pc_2 = pc$x[, 2],
           tissue = tissue_gene_expression$y) %>%
  ggplot(aes(pc_1, pc_2, color = tissue)) +
  geom_point()

# We can see that liver clusters alone in the lower right-hand corner of the plot.


# Q2
# The predictors for each observation are measured 
# using the same device and experimental procedure. 
# This introduces biases that can affect all the predictors 
# from one observation. 

# For each observation, compute the average across 
# all predictors, and then plot this against 
# the first PC with color representing tissue. 

x_1 = tissue_gene_expression$x[,1]
x_avg = rowMeans(tissue_gene_expression$x)
tissue = tissue_gene_expression$y
x.df <- data.frame(x_1, x_avg)

x <- tissue_gene_expression$x
pc <- prcomp(x)
pc
pc_avg = rowMeans(pc$x)
x_avg = rowMeans(tissue_gene_expression$x)
pc_1 = pc$x[,1]
tissue = tissue_gene_expression$y
data.frame(x_avg, pc_1, tissue) %>%
  ggplot(aes(pc_1, x_avg, color = tissue)) +
  geom_point() 

# Report the correlation.
cor(x_avg, pc_1)
# [1] 0.5969088 # YES!!!  TYJ!!!!!!!!!!

# What is the correlation?
# [1] 0.5969088

# Explanation from the course web site
# The plot and correlation can be generated using 
# the following code:
  
avgs <- rowMeans(tissue_gene_expression$x)
data.frame(pc_1 = pc$x[, 1],
           avg = avgs,
           tissue = tissue_gene_expression$y) %>%
  ggplot(aes(avgs, pc_1, color = tissue)) +
  geom_point()
cor(avgs, pc$x[, 1])



# Q3
# We see an association with the first PC 
# and the observation averages. Redo the PCA 
# but only after removing the center. 
# Part of the code is provided for you.

#BLANK
# option #4 is correct
x <- with(tissue_gene_expression, sweep(x, 1, rowMeans(x)))
pc <- prcomp(x)
data.frame(pc_1 = pc$x[,1], pc_2 = pc$x[,2], 
           tissue = tissue_gene_expression$y) %>%
  ggplot(aes(pc_1, pc_2, color = tissue)) +
  geom_point()

# Which line of code should be used to replace #BLANK in the code block above?


# Q4
# For the first 10 PCs, make a boxplot showing 
# the values for each tissue.

boxplot(pc$x[, 1:10]) 

data.frame(tissue, pc$x[,7]) %>%
  ggplot(aes(tissue, x[,1])) +
  geom_boxplot() 

data.frame(tissue, pc$x[,7]) %>%
  group_by(tissue) %>% 
  summary()
  

# For the 7th PC, which two tissues have 
# the greatest median difference?


#
# Select the TWO tissues that have 
# the greatest median difference.
# cerebellum # is correct
# endometrium # wrong
# hippocampus # wrong
# colon # wrong
# placenta is corrrect

# Explanation
# The boxplots for the first 10 PCs can be made using this code:
  
  for(i in 1:10){
    boxplot(pc$x[,i] ~ tissue_gene_expression$y, main = paste("PC", i))
  }


# Q5
# Plot the percent variance explained by PC number. 
# Hint: use the summary function.

summary(pc)$importance 

plot(summary(pc)$importance[3,])

# How many PCs are required to reach 
# a cumulative percent variance explained greater than 50%?
#   
#   3  

# Explanation
# The plot can be made using the following code: 
plot(summary(pc)$importance[3,])










































