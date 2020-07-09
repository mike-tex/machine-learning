## Breast Cancer Project Part 1

# The brca dataset from the dslabs package contains information 
# about breast cancer diagnosis biopsy samples for tumors 
# that were determined to be either benign (not cancer) 
# and malignant (cancer). The brca object is a list consisting of:
#   
# * brca$y: a vector of sample classifications ("B" = benign or "M" = malignant)
# * brca$x: a matrix of numeric features describing properties of the shape and size of cell nuclei extracted from biopsy microscope images
# 
# For these exercises, load the data by setting your options 
# and loading the libraries and data as shown in the code here:
  
options(digits = 3)
library(matrixStats)
library(tidyverse)
library(caret)
library(dslabs)
data(brca)

# The exercises in this assessment are available to 
# Verified Learners only and are split into four parts, 
# all of which use the data described here.
# 
# IMPORTANT: Some of these exercises use dslabs datasets 
# that were added in a July 2019 update. Make sure your package 
# is up to date with the command install.packages("dslabs").

??brca

# Question 1: Dimensions and properties
# How many samples are in the dataset?
dim(brca$x)  
569

# How many predictors are in the matrix?
30

# What proportion of the samples are malignant?
mean(brca$y == "M")
# [1] 0.373
# 
# Which column number has the highest mean?
which.max(colMeans(brca$x))
# area_worst 
# 24
# 
# Which column number has the lowest standard deviation?
which.min(colSds(brca$x))
# [1] 20

# Question 2: Scaling the matrix
# Use sweep() two times to scale each column: subtract 
# the column mean, then divide by the column standard deviation.

s1 <- sweep(brca$x, 2, colMeans(brca$x), "-")
s2 <- sweep(s1, 2, colSds(s1), "/")

# After scaling, what is the standard deviation 
# of the first column?
sd(s2[,1]) 
# [1] 1

# Explanation from the course website
# The standard deviation can be found using the following code:
  
x_centered <- sweep(brca$x, 2, colMeans(brca$x))
x_scaled <- sweep(x_centered, 2, colSds(brca$x), FUN = "/")

sd(x_scaled[,1])

# After scaling, what is the median value of the first column?
median(s2[,1])

# Explanation
# The median value can be found using the following code:
median(x_scaled[,1])


# Question 3: Distance
# Calculate the distance between all samples 
# using the scaled matrix.

x_centered <- sweep(brca$x, 2, colMeans(brca$x))
x_scaled <- sweep(x_centered, 2, colSds(brca$x), FUN = "/")

d <- as.matrix(dist(x_scaled))
dim(as.matrix(d))



# What is the average distance between the first sample, 
# which is benign, and other benign samples?
# z <- x[,1]
# sd(dist(x) - dist(z)*sqrt(2))
idx <- which(brca$y == "B")

mean(d[1, idx[-1]])
# [1] 4.41

# Explanation
# The average distance can be found using the following code:
  
d_samples <- dist(x_scaled)
dist_BtoB <- as.matrix(d_samples)[1, brca$y == "B"]
mean(dist_BtoB[2:length(dist_BtoB)])


# What is the average distance between the first sample 
# and malignant samples?

mean(d[1,-idx])
# [1] 7.12

# Explanation
# The average distance can be found using the following code:
  
dist_BtoM <- as.matrix(d_samples)[1, brca$y == "M"]
mean(dist_BtoM)


# Question 4: Heatmap of features
# Make a heatmap of the relationship between features 
# using the scaled matrix.

# Which of these heatmaps is correct?
# To remove column and row labels like the images below, 
# use labRow = NA and labCol = NA.

heatmap(as.matrix(dist(t(x_scaled))), labRow = NA, labCol = NA)

# Explanation
# The correct heatmap can be generated using the following code:
  
d_features <- dist(t(x_scaled))
heatmap(as.matrix(d_features), labRow = NA, labCol = NA)


# Question 5: Hierarchical clustering
# Perform hierarchical clustering on the 30 features. Cut the tree into 5 groups.

plot(hclust(dist(t(x_scaled))))

# All but one of the answer options are in the same group.
# 
# Which is in a different group?
# concavity_mean


# Explanation
# The hierarchical clustering can be done using the following code:
  
h <- hclust(d_features)
groups <- cutree(h, k = 5)
split(names(groups), groups)



##
## Breast Cancer Project Part 2
##

# Question 6: PCA: proportion of variance
# Perform a principal component analysis of the scaled matrix.

pca <- prcomp(x_scaled)

# What proportion of variance is explained by the first principal component?
   
summary(pca)
# 0.443

# Explanation
# The proportion of variance explained can be determined 
# using the following code:
pca <- prcomp(x_scaled)
summary(pca)    # see PC1 Cumulative Proportion

# How many principal components are required to explain at least 90% of the variance?

# Explanation
# The number of principal components can be determined 
# using the following code:
pca <- prcomp(x_scaled)
summary(pca)     # first value of Cumulative Proportion that exceeds 0.9: PC7


# Question 7: PCA: plotting PCs
# Plot the first two principal components with color 
# representing tumor type (benign/malignant).

data.frame(pca$x[,1:2], tumor = brca$y) %>%
  ggplot(aes(PC1, PC2, fill = tumor))+
  geom_point(cex = 3, pch = 21) +
  coord_fixed(ratio = 1)

# Which of the following is true?

# Malignant tumors tend to have larger values of PC1 
# than benign tumors.

# Explanation
# The plot can be made using the following code:
  
data.frame(pca$x[,1:2], type = brca$y) %>%
  ggplot(aes(PC1, PC2, color = type)) +
  geom_point() + 
  geom_boxplot()

# From the plot, you can see that the benign tumors 
# tend to have smaller values of PC1 and that 
# the malignant tumors have larger values of PC1.
# PC2 values have a similar spread for both benign 
# and malignant tumors.


# Question 8: PCA: PC boxplot
# Make a boxplot of the first 10 PCs grouped by tumor type.

df <- data.frame(pca$x[,1:10], tumor = brca$y)

library(ggplot2)
p1 <-   ggplot(data = df) +
  geom_boxplot(aes(tumor, df[, 1])) 

p2 <-   ggplot(data = df) +
  geom_boxplot(aes(tumor, df[, 2])) 

p3 <-   ggplot(data = df) +
  geom_boxplot(aes(tumor, df[, 3])) 

p4 <-   ggplot(data = df) +
  geom_boxplot(aes(tumor, df[, 4])) 

p5 <-   ggplot(data = df) +
  geom_boxplot(aes(tumor, df[, 5])) 

p6 <-   ggplot(data = df) +
  geom_boxplot(aes(tumor, df[, 6])) 

p7 <-   ggplot(data = df) +
  geom_boxplot(aes(tumor, df[, 7])) 

p8 <-   ggplot(data = df) +
  geom_boxplot(aes(tumor, df[, 8])) 

p9 <-   ggplot(data = df) +
  geom_boxplot(aes(tumor, df[, 9])) 

p10 <-   ggplot(data = df) +
  geom_boxplot(aes(tumor, df[, 10])) 

library(gridExtra)

plot_grid(ps)
grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, ncol = 5)


# Which PCs are significantly different enough 
# by tumor type that there is no overlap in 
# the interquartile ranges (IQRs) for benign 
# and malignant samples?
#   Select ALL that apply.
# PC1

# Explanation
# The boxplot can be generated using the following code:
  
data.frame(type = brca$y, pca$x[, 1:10]) %>%
  gather(key = "PC", value = "value",-type) %>%
  ggplot(aes(PC, value, fill = type)) +
  geom_boxplot()

# When you look at the boxplot, you can see that the IQRs 
# overlap for PCs 2 through 10 but not for PC1.


# Breast Cancer Project Part 3

options(digits = 3)
library(matrixStats)
library(tidyverse)
library(caret)
library(dslabs)
data(brca)

x_centered <- sweep(brca$x, 2, colMeans(brca$x))
x_scaled <- sweep(x_centered, 2, colSds(brca$x), FUN = "/")

# Set the seed to 1, then create a data partition 
# splitting brca$y and the scaled version of the brca$x matrix 
# into a 20% test set and 80% train using the following code:

# set.seed(1) # if using R 3.5 or earlier
set.seed(1, sample.kind = "Rounding")    # if using R 3.6 or later
test_index <- createDataPartition(brca$y, times = 1, p = 0.2, list = FALSE)
test_x <- x_scaled[test_index,]
test_y <- brca$y[test_index]
train_x <- x_scaled[-test_index,]
train_y <- brca$y[-test_index]

# You will be using these training and test sets throughout 
# the exercises in Parts 3 and 4. Save your models as you go, 
# because at the end, you'll be asked to make 
# an ensemble prediction and to compare the accuracy 
# of the various models!


# Question 9: Training and test sets
# Check that the training and test sets have similar 
# proportions of benign and malignant tumors.

# What proportion of the training set is benign?
#   
mean(train_y == "B")  
# [1] 0.628
# 
# What proportion of the test set is benign?
mean(test_y == "B")
# [1] 0.626  


# Question 10a: K-means Clustering
# The predict_kmeans() function defined here 
# takes two arguments - a matrix of observations x 
# and a k-means object k - and assigns 
# each row of x to a cluster from k.

predict_kmeans <- function(x, k) {
  centers <- k$centers    # extract cluster centers
  # calculate distance to cluster centers
  distances <- sapply(1:nrow(x), function(i){
    apply(centers, 1, function(y) dist(rbind(x[i,], y)))
  })
  max.col(-t(distances))  # select cluster with min distance to center
}

# Set the seed to 3. Perform k-means clustering 
# on the training set with 2 centers and assign 
# the output to k. Then use the predict_kmeans() function 
# to make predictions on the test set.

set.seed(3, sample.kind = "Rounding")
k <- kmeans(train_x, centers = 2)
pk <- predict_kmeans(test_x, k)

pkf[1:length(pk)] <- "B"
pkf[pk == 2] <- "M"
pkf
levels(pkf) <- levels(test_y)
levels(pkf)
factor(pkf)

# What is the overall accuracy?
mean(pkf == test_y) 
# [1] 0.922
#   unanswered  

# Question 10b: K-means Clustering
# What proportion of benign tumors are correctly identified?
bt_idx <- which(test_y == "B")
mean(pkf[bt_idx] == test_y[bt_idx])
# [1] 0.986  
# 
# What proportion of malignant tumors are correctly identified?
mean(pkf[-bt_idx] == test_y[-bt_idx])
# [1] 0.814
#   
#   unanswered  


# Question 11: Logistic regression model
# Fit a logistic regression model on the training set 
# using all predictors. Ignore warnings about the algorithm 
# not converging. Make predictions on the test set.

# glm_fit <- train_set %>%
#   mutate(y = as.numeric(sex == "Female")) %>%
#   glm(y ~ height, data=., family = "binomial")
# # We can obtain prediction using the predict function:
# p_hat_logit <- predict(glm_fit, 
#                newdata = test_set, type = "response")

glm_fit <- train(x = train_x, y = train_y, method = "glm")
p_hat_glm <- predict(glm_fit, newdata = test_x, 
                     type = "raw")

glm_fit$results$Accuracy # Do not use, wrong answer

# What is the accuracy of the logistic regression model 
# on the test set?
mean(p_hat_glm == test_y) 
# [1] 0.957 or 0.939

# Explanation
# The accuracy of the logistic regression model 
# can be calculated using the following code:
train_glm <- train(train_x, train_y,
                   method = "glm")
glm_preds <- predict(train_glm, test_x)
mean(glm_preds == test_y)


# Question 12: LDA and QDA models
# Train an LDA model and a QDA model on the training set. 
# Make predictions on the test set using each model.

lda_fit <- train(train_x, train_y,
                 method = "lda")
p_hat_lda <- predict(lda_fit, newdata = test_x)
qda_fit <- train(train_x, train_y,
                 method = "qda")
p_hat_qda <- predict(qda_fit, newdata = test_x)


# What is the accuracy of the LDA model on the test set?
mean(p_hat_lda == test_y) 
# [1] 0.991 or 0.974

# Explanation
# The accuracy can be determined using the following code:
train_lda <- train(train_x, train_y,
                   method = "lda")
lda_preds <- predict(train_lda, test_x)
mean(lda_preds == test_y)


# What is the accuracy of the QDA model on the test set?
mean(p_hat_qda == test_y) 
# [1] 0.957 or 0.948

# Explanation
# The accuracy can be determined using the following code:
train_qda <- train(train_x, train_y,
                   method = "qda")
qda_preds <- predict(train_qda, test_x)
mean(qda_preds == test_y)


# Question 13: Loess model
# Set the seed to 5, then fit a loess model 
# on the training set with the caret package. 
# You will need to install the gam package 
# if you have not yet done so. Use the default tuning grid. 
# This may take several minutes; ignore warnings. 
# Generate predictions on the test set.
library(caret)
library(gam)

set.seed(5, sample.kind = "Rounding")
loess_fit <- train(train_x, train_y,
                 method = "gamLoess")
p_hat_loess <- predict(loess_fit, newdata = test_x)

# What is the accuracy of the loess model on the test set?
mean(p_hat_loess == test_y) 
# [1] 0.983 or 0.93


# Explanation
# The accuracy can be determined using the following code:
# set.seed(5)
set.seed(5, sample.kind = "Rounding")    # simulate R 3.5
train_loess <- train(train_x, train_y,
                     method = "gamLoess")
loess_preds <- predict(train_loess, test_x)
mean(loess_preds == test_y)



##
## Breast Cancer Project Part 4
##

# Question 14: K-nearest neighbors model
# Set the seed to 7, then train a k-nearest neighbors model 
# on the training set using the caret package. 
# Try odd values of  k  from 3 to 21 
# (use tuneGrid with seq(3, 21, 2)). 
# Use the final model to generate predictions on the test set.

modelLookup("knn")

set.seed(7, sample.kind = "Rounding")
grid <- expand.grid(k = seq(3, 21, 2))
knn_fit <- train(train_x, train_y,
                   method = "knn",
                   tuneGrid = grid)
p_hat_knn <- predict(knn_fit, newdata = test_x)

# What is the final value of  k  used in the model?
knn_fit$results
# 21 or 9

# xplanation
# The value of  k  can be determined using the following code:
#  set.seed(7)
set.seed(7, sample.kind = "Rounding")    # simulate R 3.5
tuning <- data.frame(k = seq(3, 21, 2))
train_knn <- train(train_x, train_y,
                   method = "knn", 
                   tuneGrid = tuning)
train_knn$bestTune

# What is the accuracy of the kNN model on the test set?
mean(p_hat_knn == test_y) 
# [1] 0.948 or 0.974
# 
# Explanation
# The accuracy can be determined using the following code:
knn_preds <- predict(train_knn, test_x)
mean(knn_preds == test_y)



## Question 15a: Random forest model
# Set the seed to 9, then train a random forest model 
# on the training set using the caret package. 
# Test mtry values of 3, 5, 7 and 9. Use the argument 
# importance = TRUE so that feature importance 
# can be extracted. Generate predictions on the test set.

library(randomForest)

modelLookup("rf")

grid <- expand.grid(mtry = seq(3, 9, 2))
set.seed(9, sample.kind = "Rounding")
train_rf <- train(train_x, train_y,
                 method = "rf",
                 tuneGrid = grid)

fit_rf <- randomForest(train_x, train_y, 
                       minNode = train_rf$bestTune$mtry, 
                       importance=TRUE)

p_hat_rf <- predict(train_rf, newdata = test_x)
y_hat_rf <- predict(fit_rf, newdata = test_x)
cm <- confusionMatrix(p_hat_rf, test_y) # use train_rf
# cm <- confusionMatrix(y_hat_rf, test_y) # don't use fit_rf
cm$overall["Accuracy"]
# Accuracy 
# 0.974 # correct # cm <- confusionMatrix(p_hat_rf, test_y)



# What value of mtry gives the highest accuracy?
train_rf$bestTune$mtry
#     mtry
# 1    3   ### correct

# What is the accuracy of the random forest model 
# on the test set?
cm <- confusionMatrix(p_hat_rf, test_y)
cm$overall["Accuracy"]
# Accuracy 
# 0.974

mean(p_hat_rf == test_y) # use train_rf
# 0.974 # cm <- confusionMatrix(p_hat_rf, test_y)


# What is the most important variable 
# in the random forest model?
importance(fit_rf)
varImp(fit_rf) %>% arrange(desc(B))
# concave_pts_worst
#


# Question 15b: Random forest model
# Consider the top 10 most important variables 
# in the random forest model.

# Which set of features is most important 
# for determining tumor type?

importance(fit_rf) %>% 
  data.frame() %>% 
  arrange(desc(B))

fit_rf$importance



# Question 16a: Creating an ensemble
# Create an ensemble using the predictions 
# from the 7 models created in the previous exercises: 
# k-means, logistic regression, LDA, QDA, loess, 
# k-nearest neighbors, and random forest. 
# Use the ensemble to generate a majority prediction 
# of the tumor type (if most models suggest the tumor 
# is malignant, predict malignant).

models <- c(fit_rf, glm_fit, k, lda_fit, 
          qda_fit, loess_fit, knn_fit)


pred <- sapply(models, function(x){
  x <- 
  y_hat <- predict(x, newdata = test_x)
  y_hat = y_hat
})

rm(pred_y_hat)
pred_y_hat <- p_hat_rf
pred_y_hat <- cbind(pred_y_hat, p_hat_glm)
pred_y_hat <- cbind(pred_y_hat, pk)
pred_y_hat <- cbind(pred_y_hat, p_hat_lda)
pred_y_hat <- cbind(pred_y_hat, p_hat_qda)
pred_y_hat <- cbind(pred_y_hat, p_hat_loess)
pred_y_hat <- cbind(pred_y_hat, p_hat_knn)

dim(pred_y_hat)

y_hat_prime <- ifelse(rowMeans(pred_y_hat) > 1.5, 2, 1)
mean(y_hat_prime == as.numeric(test_y))


# What is the accuracy of the ensemble prediction?
y_hat_prime <- ifelse(rowMeans(pred_y_hat) > 1.5, 2, 1)
mean(y_hat_prime == as.numeric(test_y))
# [1] 0.983

# Question 16b: Creating an ensemble
# Make a table of the accuracies of the 7 models 
# and the accuracy of the ensemble model.

q16b <- cbind(pred_y_hat, y_hat_prime)
colMeans(q16b == as.numeric(test_y)) 

# Which of these models has the highest accuracy?
# lda
























