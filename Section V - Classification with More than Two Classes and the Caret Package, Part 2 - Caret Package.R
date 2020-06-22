## Section 5: Classification with More than Two Classes
## and the Caret Package
## 5.2: Caret Package
##   Caret Package

# Caret package links
# http://topepo.github.io/caret/available-models.html
# http://topepo.github.io/caret/train-models-by-tag.html

cat("\014")

library(tidyverse)
library(dslabs)
data("mnist_27")

library(caret)

train_glm <- train(y ~ .,
                   method = "glm",
                   data = mnist_27$train)
train_knn <- train(y ~ .,
                   method = "knn",
                   data = mnist_27$train)

y_hat_glm <- predict(train_glm,
                     mnist_27$test, type = "raw")
y_hat_knn <- predict(train_knn,
                     mnist_27$test, type = "raw")

confusionMatrix(y_hat_glm,
                mnist_27$test$y)$overall[["Accuracy"]]
confusionMatrix(y_hat_knn,
                mnist_27$test$y)$overall[["Accuracy"]]


##
## Tuning Parameters with Caret
##

getModelInfo("knn")
modelLookup("knn")

train_knn <- train(y ~ ., method = "knn",
                   data = mnist_27$train)
ggplot(train_knn, highlight = TRUE)

train_knn <- train(
  y ~ .,
  method = "knn",
  data = mnist_27$train,
  tuneGrid = data.frame(k = seq(9, 71, 2))
)
ggplot(train_knn, highlight = TRUE)
train_knn$bestTune
train_knn$finalModel
confusionMatrix(predict(train_knn, 
                        mnist_27$test, type = "raw"),
                mnist_27$test$y)$overall["Accuracy"]

control <- trainControl(method = "cv", number = 10, p = .9)
train_knn_cv <- train(
  y ~ .,
  method = "knn",
  data = mnist_27$train,
  tuneGrid = data.frame(k = seq(9, 71, 2)),
  trControl = control
)
ggplot(train_knn_cv, highlight = TRUE)

train_knn$results %>%
  ggplot(aes(x = k, y = Accuracy)) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(
    x = k,
    ymin = Accuracy - AccuracySD,
    ymax = Accuracy + AccuracySD
  ))

plot_cond_prob <- function(p_hat = NULL) {
  tmp <- mnist_27$true_p
  if (!is.null(p_hat)) {
    tmp <- mutate(tmp, p = p_hat)
  }
  tmp %>% ggplot(aes(x_1, x_2, z = p, fill = p)) +
    geom_raster(show.legend = FALSE) +
    scale_fill_gradientn(colors = 
                           c("#F8766D", "white", "#00BFC4")) +
    stat_contour(breaks = c(0.5), color = "black")
}

plot_cond_prob(predict(train_knn, 
                       mnist_27$true_p, 
                       type = "prob")[, 2])

install.packages("gam")
modelLookup("gamLoess")

grid <- expand.grid(span = seq(0.15, 0.65, len = 10), 
                    degree = 1)

train_loess <- train(y ~ .,
                     method = "gamLoess",
                     tuneGrid = grid,
                     data = mnist_27$train)

getModelInfo(train_loess)

ggplot(train_loess, highlight = TRUE)

confusionMatrix(data = predict(train_loess, 
                               mnist_27$test),
                reference = 
                  mnist_27$test$y)$overall["Accuracy"]

p1 <- plot_cond_prob(predict(train_loess,
                             mnist_27$true_p,
                             type = "prob")[, 2])
p1


##
## Comprehension Check: Caret Package
##

# These exercises take you through an analysis 
# using the tissue_gene_expression dataset.

# Q1
# Load the rpart package and then use the caret::train() 
# function with method = "rpart" to fit a classification tree 
# to the tissue_gene_expression dataset. Try out cp values 
# of seq(0, 0.1, 0.01). Plot the accuracies to report 
# the results of the best model. Set the seed to 1991.
 
library(tidyverse)
library(caret)
library(rpart)
library(dslabs)

# getModelInfo("rpart")
modelLookup("rpart")$parameter
# "cp"

data(tissue_gene_expression)
y <- tissue_gene_expression$y
x <- tissue_gene_expression$x

set.seed(1991, sample.kind = "Rounding")

train_rpart <-
  train(x, y,
        method = "rpart",
        # data = tissue_gene_expression,
        tuneGrid = data.frame(cp = seq(0, 0.1, 0.01)))
ggplot(train_rpart, highlight = T)   

train_rpart$bestTune
#   cp
# 1  0

# Which value of cp gives the highest accuracy?
  
#  0

# Explanation from web site

# The following code can be used to do generate 
# the plot and get the value of cp:
  
library(caret)
library(rpart)          
library(dslabs)
library(dplyr)

set.seed(1991, sample.kind = "Rounding")
data("tissue_gene_expression")

fit <- with(tissue_gene_expression, 
            train(x, y, method = "rpart",
                  tuneGrid = data.frame(cp = seq(0, 0.1, 0.01))))

ggplot(fit)

# Q2
# Note that there are only 6 placentas in the dataset. 
# By default, rpart() requires 20 observations 
# before splitting a node. That means that it is difficult 
# to have a node in which placentas are the majority. 
# Rerun the analysis you did in the exercise in Q1, 
# but this time, allow rpart() to split any node by using 
# the argument control = rpart.control(minsplit = 0). 
# Look at the confusion matrix again to determine whether 
# the accuracy increases. Again, set the seed to 1991.

rm(list = ls())
data("tissue_gene_expression")
set.seed(1991, sample.kind = "Rounding")
fit_rpart <- with(tissue_gene_expression, 
            train(x, y, method = "rpart",
                  control = rpart.control(minsplit = 0),
                  tuneGrid = data.frame(cp = seq(0, 0.1, 0.01))))

ggplot(fit_rpart, highlight = T)   
fit_rpart$bestTune
fit_rpart

# cp    Accuracy   Kappa    
# 0.00  0.9147869  0.8966977
# 0.01  0.9097380  0.8906158
# 0.02  0.9024233  0.8816999
# 0.03  0.8947150  0.8723923
# 0.04  0.8894580  0.8659688
# 0.05  0.8827699  0.8576901
# 0.06  0.8760811  0.8494140
# 0.07  0.8615446  0.8310576
# 0.08  0.8615446  0.8310576
# 0.09  0.8500201  0.8165422
# 0.10  0.8470350  0.8127472

fit_rpart[[4]]$Accuracy %>% max()
# [1] 0.9147869


# What is the accuracy now?
  
# 0.9147869  ## YES!!!!
# Answer:  0.9141

## Explanation from the web site
# The following code can be used to re-run the analysis 
# and view the confusion matrix:
  
library(rpart)
data("tissue_gene_expression")
set.seed(1991, sample.kind = "Rounding")
fit_rpart <- with(tissue_gene_expression, 
          train(x, y, method = "rpart",
                tuneGrid = data.frame(cp = seq(0, 0.10, 0.01)),
                control = rpart.control(minsplit = 0)))
ggplot(fit_rpart)
confusionMatrix(fit_rpart)

#  Accuracy (average) : 0.9141


# Q3
# Plot the tree from the best fitting model 
# of the analysis you ran in Q2.
# 
library(rpart)
data("tissue_gene_expression")
set.seed(1991, sample.kind = "Rounding")
fit_rpart <- with(tissue_gene_expression, rpart(y ~ x))

plot(fit_rpart, margin = 0.1)
text(fit_rpart, cex = 0.75)


# Which gene is at the first split?

# GPA33

# Explanation from the web site
# The first split is at GPA33 >= 8.794. 
# The following code will give the tree:
  
plot(fit_rpart$finalModel)
text(fit_rpart$finalModel)


## Q4
# We can see that with just seven genes, we are able 
# to predict the tissue type. Now let's see if we can predict 
# the tissue type with even fewer genes using a Random Forest. 
# Use the train() function and the rf method to train 
# a Random Forest model and save it to an object called fit. 
# Try out values of mtry ranging from seq(50, 200, 25) 
# (you can also explore other values on your own). 
# What mtry value maximizes accuracy? To permit small nodesize 
# to grow as we did with the classification trees, use 
# the following argument: nodesize = 1.
# 
# Note: This exercise will take some time to run. 
# If you want to test out your code first, try using smaller 
# values with ntree. Set the seed to 1991 again.
# 
# What value of mtry maximizes accuracy?  
# 100 or 75 from looking at discussion board?

cat("\014") # clear console
rm(list = ls()) # clear environment

library(randomForest)
library(caret)
library(dslabs)

modelLookup("rf")$parameter
# [1] "mtry"
data("tissue_gene_expression")
set.seed(1991, sample.kind = "Rounding")
fit <- with(tissue_gene_expression, 
          train(x, y, method = "rf",
                tuneGrid = data.frame(mtry = seq(50, 200, 25)),
               nodesize = 1))
confusionMatrix(fit)
fit$bestTune$mtry


## Explanation

# The following code can be used to do the analysis:
  
set.seed(1991)
library(randomForest)
fit <- with(tissue_gene_expression, 
            train(x, y, method = "rf", 
                  nodesize = 1,
                  tuneGrid = data.frame(mtry = seq(50, 200, 25))))

ggplot(fit)


## Q5
# Use the function varImp() on the output of train() 
# and save it to an object called imp:
  
  imp <- varImp(fit) #BLANK
  imp

# What should replace #BLANK in the code above?


# Q6
# The rpart() model we ran above produced a tree 
# that used just seven predictors. Extracting 
# the predictor names is not straightforward, but can be done. 
# If the output of the call to train was fit_rpart, 
# we can extract the names like this:
    
tree_terms <- as.character(unique(fit_rpart$finalModel$frame$var[!(fit_rpart$finalModel$frame$var == "<leaf>")]))
tree_terms
  
# Calculate the variable importance in 
# the Random Forest call from Q4 for these seven predictors 
# and examine where they rank.

y <- tissue_gene_expression$y
x <- tissue_gene_expression$x[,terms]

library(randomForest)

set.seed(1991)
fit <-train(x, y, method = "rf", 
                  nodesize = 1,
                  tuneGrid = data.frame(mtry = seq(50, 200, 25)))


imp7 <- varImp(fit, data = tree_terms)
imp7
  
# What is the importance of the CFHR4 gene in the Random Forest call?
  #   Enter a number.
  # 35.03  
  
# What is the rank of the CFHR4 gene in the Random Forest call?
  #   Enter a number.
  # 7  


# Explanation
# The following code can be used to calculate 
# the rank and importance in the Random Forest call 
# for the predictors from the rpart() model:
  
data_frame(term = rownames(imp$importance), 
           importance = imp$importance$Overall) %>%
mutate(rank = rank(-importance)) %>% arrange(desc(importance)) %>%
filter(term %in% tree_terms)














