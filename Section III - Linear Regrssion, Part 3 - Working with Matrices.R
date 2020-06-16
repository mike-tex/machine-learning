## Section 3: Linear Regression for
## Prediction, Smoothing, and Working with Matrices
##   3.3: Working with Matrices
##   Matrices

library(tidyverse)
library(dslabs)
if(!exists("mnist")) mnist <- read_mnist()

class(mnist$train$images)

x <- mnist$train$images[1:1000,] 
y <- mnist$train$labels[1:1000]


# Matrix Notation

length(x[,1])
x_1 <- 1:5
x_2 <- 6:10
cbind(x_1, x_2)
dim(x)
dim(x_1)
dim(as.matrix(x_1))
dim(x)


## Converting a Vector to a Matrix

my_vector <- 1:15

# fill the matrix by column
mat <- matrix(my_vector, 5, 3)
mat

# fill by row
mat_t <- matrix(my_vector, 3, 5, byrow = TRUE)
mat_t
identical(t(mat), mat_t)
matrix(my_vector, 5, 5)
grid <- matrix(x[3,], 28, 28)
image(1:28, 1:28, grid)

# flip the image back
image(1:28, 1:28, grid[, 28:1])


## Row and Column Summaries and Apply

sums <- rowSums(x)
avg <- rowMeans(x)

data_frame(labels = as.factor(y), row_averages = avg) %>%
  qplot(labels, row_averages, data = ., geom = "boxplot")

avgs <- apply(x, 1, mean)
sds <- apply(x, 2, sd)


## Filtering Columns Based on Summaries

library(matrixStats)

sds <- colSds(x)
qplot(sds, bins = "30", color = I("black"))
image(1:28, 1:28, matrix(sds, 28, 28)[, 28:1])

#extract columns and rows
x[ ,c(351,352)]
x[c(2,3),]
new_x <- x[ ,colSds(x) > 60]
dim(new_x)
class(x[,1])
dim(x[1,])

#preserve the matrix class
class(x[ , 1, drop=FALSE])
dim(x[, 1, drop=FALSE])


## Indexing with Matrices and Binarizing the Data

# We can use logical operations with matrices:
mat <- matrix(1:15, 5, 3)
mat[mat > 6 & mat < 12] <- 0

# We can also binarize the data using just matrix operations:
bin_x <- x
bin_x[bin_x < 255/2] <- 0 
bin_x[bin_x > 255/2] <- 1

# Code
# index with matrices
mat <- matrix(1:15, 5, 3)
as.vector(mat)
qplot(as.vector(x), bins = 30, color = I("black"))
new_x <- x
new_x[new_x < 50] <- 0

mat <- matrix(1:15, 5, 3)
mat[mat < 3] <- 0
mat

mat <- matrix(1:15, 5, 3)
mat[mat > 6 & mat < 12] <- 0
mat

# binarize the data
bin_x <- x
bin_x[bin_x < 255/2] <- 0
bin_x[bin_x > 255/2] <- 1
bin_X <- (x > 255/2)*1


## Vectorization for Matrices and Matrix Algebra Operations

#scale each row of a matrix
(x - rowMeans(x)) / rowSds(x)

#scale each column
t(t(x) - colMeans(x))

#take each entry of a vector and subtracts it from 
# the corresponding row or column
x_mean_0 <- sweep(x, 2, colMeans(x))

#divide by the standard deviation
x_mean_0 <- sweep(x, 2, colMeans(x))
x_standardized <- sweep(x_mean_0, 2, colSds(x), FUN = "/")



## Comprehension Check: Working with Matrices

# Q1
# Which line of code correctly creates a 100 by 10 matrix 
# of randomly generated normal numbers and assigns it to x?

x <- matrix(rnorm(100*10), 100, 10)
dim(x)


# Q2
# Write the line of code that would give you 
# the specified information about the matrix x 
# that you generated in q1. Do not include any spaces 
# in your line of code.

# Dimension of x
dim(x) 
#  
# Number of rows of x
nrow(x)
#   
# Number of columns of x
ncol(x) 
#  


# Q3
# Which of the following lines of code would 
# add the scalar 1 to row 1, the scalar 2 to row 2, 
# and so on, for the matrix x?
# Select ALL that apply.

head(x)
head(x + seq(nrow(x)))

## x <- head(1:nrow(x)) ## NO

## head(sweep(x, 2, 1:nrow(x),"+")) ## 2 = col, error

head(sweep(x, 1, 1:nrow(x),"+"))


## Q4
# Which of the following lines of code would add 
# the scalar 1 to column 1, the scalar 2 to column 2, 
# and so on, for the matrix x?
# Select ALL that apply.

head(x)

head(sweep(x, 2, 1:ncol(x), FUN = "+"))


# Q5
# Which code correctly computes the average of each row of x?

rowMeans(x)

colMeans(x)


# Q6
# For each observation in the mnist training data, 
# compute the proportion of pixels that are in the grey area, 
# defined as values between 50 and 205 (but not including 
# 50 and 205). (To visualize this, you can make a boxplot 
# by digit class.)

# What proportion of the 60000*784 pixels in 
# the mnist training data are in the grey area overall, 
# defined as values between 50 and 205?

names(mnist$train)

mean(mnist$train$images > 50 & mnist$train$images < 205)



## Explanation from the website

# The matrix and plot can be calculated using 
# the following code:
  
mnist <- read_mnist()
y <- rowMeans(mnist$train$images>50 & mnist$train$images<205)
qplot(as.factor(mnist$train$labels), y, geom = "boxplot")



















































