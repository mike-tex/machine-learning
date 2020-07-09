## Section 4: Distance, Knn, Cross-validation,
## and Generative Models
##   4.2: Cross-validation
##   Bootstrap

library(tidyverse)

n <- 10^6
income <- 10^(rnorm(n, log10(45000), log10(3)))
qplot(log10(income), bins = 30, color = I("black"))

m <- median(income)
m

# set.seed(1995)
set.seed(1995, sample.kind="Rounding") # if using R 3.6 or later
N <- 250
X <- sample(income, N)
M<- median(X)
M

library(gridExtra)
B <- 10^4

M <- replicate(B, {
  X <- sample(income, N)
  median(X)
})

p1 <- qplot(M, bins = 30, color = I("black"))
p2 <- qplot(sample = scale(M)) + geom_abline()
grid.arrange(p1, p2, ncol = 2)

mean(M)
sd(M)

B <- 10^4
M_star <- replicate(B, {
  X_star <- sample(X, N, replace = TRUE)
  median(X_star)
})

tibble(monte_carlo = sort(M), bootstrap = sort(M_star)) %>%
  qplot(monte_carlo, bootstrap, data = .) + 
  geom_abline()

quantile(M, c(0.05, 0.95))
quantile(M_star, c(0.05, 0.95))

median(X) + 1.96 * sd(X) / sqrt(N) * c(-1, 1)

mean(M) + 1.96 * sd(M) * c(-1,1)

mean(M_star) + 1.96 * sd(M_star) * c(-1, 1)


## Comprehension Check: Bootstrap

# Q1
# The createResample() function can be used 
# to create bootstrap samples. For example, 
# we can create the indexes for 10 bootstrap samples 
# for the mnist_27 dataset like this:
  
library(dslabs)
library(caret)
data(mnist_27)
# set.seed(1995) # if R 3.6 or later, set.seed(1995, sample.kind="Rounding")
set.seed(1995, sample.kind="Rounding")
indexes <- createResample(mnist_27$train$y, 10)

# How many times do 3, 4, and 7 appear 
# in the first resampled index?

### in the FIRST resample

for (n in c(3, 4, 7)) {
  times <- 0
  for (i in 1:10) {
    times <- times + sum(indexes[[i]] == n)
  }
  print(paste0(n, " = ", times))
}

# [1] "3 = 1"
# [1] "4 = 4"
# [1] "7 = 0"


# Q2
# We see that some numbers appear more than once 
# and others appear no times. This has to be this way 
# for each dataset to be independent. Repeat the exercise 
# for all the resampled indexes.

# What is the total number of times that 3 appears in all 
# of the resampled indexes?

for (n in c(3, 4, 7)) {
  times <- 0
  for (i in 1:10) {
    times <- times + sum(indexes[[i]] == n)
  }
  print(paste0(n, " = ", times))
}

# [1] "3 = 11"


## Q3
# A random dataset can be generated using the following code:
  
y <- rnorm(100, 0, 1)

# Estimate the 75th quantile, which we know is qnorm(0.75), 
# with the sample quantile: quantile(y, 0.75).
# 
# Set the seed to 1 and perform a Monte Carlo simulation 
# with 10,000 repetitions, generating the random dataset 
# and estimating the 75th quantile each time. What is 
# the expected value and standard error of the 75th quantile?

B <- 10000
set.seed(1, sample.kind = "Rounding")
Q75 <- replicate(B, {
  y <- rnorm(100, 0, 1)
  quantile(y, 0.75)
})
#   
# Expected value
mean(Q75)
# [1] 0.6656107
# 
# Standard error
sd(Q75)
# [1] 0.1353809

# Explanation from web site

# The following code can be used to run the simulation 
# and calculate the expected value and standard error:
  
  set.seed(1)
B <- 10000
q_75 <- replicate(B, {
  y <- rnorm(100, 0, 1)
  quantile(y, 0.75)
})

mean(q_75)
sd(q_75)


## Q4
# In practice, we can't run a Monte Carlo simulation. 
# Use the sample:

library(tidyverse)
library(dslabs)
library(caret)

set.seed(1, sample.kind = "Rounding")
y <- rnorm(100, 0, 1)

# Set the seed to 1 again after generating y 
# and use 10 bootstrap samples to estimate 
# the expected value and standard error of the 75th quantile.

N <- 10
B <- 10
set.seed(1, sample.kind = "Rounding")
idx <- createResample(y, times = N, list = F)

Q75 <- rep(0, 10)

for (i in 1:N) {
  Q75[i] <- quantile(y[idx[,i]], 0.75)
}

mean(Q75)
sd(Q75)

# > mean(Q75)
# [1] 0.7312648  ## YES!!!
# > sd(Q75)
# [1] 0.07419278 ## YES!!!

## Explanation from web page

#The following code can be used to take 10 bootstrap samples 
# and calculate the expected value and standard error:
  
set.seed(1, sample.kind="Rounding") # if R 3.6 or later
indexes <- createResample(y, 10)
q_75_star <- sapply(indexes, function(ind){
  y_star <- y[ind]
  quantile(y_star, 0.75)
})
mean(q_75_star)
sd(q_75_star)


## Q5
# Repeat the exercise from Q4 but with 10,000 bootstrap samples instead of 10. Set the seed to 1.

set.seed(1, sample.kind="Rounding") # if R 3.6 or later
indexes <- createResample(y, 10000)
q_75_star <- sapply(indexes, function(ind){
  y_star <- y[ind]
  quantile(y_star, 0.75)
})
mean(q_75_star)
sd(q_75_star)

# Expected value
# > mean(q_75_star)
# [1] 0.6737512  ## YES!!!
# Standard error
# > sd(q_75_star)
# [1] 0.0930575  ## YES!!!

## Q6
# When doing bootstrap sampling, the simulated samples are
# drawn from the empirical distribution of the original data.
# 
# False: The bootstrap is particularly useful in situations 
# in which a tractable variance formula exists.

























































































