## Section 2: Machine Learning Basics
##   2.2: Conditional Probabilities
##   Conditional probabilities

# Comprehension Check: Conditional Probabilities Part 1

# Q1
# In a previous module, we covered Bayes' theorem 
# and the Bayesian paradigm. Conditional probabilities 
# are a fundamental part of this previous covered rule.
# 
# P(A|B)=P(B|A)P(A)P(B)

# We first review a simple example to go over 
# conditional probabilities.
# 
# Assume a patient comes into the doctor’s office 
# to test whether they have a particular disease.
# 
# The test is positive 85% of the time when tested 
# on a patient with the disease (high sensitivity): 
# P(test+|disease)=0.85

# The test is negative 90% of the time when tested 
# on a healthy patient (high specificity): 
# P(test−|heathy)=0.90

# The disease is prevalent in about 2% of the community: 
# P(disease)=0.02

# Using Bayes' theorem, calculate the probability 
# that you have the disease if the test is positive.
# 
# Enter your answer as a percentage or decimal 
# (eg "50%" or "0.50").

.02 * .85
# [1] 0.017

# P(A|B)=P(B|A)P(A)P(B)

# P(disease | test+) = 
#    P(test+|disease) * P(positive) * P(disease)
0.85 * P(positive) * 0.02

tp <- .02 * .85
fp <- .98 * .1

tp / (tp + fp)
# [1] 0.1478261


# The following 4 questions (Q2-Q5) all relate 
# to implementing this calculation using R.
# 
# We have a hypothetical population of 1 million individuals 
# with the following conditional probabilities 
# as described below:
#   
# # The test is positive 85% of the time when tested 
# on a patient with the disease (high sensitivity):  
#   P(test+|disease)=0.85 
# # The test is negative 90% of the time when tested 
# on a healthy patient (high specificity):  
#   P(test−|heathy)=0.90 
# # The disease is prevalent in about 2% of the community:  
# P(disease)=0.02 
# Here is some sample code to get you started:
  
# set.seed(1) # 
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
disease <- sample(c(0,1), 
                  size=1e6, 
                  replace=TRUE, 
                  prob=c(0.98,0.02))
test <- rep(NA, 1e6)
test[disease==0] <- 
  sample(c(0,1), 
         size=sum(disease==0), 
         replace=TRUE, 
         prob=c(0.90,0.10))
test[disease==1] <- 
  sample(c(0,1), 
         size=sum(disease==1), 
         replace=TRUE, 
         prob=c(0.15, 0.85))

# Q2
# What is the probability that a test is positive?
tp <- .85 * .02
fp <- .10 * .98
tp; fp; 
tp + fp
# 0.115

mean(test == 1)
# [1] 0.114509


# Q3
# What is the probability that an individual 
# has the disease if the test is negative?

tn <- .9 * .98  # true negative
fn <- .15 * .02 # false negative
fn / (tn + fn)
[1] 0.003389831


# Q4
# # What is the probability that you have the disease 
# if the test is positive?
# # Remember: calculate the conditional probability 
# the disease is positive assuming a positive test.

tp <- .85 * .02 # true positive
fp <- .10 * .98 # false positive
Prob_of_Disease_if_Test_Positive <- 
  tp / (tp + fp) # P(disease | test+)
Prob_of_Disease_if_Test_Positive

# Q5
# Compare the prevalence of disease in people 
# who test positive to the overall prevalence of disease.
# 
# If a patient's test is positive, how much does that 
# increase their risk of having the disease?
#
# First calculate the probability of having 
# the disease given a positive test, then divide 
# by the probability of having the disease.

# P(test+|disease) = 0.85
# P(test−|healthy) = 0.90
# P(disease) = 0.02


tp <- .85 * .02 # true positive i.e. P(disease | test+)
fp <- .10 * .98 # false positive
pd <- tp + fp # probability of having the disease based on testing
Prob_of_Disease_if_Test_Positive <- 
  tp / (tp + fp) # P(disease | test+)
# [1] 0.1478261

tp / mean(test == 1)
# 0.1484599

mean(test[disease==1] == 1) # prob having disease if test+ from sample
# [1] 0.8461191
(tp / .02) # # prob having disease if test+, calculated
# [1] 0.85
mean(test == 1)
mean(test[disease==1] == 1) / mean(test == 1)
# [1] 7.389106

###########################
#
#    ANSWER = 7.389106 !!!!
#
###########################

(tp / .02) / (tp + fp) # [1] 7.391304

## used the expected probability of 0.02
## "by how many times does their risk of 
## having the disease increase" would be a clearer wording.

### tp / .02 ### NO
## .85 ## NO

## tp / (tp / (tp + fp)) ## NO
# 0.115 # NO

# tp / (tp + fp) # NO
# 0.1478261 ## NO


# To estimate prevalence, researchers randomly select a sample 
# (smaller group) from the entire population they want 
# to describe. ...
# For a representative sample, prevalence is the number 
# of people in the sample with the characteristic of interest, 
# divided by the total number of people in the sample.


rm(list = ls())

# Q6
# We are now going to write code 
# to compute conditional probabilities 
# for being male in the heights dataset. 
# Round the heights to the closest inch. 
# Plot the estimated conditional probability  
# P(x)=Pr(Male|height=x)  for each  x .
# 
# Part of the code is provided here:
  
library(dslabs)
data("heights")
# MISSING CODE
qplot(height, p, data =.)

# Which of the following blocks of code can be used 
# to replace # MISSING CODE to make the correct plot?

library(dslabs)
library(tidyverse)

# option 4 - YES
data("heights")
# start missing code
heights %>% 
  mutate(height = round(height)) %>%
  group_by(height) %>%
  summarize(p = mean(sex == "Male")) %>%
  # end missing code
  qplot(height, p, data =.)



## Q7
# In the plot we just made in Q6 we see 
# high variability for low values of height. 
# This is because we have few data points. 
# This time use the quantile  0.1,0.2,…,0.9  
# and the cut() function to assure 
# each group has the same number of points. 
# Note that for any numeric vector x, 
# you can create groups based on quantiles 
# like this: cut(x, quantile(x, seq(0, 1, 0.1)), 
# include.lowest = TRUE).

# Part of the code is provided here:
  
ps <- seq(0, 1, 0.1)
heights %>% 
  # MISSING CODE
  group_by(g) %>%
  summarize(p = mean(sex == "Male"), height = mean(height)) %>%
  qplot(height, p, data =.)

# Which of the following lines of code can be used 
# to replace # MISSING CODE to make the correct plot?

# option 2 # YES
ps <- seq(0, 1, 0.1)
heights %>% 
  # MISSING CODE
  mutate(g = cut(height, 
                 quantile(height, ps), 
                 include.lowest = TRUE)) %>%
  # end missing code
  group_by(g) %>%
  summarize(p = mean(sex == "Male"), 
            height = mean(height)) %>%
  qplot(height, p, data =.)



# Q8
# You can generate data from a bivariate normal distrubution 
# using the MASS package using the following code:
  
Sigma <- 9*matrix(c(1,0.5,0.5,1), 2, 2)
dat <- MASS::mvrnorm(n = 10000, c(69, 69), Sigma) %>%
  data.frame() %>% setNames(c("x", "y"))

# And you can make a quick plot using plot(dat).
# 
# Using an approach similar to that used 
# in the previous exercise, let's estimate 
# the conditional expectations and make a plot. 
# Part of the code has again been provided for you:

ps <- seq(0, 1, 0.1)
dat %>% 
	# MISSING CODE
	qplot(x, y, data =.)

# Which of the following blocks of code can be used 
# to replace # MISSING CODE to make the correct plot?

rm(list = ls())

Sigma <- 9*matrix(c(1,0.5,0.5,1), 2, 2)
dat <- MASS::mvrnorm(n = 10000, c(69, 69), Sigma) %>%
  data.frame() %>% setNames(c("x", "y"))

plot(dat)

# option 1 # YES
ps <- seq(0, 1, 0.1)
dat %>% 
  # MISSING CODE
  mutate(g = cut(x, quantile(x, ps), 
                 include.lowest = TRUE)) %>%
  group_by(g) %>%
  summarize(y = mean(y), x = mean(x)) %>%
  # end missing code
  qplot(x, y, data =.)

# option 2 # no, didn't use include.lowest = TRUE
# option 3 # NO, didn't group_by g
# option 4 # NO, didn't get mean of x & y






































































