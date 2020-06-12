## Machine Learning
## Introduction
## Assessment: Programming Skills

library(dslabs)
library(tidyverse)

data(heights)
heights

class(heights)
str(heights)

nrow(heights)

heights$height[777]

# Q5
max(heights$height)
which.min(heights$height)

# Q6
mean(heights$height)
median(heights$height)

# Q7: Conditional Statements- 1
# What proportion of individuals in the dataset are male?

mean(heights$sex == "Male")


# Q8: Conditional Statements - 2
# How many individuals are taller than 78 inches 
# (roughly 2 meters)?

data("heights")
heights %>% 
  filter(height > 78 & sex == "Female") %>% 
  summarize(n = n())

sum(heights$height > 78 & heights$sex == "Female")








































