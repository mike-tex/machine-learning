## Section 6: Model Fitting and Recommendation Systems
##   6.3: Regularization
##   Regularization

library(dslabs)
library(tidyverse)
library(caret)
data("movielens")
set.seed(755)
test_index <- createDataPartition(y = movielens$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- movielens[-test_index,]
test_set <- movielens[test_index,]
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
mu_hat <- mean(train_set$rating)
naive_rmse <- RMSE(test_set$rating, mu_hat)
rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i
model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = model_1_rmse ))
user_avgs <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  >     left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))


# Here are 10 of the largest mistakes 
# that we made when only using the movie

test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual))) %>% 
  select(title,  residual) %>% slice(1:10) %>% knitr::kable()


# To see what's going on, let's look at the top 10 best
# movies in the top 10 worst movies based on the estimates 
# of the movie effect b-hat_i.

movie_titles <- movielens %>% 
  select(movieId, title) %>%
  distinct()
movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i) %>% 
  slice(1:10) %>%  
  knitr::kable()


# the top 10 worst movies
movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i) %>% 
  slice(1:10) %>%  
  knitr::kable()


# look at how often they were rated
train_set %>% dplyr::count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()


# the same for the bad movies
train_set %>% dplyr::count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()



## introduce the concept of regularization

# compute these regularized estimates
# of b_i using lambda equals to 3.0

lambda <- 3
mu <- mean(train_set$rating)
movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 


# To see how the estimates shrink, let's make
# a plot of the regularized estimate versus 
# the least square estimates with the size of the circle 
# telling us how large ni was.  You can see that 
# when n is small, the values are shrinking more towards zero.

data_frame(original = movie_avgs$b_i, 
           regularlized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)


# look at our top 10 best movies based on the estimates
# we got when using regularization

train_set %>%
  dplyr::count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()


# look at the worst movies and the worst

train_set %>%
  dplyr::count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()


# the residual mean squared error 
# all the way down to 0.885 from 0.986.

predicted_ratings <- test_set %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred

model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie Effect Model",  
                                     RMSE = model_3_rmse ))
rmse_results %>% knitr::kable()


# lambda is a tuning parameter.
# We can use cross-validation to choose it.

lambdas <- seq(0, 10, 0.25)
mu <- mean(train_set$rating)
just_the_sum <- train_set %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())
rmses <- sapply(lambdas, function(l){
  predicted_ratings <- test_set %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})


# we see why we picked 3.0 as lambda

qplot(lambdas, rmses)  
lambdas[which.min(rmses)]


# We can also use regularization to estimate the user effect.
# again use cross-validation to pick lambda.
# The code looks like this, and we see what lambda minimizes 
# our equation

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

# we see what lambda minimizes our equation

qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <-
  bind_rows(rmse_results,
            data_frame(method = "Regularized Movie + User Effect Model",
                       RMSE = min(rmses)))
rmse_results %>% knitr::kable()



#############################################
##                                         ##
##   Comprehension Check: Regularization   ##
##                                         ##
#############################################

# The exercises in Q1-Q8 work with a simulated dataset 
# for 1000 schools. This pre-exercise setup walks you 
# through the code needed to simulate the dataset.
# 
# If you have not done so already since the Titanic Exercises, 
# please restart R or reset the number of digits 
# that are printed with options(digits=7).

options(digits=7)
library(dplyr)

# An education expert is advocating for smaller schools. 
# The expert bases this recommendation on the fact 
# that among the best performing schools, many are 
# small schools. Let's simulate a dataset for 1000 schools. 
# First, let's simulate the number of students 
# in each school, using the following code:
  
# set.seed(1986) #for R 3.5 or earlier
set.seed(1986, sample.kind="Rounding") #if using R 3.6 or later, use `` instead
n <- round(2^rnorm(1000, 8, 1))


# Now let's assign a true quality for each school 
# that is completely independent from size. This is 
# the parameter we want to estimate in our analysis. 
# The true quality can be assigned using the following code:

# set.seed(1) #for R 3.5 or earlier
set.seed(1, sample.kind="Rounding") #if using R 3.6 or later, use `` instead
mu <- round(80 + 2*rt(1000, 5))
range(mu)
schools <- data.frame(id = paste("PS",1:1000),
                      size = n,
                      quality = mu,
                      rank = rank(-mu))


# We can see the top 10 schools using this code: 
 
schools %>% top_n(10, quality) %>% arrange(desc(quality))


# Now let's have the students in the school take a test. 
# There is random variability in test taking, so we will 
# simulate the test scores as normally distributed with 
# the average determined by the school quality with 
# a standard deviation of 30 percentage points. This code 
# will simulate the test scores:
   
# set.seed(1) #for R 3.5 or earlier
set.seed(1, sample.kind="Rounding") #if using R 3.6 or later, use `` instead
mu <- round(80 + 2*rt(1000, 5))
range(mu)

scores <- sapply(1:nrow(schools), function(i){
  scores <- rnorm(schools$size[i], schools$quality[i], 30)
  scores
})
schools <- schools %>% mutate(score = sapply(scores, mean))


# Q1
# What are the top schools based on the average score? 
# Show just the ID, size, and the average score.

# Report the ID of the top school and average score 
# of the 10th school.

# What is the ID of the top school?
#   Note that the school IDs are given 
#   in the form "PS x" - where x is a number. 
#   Report the number only.

schools %>% 
  arrange(desc(score)) %>% 
  slice_head(n = 1) %>% 
  pull(id)

# 191 # no, want ranking by score, not quality
# 567
 
# What is the average score of the 10th school?
mean(unlist(scores[756]))

schools %>% 
  arrange(desc(score)) %>% 
  slice_head(n = 10) %>% 
  pull(score) %>% .[10]

# [1] 86.31867
# 87.95731

# Explanation from the course web site
# The ID, size, and average score of the top schools 
# can be identified using this code: 
schools %>% top_n(10, score) %>%
  arrange(desc(score)) %>%
  select(id, size, score)



# Q2
# Compare the median school size to the median school size 
# of the top 10 schools based on the score.

# What is the median school size overall?
schools %>% 
  summarize(mid = (median(size)))
# 261

# What is the median school size of the of the top 10 schools 
# based on the score?
schools %>% 
  arrange(desc(score)) %>% 
  slice_head(n = 10) %>% 
  summarize(mid = (median(size)))
# 185.5

# Explanation from the course web site
# The median school sizes can be compared using 
# the following code:
  
median(schools$size)
schools %>% top_n(10, score) %>% .$size %>% median()



# Q3
# According to this analysis, it appears 
# that small schools produce better test scores 
# than large schools. Four out of the top 10 schools 
# have 100 or fewer students. But how can this be? 
# We constructed the simulation so that quality 
# and size were independent. Repeat the exercise for 
# the worst 10 schools.

# What is the median school size of the bottom 10 schools 
# based on the score?
schools %>% 
  arrange((score)) %>% 
  slice_head(n = 10) %>% 
  summarize(mid = (median(size)))
# 219

# Explanation from the course web site
# The median school size for the bottom 10 schools 
# can be found using the following code:
  
median(schools$size)
schools %>% top_n(-10, score) %>% .$size %>% median()


# Q4
# From this analysis, we see that the worst schools 
# are also small. Plot the average score versus school size 
# to see what's going on. Highlight the top 10 schools based 
# on the true quality.
# 
# What do you observe?
# The standard error of the score has larger variability 
# when the school is smaller, which is why both the best 
# and the worst schools are more likely to be small.

library(ggplot2)

plot(schools$score, schools$size)

top_10 <- schools %>% 
  arrange(desc(score)) %>% 
  slice_head(n = 10) %>% 
  tibble()

ggplot() +
  geom_point(aes(x = score, y = size), data = schools) +
  geom_smooth(aes(x = score, y = size),
              method = "lm",
              formula = y ~ x,
              color = "blue", 
              data = schools) +
  geom_point(
    aes(score, size),
    data = top_10,
    color = "red",
    inherit.aes = F
  )

x <- schools$size
y <- schools$score

ggplot() +
  geom_point(aes(x = x, y = y)) +
  geom_smooth(aes(x = x, y = y),
              method = "lm",
              formula = y ~ x,
              color = "blue") +
  geom_point(
    aes(size, score),
    data = top_10,
    color = "red",
    inherit.aes = F
  )

# Explanation from the course web site
# You can generate the plot using the following code:
  
schools %>% ggplot(aes(size, score)) +
  geom_point(alpha = 0.5) +
  geom_point(data = filter(schools, rank <= 10), col = 2)

# We can see that the standard error of the score 
# has larger variability when the school is smaller. 
# This is a basic statistical reality we learned 
# in PH125.3x: Data Science: Probability 
# and PH125.4x: Data Science: Inference and Modeling courses! 
# Note also that several of the top 10 schools based 
# on true quality are also in the top 10 schools based 
# on the exam score: 
schools %>% top_n(10, score) %>% arrange(desc(score)).



# Q5
# Let's use regularization to pick the best schools. 
# Remember regularization shrinks deviations 
# from the average towards 0. To apply regularization here, 
# we first need to define the overall average for all schools, 
# using the following code:

overall <- mean(sapply(scores, mean))

# Then, we need to define, for each school, how it deviates 
# from that average.

# Write code that estimates the score above the average 
# for each school but dividing by n+α instead of  n , 
# with  n  the school size and α a regularization parameter. 
# Try  α=25 .
# 

lambda <- 25
mu <- mean(sapply(scores, mean))
score_est <- schools %>% 
  group_by(id) %>% 
  summarize(id = id, 
            b_i = sum(score - mu)/(size + lambda), 
            n_i = size,
            mu = mu, 
            score = score) %>% 
 tibble()



predicted_score <- 
  score_est %>% 
  # left_join(b_i, by = "movieId") %>%
  # left_join(b_u, by = "userId") %>%
  mutate(id = id,
         pred = mu + (n_i * b_i)) #%>%
  #.$pred

predicted_score %>% 
  arrange(desc(pred)) %>% 
  slice(1:10)


scores[1]
sum(scores[1] - overall)

# What is the ID of the top school with regularization?
#   Note that the school IDs are given 
#   in the form "PS x" - where x is a number. 
#   Report the number only.
191

# What is the regularized score of the 10th school?
87.2

# explanation from the course web site
# The regularization and reporting of scores can be done 
# using the following code:
  
alpha <- 25
score_reg <-
  sapply(scores, function(x)
    overall + sum(x - overall) / (length(x) + alpha))
schools %>% mutate(score_reg = score_reg) %>%
  top_n(10, score_reg) %>% arrange(desc(score_reg))



# Q6
# Notice that this improves things a bit. 
# The number of small schools that are not highly ranked 
# is now lower. Is there a better  α ? Using values 
# of  α  from 10 to 250, find the  α  that minimizes the RMSE
# for quality.
# 

library(caret)

overall <- mean(sapply(scores, mean))
alpha <- seq(10, 250)
rmses <- sapply(alpha, function(a) { 
  score_reg <- sapply(scores, function(x) 
    overall + sum(x - overall) / (length(x) + a))
  RMSE(score_reg, schools$quality)
})


# What value of  α  gives the minimum RMSE (using quality)?

which.min(x = rmses) + 9
# 1 # NO
# 10 # NO
# 135

# Explanation from the course web site
# The value of  α  that minimizes the MSE 
# can be calculated using the following code:
  
  alphas <- seq(10,250)
rmse <- sapply(alphas, function(alpha){
  score_reg <- 
    sapply(scores, 
           function(x) 
             overall+sum(x-overall)/(length(x)+alpha))
  sqrt(mean((score_reg - schools$quality)^2))
})
plot(alphas, rmse)
alphas[which.min(rmse)]

# Q7
# Rank the schools based on the average obtained with 
# the best α. Note that no small school is incorrectly included.

alpha <- 135
score_reg <-
  sapply(scores, function(x)
    overall + sum(x - overall) / (length(x) + alpha))
schools %>% mutate(score_reg = score_reg) %>%
  top_n(10, score_reg) %>% arrange(desc(score_reg))



# What is the ID of the top school now?
#   Note that the school IDs are given 
#   in the form "PS x" - where x is a number. 
#   Report the number only.
191

# What is the regularized average score of the 10th school now?
85.48132


# Q8
# A common mistake made when using regularization 
# is shrinking values towards 0 that are not centered around 0. 
# For example, if we don't subtract the overall average 
# before shrinking, we actually obtain a very similar result. 
# Confirm this by re-running the code from the exercise 
# in Q6 but without removing the overall mean.

overall <- mean(sapply(scores, mean))
alpha <- seq(10, 250)
rmses <- sapply(alpha, function(a) { 
  score_reg <- sapply(scores, function(x) 
   sum(x) / (length(x) + a))
  RMSE(score_reg, schools$quality)
})


# What value of  α  gives the minimum RMSE here?
10

# # Explanation
# # The code here is nearly the same as in Q6, 
# but we don't subtract the overall mean. The value of  α  
# that minimizes the RMSE can be calculated using 
# the following code:

alphas <- seq(10,250)
rmse <- sapply(alphas, function(alpha){
  score_reg <- 
    sapply(scores, 
           function(x) sum(x)/(length(x)+alpha))
  sqrt(mean((score_reg - schools$quality)^2))
})
plot(alphas, rmse)
alphas[which.min(rmse)]  














































































