##  Section 6: Model Fitting and Recommendation Systems
##   6.2: Recommendation Systems
##   Recommendation Systems

## Key points
# * Recommendation systems are more complicated 
#   machine learning challenges because each outcome 
#   has a different set of predictors. For example, 
#   different users rate a different number of movies 
#   and rate different movies.
# * To compare different models or to see how well 
#   we’re doing compared to a baseline, we will use 
#   root mean squared error (RMSE) as our loss function. 
#   We can interpret RMSE similar to standard deviation.

# Code
# Please refer to the textbook for an updated version of the code that may contain some corrections.

library(dslabs)
library(tidyverse)
data("movielens")

head(movielens)

movielens %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

keep <- movielens %>%
  dplyr::count(movieId) %>%
  top_n(5) %>%
  pull(movieId)
tab <- movielens %>%
  filter(userId %in% c(13:20)) %>% 
  filter(movieId %in% keep) %>% 
  select(userId, title, rating) %>% 
  spread(title, rating)
tab %>% knitr::kable()

users <- sample(unique(movielens$userId), 100)
rafalib::mypar()
movielens %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")

movielens %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

movielens %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")

library(caret)
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



##
## Building the Recommendation System
##

mu_hat <- mean(train_set$rating)
mu_hat

naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse

predictions <- rep(2.5, nrow(test_set))
RMSE(test_set$rating, predictions)

rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)

fit <- lm(rating ~ as.factor(userId), data = movielens)
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = model_1_rmse ))

rmse_results %>% knitr::kable()

train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

# lm(rating ~ as.factor(movieId) + as.factor(userId))
user_avgs <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()



##
## Comprehension Check: Recommendation Systems
##

# The following exercises all work with the movielens data, 
# which can be loaded using the following code:
  
library(tidyverse)
library(lubridate)
library(dslabs)
data("movielens")
View(movielens)

# Q1
# Compute the number of ratings for each movie 
# and then plot it against the year the movie came out. 
# Use the square root transformation on the y-axis 
# when plotting.

movielens %>% group_by(movieId, year) %>% 
  summarise(num_ratings = n()) %>% 
  ungroup() %>% group_by(year) %>% 
  summarise(med_num_ratings = median(num_ratings)) %>% 
  ggplot(aes(year, med_num_ratings)) +
  geom_point()
# 
# What year has the highest median number of ratings?

med_num <- movielens %>% group_by(movieId, year) %>% 
  summarise(num_ratings = n()) %>% 
  ungroup() %>% group_by(year) %>% 
  summarise(med_num_ratings = median(num_ratings)) 

med_num$year[which.max(med_num$med_num_ratings)]

## Explanation from the course web site
# The following code will generate the plot:
  
movielens %>% group_by(movieId) %>%
  summarize(n = n(), year = as.character(first(year))) %>%
  qplot(year, n, data = ., geom = "boxplot") +
  coord_trans(y = "sqrt") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# From the plot, you can see that the year 
# with the highest median number of ratings is 1995.


## Q2
# We see that, on average, movies that came out after 1993 
# get more ratings. We also see that with newer movies, 
# starting in 1993, the number of ratings decreases with year: 
# the more recent a movie is, the less time users have had 
# to rate it.

# Among movies that came out in 1993 or later, 
# select the top 25 movies with the highest average number 
# of ratings per year (n/year), and caculate 
# the average rating of each of them. To calculate number 
# of ratings per year, use 2018 as the end year.
# 

top_25 <- movielens %>% 
  filter(between(year, 1993, 2018)) %>% 
  group_by(movieId) %>%
  summarize(n = n(), year = as.character(first(year))) %>% 
  arrange(desc(n)) 

top_25

top_25_avg <- left_join(top_25, movielens, by = "movieId") %>%
  group_by(movieId) %>%
  summarise(
    year = as.character(first(year)), # added
    avg_rating = mean(rating),
    title = first(title)
  )

# What is the average rating 
# for the movie The Shawshank Redemption 
# ("Shawshank Redemption, The")?

top_25_avg$avg_rating[
  top_25_avg$title == "Shawshank Redemption, The"]
# [1] 4.487138

# What is the average number of ratings per year 
# for the movie Forrest Gump?
## HINT:  use release year to 2018 (24) for the denominator!!!
# [1] 14.20833 # 341 / 24 = 2018 - 1994



movielens %>% 
  mutate(review_yr = year(as_datetime(timestamp))) %>% 
  filter(between(review_yr, 1993, 2018) &
           title == "Forrest Gump") %>% 
  group_by(review_yr) %>% 
  summarise(n = n()) %>% 
  summarise(n_sum = sum(n), n_avg = mean(n), n_years = nrow(.))

diff.Date(c(1994, 2018))

movielens %>% 
  filter(title == "Forrest Gump") %>% 
  summarise(n = n())
  group_by(year) %>% 
  summarize(n = n())         

# [1] 14.20833 # 341 / 24 = 2018 - 1994

movielens %>% 
  mutate(review_yr = year(as_datetime(timestamp))) %>% 
  filter(year >= 1993 & year <= 2018) %>%  
  group_by(title) %>% 
  summarize(N=n(), mu=mean(rating)) %>%  
  arrange(desc(N))

# Explanation from the course web site

# The top 25 movies with the most ratings per year, 
# along with their average ratings, can be found 
# using the following code:
  
movielens %>%
  filter(year >= 1993) %>%
  group_by(movieId) %>%
  summarize(
    n = n(),
    years = 2018 - first(year),
    title = title[1],
    rating = mean(rating)
  ) %>%
  mutate(rate = n / years) %>%
  top_n(25, rate) %>%
  arrange(desc(rate)) %>% View()


# Q3
# From the table constructed in Q2, we can see that 
# the most frequently rated movies tend to have 
# above average ratings. This is not surprising: 
# more people watch popular movies. To confirm this, 
# stratify the post-1993 movies by ratings per year 
# and compute their average ratings. To calculate number 
# of ratings per year, use 2018 as the end year. 
# Make a plot of average rating versus ratings per year 
# and show an estimate of the trend.



movielens %>%
  mutate(rev_yr = year(as_datetime(timestamp))) %>% 
  filter(between(rev_yr, 1993, 2018)) %>%
  group_by(movieId) %>%
  summarize(
    n = n(),
    years = 2018 - first(year),
    title = title[1],
    rating = mean(rating)
  ) %>%
  mutate(rate = n / years) %>%
  top_n(25, rate) %>%
  arrange(desc(rate)) %>% 
  ggplot(aes(rate, rating)) +
  geom_point() +
  geom_smooth(method = "lm", formula = y ~ x)


# What type of trend do you observe?

# The more often a movie is rated, 
# the higher its average rating.


# Q4
# Suppose you are doing a predictive analysis 
# in which you need to fill in the missing ratings 
# with some value.

# Given your observations in the exercise in Q3, 
# which of the following strategies would be most appropriate?

# Fill in the missing values with a lower value than 
# the average rating across all movies.

avg_rating <- movielens %>%
  filter(between(year(as_datetime(timestamp)),
                 1993, 2018)) %>% 
  summarize(mean(rating))

ml_test <- movielens %>%
  filter(between(year(as_datetime(timestamp)),
                 1993, 2018))
library(caret)
idx <- createDataPartition(factor(ml_test$rating), list = F)

ml_test_0 <- ml_test
ml_test_avg <- ml_test
ml_test_0$rating[idx] <- 0
ml_test_avg$rating[idx] <- mean(ml_test$rating)

avg_rating
mean(ml_test$rating)
mean(ml_test_0$rating)
mean(ml_test_avg$rating)

class(ml_test_avg$rating)

# Explanation from the course web site: 
# Because a lack of ratings is associated with lower ratings, 
# it would be most appropriate to fill in the missing value 
# with a lower value than the average. You should try out 
# different values to fill in the missing value 
# and evaluate prediction in a test set.


# Q5
# The movielens dataset also includes a time stamp. 
# This variable represents the time and data in which 
# the rating was provided. The units are seconds 
# since January 1, 1970. Create a new column date 
# with the date.

# Which code correctly creates this new column?

mutate(movielens, date = as_datetime(timestamp))


# Q6
# Compute the average rating for each week 
# and plot this average against date. 
# Hint: use the round_date() function before you group_by().
# 

library(dslabs)
library(dplyr)
library(lubridate)
data("movielens")

movielens %>%
  mutate(date = round_date(as_date(timestamp)),
         unit = "weeks", week_start = 1) %>%
  filter(between(year(date), 1993, 2018)) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() #+
  geom_smooth(method = "lm", formula = y ~ x)


# What type of trend do you observe?











































