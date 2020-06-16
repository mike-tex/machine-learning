## Section 3: Linear Regression for
## Prediction, Smoothing, and Working with Matrices
##   3.2: Smoothing
##   Introduction to Smoothing

.rs.restartR()
rm(list = ls())

data("polls_2008")
qplot(day, margin, data = polls_2008)


## Bin Smoothing and Kernels

# bin smoothers
span <- 7
fit <- with(polls_2008,
            ksmooth(
              day,
              margin,
              x.points = day,
              kernel = "box",
              bandwidth = span
            ))
polls_2008 %>% mutate(smooth = fit$y) %>%
  ggplot(aes(day, margin)) +
  geom_point(size = 3,
             alpha = .5,
             color = "grey") +
  geom_line(aes(day, smooth), color = "red")

# kernel
span <- 7
fit <- with(polls_2008,
            ksmooth(
              day,
              margin,
              x.points = day,
              kernel = "normal",
              bandwidth = span
            ))
polls_2008 %>% mutate(smooth = fit$y) %>%
  ggplot(aes(day, margin)) +
  geom_point(size = 3,
             alpha = .5,
             color = "grey") +
  geom_line(aes(day, smooth), color = "red")


## Local Weighted Regression (loess)

total_days <- diff(range(polls_2008$day))
span <- 21 / total_days

fit <- loess(margin ~ day, 
             degree = 1, 
             span = span, 
             data = polls_2008)

polls_2008 %>% mutate(smooth = fit$fitted) %>% 
  ggplot(aes(day, margin)) +
  geom_point(size = 3, alpha = .5, color = "grey") +
  geom_line(aes(day, smooth), color = "red")

# polls_2008 %>% ggplot(aes(day, margin)) +
#   geom_point() + 
#   geom_smooth(color="red", 
#               span = 0.15, 
#               method.args = list(degree=1))

span <- 7
fit <- 
  with(polls_2008, ksmooth(day, margin, 
                           kernel = "normal",
                           bandwidth = span))

polls_2008 %>% mutate(smooth = fit$y) %>%
  ggplot(aes(day, margin)) +
  geom_point(size = 3, alpha = .5, color = "grey") + 
  geom_line(aes(day, smooth), color="red")


polls_2008 %>% ggplot(aes(day, margin)) +
  geom_point() +
  geom_smooth()

polls_2008 %>% ggplot(aes(day, margin)) +
  geom_point() +
  geom_smooth(color = "red",
              span = 0.15,
              method.args = list(degree = 1))
# Warning message:
#   Computation failed in `stat_smooth()`:
#   'what' must be a function or character string 


## Comprehension Check: Smoothing

# Q1
# In the Wrangling course of this series, PH125.6x, 
# we used the following code to obtain mortality counts 
# for Puerto Rico for 2015-2018:

.rs.restartR()
rm(list = ls())
  
library(tidyverse)
library(lubridate)
library(purrr)
library(pdftools)

fn <- system.file("extdata", 
                  "RD-Mortality-Report_2015-18-180531.pdf", 
                  package="dslabs")
dat <- map_df(str_split(pdf_text(fn), "\n"), function(s){
  s <- str_trim(s)
  header_index <- str_which(s, "2015")[1]
  tmp <- str_split(s[header_index], "\\s+", simplify = TRUE)
  month <- tmp[1]
  header <- tmp[-1]
  tail_index  <- str_which(s, "Total")
  n <- str_count(s, "\\d+")
  out <- c(1:header_index, which(n==1), 
           which(n>=28), tail_index:length(s))
  s[-out] %>%
    str_remove_all("[^\\d\\s]") %>%
    str_trim() %>%
    str_split_fixed("\\s+", n = 6) %>%
    .[,1:5] %>%
#    as_data_frame() %>% 
    as_tibble() %>% 
    setNames(c("day", header)) %>%
    mutate(month = month,
           day = as.numeric(day)) %>%
    gather(year, deaths, -c(day, month)) %>%
    mutate(deaths = as.numeric(deaths))
}) %>%
  mutate(month = recode(month, "JAN" = 1, "FEB" = 2, "MAR" = 3, 
                        "APR" = 4, "MAY" = 5, "JUN" = 6, 
                        "JUL" = 7, "AGO" = 8, "SEP" = 9, 
                        "OCT" = 10, "NOV" = 11, "DEC" = 12)) %>%
  mutate(date = make_date(year, month, day)) %>%
  filter(date <= "2018-05-01")
dat <- na.omit(dat)

# Use the loess() function to obtain a smooth estimate 
# of the expected number of deaths as a function of date. 
# Plot this resulting smooth function. Make the span about 
# two months long.

# Which of the following plots is correct?

total_days <- as.numeric(diff(range(dat$date)))
span <- 60 / total_days

fit <- loess(deaths ~ as.numeric(date), 
             degree = 1, 
             span = span, 
             data = dat)

dat %>% mutate(smooth = fit$fitted) %>% 
  ggplot(aes(date, deaths)) +
  geom_point(size = 3, alpha = .5, color = "grey") +
  geom_line(aes(date, smooth), color = "red")

## NOT #2 OPTION
## Option #1, had to used na.omit to get rid of NAs

# Explanation

# The following code makes the correct plot:
  
span <- 60 / as.numeric(diff(range(dat$date)))
fit <- dat %>% mutate(x = as.numeric(date)) %>% 
  loess(deaths ~ x, data = ., span = span, degree = 1)
dat %>% mutate(smooth = predict(fit, as.numeric(date))) %>%
  ggplot() +
  geom_point(aes(date, deaths)) +
  geom_line(aes(date, smooth), lwd = 2, col = "red")

# The second plot uses a shorter span, 
# the third plot uses the entire timespan, 
# and the fourth plot uses a longer span.


# Q2
# Work with the same data as in Q1 to plot smooth estimates 
# against day of the year, all on the same plot, 
# but with different colors for each year.

# Which code produces the desired plot?


# Option #4
dat %>% mutate(smooth = predict(fit,  as.numeric(date)),
    day = yday(date),
    year = as.character(year(date))) %>%
  ggplot(aes(day, smooth, col = year)) +
  geom_line(lwd = 2)


# Q3
# Suppose we want to predict 2s and 7s in the 
# mnist_27 dataset with just the second covariate. 
# Can we do this? On first inspection it appears 
# the data does not have much predictive power.

# In fact, if we fit a regular logistic regression 
# the coefficient for x_2 is not significant!
  
# This can be seen using this code:
  
rm(list = ls())

library(broom)
library(dslabs)
data(mnist_27)
mnist_27$train %>% 
  glm(y ~ x_2, family = "binomial", data = .) %>% 
  tidy()

# Plotting a scatterplot here is not useful since y is binary:
  
qplot(x_2, y, data = mnist_27$train)

# Fit a loess line to the data above and plot the results. 
# What do you observe?

fit <- loess(as.numeric(y) ~ x_2, data = mnist_27$train)
fit

mnist_27$train %>% 
  mutate(smooth = fit$fitted) %>%
  ggplot() +
  geom_point(aes(x_2, y)) +
  geom_line(aes(x_2, smooth))

## NOT OPTION #2 !!!!
#  The answer is option #4 ... look at the graph,
#  the line is either closer to 2 or 7

## Explanation
# 
# Note that there is indeed predictive power, 
# but that the conditional probability is non-linear.
# 
# The loess line can be plotted using the following code:
  
  mnist_27$train %>% 
  mutate(y = ifelse(y=="7", 1, 0)) %>%
  ggplot(aes(x_2, y)) + 
  geom_smooth(method = "loess")


















