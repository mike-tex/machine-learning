## Section 6: Model Fitting and Recommendation Systems
##   6.3: Regularization
##   Matrix Factorization


train_small <- movielens %>% 
  group_by(movieId) %>%
  #3252 is Scent of a Woman used in example
  filter(n() >= 50 | movieId == 3252) %>% ungroup() %>% 
  group_by(userId) %>%
  filter(n() >= 50) %>% ungroup()

y <- train_small %>% 
  select(userId, movieId, rating) %>%
  spread(movieId, rating) %>%
  as.matrix()


# add row names and column names.

rownames(y)<- y[,1]
y <- y[,-1]
colnames(y) <- with(movie_titles, 
                    title[match(colnames(y), movieId)])


# convert these residuals 
# by removing the column and row averages.

y <- sweep(y, 1, rowMeans(y, na.rm=TRUE))
y <- sweep(y, 2, colMeans(y, na.rm=TRUE))


# Here's a plot of the residuals 
# for The Godfather and The Godfather II.

m_1 <- "Godfather, The"
m_2 <- "Godfather: Part II, The"
qplot(y[ ,m_1], y[,m_2], xlab = m_1, ylab = m_2)


# plot for The Godfather and Goodfellas

m_1 <- "Godfather, The"
m_3 <- "Goodfellas"
qplot(y[ ,m_1], y[,m_3], xlab = m_1, ylab = m_3)


# here's a correlation between 
# You've Got Mail and Sleepless in Seattle

m_4 <- "You've Got Mail" 
m_5 <- "Sleepless in Seattle" 
qplot(y[ ,m_4], y[,m_5], xlab = m_4, ylab = m_5)


# look at the pairwise correlation for these five movies,
# we can see that there's a positive correlation 
# between the gangster movies
# Godfathers and Goodfellas, and then there's
# a positive correlation between the romantic comedies You've
# Got Mail and Sleepless in Seattle.
# We also see a negative correlation between the gangster
# movies and the romantic comedies.

cor(y[, c(m_1, m_2, m_3, m_4, m_5)], 
    use="pairwise.complete") %>% 
  knitr::kable()


# We have a matrix r and we factorized it into two things, 
# the vector p, and the vector q.
# Now we should be able to explain much more of the variance

set.seed(1)
options(digits = 2)
Q <- matrix(c(1 , 1, 1, -1, -1), ncol=1)
rownames(Q) <- c(m_1, m_2, m_3, m_4, m_5)
P <- matrix(rep(c(2,0,-2), c(3,5,4)), ncol=1)
rownames(P) <- 1:nrow(P)

X <- jitter(P%*%t(Q))
X %>% knitr::kable(align = "c")

cor(X)

t(Q) %>% knitr::kable(aling="c")

P

set.seed(1)
options(digits = 2)
m_6 <- "Scent of a Woman"
Q <- cbind(c(1 , 1, 1, -1, -1, -1), 
           c(1 , 1, -1, -1, -1, 1))
rownames(Q) <- c(m_1, m_2, m_3, m_4, m_5, m_6)
P <- cbind(rep(c(2,0,-2), c(3,5,4)), 
           c(-1,1,1,0,0,1,1,1,0,-1,-1,-1))/2
rownames(P) <- 1:nrow(X)

X <- jitter(P%*%t(Q), factor=1)
X %>% knitr::kable(align = "c")

cor(X)

t(Q) %>% knitr::kable(align="c")

P

six_movies <- c(m_1, m_2, m_3, m_4, m_5, m_6)
tmp <- y[,six_movies]
cor(tmp, use="pairwise.complete")


##
## SVD and PCA
## 

# textbook section on singular value decomposition (SVD) 
# and principal component analysis (PCA).


# To compute the decomposition, will make all the NAs zero.
# The PCA function returns a component with the variability
# of each of the principal components and we can access it 
# like this and plot it

y[is.na(y)] <- 0
y <- sweep(y, 1, rowMeans(y))
pca <- prcomp(y)

dim(pca$rotation)

dim(pca$x)

plot(pca$sdev)


# ust with a few of these principal components
# we already explain a large percent of the data.

var_explained <- cumsum(pca$sdev^2/sum(pca$sdev^2))
plot(var_explained)


# To see that the principal components are actually capturing 
# something important about the data, we can make a plot 
# of for example, the first two principal components, 
# but now label the points with the movie
# that each one of those points is related to.

library(ggrepel)
pcs <- data.frame(pca$rotation, name = colnames(y))
pcs %>%  ggplot(aes(PC1, PC2)) + geom_point() +
  geom_text_repel(aes(PC1, PC2, label = name),
                  data = filter(pcs, PC1 < -0.1 |
                                  PC1 > 0.1 |
                                  PC2 < -0.075 |
                                  PC2 > 0.1))


# The first principle component shows the difference
# between critically acclaimed movies on one side.

pcs %>% select(name, PC1) %>% arrange(PC1) %>% slice(1:10)

pcs %>% select(name, PC1) %>% arrange(desc(PC1)) %>% slice(1:10)

pcs %>% select(name, PC2) %>% arrange(PC2) %>% slice(1:10)

pcs %>% select(name, PC2) %>% arrange(desc(PC2)) %>% slice(1:10)




##############################################
##
## Comprehension Check: Matrix Factorization
##
##############################################

# In this exercise set, we will be covering a topic 
# useful for understanding matrix factorization: 
# the singular value decomposition (SVD). 
# SVD is a mathematical result that is widely used 
# in machine learning, both in practice 
# and to understand the mathematical properties 
# of some algorithms. This is a rather advanced topic 
# and to complete this exercise set you will have 
# to be familiar with linear algebra concepts such 
# as matrix multiplication, orthogonal matrices, 
# and diagonal matrices.

# The SVD tells us that we can decompose an  N×p  matrix  Y  
# with  p<N  as Y=UDV⊤ with  U  and  V  orthogonal of dimensions  
# N×p  and  p×p  respectively and  D  a  p×p  diagonal matrix 
# with the values of the diagonal decreasing: 
# d1,1≥d2,2≥…dp,p 

# In this exercise, we will see one of the ways 
# that this decomposition can be useful. To do this, 
# we will construct a dataset that represents grade scores 
# for 100 students in 24 different subjects. The overall average 
# has been removed so this data represents the percentage point 
# each student received above or below the average test score. 
# So a 0 represents an average grade (C), a 25 is a high grade 
# (A+), and a -25 represents a low grade (F). You can simulate 
# the data like this:

library(tidyverse)

# set.seed(1987)
#if using R 3.6 or later, use `
set.seed(1987, sample.kind="Rounding") #` instead
n <- 100
k <- 8
Sigma <- 64  * 
  matrix(c(1, .75, .5, .75, 1, .5, .5, .5, 1), 3, 3) 
m <- MASS::mvrnorm(n, rep(0, 3), Sigma)
m <- m[order(rowMeans(m), decreasing = TRUE),]
y <- m %x% matrix(rep(1, k), 
                  nrow = 1) + matrix(rnorm(matrix(n*k*3)), 
                                     n, k*3)
colnames(y) <- c(paste(rep("Math",k), 1:k, sep="_"),
                 paste(rep("Science",k), 1:k, sep="_"),
                 paste(rep("Arts",k), 1:k, sep="_"))

# Our goal is to describe the student performances 
# as succinctly as possible. For example, we want to know 
# if these test results are all just a random independent numbers. 
# Are all students just about as good? Does being good 
# in one subject  imply you will be good in another? 
#   How does the SVD help with all this? We will go step by step 
# to show that with just three relatively small pairs 
# of vectors we can explain much of the variability 
# in this  100×24  dataset. 
 
# Q1
# You can visualize the 24 test scores for the 100 students 
# by plotting an image:
  
my_image <- function(x, zlim = range(x), ...) {
  colors = rev(RColorBrewer::brewer.pal(9, "RdBu"))
  cols <- 1:ncol(x)
  rows <- 1:nrow(x)
  image(
    cols,
    rows,
    t(x[rev(rows), , drop = FALSE]),
    xaxt = "n",
    yaxt = "n",
    xlab = "",
    ylab = "",
    col = colors,
    zlim = zlim,
    ...
  )
  abline(h = rows + 0.5, v = cols + 0.5)
  axis(side = 1, cols, colnames(x), las = 2)
}

my_image(y)

# How would you describe the data based on this figure?

#   The students that test well are at the top of 
#   the image and there seem to be three groupings by subject.
#   (i.e. red is high, blue is low)

# Explanation
# The following code can be used to estimate 
# the accuracy of the LDA:
  
fit_lda <- train(x, y, method = "lda")
fit_lda$results["Accuracy"]


# Q2
# You can examine the correlation between 
# the test scores directly like this:
  
my_image(cor(y), zlim = c(-1,1))
range(cor(y))
axis(side = 2, 1:ncol(y), rev(colnames(y)), las = 2)

# Which of the following best describes what you see?

#   There is correlation among all tests, but higher 
#   if the tests are in science and math 
#   and even higher within each subject.

# Explanation
# The following code can be used to make the plot:
  
t(fit_lda$finalModel$means) %>% data.frame() %>%
  mutate(predictor_name = rownames(.)) %>%
  ggplot(aes(cerebellum, hippocampus, label = predictor_name)) +
  geom_point() +
  geom_text() +
  geom_abline()


# Q3
# Use the function svd() to compute the SVD of y. 
# This function will return U, V, and the diagonal entries of D.

s <- svd(y)
names(s)

# You can check that the SVD works by typing:
  
y_svd <- s$u %*% diag(s$d) %*% t(s$v)
max(abs(y - y_svd))

# Compute the sum of squares of the columns of Y 
# and store them in ss_y. Then compute the sum of squares 
# of columns of the transformed YV and store them in ss_yv. 
# Confirm that sum(ss_y) is equal to sum(ss_yv).

ss_y <- colSums(y^2)
ss_yv <- colSums((y%*%s$v)^2)  

  
# 
# What is the value of sum(ss_y) 
# (and also the value of sum(ss_yv))?
sum(ss_y)  
sum(ss_yv)  
# 175434.6


# Explanation from the course web site

ss_y <- apply(y^2, 2, sum)
ss_yv <- apply((y%*%s$v)^2, 2, sum)
sum(ss_y)
sum(ss_yv)


# Q4
# We see that the total sum of squares is preserved. 
# This is because  V  is orthogonal. Now to start understanding 
# how  YV  is useful, plot ss_y against the column number 
# and then do the same for ss_yv.

plot(1:24, ss_y)
plot(1:24, ss_yv)
# 
# What do you observe?
#   ss_yv is decreasing and close to 0 
#   for the 4th column and beyond.


# Q5
# Now notice that we didn't have to compute ss_yv 
# because we already have the answer. How? Remember 
# that  YV=UD  and because  U  is orthogonal, we know 
# that the sum of squares of the columns of  UD  
# are the diagonal entries of  D  squared. 
# Confirm this by plotting the square root of ss_yv versus 
# the diagonal entries of  D .

plot(sqrt(ss_yv), s$d) # this is the correct plot

# Which of these plots is correct?
# option #1

# Explanation from the course web site
# This plot can be generated using the following code:

data.frame(x = sqrt(ss_yv), y = s$d) %>%
  ggplot(aes(x, y)) +
  geom_point()



# Q6
# So from the above we know that the sum of squares 
# of the columns of  Y  (the total sum of squares) 
# adds up to the sum of s$d^2 and that the transformation  YV  
# gives us columns with sums of squares equal to s$d^2. 
# Now compute the percent of the total variability 
# that is explained by just the first three columns of  YV .

# from 33.5.4 Principal component analysis:
# the total variability in our data can be defined as 
# the sum of the sum of squares of the columns.

sum(colSums((y %*% s$v)[,1:3]^2)) /
  sum(ss_yv)

sum(ss_yv[1:3]) / sum(ss_yv)

# What proportion of the total variability is explained 
# by the first three columns of  YV ?
#   Enter a decimal, not the percentage.
# 0.999797463 # NO
# [1] 0.9877922

# Explanation from the course web site
# The total variability explained can be calculated 
#using the following code: 

sum(s$d[1:3]^2) / sum(s$d^2).

# We see that almost 99% of the variability 
# is explained by the first three columns of  YV=UD . 
# So we get the sense that we should be able to explain 
# much of the variability and structure we found 
# while exploring the data with a few columns.


# Q7
# Use the sweep function to compute UD without constructing 
# diag(s$d) or using matrix multiplication.

s$u %*% s$d
identical(s$u %*% diag(s$d), s$u %*% t(diag(s$d)))

# Which code is correct?
# option #2
identical(s$u %*% diag(s$d), sweep(s$u, 2, s$d, FUN = "*"))


# Q8
# We know that  U1d1,1 , the first column of  UD , 
# has the most variability of all the columns of  UD . 
# Earlier we looked at an image of  Y  using my_image(y), 
# in which we saw that the student to student variability 
# is quite large and that students that are good 
# in one subject tend to be good in all. This implies 
# that the average (across all subjects) for each student 
# should explain a lot of the variability. Compute the 
# average score for each student, plot it against  U1d1,1 , 
# and describe what you find.
# 

svd(y)

UD <- sweep(s$u, 2, s$d, FUN = "*")
UD[,1]
stu_avg <- rowMeans(y)
pca <- prcomp(y)

plot(stu_avg, -s$u[,1] * s$d[1]) 
plot(stu_avg, -UD[,1]) 
plot(stu_avg, -pca$x[,1])

# What do you observe?
# There is a linearly increasing relationship between the average score for each student and  U1d1,1 .

# Explanation
# You can generate the plot using 
plot(-s$u[,1]*s$d[1], rowMeans(y))


# Q9
# We note that the signs in SVD are arbitrary because:
  
#   UDV⊤=(−U)D(−V)⊤ 
# With this in mind we see that the first column of  UD  
# is almost identical to the average score 
# for each student except for the sign.
# 
# This implies that multiplying  Y  by the first column 
# of  V  must be performing a similar operation to taking 
# the average. Make an image plot of  V  and describe 
# the first column relative to others and how this relates 
# to taking an average.

my_image(s$v)

# How does the first column relate to the others, 
# and how does this relate to taking an average?

#   The first column is very close to being a constant, 
#   which implies that the first column of YV is 
#   the sum of the rows of Y multiplied by some constant, 
#   and is thus proportional to an average.


# Explanation from the course web site
# The image plot can be made using 
my_image(s$v).


# Q10

# Explanation
# The plot can be made using the following code:
  
plot(s$u[, 1], ylim = c(-0.25, 0.25))
plot(s$v[, 1], ylim = c(-0.25, 0.25))
with(s, 
     my_image((u[, 1, drop = FALSE] * 
                 d[1]) %*% t(v[, 1, drop = FALSE])))
my_image(y)


# Q11
# Explanation
# The plot can be made using the following code:
  
plot(s$u[, 2], ylim = c(-0.5, 0.5))
plot(s$v[, 2], ylim = c(-0.5, 0.5))
with(s,
     my_image((u[, 2, drop = FALSE] *
                 d[2]) %*% t(v[, 2, drop = FALSE])))
my_image(resid)


# Q12
# The second column clearly relates to a student's difference 
# in ability in math/science versus the arts. We can see 
# this most clearly from the plot of s$v[,2]. Adding 
# the matrix we obtain with these two columns will help 
# with our approximation:
# 
# Y≈d1,1U1V⊤1+d2,2U2V⊤2 
# We know it will explain 
# sum(s$d[1:2]^2)/sum(s$d^2) * 100 percent 
# of the total variability. We can compute new residuals 
# like this:

resid <- 
  y - with(s,
           sweep(u[, 1:2], 2, d[1:2], FUN="*") 
           %*% t(v[, 1:2]))
my_image(cor(resid), zlim = c(-1,1))
axis(side = 2, 1:ncol(y), rev(colnames(y)), las = 2)


# and see that the structure that is left is driven by 
# the differences between math and science. Confirm this 
# by first plotting  U3 , then plotting  V⊤3  using 
# the same range for the y-axis limits, then making 
# an image of  U3d3,3V⊤3  and comparing it to the image of resid.

# Explanation
# 
# This plot can be made using the following code:
  
plot(s$u[, 3], ylim = c(-0.5, 0.5))
plot(s$v[, 3], ylim = c(-0.5, 0.5))
with(s, my_image((u[, 3, drop = FALSE] * d[3]) %*% t(v[, 3, drop = FALSE])))
my_image(resid)


# Q13 - UNGRADED
# The third column clearly relates to a student's difference 
# in ability in math and science. We can see this most clearly 
# from the plot of s$v[,3]. 

# We know it will explain: 
# sum(s$d[1:3]^2)/sum(s$d^2) * 100 percent 
# of the total variability. We can compute 
# new residuals like this:

resid <- y - with(s,sweep(u[, 1:3], 2, d[1:3], FUN="*") %*% t(v[, 1:3]))
my_image(cor(resid), zlim = c(-1,1))
axis(side = 2, 1:ncol(y), rev(colnames(y)), las = 2)

# Explanation
# 
# These plots can be made using the following code:
  
y_hat <-
  with(s, sweep(u[, 1:3], 2, d[1:3], FUN = "*") 
       %*% t(v[, 1:3]))
my_image(y, zlim = range(y))
my_image(y_hat, zlim = range(y))
my_image(y - y_hat, zlim = range(y))


##
## Comprehension Check: Clustering
## 

# These exercises will work with the tissue_gene_expression dataset, which is part of the dslabs package.

library(dslabs)
data("tissue_gene_expression")

# Q1
# Load the tissue_gene_expression dataset. 
# Remove the row means and compute the distance 
# between each observation. Store the result in d.
# 
# Which of the following lines of code correctly does this 
# computation?

d <- dist(tissue_gene_expression$x - 
            rowMeans(tissue_gene_expression$x))


# Q2
# Make a hierarchical clustering plot and add 
# the tissue types as labels.

# The hclust function implements this algorithm and it
# takes a distance as input.

h <- hclust(d)

# We can see the resulting groups using a dendrogram.
 
plot(h, cex = 0.65, main = "", xlab = "")

# You will observe multiple branches.
# 
# Which tissue type is in the branch farthest to the left?

# liver


# Q3
# Run a k-means clustering on the data with  K=7 . 
# Make a table comparing the identified clusters 
# to the actual tissue types. Run the algorithm 
# several times to see how the answer changes.

# # 34.2 k-means from the book:
# # The kmeans function included in R-base 
# # does not handle NAs. For illustrative purposes we
# # will fill out the NAs with 0s. In general, 
# # the choice of how to fill in missing data, or if one
# # should do it at all, should be made with care.
# 
# # x_0 <- x
# # x_0[is.na(x_0)] <- 0
# # k <- kmeans(x_0, centers = 10)
# 
# # The cluster assignments are in the cluster component:
# 
# # groups <- k$cluster
# 
# # Note that because the first center is chosen 
# # at random, the final clusters are random. We
# # impose some stability by repeating 
# # the entire function several times and averaging the
# # results. The number of random starting values 
# # to use can be assigned through the nstart
# # argument.
# 
# # k <- kmeans(x_0, centers = 10, nstart = 25)

k <- kmeans(tissue_gene_expression$x, centers = 7)
table(k$cluster, tissue_gene_expression$y)

# What do you observe for the clustering of the liver tissue?

# Liver is classified in a single cluster 
# roughly 20% of the time and in more than one cluster 
# roughly 80% of the time.

# Explanation
# The clustering and the table can be generated 
# using the following code:
  
cl <- kmeans(tissue_gene_expression$x, centers = 7)
table(cl$cluster, tissue_gene_expression$y)


# Q4
# Select the 50 most variable genes. 
# Make sure the observations show up 
# in the columns, that the predictor 
# are centered, and add a color bar to show 
# the different tissue types. 
# Hint: use the ColSideColors argument 
# to assign colors. Also, 
# use col = RColorBrewer::brewer.pal(11, "RdBu") 
# for a better use of colors.

# Part of the code is provided for you here:
  
library(RColorBrewer)
sds <- matrixStats::colSds(tissue_gene_expression$x)
ind <- order(sds, decreasing = TRUE)[1:50]
colors <- brewer.pal(7, "Dark2")[as.numeric(tissue_gene_expression$y)]
#BLANK

#Which line of code should replace #BLANK in the code above?
# option 1
heatmap(t(tissue_gene_expression$x[,ind]), 
        col = brewer.pal(11, "RdBu"), 
        scale = "row", ColSideColors = colors)










