## Credit Risk

tr1 <- read.csv(file.choose())
ts1 <- read.csv(file.choose())

##Duplicated

sum(duplicated(tr1))

##Outlier

library(DMwR)

sum(!complete.cases(tr1))
colMeans(is.na(tr1)) * 100

missrow1 <- which(!complete.cases(tr1))
miss1 <- tr1[missrow1,]
notmiss1 <- tr1[-missrow1,]

notmiss2 <- notmiss1[, -21]
notmiss2 <- notmiss2[, - c(3, 4, 8, 10)]
notmiss2 <- as.matrix(notmiss2)

lof <- lofactor(data = notmiss2, k = 5)


notmiss2 <- data.frame(notmiss2)
notmiss2$lof <- lof
hist(notmiss2$lof, breaks = 40)
sum(notmiss2$lof > 3)
notmiss2 <- cbind(notmiss2[, 1:2], notmiss1[, 3:4], notmiss2[, 3:5], notmiss1[, 8], notmiss2[, 6], notmiss1[, 10], notmiss2[, 7:16], notmiss1[, 21], notmiss2[, 17])
names(notmiss2)[8:10] <- c("home_ownership", "dti", "purpose")
names(notmiss2)[21:22] <- c("bad_loans", "lof")
notmiss2 <- notmiss2[notmiss2$lof < 4,]
notmiss2 <- notmiss2[, -22]
tr2 <- rbind(miss1, notmiss2)
colMeans(is.na(tr2)) * 100

##Missing Value
tr2$payment_inc_ratio[is.na(tr2$payment_inc_ratio)] <- round(mean(tr2$payment_inc_ratio, na.rm = TRUE))
tr2$delinq_2yrs[is.na(tr2$delinq_2yrs)] <- round(mean(tr2$delinq_2yrs, na.rm = TRUE))
tr2$delinq_2yrs_zero[is.na(tr2$delinq_2yrs_zero)] <- round(mean(tr2$delinq_2yrs_zero, na.rm = TRUE))
tr2$inq_last_6mths[is.na(tr2$inq_last_6mths)] <- round(mean(tr2$inq_last_6mths, na.rm = TRUE))
tr2$open_acc[is.na(tr2$open_acc)] <- round(mean(tr2$open_acc, na.rm = TRUE))
tr2$pub_rec[is.na(tr2$pub_rec)] <- round(mean(tr2$pub_rec, na.rm = TRUE))
tr2$pub_rec_zero[is.na(tr2$pub_rec_zero)] <- round(mean(tr2$pub_rec_zero, na.rm = TRUE))
colMeans(is.na(tr2)) * 100

##balancing
library(DMwR)
table(tr2$bad_loans)
tr2$bad_loans <- as.factor(tr2$bad_loans)
tr3 <- SMOTE(bad_loans ~ ., tr2, perc.over = 200,
             k = 5, perc.under = 100)
table(tr3$bad_loans)
tr4 <- tr3

##Modeling
set.seed(1234)
sm1 <- sample(1:nrow(tr3), nrow(tr3) / 3, replace = FALSE)
test1 <- tr3[sm1,]
train1 <- tr3[-sm1,]

###Decision Tree

library(rpart)
library(rpart.plot)
tr3 <- tr4
fit1 <- rpart(bad_loans ~ ., method = "class", data = train1)
rpart.plot(fit1, type = 4, extra = 2, cex = 0.7)
pred1 <- predict(fit1, test1)
p1 <- ifelse(pred1[, 2] > 0.5, 1, 0)
table(p1, test1$bad_loans)
mean(p1 == test1$bad_loans)

library(ROSE)
accuracy.meas(test1$bad_loans, pred1[, 2])
res1 <- predict(fit1, ts1, type = "class")
write.table(res1, "D:/Credit Risk/Res.csv",
            row.names = FALSE, col.names = FALSE, quote = FALSE)

fit2 <- rpart(bad_loans ~ ., data = train1, method = "class", control = rpart.control(minsplit = 5, maxdepth = 8))
rpart.plot(fit2, type = 3, extra = 2, cex = 0.5)
pred2 <- predict(fit2, test1, type = "class")
table(pred2, test1$bad_loans)
mean(pred2 == test1$bad_loans)
library(ROSE)
accuracy.meas(test1$bad_loans, pred2)


###Naive Bayes
library(e1071)
tr3 <- tr4
set.seed(1234)
sm1 <- sample(1:nrow(tr3), nrow(tr3) / 3, replace = FALSE)
test1 <- tr3[sm1,]
train1 <- tr3[-sm1,]
nb1 <- naiveBayes(bad_loans ~ ., data = train1)
pred3 <- predict(nb1, test1, type = "class")
table(pred3, test1$bad_loans)
mean(pred3 == test1$bad_loans)
accuracy.meas(test1$bad_loans, pred3)

###Neural network

library(e1071)
library(RWeka)
library(nnet)

tr3 <- Normalize(bad_loans ~ ., tr3)
levels(tr3$pymnt_plan)
tr3$pymnt_plan <- as.numeric(tr3$pymnt_plan)
levels(tr3$grade)
tr3$grade <- as.numeric(tr3$grade)
levels(tr3$home_ownership)
tr3$home_ownership <- as.numeric(tr3$home_ownership)
levels(tr3$purpose)
tr3$purpose <- as.numeric(tr3$purpose)
tr3$loan_amnt <- as.numeric(tr3$loan_amnt)
tr3$funded_amnt <- as.numeric(tr3$funded_amnt)
tr3$short_emp <- as.numeric(tr3$short_emp)
tr3$emp_length_num <- as.numeric(tr3$emp_length_num)
tr3$bad_loans <- as.numeric(tr3$bad_loans)
for (i in 12:19) {
    tr3[, i] <- as.numeric(tr3[, i])
}
str(tr3)
set.seed(1234)
sm1 <- sample(1:nrow(tr3), nrow(tr3) / 3, replace = FALSE)
test1 <- tr3[sm1,]
train1 <- tr3[-sm1,]
nm1 <- names(tr3)[-21]
fr1 <- as.formula(paste("bad_loans~", paste(nm1, collapse = "+")))
nn1 <- nnet(fr1, train1, size = 10)
tune1 <- tune.nnet(bad_loans ~ ., data = tr3, size = 5:10, decay = seq(0.01, 0.1, 0.02), tunecontrol = tune.control(sampling = "fix"))
tune1$best.parameters
nn2 <- nnet(fr1, train1, size = 10, maxit = 150, abstol = 1.0e-5, decay = 0.01)

library(devtools)
source_url("https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet-plot-update.r")
x11()
plot.nnet(nn1)
pred4 <- predict(nn1, test1)
table(pred4, test1$bad_loans)
mean(pred4 == test1$bad_loans)
library(ROSE)
accuracy.meas(test1$bad_loans, pred4)
