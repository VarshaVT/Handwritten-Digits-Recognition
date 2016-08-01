
#################  Optical Recognition of Handwritten Digits  ##################

##Set the directory
getwd()
setwd("C:\\Users\\Varsha\\Music\\AnalyticsPracticum\\uci_files-selected")
fig.directory <- "./figures/"

#===================================================================================================================================

##Load the data
tra <- read.table(file = "C:\\Users\\Varsha\\Music\\AnalyticsPracticum\\uci_files-selected\\optdigits-orig_tra_linear.dat")
cv <- read.table(file = "C:\\Users\\Varsha\\Music\\AnalyticsPracticum\\uci_files-selected\\optdigits-orig_cv_linear.dat")

tra <- as.data.frame(tra)
cv <- as.data.frame(cv)

#===================================================================================================================================

##
##install the packages
packages <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg))
    install.packages(new.pkg, dependencies = TRUE, repos='http://cran.rstudio.com/')
  sapply(pkg, require, character.only = TRUE)
}

packages(c("RColorBrewer", "ElemStatLearn", "foreign", "tree", "RWeka", 
           "rpart", "maptree", "e1071", "cluster", "class", "FNN", "randomForest"))

#===================================================================================================================================

##Data Exploration of Training dataset
head(tra)
str(tra)
dim(tra)
attributes(tra)
nrow(tra)
ncol(tra)
row.names(tra)
colnames(tra)
summary(tra)
sapply(tra[1,], class)

##Data Exploration of validation dataset
head(cv)
str(cv)
dim(cv)
attributes(cv)
nrow(cv)
ncol(cv)
row.names(cv)
colnames(cv)
summary(cv)
sapply(cv[1,], class)

#===================================================================================================================================

## Check for missing values
sum(is.na(tra))
sum(is.na(cv))

##Change the label of last column of training set and make it as factor
tra[,1025] <- as.factor(tra[,1025])
colnames(tra) <- c(paste("X.", 1:1024, sep = ""), "Y")
class(tra[,1025])

###See the digits 0-9
levels(tra[,1025])
sapply(tra[1,], class)

##Change the label of last column of validation set and make it as factor
cv[,1025] <- as.factor(cv[,1025])
colnames(cv) <- c(paste("X.", 1:1024, sep = ""), "Y")
class(cv[,1025])

###See the digits 0-9
levels(cv[,1025])
sapply(cv[1,], class)

#===================================================================================================================================

##As the columns 1025 and 1026 are exact same drop column 1026
tra <- tra[, -1026]
cv <- cv[, -1026]

## Move the label to first column
tra <- tra[c(1025,1:1024)]
cv <- cv[c(1025,1:1024)]

## Set the colors for visualization of digits
digit_colors <- c("red", "white")

#"colorRampPalette": return functions that interpolate a set of given colors to create new color palettes
more_colors <- colorRampPalette(colors = digit_colors)
colors.plot <- colorRampPalette(brewer.pal(10, "Set3"))

#===================================================================================================================================

###
### Display digits of training data set by calculating the average of each digit
###

par(mfrow = c(4, 3), pty = "s", mar = c(1, 1, 1, 1), xaxt = "n", yaxt = "n")
digits.0_9 <- array(dim = c(10, 32 * 32))
for (dig in 0:9) {
  print(dig)
  digits.0_9[dig + 1, ] <- apply(tra[tra[, 1] == dig, -1], 2, sum)
  digits.0_9[dig + 1, ] <- digits.0_9[dig + 1, ]/max(digits.0_9[dig + 1, ]) * 1023
  z <- array(digits.0_9[dig + 1, ], dim = c(32, 32))
  z <- z[, 32:1]  ##right side up
  image(1:32, 1:32, z, main = dig, col = more_colors(1024))
}

head(digits.0_9)

##creating pdf
#pdf(file=paste0(fig.directory,'trainDigit.pdf'),)
par(mfrow = c(4, 4), pty = "s", mar = c(3, 3, 3, 3), xaxt = "n", yaxt = "n")
for (i in 1:10) {
  z <- array(as.vector(as.matrix(tra[i, -1])), dim = c(32, 32))
  z <- z[, 32:1]  ##right side up
  image(1:32, 1:32, z, main = tra[i, 1], col = more_colors(1024))
  print(i)
}

#==============================================================================================================================

## Total numbers in training dataset
dig_lable <- table(tra$Y)
par(mfrow = c(1, 1))
par(mar = c(5, 4, 4, 2) + 0.1)  # increase y-axis margin.
plot <- plot(tra$Y, col = colors.plot(10), main = "Total Number of Digits in Training Set", 
             ylim = c(0, 300), ylab = "Examples Number")
text(x = plot, y = dig_lable+20, labels = dig_lable, cex = 0.75)

par(mfrow = c(1, 1))
percentage <- round(dig_lable/sum(dig_lable) * 100)
labels <- paste0(row.names(dig_lable), " (", percentage, "%) ")  # add percents to labels
pie(dig_lable, labels = labels, col = colors.plot(10), main = "Total Number of Digits (Training Set)")

## Total numbers in validation dataset
dig_lable <- table(cv$Y)
par(mfrow = c(1, 1))
par(mar = c(5, 4, 4, 2) + 0.1)  # increase y-axis margin.
plot <- plot(cv$Y, col = colors.plot(10), main = "Total Number of Digits (Validation Set)", 
             ylim = c(0, 200), ylab = "Examples Number")
text(x = plot, y = dig_lable + 15, labels = dig_lable, cex = 0.75)

par(mfrow = c(1, 1))
percentage <- round(dig_lable/sum(dig_lable) * 100)
labels <- paste0(row.names(dig_lable), " (", percentage, "%) ")  # add percents to labels
pie(dig_lable, labels = labels, col = colors.plot(10), main = "Total Number of Digits (Validation Set)")


#====================================================================================================================================

        #####  MODEL 1  #####

##Classification. Predictive Model. RPart
#proc.time() determines how much real and CPU time (in seconds) the currently running R process has already taken.
x <- proc.time()
fit.rpart <- rpart(tra$Y ~ ., method = "class", data = tra)
proc.time() - x

#printcp() Displays the cp table for fitted rpart object
printcp(fit.rpart)

plot(fit.rpart, uniform = TRUE, main = "Classification (RPART). Tree of Handwritten Digit Recognition ")
text(fit.rpart, all = TRUE, cex = 0.75)

draw.tree(fit.rpart, cex = 0.5, nodeinfo = TRUE, col = gray(0:8/8))

###Time caculation for prediction
y <- proc.time()
prediction.rpart <- predict(fit.rpart, newdata = cv, type = "class")
proc.time() - y

#Confusion Matrix (RPart)
table(`Actual` = cv$Y, `Predicted` = prediction.rpart)

error.rate.rpart <- sum(cv$Y != prediction.rpart)/nrow(cv)
print(paste0("Accuracy: ", 1 - error.rate.rpart))

#Predict Digit for Example 1 (RPart)
row <- 1
prediction.digit <- as.vector(predict(fit.rpart, newdata = cv[row, ], type = "class"))
print(paste0("Actual Digit: ", as.character(cv$Y[row]))) 

print(paste0("Predicted Digit: ", prediction.digit))

z <- array(as.vector(as.matrix(cv[row, -1])), dim = c(32, 32))
z <- z[, 32:1]  ##right side up
par(mfrow = c(1, 3), pty = "s", mar = c(1, 1, 1, 1), xaxt = "n", yaxt = "n")
image(1:32, 1:32, z, main = cv[row, 1], col = more_colors(1024))

##Errors with tree based methods (rpart)
errors <- as.vector(which(cv$Y != prediction.rpart))
print(paste0("Error Numbers: ", length(errors))) 

predicted <- as.vector(prediction.rpart)
par(mfrow = c(29, 8), pty = "s", mar = c(.5, .5, .5, .5), xaxt = "n", yaxt = "n")
for (i in 1:length(errors)) {
  z <- array(as.vector(as.matrix(cv[errors[i], -1])), dim = c(32, 32))
  z <- z[, 32:1]  ##right side up
  image(1:32, 1:32, z, main = paste0("act:", as.character(cv$Y[i]), 
                                     " - pre:", predicted[errors[i]]), col = more_colors(1024))
}

#========================================================================================================================

#####  MODEL 2  #####


##
##Classification. Predictive Model. Naive Bayes Algorithm
##

#Naive Bays Algorithm{e1071}: Computes the conditional a-posterior probabilities of a 
#categorical class variable given independent predictor variables using the Bayes rule.

x <- proc.time()
fit.naiveBayes <- naiveBayes(tra$Y ~ ., data = tra)
proc.time() - x

summary(fit.naiveBayes)

###Time caculation for prediction
y <- proc.time()
prediction.naiveBayes <- predict(fit.naiveBayes, newdata = cv, type = "class")
proc.time() - y

#Confusion Matrix (naiveBayes)
table(`Actual` = cv$Y, `Predicted` = prediction.naiveBayes)

error.rate.naiveBayes <- sum(cv$Y != prediction.naiveBayes)/nrow(cv)
print(paste0("Accuracy: ", 1 - error.rate.naiveBayes)) 

##Predict Digit for Example 1 (naiveBayes)
# All Prediction for Row 1
row <- 1
prediction.digit <- as.vector(predict(fit.naiveBayes, newdata = cv[row, ], type = "class"))
print(paste0("Actual Digit: ", as.character(cv$Y[row]))) 

print(paste0("Predicted Digit: ", prediction.digit)) #[1] "Predicted Digit: 5"

z <- array(as.vector(as.matrix(cv[row, -1])), dim = c(32, 32))
z <- z[, 32:1]  ##right side up
par(mfrow = c(1, 3), pty = "s", mar = c(1, 1, 1, 1), xaxt = "n", yaxt = "n")
image(1:32, 1:32, z, main = cv[row, 1], col = more_colors(1024))

##Errors with Naive Bayes
errors <- as.vector(which(cv$Y != prediction.naiveBayes))
print(paste0("Error Numbers: ", length(errors))) 

predicted <- as.vector(prediction.naiveBayes)
par(mfrow = c(16, 8), pty = "s", mar = c(1, 1, 1, 1), xaxt = "n", yaxt = "n")
for (i in 1:length(errors)) {
  z <- array(as.vector(as.matrix(cv[errors[i], -1])), dim = c(32, 32))
  z <- z[, 32:1]  ##right side up
  image(1:32, 1:32, z, main = paste0("act:", as.character(cv$Y[i]), 
                                     " - pre:", predicted[errors[i]]), col = more_colors(1024))
}

#========================================================================================================================

#####  MODEL 3  #####


##
##Classification. Predictive Model. SVM (Support Vector Machine) Algorithm
##

#svm {e1071}:is used to train a support vector machine. 
#It can be used to carry out general regression and classification (of nu and epsilon-type), 
#as well as density-estimation.

x <- proc.time()
fit.svm <- svm(tra$Y ~ ., method = "class", data = tra)
proc.time() - x

summary(fit.svm) #Number of Support Vectors:  1141

##plot(model.svm, tra, tra$Y ~z )


###Time caculation for prediction
y <- proc.time()
prediction.svm <- predict(fit.svm, newdata = cv, type = "class")
proc.time() - y

##Confusion Matrix (SVM)
table(`Actual` = cv$Y, `Predicted` = prediction.svm)

error.rate.svm <- sum(cv$Y != prediction.svm)/nrow(cv)
print(paste0("Accuracy (Precision): ", 1 - error.rate.svm)) 

#Predict Digit for Example 1 (SVM)
# All Prediction for Row 1
row <- 1
prediction.digit <- as.vector(predict(fit.svm, newdata = cv[row, ], type = "class"))
print(paste0("Actual Digit: ", as.character(cv$Y[row]))) 

print(paste0("Predicted Digit: ", prediction.digit)) 

z <- array(as.vector(as.matrix(cv[row, -1])), dim = c(32, 32))
z <- z[, 32:1]  ##right side up
par(mfrow = c(1, 3), pty = "s", mar = c(1, 1, 1, 1), xaxt = "n", yaxt = "n")
image(1:32, 1:32, z, main = cv[row, 1], col = more_colors(1024))

##Errors with Support Vector Machine (SVM)
errors <- as.vector(which(cv$Y != prediction.svm))
print(paste0("Error Numbers: ", length(errors))) 

predicted <- as.vector(prediction.svm)
par(mfrow = c(6, 4), pty = "s", mar = c(1, 1, 1, 1), xaxt = "n", yaxt = "n")
for (i in 1:length(errors)) {
  z <- array(as.vector(as.matrix(cv[errors[i], -1])), dim = c(32, 32))
  z <- z[, 32:1]  ##right side up
  image(1:32, 1:32, z, main = paste0("act:", as.character(cv$Y[i]), 
                                     " - prd:", predicted[errors[i]]), col = more_colors(1024))
}


#========================================================================================================================


#####  MODEL 4  #####


##
##Classification. Fast Nearest Neighbors (FNN) Algorithm
##

x <- proc.time()
# Avoid Name Collision (knn)
fit.fnn <- FNN::knn(tra[, -1], cv[, -1], tra$Y, 
                    k = 10, algorithm = "cover_tree")
proc.time() - x

summary(fit.fnn)

##Confusion Matrix (FNN)
table(`Actual` = cv$Y, `Predicted` = fit.fnn)

error.rate.fnn <- sum(cv$Y != fit.fnn)/nrow(cv)
print(paste0("Accuracy: ", 1 - error.rate.fnn)) 

##Predict Digit for Example 1 (FNN)
row <- 1
prediction.digit <- fit.fnn[row]
print(paste0("Actual Digit: ", as.character(cv$Y[row]))) 
print(paste0("Predicted Digit: ", prediction.digit)) 

par(mfrow = c(1, 3), pty = "s", mar = c(1, 1, 1, 1), xaxt = "n", yaxt = "n")
z <- array(as.vector(as.matrix(cv[row, -1])), dim = c(32, 32))
z <- z[, 32:1]  ##right side up
image(1:32, 1:32, z, main = cv[row, 1], col = more_colors(1024))

##Errors with Fast Nearest Neighbors (FNN)
errors <- as.vector(which(cv$Y != fit.fnn))
print(paste0("Error Numbers: ", length(errors))) 

predicted <- as.vector(fit.fnn)
par(mfrow = c(6, 4), pty = "s", mar = c(1, 1, 1, 1), xaxt = "n", yaxt = "n")
for (i in 1:length(errors)) {
  z <- array(as.vector(as.matrix(cv[errors[i], -1])), dim = c(32, 32))
  z <- z[, 32:1]  ##right side up
  image(1:32, 1:32, z, main = paste0("act:", as.character(cv$Y[i]), 
                                     " - pre:", predicted[errors[i]]), col = more_colors(1024))
}


#========================================================================================================================

#####  MODEL 5  #####

###
###Classification. Predictive Model. Random Forest Algorithm
###

x <- proc.time()
fit.randomForest <- randomForest(tra$Y ~ ., data = tra, method = "class", ntree=200)
proc.time() - x


summary(fit.randomForest)

###Time caculation for prediction
y <- proc.time()
prediction.randomForest <- predict(fit.randomForest, newdata = cv, type = "class")
proc.time() - y

##Confusion Matrix Random Forest
table(`Actual` = cv$Y, `Predicted` = prediction.randomForest)

error.rate.randomForest <- sum(cv$Y != prediction.randomForest)/nrow(cv)
print(paste0("Accuracy: ", 1 - error.rate.randomForest)) 

#Predict Digit for Example 1 (Random Forest)
# All Prediction for Row 1
row <- 1
prediction.digit <- as.vector(predict(fit.randomForest, newdata = cv[row, ], type = "class"))
print(paste0("Actual Digit: ", as.character(cv$Y[row]))) 

print(paste0("Predicted Digit: ", prediction.digit)) 

z <- array(as.vector(as.matrix(cv[row, -1])), dim = c(32, 32))
z <- z[, 32:1]  ##right side up
par(mfrow = c(1, 3), pty = "s", mar = c(1, 1, 1, 1), xaxt = "n", yaxt = "n")
image(1:32, 1:32, z, main = cv[row, 1], col = more_colors(1024))

##Errors with Random Forest
errors <- as.vector(which(cv$Y != prediction.randomForest))
print(paste0("Error Numbers: ", length(errors))) 

predicted <- as.vector(prediction.randomForest)
par(mfrow = c(5, 4), pty = "s", mar = c(1, 1, 1, 1), xaxt = "n", yaxt = "n")
for (i in 1:length(errors)) {
  z <- array(as.vector(as.matrix(cv[errors[i], -1])), dim = c(32, 32))
  z <- z[, 32:1]  ##right side up
  image(1:32, 1:32, z, main = paste0("act:", as.character(cv$Y[i]), 
                                     " - pre:", predicted[errors[i]]), col = more_colors(1024))
}

#========================================================================================================================


#####  MODEL 6  #####

##
##Classification. k-Nearest Neighbors (kNN) Algorithm
##

#IBk(RWeka): provides a k-nearest neighbors classifier
x <- proc.time()
##Knn is also provided by Weka as a class "IBk"
fit.knn <- IBk(tra$Y ~ ., data = tra) #IBk(): R interfaces to Weka lazy learners
proc.time() - x

summary(fit.knn) ##Correctly Classified Instances=1934(100%)


###Time caculation for prediction
y <- proc.time()
prediction.knn <- predict(fit.knn, newdata = cv, type = "class")
proc.time() - y

#Confusion Matrix (kNN)
table(`Actual` = cv$Y, `Predicted` = prediction.knn)

error.rate.knn <- sum(cv$Y != prediction.knn)/nrow(cv)
print(paste0("Accuracy: ", 1 - error.rate.knn)) 

##Predict Digit for Example 1 (kNN)
row <- 1
prediction.digit <- as.vector(predict(fit.knn, newdata = cv[row, ], type = "class"))
print(paste0("Actual Digit: ", as.character(cv$Y[row])))

print(paste0("Predicted Digit: ", prediction.digit)) 
par(mfrow = c(1, 3), pty = "s", mar = c(1, 1, 1, 1), xaxt = "n", yaxt = "n")
z <- array(as.vector(as.matrix(cv[row, -1])), dim = c(32, 32))
z <- z[, 32:1]  ##right side up
image(1:32, 1:32, z, main = cv[row, 1], col = more_colors(1024))

##Errors with K Nearest Neighbours (KNN)
errors <- as.vector(which(cv$Y != prediction.knn))
print(paste0("Error Numbers: ", length(errors))) 

predicted <- as.vector(prediction.knn)
par(mfrow = c(4, 4), pty = "s", mar = c(1, 1, 1, 1), xaxt = "n", yaxt = "n")
for (i in 1:length(errors)) {
  z <- array(as.vector(as.matrix(cv[errors[i], -1])), dim = c(32, 32))
  z <- z[, 32:1]  ##right side up
  image(1:32, 1:32, z, main = paste0("act:", as.character(cv$Y[i]), 
                                     " - pre:", predicted[errors[i]]), col = more_colors(1024))
}


#========================================================================================================================



#==========================================================================================================================
###PLOTS: Actual vs Predicted ####

### Model1: rpart
prediction.rpart1 <- as.data.frame(prediction.rpart)
View(prediction.rpart1)
actual.rpart <- as.data.frame(cv$Y)
View(actual.rpart)
rpart <- cbind(actual.rpart, prediction.rpart1)
View(rpart)
write.csv(rpart, file = "C:\\Users\\Varsha\\Music\\AnalyticsPracticum\\uci_files-selected\\rpartResult.csv")
scatter.smooth(rpart$`cv$Y`, rpart$prediction.rpart)
with(rpart, scatter.smooth(rpart$prediction.rpart, rpart$`cv$Y`, main="Tree (Error: 232, Total: 946)",  xlab = "Predicted Digits", ylab = "Actual Digits", lpars = list(col = "red", lwd = 1, lty = 1)))


### Model2: Naive Bayes Algorithm
prediction.naiveBayes1 <- as.data.frame(prediction.naiveBayes)
View(prediction.naiveBayes1)
actual.naiveBayes <- as.data.frame(cv$Y)
View(actual.naiveBayes)
naiveBayes <- cbind(actual.naiveBayes, prediction.naiveBayes1)
View(naiveBayes)
write.csv(rpart, file = "C:\\Users\\Varsha\\Music\\AnalyticsPracticum\\uci_files-selected\\naiveBayesResult.csv")
with(naiveBayes, scatter.smooth(naiveBayes$prediction.naiveBayes, naiveBayes$`cv$Y`, main="NaiveBays (Error: 127, Total: 946)",  xlab = "Predicted Digits", ylab = "Actual Digits", lpars = list(col = "red", lwd = 1, lty = 1)))



### Model3: Support Vector Machine
prediction.svm1 <- as.data.frame(prediction.svm)
View(prediction.svm1)
actual.svm <- as.data.frame(cv$Y)
View(actual.svm)
svm <- cbind(actual.svm, prediction.svm1)
View(svm)
write.csv(knn, file = "C:\\Users\\Varsha\\Music\\AnalyticsPracticum\\uci_files-selected\\svmResult.csv")
with(svm, scatter.smooth(svm$prediction.svm, svm$`cv$Y`, main="SVM (Error: 24, Total: 946)",  xlab = "Predicted Digits", ylab = "Actual Digits", lpars = list(col = "red", lwd = 1, lty = 1)))

### Model4: FNN::KNN
prediction.digit1 <- as.data.frame(prediction.digit)
View(prediction.digit1)
actual.fnn <- as.data.frame(cv$Y)
View(actual.fnn)
fnn <- cbind(actual.fnn, prediction.digit1)
View(fnn)
write.csv(fnn, file = "C:\\Users\\Varsha\\Music\\AnalyticsPracticum\\uci_files-selected\\fnnResult.csv")
scatter.smooth(fnn$`cv$Y`, fnn$prediction.digit)
with(fnn, scatter.smooth(fnn$prediction.digit, fnn$`cv$Y`, main="FNN::knn (Error: 23, Total: 946)",  xlab = "Predicted Digits", ylab = "Actual Digits", lpars = list(col = "red", lwd = 1, lty = 1)))

### Model5: Random Forest
prediction.randomForest1 <- as.data.frame(prediction.randomForest)
View(prediction.randomForest1)
actual.randomForest <- as.data.frame(cv$Y)
View(actual.randomForest)
randomForest <- cbind(actual.randomForest, prediction.randomForest1)
View(randomForest)
write.csv(randomForest, file = "C:\\Users\\Varsha\\Music\\AnalyticsPracticum\\uci_files-selected\\randomForestResult.csv")
scatter.smooth(randomForest$`cv$Y`, randomForest$prediction.randomForest)
with(randomForest, scatter.smooth(randomForest$prediction.randomForest, randomForest$`cv$Y`, main="Random Forest (Error: 17, Total: 946)",  xlab = "Predicted Digits", ylab = "Actual Digits", lpars = list(col = "red", lwd = 1, lty = 1)))

### Model6: RWeka::IBk
prediction.knn1 <- as.data.frame(prediction.knn)
View(prediction.knn1)
actual.knn <- as.data.frame(cv$Y)
View(actual.knn)
knn <- cbind(actual.knn, prediction.knn1)
View(knn)
write.csv(knn, file = "C:\\Users\\Varsha\\Music\\AnalyticsPracticum\\uci_files-selected\\knnResult.csv")
scatter.smooth(knn$`cv$Y`, knn$prediction.knn)
with(knn, scatter.smooth(knn$prediction.knn, knn$`cv$Y`, main="RWeka::KNN (Error: 13, Total: 946)",  xlab = "Predicted Digits", ylab = "Actual Digits", lpars = list(col = "red", lwd = 1, lty = 1)))



HandwricttenDiditsPrediction <- cbind(cv$Y, rpart$prediction.rpart, naiveBayes$prediction.naiveBayes, svm$prediction.svm, fnn$prediction.digit, randomForest$prediction.randomForest, knn$prediction.knn)
colnames(HandwricttenDiditsPrediction)  <-  c("ActualDigits", "rprtPrediction", "naiveBayesPrediction", "svmPrediction", "FNNPrediction", "randomForestPrediction", "kNNPrediction")

write.csv(HandwricttenDiditsPrediction, file = "C:\\Users\\Varsha\\Music\\AnalyticsPracticum\\uci_files-selected\\HandwricttenDiditsPrediction.csv")
