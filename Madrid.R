install.packages('caret', dependencies = TRUE)
install.packages("caTools")
install.packages("quantreg")
install.packages("car")
install.packages("GGally")
install.packages("psych")
install.packages("corrplot")
install.packages("MASS")
install.packages("broom")
install.packages("WVPlots")
library(readxl)
library(ggplot2)
library(caret)
library(caTools)
library(quantreg)
library(car)
library(GGally)
library(psych)
library(corrplot)
library(MASS)
library(broom)
library(WVPlots)
library(dplyr)
library(tidyverse)
madrid <- read_excel("C:\\Users\\HP\\OneDrive\\Desktop\\madrid_real_estate.xlsx", 
                     sheet = "cleaned_dataset")
glimpse(madrid)

#see how many unique locations are there in the dataset
unique(madrid$area)

#So there are 145 unique locations in the dataset.
#Now lets's aggregate the data by location and see how many properties are listed 
#for each location.

madrid_location_unique <- madrid %>% count(area) %>% arrange(n)
madrid_location_unique

#Now we'll add a coloumn of each area in sq-ft as most people use them as a measure.

madrid <- madrid %>% mutate(area_sq_ft = sq_mt_built * 10.8)
glimpse(madrid)

#So now we'll see if there are any unsual sized bedrooms and thus properties by 
#making sure that each bedroom is over 300
madrid <- madrid %>% mutate(sq_ft_per_bedroom = area_sq_ft / n_rooms) %>% 
                            arrange(sq_ft_per_bedroom)
glimpse(madrid)

#Now according to architects guide in spain a single bedroom should atleast be 9 
#sq-m or 100 sq-ft. 
#So we remove the properties below that.
madrid <- madrid %>% filter(sq_ft_per_bedroom > 100)
glimpse(madrid)

#Now we'll add a price per sq feet to the data
madrid <- madrid %>% mutate(buy_price_per_sq_ft = buy_price / area_sq_ft)
glimpse(madrid)

#Now we'll see if there are any anomalies in the price.
summary(madrid$buy_price_per_sq_ft)
madrid <- madrid[order(madrid$area),]
glimpse(madrid)

summary(madrid$buy_price_per_sq_ft)

#We can see that the min price is 41 Euros per sq-ft which is very less. 
#So, we want to filter out properties which are 1 standard deviation away from 
#the mean in each area.
madrid_1 <- madrid %>% group_by(area) %>% 
             filter(buy_price_per_sq_ft <= quantile(buy_price_per_sq_ft, 0.84), buy_price_per_sq_ft >= quantile(buy_price_per_sq_ft, 0.16))

madrid_1
summary(madrid_1$buy_price_per_sq_ft)

#So now our data points have reduced from 17,796 to 12,078. 
#But these are the ones we want to feed into our model because they make up 68% of the data 
#Or are 1 standard deviation away from meam, i.e. the ideal data points for our model.
mean_buy_price <- mean(madrid_1$buy_price_per_sq_ft)
mean_buy_price
sd_buy_price <- sd(madrid_1$buy_price_per_sq_ft)
sd_buy_price

#So we know our mean and standard deviation of buy price per sq-ft.
#Now we will plot the price_per_sq_ft with the number of data points to get a basic idea of our data.
price_per_sq_ft <- madrid_1$buy_price_per_sq_ft
price_variration_plot <- hist(price_per_sq_ft, breaks = 40, main = "Price per sq-ft in Madrid", xlab = "Price per sq-ft", col = "lightblue", prob = TRUE) 
x <- seq(min(price_per_sq_ft), max(price_per_sq_ft), length = 60)
f <- dnorm(x, mean = mean(price_per_sq_ft), sd = sd(price_per_sq_ft))
lines(x, f, col = "red", lwd = 3)


#We can see from the plot that our data is bi-modal, so we cannot apply lsr. We need to do quantile regression.

#Now we take a look at the number of bathrooms in the dataset
summary(madrid_1$n_bathrooms)
bathrooms_plot <- hist(madrid_1$n_bathrooms, breaks = 14, main = "Frequency of Bathrooms", xlab = "No. of Bathrooms", xlim = c(0,7), col = "lightgreen")
bathrooms_plot

#So we see from the graph that there are mostly 1-2 bathrooms in the dataset.
#Now we have cleaned and wrangled our data and it is ready for model building.

glimpse(madrid_1)

#Now that we are building a model we need to remove some coloumns which don't necessarily help our model.
#Like the buy_price_per_sq_mt & buy_price_per_sq_ft.
madrid_2 <- subset(madrid_1, select = -c(buy_price_per_sq_mt, buy_price_per_sq_ft))
glimpse(madrid_2)

#Now we convert all the logical vectors to numeric.
cols <- sapply(madrid_2, is.logical)
madrid_2[,cols] <- lapply(madrid_2[,cols], as.numeric)

glimpse(madrid_2)


#Machine learning models can't interpret Text data. 
#So we have to convert all text data into numeric coloumns,
#One way to do that is to use One Hot Encoding.

dmy <- dummyVars(" ~ .", data = madrid_2, fullRank = T)
madrid_3 <- data.frame(predict(dmy, newdata = madrid_2))

glimpse(madrid_3)
dim(madrid_3)

#Changing names of coloumns so that they can be easily used in the model.
f <- paste0(rep(LETTERS[1:26], 7), rep(1:7, each = 26))
n <- f[1:165]

n

colnames(madrid_3) <- n
madrid_3


#Now we check all the NA's & NAN's & INF's one more time and replace them with a 0.

data_new <- madrid_3
data_new[is.na(data_new) | data_new == "Inf"] <- NA 
data_new[is.infinite(data_new)] <- NA

glimpse(madrid_3)

is.nan.data.frame <- function(x)
  do.call(cbind, lapply(x, is.nan))
madrid_3[is.nan(madrid_3)] <- 0


madrid_3

madrid_3[madrid_3 == -Inf] <- 0
madrid_3

#Now we remove the id & sq_ft_per_bedroom coloumn because they don't add much to the regression.
madrid_3 <- madrid_3[, 2:164]
glimpse(madrid_3)
summary(madrid_3)

#Now we convert all the data in our data frame to the type numeric so that regression can be done.
df
glimpse(df)
dim(df)
f_df <- as.data.frame(df)
f_df
glimpse(f_df)
dim(f_df)

final_df <- f_df[, -c(163, 146)]
final_df
glimpse(final_df)
dim(final_df)
#Now we apply the formula for quantile regression and make the model.
attach(final_df)
summary(final_df)

#Now let's check the co-relation between all the independent variables.
ind_var <- final_df[, -c(147)]
glimpse(ind_var)
dim(ind_var)
c <- cor(ind_var)
c
#Now since the correlation of all the colomns are not visible we set a limit of 0.4 as max co-relation.
c_1 <- findCorrelation(cor(ind_var), cutoff = 0.7, verbose = FALSE, exact = TRUE)
c_1
c[,147]

#So we need to remove the colomns we got from c_1 who have a correlation more than 0.7.
final_df <- final_df[, -c(148)]
glimpse(final_df)
dim(final_df)


#Now we plot the scatterplot to see the corelation:
pairs(final_df, col="blue", main="Scatterplots")
y <- final_df$S6
y 
x <- final_df[,-c(147)]
x
glimpse(x)
dim(x)

cor(y,x)

#Now we plot a histogram and density function line for dependent variable.
his <- hist(y, prob=TRUE, col = "blue", border = "black")
lines(density(y))
his

#Now we split our dataset into Test Data an Train Data for the ML-Model.
# Splitting the dataset into the Training set and Test set.

set.seed(123)
data <- final_df
head(data)
dim(data)
split1<- sample(c(rep(0, 0.8 * nrow(data)), rep(1, 0.2 * nrow(data))))
split1
table(split1)

#Now we create our train data.
train <- data[split1 == 0, ]
train <- as.data.frame(train)
head(train)
glimpse(train)
dim(train)

#Now we create our test data.
test <- data[split1== 1, ]
head(test)
dim(test)


model <- rq(S6 ~ O6 + P6 + R6 + U6 + V6 + W6 + X6 + Y6 + Z6 + A7 + B7 + C7 + D7 + E7 + F7 + G7 + B1 + C1 + D1 + E1 + F1 + G1 + H1 + I1 + J1 + K1 + L1 + M1 + N1 + O1 + P1 + Q1 + R1 + S1 + T1 + U1 + V1 + W1 + X1 + Y1 + Z1 + A2 + B2 + C2 + D2 + E2 + F2 + G2 + H2 + I2 + J2 + K2 + L2 + M2 + N2 + O2 + P2 + Q2 + R2 + S2 + T2 + U2 + V2 + W2 + X2 + Y2 + Z2 + A3 + B3 + C3 + D3 + E3 + F3 + G3 + H3 + I3 + J3 + K3 + L3 + M3 + N3 + O3 + P3 + Q3 + R3 + S3 + T3 + U3 + V3 + W3 + X3 + Y3 + Z3 + A4 + B4 + C4 + D4 + E4 + F4 + G4 + H4 + I4 + J4 + K4 + L4 + M4 + N4 + O4 + P4 + Q4 + R4 + S4 + T4 + U4 + V4 + W4 + X4 + Y4 + Z4 + A5 + B5 + C5 + D5 + E5 + F5 + G5 + H5 + I5 + J5 + K5 + L5 + M5 + N5 + O5 + P5 + Q5 + R5 + S5 + T5 + U5 + V5 + W5 + X5 + Y5 + Z5 + A6 + B6 + C6 + D6 + E6 + F6 + G6 + H6 + I6 + J6 + K6 + L6 + M6 + N6, tau = 0.8, train)

sum <- summary.rq(model, se = "iid", covariance = TRUE)
sum


#Our model looks like it’s pretty good! But do we know that we need each and every variable?
#It’s hard to tell just looking at the p-values. To find the ‘best’ combination of predictor variables, we will use Akaike Information Criterion (AIC).

check <- stepAIC(model, direction = "both")
check

#So our AIC check reveal that for the best fit we need to remove coloumn R6 & V6.
f_model <- rq(S6 ~ O6 + P6 + U6 + W6 + X6 + Y6 + Z6 + A7 + B7 + C7 + D7 + E7 + 
                    F7 + G7 + B1 + C1 + D1 + E1 + F1 + G1 + H1 + I1 + J1 + K1 + 
                    L1 + M1 + N1 + O1 + P1 + Q1 + R1 + S1 + T1 + U1 + V1 + W1 + 
                    Y1 + Z1 + A2 + B2 + C2 + D2 + E2 + F2 + G2 + H2 + I2 + J2 + 
                    K2 + L2 + M2 + N2 + O2 + P2 + Q2 + R2 + S2 + T2 + U2 + V2 + 
                    W2 + X2 + Y2 + Z2 + A3 + B3 + C3 + D3 + E3 + F3 + G3 + H3 + 
                    I3 + J3 + K3 + L3 + M3 + N3 + O3 + P3 + Q3 + R3 + S3 + T3 + 
                    U3 + V3 + W3 + X3 + Y3 + Z3 + A4 + B4 + C4 + D4 + E4 + F4 + 
                    G4 + H4 + I4 + J4 + K4 + L4 + M4 + N4 + O4 + P4 + Q4 + R4 + 
                    S4 + T4 + U4 + V4 + W4 + X4 + Y4 + Z4 + A5 + B5 + C5 + D5 + 
                    E5 + F5 + G5 + H5 + I5 + J5 + K5 + L5 + M5 + N5 + O5 + P5 + 
                    Q5 + R5 + S5 + T5 + U5 + V5 + W5 + X5 + Y5 + Z5 + A6 + B6 + 
                    C6 + D6 + E6 + F6 + G6 + H6 + I6 + J6 + K6 + L6 + M6 + N6, tau = 0.8, train)

f_summary <- summary.rq(f_model, se = "iid", covariance = TRUE)
f_summary

f_check <- stepAIC(final_model, direction = "both")

#So we see that our model still has coloumns which have p-values higher than 0.05. 
#Let's remove those values and check our model again.

final_model <- rq(S6 ~ O6 + P6 + U6 + W6 + X6 + Y6 + Z6 + A7 + C7 + D7 + E7 + 
                 F7 + G7 + B1 + C1 + D1 + E1 + F1 + G1 + H1 + I1 + J1 + K1 + 
                 L1 + M1 + N1 + O1 + P1 + Q1 + R1 + S1 + T1 + U1 + V1 + W1 + 
                 Y1 + Z1 + A2 + B2 + C2 + D2 + E2 + F2 + G2 + H2 + I2 + J2 + 
                 K2 + L2 + M2 + N2 + O2 + P2 + Q2 + R2 + S2 + T2 + U2 + 
                 W2 + X2 + Y2 + Z2 + B3 + C3 + D3 + E3 + F3 + G3 + H3 + 
                 I3 + J3 + K3 + L3 + M3 + N3 + O3 + P3 + Q3 + R3 + S3 + T3 + 
                 U3 + V3 + W3 + X3 + Y3 + Z3 + A4 + B4 + C4 + D4 + E4 + F4 + 
                 G4 + H4 + I4 + J4 + K4 + L4 + M4 + N4 + O4 + P4 + Q4 + R4 + 
                 S4 + T4 + U4 + V4 + W4 + X4 + Y4 + Z4 + A5 + B5 + C5 + D5 + 
                 E5 + F5 + G5 + H5 + I5 + J5 + K5 + L5 + M5 + N5 + O5 + P5 + 
                 Q5 + R5 + S5 + T5 + U5 + V5 + W5 + X5 + Y5 + Z5 + A6 + B6 + 
                 C6 + D6 + E6 + F6 + G6 + H6 + I6 + J6 + K6 + L6 + M6 + N6, tau = 0.8, train)

final_summary <- summary.rq(final_model, se = "iid", covariance = TRUE)
final_summary


#So we see that our model indeed has the lowest AIC value & thus has all the necessary variables.
p_train <- predict(final_model)
p_train

MSE_train <- sum((p_train - train$S6)^2)/(nrow(train) - 2)
RMSE_train <- prettyNum(sqrt(MSE_train),
                        digits=0,
                        big.mark = ",")
MSE_train
RMSE_train


#So now we can see that model p value and predictors p value are less than 0.05.
#So our model is statistically significant.

#Now let's use our test data to predict Test set results.
P <- predict(final_model, newdata = test)
P

error <- P - test[["S6"]]
error

#Calculate RMSE.
MSE_test <- mean((P - test$S6)^2)
RMSE_test <- prettyNum(sqrt(MSE_test), 
                       digits=0, big.mark = ",")

MSE_test
RMSE_test

#Validating Model.
val <- cbind(P, error)
val_1 <- cbind(test, val)

#Now let's make the Gain curve.
GainCurvePlot(val_1, "P", "S6", "Property Price Model")

#A relative Gini score close to 1 means the model sorts responses well.
#And since our relative Gini score is 0.98 we can say that our model predicts well and thus is a good fit.
final_model
