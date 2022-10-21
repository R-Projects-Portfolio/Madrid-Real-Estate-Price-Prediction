install.packages('caret', dependencies = TRUE)
install.packages("caTools")
install.packages("car")
install.packages("GGally")
install.packages("psych")
install.packages("corrplot")
install.packages("MASS")
install.packages("broom")
install.packages("WVPlots")
install.packages("quantreg")
install.packages("shiny")
install.packages("RColorBrewer")

library(quantreg)
library(shiny)
library(readxl)
library(ggplot2)
library(caret)
library(caTools)
library(car)
library(GGally)
library(psych)
library(corrplot)
library(MASS)
library(broom)
library(WVPlots)
library(dplyr)
library(tidyverse)
library(RColorBrewer)

#Now let's import our data into R.
madrid <- read_excel("C:\\Users\\HP\\OneDrive\\Documents\\Projects\\Datasets for Projects\\R\\madrid_real_estate.xlsx", 
                     sheet = "cleaned_dataset")
glimpse(madrid)
dim(madrid)


#Let's see how many unique locations are there in the data-set.
unique(madrid$area)

#So there are 145 unique locations in the data-set.
#Now lets aggregate the data by location and see how many properties are listed 
#for each location.

madrid_location_unique <- madrid %>% count(area) %>% arrange(n)
madrid_location_unique

#Then we'll add a column of each area in sq-ft as most people use them as a measure.

madrid_1 <- madrid %>% mutate(area_sq_ft = sq_mt_built * 10.8)
glimpse(madrid_1)

#So now we'll see if there are any unusual sized bedrooms and thus properties by 
#making sure that each bedroom is over 300 sq-ft.
madrid_2 <- madrid_1 %>% mutate(sq_ft_per_bedroom = area_sq_ft / n_rooms) %>% 
  arrange(sq_ft_per_bedroom)
glimpse(madrid_2)

#Now according to architects guide in Spain, a single bedroom should atleast be 9 sq-m or 100 sq-ft. 
#So we remove the properties below that.
madrid_3 <- madrid_2 %>% filter(sq_ft_per_bedroom > 100)
glimpse(madrid_3)

#Now we'll add a buy_price_per_sq_ft column to the data.
madrid_4 <- madrid_3 %>% mutate(buy_price_per_sq_ft = buy_price / area_sq_ft)
glimpse(madrid_4)

#Now we'll see if there are any anomalies in the price.
summary(madrid_4$buy_price_per_sq_ft)

madrid_5 <- madrid_4[order(madrid_4$area),]
glimpse(madrid_5)
dim(madrid_5)
summary(madrid_5$buy_price_per_sq_ft)


#We can see that the min price is 41 Euros per sq-ft which is very less. 
#So, we want to filter out properties which are 1 standard deviation away from 
#the mean in each area.
madrid_6 <- madrid_5 %>% group_by(area) %>% 
  filter(buy_price_per_sq_ft <= quantile(buy_price_per_sq_ft, 0.84), buy_price_per_sq_ft >= quantile(buy_price_per_sq_ft, 0.16))

dim(madrid_6)
summary(madrid_6$buy_price_per_sq_ft)

#So now our data points have reduced from 17,806 to 12,078. 
#But these are the ones we want to feed into our model because they make up 68% of the data 
#Or are 1 standard deviation away from mean, i.e. the ideal data points for our model.
mean_buy_price <- mean(madrid_6$buy_price_per_sq_ft)
mean_buy_price
sd_buy_price <- sd(madrid_6$buy_price_per_sq_ft)
sd_buy_price

#So we know our mean and standard deviation of buy price per sq-ft.
#Now we will plot the price_per_sq_ft with the number of data points to get a basic idea of our data.
price_per_sq_ft <- madrid_6$buy_price_per_sq_ft
price_variration_plot <- hist(price_per_sq_ft, breaks = 40, main = "Price per sq-ft in Madrid", xlab = "Price per sq-ft", col = "lightblue", prob = TRUE) 
x <- seq(min(price_per_sq_ft), max(price_per_sq_ft), length = 60)
f <- dnorm(x, mean = mean(price_per_sq_ft), sd = sd(price_per_sq_ft))
lines(x, f, col = "red", lwd = 3)

#We can see from the plot that our data is bi-modal, so we cannot apply Linear Regression. We need to do quantile regression.

#Now we take a look at the number of bathrooms in the data-set
summary(madrid_6$n_bathrooms)
bathrooms_plot <- hist(madrid_6$n_bathrooms, breaks = 14, main = "Frequency of Bathrooms", xlab = "No. of Bathrooms", xlim = c(0,7), col = "lightgreen")
bathrooms_plot


#So we see from the graph that there are mostly 1-2 bathrooms in the data-set.
#Now we have cleaned and wrangled our data and it is ready for model building.

glimpse(madrid_6)

#Now that we are building a model we need to remove some columns which don't necessarily help our model.
#Like the buy_price_per_sq_mt & buy_price_per_sq_ft.
madrid_7 <- subset(madrid_6, select = -c(id, buy_price_per_sq_mt, area_sq_ft, sq_ft_per_bedroom, buy_price_per_sq_ft))
glimpse(madrid_7)

#Now we convert all the logical vectors to numeric.
cols <- sapply(madrid_7, is.logical)
madrid_7[,cols] <- lapply(madrid_7[,cols], as.numeric)

glimpse(madrid_7)

view(madrid_7)
summary(madrid_7)
str(madrid_7)

is.factor(madrid_7$area)
is.factor(madrid_7$house_type)
is.factor(madrid_7$energy_certificate)

madrid_7$area = as.factor(madrid_7$area)
madrid_7$house_type = as.factor(madrid_7$house_type)
madrid_7$energy_certificate = as.factor(madrid_7$energy_certificate)

str(madrid_7)

#Checking NA's NAN's % INF's.
data_new <- madrid_7
data_new[is.na(data_new) | data_new == "Inf"] <- NA 
data_new[is.infinite(data_new)] <- NA

glimpse(madrid_7)

is.nan.data.frame <- function(x)
  do.call(cbind, lapply(x, is.nan))
madrid_7[is.nan(madrid_7)] <- 0


madrid_7

madrid_7[madrid_7 == -Inf] <- 0
madrid_7

glimpse(madrid_7)

sum(is.na(madrid_7))


#Converting some coloumns to factors.
area.type = sapply(madrid_7$area, as.numeric)
area.type

house.type = sapply(madrid_7$house_type, as.numeric)
house.type

energy.certificate = sapply(madrid_7$energy_certificate, as.numeric)
energy.certificate
factor(energy.certificate)

#Now combine all these new columns into main row.
madrid_8 <- cbind(madrid_7, area_id = area.type)
glimpse(madrid_8)

madrid_9 <- cbind(madrid_8, housetype = house.type)
glimpse(madrid_9)

madrid_10 <- cbind(madrid_9, energy_cert = energy.certificate)
glimpse(madrid_10)

#Now we'll remove the old area, house type and energy certificate coloumns.

dataset = madrid_10[, c("area_id", "sq_mt_built", "n_rooms", "n_bathrooms", "floor", "housetype", "is_new_development", "has_lift", "has_parking", "energy_cert", "buy_price")]
view(dataset)
glimpse(dataset)

#Let's  check the correlation of all the independent variables.
x <- dataset[,-c(11)]

correlation_indep_var <- cor(x)
correlation_indep_var

c <- findCorrelation(correlation_indep_var, cutoff = 0.7, verbose = FALSE, exact = TRUE)
c

#So we see that the correlation on n_bathrooms with n_rooms & sq_mt_built is very high, which means that we can remove it and the model will still behave the same.
dataset_2 <- dataset[,-c(4,5,6,10)]
glimpse(dataset_2)

#Plotting buy price by number of rooms
m <- dataset_2 %>% group_by(n_rooms)
glimpse(m)

display.brewer.pal(n = 13, name = 'Spectral')
boxplot(buy_price ~ n_rooms, data = m, xlab = "Number of Rooms",
        ylab = "Buy Price", main = "Buy Price by No. of Rooms", col = brewer.pal(n = 14, name = 'Spectral'))

#We see that some properties have extraordinarily high prices for lesser number of bedrooms.
#So we'll remove these outliers.

dataset_3 <- dataset_2 %>% filter(buy_price <= 5000000)
glimpse(dataset_3)

display.brewer.pal(n = 13, name = 'Spectral')
boxplot(buy_price ~ n_rooms, data = dataset_3, xlab = "Number of Rooms",
        ylab = "Buy Price", main = "Buy Price by No. of Rooms", col = brewer.pal(n = 14, name = 'Spectral'), col.bg = 'Grey')

#Now we plot a histogram and density function line for dependent variable.
buy_price <- dataset_3$buy_price

his <- hist(buy_price, prob = TRUE, col = "coral", border = "black")
his
prop.table(table(buy_price))

#Now we plot the scatter plot to see the correlation:
cor_plot <- pairs(x, col = "lightblue", main = "Scatterplots")

final_dataset <- as.data.frame(dataset_3)
glimpse(final_dataset)

final_dataset <- data.frame(lapply(final_dataset,
                                   function(x) if(is.numeric(x)) round(x, 0) else x))
glimpse(final_dataset)

#Now we split our data-set into Test Data an Train Data for the ML-Model.

set.seed(123)
data <- final_dataset
head(data)
dim(data)
split <- sample(c(rep(0, 0.7 * nrow(data)), rep(1, 0.3 * nrow(data))))
split
table(split)

#Then we create our train data.
train <- data[split == 0, ]
train <- as.data.frame(train)
head(train)
glimpse(train)
dim(train)

#And then we create our test data.
test <- data[split== 1, ]
head(test)
dim(test)

#Fitting the model.
model_1 <- rq(buy_price~., tau = 0.8, train)
summary_1 <- summary.rq(model_1, se = "iid", covariance = FALSE)
summary_1


#We run an AIC test to find the best fitting variables, (so as not to over-fit) and thus the best model.

f_check <- stepAIC(model_1, direction = "both")
summary.rq(f_check)


#Now we use our model for prediction & calculate the MSE and RMSE.
pred_train <- predict(model_1)
pred_train

MSE_training <- sum((pred_train - train$buy_price)^2)/(nrow(train) - 2)
RMSE_training <- prettyNum(sqrt(MSE_training),
                           digits=0,
                           big.mark = ",")
MSE_training
RMSE_training

#Now let's use our test data to predict Test set results.
glimpse(test)
predicted_p <- predict(model_1, newdata = test)
glimpse(predicted_p)

residuals <- predicted_p - test[["buy_price"]]
residuals

#Calculate RMSE.
MSE_testing <- mean((predicted_p - test$buy_price)^2)
RMSE_testing <- prettyNum(sqrt(MSE_testing), 
                          digits=0, big.mark = ",")

MSE_testing
RMSE_testing

#Validating Model.
validating <- cbind(predicted_p, residuals)
validation <- cbind(test, validating)

#Now let's make the Gain curve.
gain_curve <- set.seed(34903490) 
gainx = 0.25  # get the predicted top 25% most valuable points as sorted by the model
# make a function to calculate the label for the annotated point
labelfun = function(gx, gy) {
  pctx = gx*100
  pcty = gy*100
  
  paste("The predicted top ", pctx, "% most valuable points by the model\n",
        "are ", pcty, "% of total actual value", sep='')
}

WVPlots::GainCurvePlotWithNotation(validation, "predicted_p", "buy_price",
                                   title = "Madrid Property Price Model",
                                   gainx = gainx, labelfun = labelfun)


# now get the top 25% actual most valuable points

labelfun = function(gx, gy) {
  pctx = gx*100
  pcty = gy*100
  
  paste("The actual top ", pctx, "% most valuable points\n",
        "are ", pcty, "% of total actual value", sep='')
}

WVPlots::GainCurvePlotWithNotation(validation, "predicted_p", "buy_price",
                                   title = "Madrid Property Price Model",
                                   gainx = gainx, labelfun = labelfun, sort_by_model = TRUE)

#A relative Gini score close to 1 means the model sorts responses well.
#And since our relative Gini score is 0.98 we can say that 
#our model predicts well and thus is a good fit.

model_1

#Cleaning the data to be passed into shiny.
glimpse(final_dataset)
data <- as.data.frame(final_dataset)

model_data <- final_dataset
glimpse(model_data)

is.factor(model_data$area_id)
is.factor(model_data$is_new_development)
is.factor(model_data$has_lift)
is.factor(model_data$has_parking)

model_data$area_id = as.factor(model_data$area_id)
model_data$is_new_development = as.factor(model_data$is_new_development)
model_data$has_lift = as.factor(model_data$has_lift)
model_data$has_parking = as.factor(model_data$has_parking)

is.factor(model_data$area_id)
is.factor(model_data$is_new_development)
is.factor(model_data$has_lift)
is.factor(model_data$has_parking)

str(model_data)

na_clean = na.omit(model_data)
summary(na_clean)
str(na_clean)

is.numeric(na_clean$area_id)
is.numeric(na_clean$sq_mt_built)
is.numeric(na_clean$n_rooms)
is.numeric(na_clean$is_new_development)
is.numeric(na_clean$has_lift)
is.numeric(na_clean$has_parking)

area.id = sapply(na_clean$area_id, as.numeric)
is.new.development = sapply(na_clean$is_new_development, as.numeric)
has.lift = sapply(na_clean$has_lift, as.numeric)
has.parking = sapply(na_clean$has_parking, as.numeric)

use_data_0 = cbind(na_clean, area.id)
use_data_0

use_data_1 = cbind(use_data_0, is.new.development)
use_data_1


use_data_2 = cbind(use_data_1, has.lift)
use_data_2

use_data_3 = cbind(use_data_2, has.parking)
use_data_3

glimpse(use_data_3)

inputdata = use_data_3[, c("area.id", "sq_mt_built", "n_rooms", "is.new.development", "has.lift", "has.parking", "buy_price")]

glimpse(inputdata)

is.numeric(inputdata$area.id)
is.numeric(inputdata$is.new.development)
is.numeric(inputdata$has.lift)
is.numeric(inputdata$has.parking)

head(inputdata)
view(inputdata)

#Now we use our model to actually predict values.


#making front end.
ui <- fluidPage(
  
  headerPanel("Madrid Real Estate Price Prediction"),
  
  sidebarPanel(
    textInput("area.id", "Enter the id of the area you want to buy the property in (1-144). ", "" ),
    textInput("sq_mt_built", "Enter the total area (sq-mt) you want. ", ""),
    textInput("n_rooms", "Enter the number of rooms you want. ", ""),
    textInput("is.new.development", "If you want newly developed property type 1, else type 0 ", ""),
    textInput("has.lift", "If you want lift on property type 1, else type 0 ", ""),
    textInput("has.parking", "If you want parking on property type 1, else type 0 ", ""),
    actionButton('go', "Predict")
  ),
  
  
  mainPanel( 
    width = 50, headerPanel("The Price of the property in Madrid is € "),
    
    textOutput("value")
  )
)

#Back-end 
server <- function(input, output) {
  data2 = reactiveValues()
  observeEvent(input$go, {
    data = inputdata
    view(inputdata)
    summary(inputdata)
    str(inputdata)
    
    
    is.numeric(inputdata$area.id)
    is.numeric(inputdata$sq_mt_built)
    is.numeric(inputdata$n_rooms)
    is.numeric(inputdata$is.new.development)
    is.numeric(inputdata$has.lift)
    is.numeric(inputdata$has.parking)
    
    data2$myarea.id = as.numeric(input$area.id)
    data2$mysq_mt_built = as.numeric(input$sq_mt_built)
    data2$myn_rooms = as.numeric(input$n_rooms)
    data2$myis.new.development = as.numeric(input$is.new.development)
    data2$myhas.lift = as.numeric(input$has.lift)
    data2$myhas.parking = as.numeric(input$has.parking)
    
    
    new.predict = data.frame(area.id = data2$myarea.id, sq_mt_built = data2$mysq_mt_built, n_rooms = data2$myn_rooms, is.new.development = data2$myis.new.development, has.lift = data2$myhas.lift, has.parking = data2$myhas.parking)
    
    model_Quant_Reg = rq(buy_price ~ area.id+sq_mt_built+n_rooms+is.new.development+has.lift+has.parking, tau = 0.8, data = inputdata)
    
    data2$op = predict(model_Quant_Reg, new.predict)
  })
  
  output$value <- renderPrint({data2$op})
}

shinyApp(ui, server)
