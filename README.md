# Madrid-Real-Estate-Price-Prediction
This is a supervised learning model which predicts the price of Real Estate in Madrid, Spain.

Data collected from Kaggle.
https://www.kaggle.com/datasets/mirbektoktogaraev/madrid-real-estate-market

The dataset consist listings from popular real estate portals of Madrid. I first cleaned the data in excel and then imported it to R.

![Data Sheet](https://user-images.githubusercontent.com/97380339/164229661-9ddcf118-9ac9-4c68-a9e0-21127acb4b8d.png)

As we can see there are 16 coloumns in the data.

![dimensions_madrid](https://user-images.githubusercontent.com/97380339/164283751-d7891fbd-e11c-46c8-94af-d9fd7465169e.png)

After looking through the data we see that there are 145 unique areas in the data-set.
Then we added a sq_ft coloumn in the data and a buy_price_per_sq_ft coloumn to get an idea of the buy price. Because the buy_price can depend on a lot of factors and thus will be varied but buy_price_per_sq_ft gives a glimpse into the variability of the data.

Now, since we're interested in the buy price, we plotted buy_price_per_sq_ft with the number of data points to check if our data is normal or not.

![Price Per Sq Ft](https://user-images.githubusercontent.com/97380339/164281667-55990fe7-5e8c-4cef-8eb7-6e1b65f17bb8.png)

As we can see from the plot, our data is bi-modal, so we cannot perform Linear Regression on our data, because normality is a condition for Linear Regression.
So, we will perform [Quantile Regression](https://en.wikipedia.org/wiki/Quantile_regression).

Now the data is cleaned and some irrelevent coloumns are removed. The coloumns which had logical vectors were converted to numeric & since our regression model can't interpret text data, the text coloumns were converted to numeric data by using Dummy Variables.

Now, there were 163 coloumns because all the UNIQUE areas & energy certificates were converted to a single numeric coloumn each.

We also rename the coloumns as A1, B1, and so on till H7 because the area coloumn had some names in spanish letters which were interpreted as symbols by our model and thus will cause an error.

So this is how our data-set looks now.

![Screenshot (38)](https://user-images.githubusercontent.com/97380339/164284985-fb875f51-b046-41cb-bddc-62623aa72563.png)
![Screenshot (37)](https://user-images.githubusercontent.com/97380339/164284998-12a0a6d5-22eb-4662-a92d-bacc51d74804.png)

![Screenshot (40)](https://user-images.githubusercontent.com/97380339/164285289-8f569a05-af89-40eb-817f-31948857dadc.png)

Now we convert the data into a data frame.
We want to predict the buy_price, so that is our dependent variable & all others are our independent variables.
So we run a correlation test on all our independent variables and remove those which have a correlation coefficent more than 0.7.

Now our final data frame (final_df) is ready. 
It has 160 coloumns & 11549 rows.

![Screenshot (43)](https://user-images.githubusercontent.com/97380339/164286491-a3cf8a2e-bf7a-4b05-9182-701491b735fa.png)

Now we filter our data into x -> which includes all independent variables & y -> the dependent variable.

We then plot a histogram and density function line for dependent variable (buy_price).

![Screenshot (61)](https://user-images.githubusercontent.com/97380339/164293096-12477e52-962a-474e-a787-e8358eecd685.png)

Now we divide our data into Test Set and Train Set. 
Since this is a supervised machine learning model, 80% of our final data set is used as Training Data, which trains the model.
And the rest 20% of the final data will be used as Testing Data which will help us to check the accuracy and fit of our model.

![Screenshot (46)](https://user-images.githubusercontent.com/97380339/164287849-2d151ca7-914f-4b79-8711-6c9e3687b46d.png)

Now that we have our train data ready, we make our Quantile Regression Model.

![Screenshot (49)](https://user-images.githubusercontent.com/97380339/164288287-64629735-ad3b-4016-a68c-9b7f3918b040.png)

Now we check the summary of the model.

![Screenshot (52)](https://user-images.githubusercontent.com/97380339/164288672-6ee7e78a-6216-4a92-8e41-7af7b89f44e3.png)

Our model looks good as most of the p-values are less than the significance value (α = 0.05).
But not all the variables have such small p-values. So to find the BEST combination of predictor variables, we will use Akaike Information Criterion (AIC). 

After applying AIC prediction we find the best fit model which has only the necessary independent variables and so our final model is ready.

![Screenshot (53)](https://user-images.githubusercontent.com/97380339/164289679-c9299eb8-3587-4af2-8358-707e2f521e4a.png)

So we see that our model indeed has the lowest AIC value & thus only the necessary variables. Also, all the p-values are less than significance value so our model is Statistically Significant.

Now we use our model for prediction & calculate the MSE and RMSE.

![Screenshot (55)](https://user-images.githubusercontent.com/97380339/164290589-21c5c64f-a6da-4d59-99fe-fefc9039024d.png)

Then we use the Test Data-Set to predict Test Set results & calulate it's MSE & RMSE. We also calculate the residual or error that our model makes in predicting.

![Screenshot (57)](https://user-images.githubusercontent.com/97380339/164291439-dbb818e0-9b2f-4757-ab5b-b87edbfd063d.png)

Then we validate our model and make a Gain Curve to check if the model predicts well.

Gain Curve.

![Screenshot (31)](https://user-images.githubusercontent.com/97380339/164292111-ca731df3-bbc6-4bb9-94e9-339c52893bc6.png)

As we can see that the Relative Gini Score of our model is 0.98 & we know that if Relative Gini Score is close to 1 then the model sorts and predicts well.
So we can safely say that we have a model which can predict the buy_price of property in Madrid very well.


