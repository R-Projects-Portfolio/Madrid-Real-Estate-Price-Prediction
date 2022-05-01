# Madrid-Real-Estate-Price-Prediction
This is a supervised learning model which predicts the price of Real Estate in Madrid, Spain.

Data collected from Kaggle.
[Kaggle](https://www.kaggle.com/datasets/mirbektoktogaraev/madrid-real-estate-market)

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

Then I cleaned the data some more and removed all the NA's, NAN's & INF's, converted text coloumns to factors and then all the columns to numerics.

![Cleaned Dataset](https://user-images.githubusercontent.com/97380339/165928474-9065bd78-b7ab-498a-b5e0-9a940feef768.png)

Then I checked the correlation of independent variables and removed those which had a correlation-coefficient greater than 0.7.

Then I plotted the buy price vs number of rooms to get a general view of my data and see if their were outliers.

![Price v Rooms](https://user-images.githubusercontent.com/97380339/165929168-67966d88-d5b9-4921-842d-cefcb37c342a.png)

Plotted a histogram for the dependent variable.

![histogram of buy price](https://user-images.githubusercontent.com/97380339/165929406-1380122d-badc-411a-873d-5aa79a50d1d3.png)

And a paired scatter plot of the independent variables to see correlation.

![indep_scatterplot](https://user-images.githubusercontent.com/97380339/165938568-fb1239a7-c70a-4e70-834c-d6ce82570bd2.png)


Then we divide the data into Test and Train & use the Train data to train our model.
When the quantile regression is done we do the AIC check to make sure our model is not over-fitted and then we get our final model.

![Final Model](https://user-images.githubusercontent.com/97380339/165948702-d5a4dc95-6315-4947-ae53-19a918fb40a2.png)


Now we use our model to predict the Test set values and check the accuracy of it.
We do so by plotting a gain curve.


![Gain Curve](https://user-images.githubusercontent.com/97380339/165949292-8ea0e93e-d4b7-48a5-8039-2de972a91f4c.png)


A relative Gini score close to 1 means the model sorts responses well.
And since our relative Gini score is 0.94 we can say that our model predicts well and thus is a good fit.

Now we try to make a web page in R Shiny, which can be used to predict the price of the property in madrid (in Euro) through our model.

This is how our webpage looks.

![Web Page](https://user-images.githubusercontent.com/97380339/165949920-4b7fce04-acec-4304-986a-0ccdad616c06.png)


You can also use the following link.

[Price Prediction of Madrid Real Esate](http://127.0.0.1:4164/)

