# Predicting Housing Prices: Project Overview

* Created a tool that predicts housing prices to help customers negotiate the price when they buy a house.
* Converted continuos data into discrete values.
* Replaced missing values with the median value using SimpleImputer.
* Built a pipeline for the model.
* Optimised Linear, Decision Tree and Random Forest Regressors using GridsearchCV to reach the best model.
* Evaluated the model on the test set.

## Code and Resources Used
* **Python Version:** 3.7
* **Packages:** pandas, numpy, sklearn, matplotlib, seaborn

## Data
The Dataset taken is the California Housing Prices from the statlib repository. The Features are following:
* longitude
* latitude
* housing_median_age
* total_rooms
* total_bedrooms
* population
* households
* median_income
* median_house_value
* ocean_proximity
* income_cat

## Data Cleaning and Preprocessing
* I observed that the 'medium_income' data is continuos, so I made it discrete.
* The attributes are for whole regions so I calculated them for each household because we are calculating per house price.
* The 'total_bedrooms' feature has missing values so we set the missing value to median value using SimpleImputer.

# EDA
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights.

![alt text](https://github.com/sandeepan1999/Predicting-House-Prices/blob/master/scatter_matrix.png "Scatter matrix")

## Building Pipelines
* Created CombinedAttributesAdder class which gives us the flexibility to add extra attributes to our pipeline.
* Created DataFrameSelector class which allows us to select entire or partial dataframe.
* Created MyLabelBinarizer class which converts text to integers,integers to one hot vectors(all 0's but only one 1).
* Separated numerical and categorical attributes.
* Added components to numerical & categorical pipeline.
* Prepared entire dataset with pipeline.

## Model Building
First, I used stratified sampling technique since we derived a new attribute called 'income_category' to split the data into train and tests sets with a test size of 20%.

I tried three different models and evaluated them using Root Mean Squared Error. I chose RMSE because it is relatively easy to interpret.

I tried three different models:
* Linear Regression - Baseline for the model
* Decision Tree Regression - Because of the sparse data I thought Decision Tree regression would be effective. 
* Random Forest Regression - Again, with the sparsity associated with the data, I thought that this would be a good fit.

## Model Performance
The Random Forest model far outperformed the other approaches.
* Random Forest : RMSE = 49201.35439552971 
* Linear Regression: RMSE = 69386.21656804287  
* Decision Tree: RMSE = 69612.56471325169

## Evaluating model
Evaluated the model on the test set using the best estimators from the GridSearch method.
* Final RMSE = 47310.59440646111



