# Forecasting-TimeSeriesAnalysis
## Objective:

1. To compare the predictive ability of the following three models on the given data and identify the best performing model. 
    - One of the four simple forecasting methods (average, naïve, seasonal naïve, or drift)
    - An Exponential smoothing model
    - An ARIMA model
2. Make an estimation of the personal consumption expenditures for the **October of 2024** based on the model’s prediction.
3. To compare models’ predictive performance using time one-step ahead rolling forecasting without re-estimation of the parameters

The R code used to process and analyse the data can be found here [link](https://github.com/Lakshya-attavar/Forecasting-TimeSeriesAnalysis/blob/main/TSA_Rcode.R).

## Data Structure & Initial Checks

The dataset contains seasonally-adjusted US personal consumption expenditure information. It has two columns containing the date and the personal consumption expenditure from January 1959 to November 2023. 

- The data is converted to a time series object with a frequency of 12 as the observations are monthly.
- The data is then checked for any missing observations and imputation is performed. In this case, missing data is handled using a simple linear interpolation method since the data has a notable trend and continuity as observed in Fig-1.1. Interpolation evaluates the missing value by averaging the values on either side of the missing value.
- Further, the data is split to create train and test sets to train the forecasting models and evaluate their performance. There are a total of 779 records in the data out of which 80% is taken as the training data and the remaining 20% is taken as the test set.

<img width="600" alt="image-2" src="https://github.com/user-attachments/assets/04667e0c-9f8f-4289-b89a-df423e9a2053">

Fig-1.1 Plot of the original time series before imputation

## Executive Summary

### Overview of Findings

Considering the accuracy as a criterion for selecting the best model, it can be concluded that Holt’s method is the best of the three. Comparing the accuracy of the drift, holt and Arima models, it is evident that Holt’s method has the lowest values for RMSE at 1145.65 and MAE at 645.241. The MPE, MAPE and MASE are also lower compared to the drift and Arima models. 

Plotting the predictions of the models with actual data in figure below show that the forecasts for Holt’s method are closest to the actual test data.

<img width="600" alt="image-3" src="https://github.com/user-attachments/assets/778a2f88-e0fb-470a-a63f-ddccb9bc3d00">

Holt’s method performs best and provides better accuracy while forecasting the test data directly and also while employing one-step ahead rolling forecasting. Comparing the models, it is very evident that one-step ahead rolling forecasting provides much better accuracies than the direct forecasting models since the model is constantly trained and updated. 

## Insights Deep Dive

### Indentification of best model:

- **Simple forecasting methods:** Of the four simple forecasting methods, the drift model is used to forecast the data as the time series displays a stable trend with no seasonality and no repeating patterns with little irregularities making the drift model a more suitable choice. The accuracy of the drift model is checked by comparing the forecast to the original time series. It is observed that the RMSE is 2545.28 and MAE is 1926.58.

<img width="600" alt="image-4" src="https://github.com/user-attachments/assets/a5400cf1-7ea2-4c5f-ae2d-a24d58d3e904">

- **Exponential smoothing methods** forecast the future values wherein larger weights are assigned to recent observations and the weights gradually reduce for the older observations. The method of choice for exponential smoothing is the Holt linear method since the data has trend only and no seasonal components. From the accuracy of Holt’s model it is observed that the RMSE is 1145.65 and MAE is 645.241.

<img width="600" alt="image-5" src="https://github.com/user-attachments/assets/de5ae84d-e223-4f9f-8f39-e07c6042a0cd">

- **ARIMA model:** ARIMA (Auto Regressive Integrated Moving Average) forecasts future values by taking into account the autoregressive, differencing and moving average components, thus combining the effect of past observations and residual errors on the current values. To select the best arima model based on the data, the auto.arima function is employed and trained using the train set. This produces a model with an order of (3,2,3). The accuracy of the ARIMA model is checked by comparing the forecast to the original time series and it is observed that the RMSE is 1593.06.65 and MAE is 1042.73.

<img width="600" alt="image-6" src="https://github.com/user-attachments/assets/0c63ba4d-e6dc-48a5-9654-0ea61e01bd7f">

- Considering the accuracy as a criterion for selecting the best model, it can be concluded that Holt’s method is the best of the three.

### Estimation of PCE for October 2024:

- The best of the three, i.e., **holt’s method** is used to estimate the personal consumption expenditure for October of 2024. The **PCE for October 2024 is 16052.81** as seen in fig below.

<img width="600" alt="image-7" src="https://github.com/user-attachments/assets/def99760-47bc-4ea6-beba-269742cb699b">
<img width="600" alt="image-8" src="https://github.com/user-attachments/assets/9832f04f-25cf-4523-952f-cda5d9d07999">


### Rolling Ahead Forecasting:

- Rolling forecasting is a technique where the forecasting model is constantly updated by iteratively training the model using new sets of data and predicting the next set of observations ahead.
- In this case, the test set is predicted in steps of one time period ahead by fitting the model for the training data (assuming train set as observations till 2010).

<img width="600" alt="image-9" src="https://github.com/user-attachments/assets/f1265263-df51-4518-868f-870f88316f8f">

- Comparing the accuracy of the models the from figure shows that Holt’s method has the lowest RMSE at 200.76 and MAE at 69.78 indicating that Holt’s model performs the best in comparison to drift model and Arima model when using one-step ahead rolling forecasting without re-estimation of parameters.

## Recommendations:

Based on the insights and findings above,

- For providing future forecasts of this data, Holt’s method with rolling forecast would be the best choice.

## Assumptions and Caveats:

These assumptions and caveats are noted below:

- There is a notable difference in errors between the train and test set which indicates poor performance of models for the test set for the initial modelling of data without rolling forecasting. Using a different split of data might prevent possible overfitting of the model.
