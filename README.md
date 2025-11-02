# Prediction of Bike Rental Count
Prediction of Rental Bikes Count using ML algorithm <br/> <br/>
[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

## Team information  :
1. Sai Manasa Yadlapalli (SJSU ID: 015999659)
* Email: saimanasa.yadlapalli@sjsu.edu 
2. Shubhada Sanjay Paithankar (SJSU ID: 016013283)
* Email: shubhadasanjay.paithankar@sjsu.edu


## [Dataset from Kaggle ](https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset?search=bike+demand+Washington)
## Deployment on Heroku:<br/> 
https://cmpe255-rental-bike-prediction.herokuapp.com/
   
## Web App Snippets:
<p float="left">
<img src="https://user-images.githubusercontent.com/87613567/206582383-c0cf8a49-066d-4e22-a2a9-8d6cfe2e61d7.png" width="330" height="190">
<img src="https://user-images.githubusercontent.com/87613567/206582650-c01fabbe-9035-41a4-99b4-a4e48e48acb2.png" width="330" height="190">
<img src="https://user-images.githubusercontent.com/99461999/206800037-c75ad8f4-e964-4366-a8a0-f13760fab756.png" width="330" height="190">
<img src="https://user-images.githubusercontent.com/87613567/206582947-04171cfb-ac30-476e-bb1a-824421bab276.png" width="330" height="190">
</p>

## Introduction :

To ensure that individuals can acquire rides when needed, a variety of bike firms, like [Zagster ](https://en.wikipedia.org/wiki/Zagster) and [Lime Bikes ](https://www.li.me/en-US/home), are present all throughout the United States. As a result of some people switching to these __cost-efficient__ and __environmentally__ friendly modes of transportation, there is a rise in demand for these bikes. This change has caused the demand for bikes to vary across different geographic areas.

## High level Architecture:
<p align="center">
<img src="https://user-images.githubusercontent.com/99461999/204049640-2b8e443d-55e3-4c18-9308-84f3e3376067.png" width ="520" height="300">
</p>

## Problem Statement: 
1. Knowing the appropriate number of bikes to station at various locations at various times in order to maximize profit and provide rides for customers is one of the issues these businesses confront. People may occasionally miss out on these bikes because they are unavailable. On the other hand, there are times when there is less need for these motorcycles and they are widely available but unutilized. Dealing with these situations and comprehending the demand for bikes on different days and in different situations becomes crucial.
2. Due to the present market crisis, US bike sharing has recently seen significant drops in its revenue. Many bicycle rental businesses are therefore having trouble remaining viable in the current market. As a result, it has made the conscious decision to create a business plan that would help it increase income as soon as the economy and market conditions are stable again. Additionally, they intend to grow their company outside of the US. As a result, they need its potential partners to predict how many bikes customers will rent under certain circumstances.


## Data Science and Machine Learning 

With the help of `machine learning` and `deep learning`, this problem could be addressed and this would ensure that the demand for the bikes are known beforehand and thus, the companies could ensure that there are `adequate bikes` present in different locations.

This project aims to use the independent factors at hand to model the demand for shared bikes. It will be used by the management and potential partners to comprehend how different features affect needs differently. To fulfill the customer's expectations and demand levels, they can adjust the business plan accordingly. Additionally, the model will help management comprehend demand patterns in a new market.
Rental Bike is based on various features like:

* Weather (cloudy, windy, rainy)
* Season (Summer, Winter)
* Temperature
* Humidity
* Wind Speed
* Holiday (Yes/No)

## Exploratory Data Analysis (EDA)

Therefore, bike rental companies would be able to comprehend the total number of bikes that must be present at different points in time and, consequently, be able to forecast the demand for bikes in the future with the aid of "data visualization" and "machine learning." As a result, the businesses would guarantee that they save millions of dollars while providing the proper assistance to various needy individuals.

## Metrics

We analyze metrics that account for continuous output variables because this is a regression problem and they base their estimates on the difference between the actual and expected output. The metrics utilized for this prediction are listed below.

* [__Mean Squared Error__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
* [__Mean Absolute Error__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)


There are a large number of machine learning models used in the prediction of the demand for `Bikes`. Below are the models that were used for prediction.

* [__Deep Neural Networks__](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
* [__K Nearest Neighbors__](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
* [__Partial Least Squares (PLS) Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html)
* [__Decision Tree Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
* [__Gradient Boosting Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
* [__Logistic Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* [__Long Short Term Memory (LSTM)__](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)


## Machine Learning Predictions and Analysis 

* In order for us to perform the __machine learning__ analysis, it is crucial to be aware of some of the features that are present in the data.
* To understand some of the underlying features, we will use a variety of __data visualizations__. Once we have a firm grasp on these features, we will use a variety of machine learning models to estimate the demand for bikes based on these features.
* After receiving the machine learning predictions, we will employ several techniques that may help us in the process of producing the models that may be applied in various ways by businesses.
* Therefore, where the demand for the bikes is predicted in advance using machine learning and deep learning, respectively, this would save the bike rental companies a lot of time and money.
[Google Colab Link ](https://colab.research.google.com/drive/19FeEZDLix2tJ6zCXj2BkHOuVItCBJ0si#scrollTo=DAW3nWDuVMqj)

## Steps to run locally:
1) git clone ``` https://github.com/shubhadapaithankar/Prediction-of-Bike-Rental-Count.git ```
2) ``` cd Prediction-of-Bike-Rental-Count```
3) Install the python dependencies using ``` pip install -r requirements.txt ```
4)  Now, run the applictaion using either of the commands ```python app.py``` or ```flask run```

## Using both hourly and daily datasets in the webapp
The web application now supports both the hourly (`hour.csv`) and daily (`day.csv`) datasets. On the main prediction page (`/index`) there is a "Dataset & Filters" panel where you can:

- Choose `day` (daily aggregated) or `hour` (hourly) dataset
- Filter by weekday (Sun...Sat)
- Filter by hour (0-23) — only applied when `hour` dataset is selected

After applying filters the page shows simple statistics (rows, total and mean `cnt`) and a preview of the filtered rows.

If you need to retrain the ML model on one of the datasets, run `model.py` which will read `day.csv` by default (the model training script was updated to use daily data). The trained model is saved to `model.pkl`.

## Conclusion:

* The models that were able to perform the best for predicting the bike demand are __Gradient Boosting Decision Regressor__ and __Deep Neural Networks__.
* __Exploratory Data Analysis (EDA)__ was performed to ensure that there is a good understanding of different features and their contribution to the output variable respectively. 
* The best machine learning models were able to generate a __mean absolute error (MAE)__ of about __23__ which is really good considering the scale of the problem at hand.

### Future Scope:
Since the current dataset only includes data for Washington, D.C., our future goals would be to forecast the number of bikes that will be rented and to shuffle those rentals based on any place and region across the entire United States.

### Project Documents:

* [__Project_Report/Document__](https://github.com/shubhadapaithankar/Prediction-of-Bike-Rental-Count/blob/main/Prediction-of-Bike-Rental-Count-Report-Tema20.pdf)
* [__Colab_Notebook__](https://github.com/shubhadapaithankar/Prediction-of-Bike-Rental-Count/blob/main/Rental_bikes_count_Prediction_Model_Comparison.ipynb)

### Reference:
1) https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset?search=bike+demand+Washington
2) https://flask.palletsprojects.com/en/2.2.x/
3) https://www.youtube.com/watch?v=4_EO4RwABbA


### Task Distribution:

<img width="566" alt="Screen Shot 2022-12-12 at 1 03 07 PM" src="https://user-images.githubusercontent.com/99461999/207153939-258badc6-3a4f-4914-8faf-b1255ca7891d.png">

