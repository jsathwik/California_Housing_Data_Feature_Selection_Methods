# Feature Selection on California Housing Price Prediction 🏠🏠

Project aims to build regression models by including the `Feature Selection Techniques` like `Filter based methods`, `Wrapper methods`, `Embedded Methods` and `Dimensionality Reduction Techniques ` like `Principal Component Analysis (PCA)`. The main theme here is to reduce the complexity of the model and improve the efficiency of the models using these techniques.

## Data 
The California housing dataset consists of `20640` data points, with each datapoint having `8 features`. This dataset was obtained from the StatLib repository -
[Dataset_Link](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)  

This dataset was derived from the 1990 U.S. census, using one row per census
block group. A block group is the smallest geographical unit for which the U.S.
Census Bureau publishes sample data (a block group typically has a population
of 600 to 3,000 people).

A household is a group of people residing within a home. The average
number of rooms and bedrooms in this dataset are provided per household.


**Features Description**

1. `MedInc`     : median income in block group
2. `HouseAge`   : median house age in block group
3. `AveRooms`   : average number of rooms per household
4. `AveBedrms`  : average number of bedrooms per household
5. `Population` : block group population
6. `AveOccup`   : average number of household members
7. `Latitude`   : block group latitude
8. `Longitude`  : block group longitude

**Target** : `Median house value` - for California districts, expressed in hundreds of thousands of dollars ($100,000).

[California Housing Dataset in Sklearn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

## Libraries Used 

**Language:** ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

**Packages:** ![sklearn](https://img.shields.io/badge/scikit-learn-orange)

## Implementation Details 📜

### Simple Regressor Model

1. Housing data can be loaded into the code by first importing `sklearn.datasets.fetch_california_housing` module , using  `fetch_california_housing()` function.

2. Use `dataset.DESCR` to understand data or Use [California Housing Dataset Sklearn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) for further details.

3. Segregate `Features` and `Target Variables` as `X`, `Y` respectively.

4. Using `sklearn.model_selection.train_test_split` split the X, Y into `train` and `test` data with defined `test_size` param as  `0.20`. Its is recommended to use `random_state` param to produce `same results` across all executions.

5. From the [California Housing Dataset Sklearn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) it is evident that Feature values are `real` (but not within any limits). I have used `sklearn.preprocessing.StandardScaler` to bring them into limits.

6.  `sklearn.linear_model.LinearRegression` module is used for implementing `Regressor Model`. 

7.  Later `r2_score` is obtained.

8. Till this point we have created regressor model without any 'feature selection' i;e plain model that includes all features. Hereafter we make use of the various feature selection techniques (their respective methods) and analyze the results.


### Filter Based Feature Selection Methods

#### Mutual Information Regression

#### 1.  Using SelectKBest

Here we make use of `sklearn.feature_selection.SelectKBest`, a filter-based feature selection method and pass parameters like  `mutual_info_regression` (a score function that computes the mutual information between variables), `k` value (numerical value that depicts top number of features to be picked.). The output obtained here are `k` best features (X_new).

Now perform the train test split on the `X_new` and `Y`, then perform general Regression implementation and calculate the r2_score.

#### 2. Using the SelectPercentile

`sklearn.feature_selection.SelectPercentile` module is imported here, the `SelectPercentile` method is used to select `highest Percentile` features of the existing features. Score_function `mutual_info_regression`, `percentile` value (numerical value that depicts 'percentile of the highest scores'). 

Similar to SelectKBest, SelectPercentile output obtained here are new set of features with percentile of highest scores.

Hereafter, generate train test split on `X_new`, `Y` and proceed with Regressor implementation, r2_score calculation.

#### 3. Using f_regression method

 `f_regression` is `correlation statistic` function can be imported from `sklearn.feature_selection.f_regression` module. We make use of the `SelectKbest` feature selection method to select `K` best features based on `f_regression` scores.

`SelectKbest` methos is passed with the `f-regression` ,'k' value params. 

Output obtained here are new set of features with  highest `f-regression scores`.

Hereafter, generate train test split on `X_new`, `Y` and proceed with Regressor implementation, r2_score calculation.

#### 4. Analyze relationship / correlation Among the features themselves  (Pearson correlation )

We make use of the `pandas` library to Convert our features into `DataFrame` format. So that corr() method can be obtianed on the newly created dataframe and obtain results.

For a visual understanding of correlation among features, make use of `seaborn` library `heatmap` and check for the features that have correlation themselves.

Decide on the features that posseses high correlation and drop them from features set. We now only have features that have negligible correlation themselves.

Now, generate train test split on `X_new`, `Y` and proceed with Regressor implementation, r2_score calculation.

### Wrapper Based Methods for Feature Selection

#### 1. Recursive Feature Elimination (RFE)

We make use of `sklearn.feature_selection.RFE` module for performing recursive feature elimination. `RFE` method has `estimator`, `n_features_to_select`, `step` as input parameters.   

As a estimator we can use either `Linear Regressor` or `Lasso Regularization`. Here we have decided to use `Lasso`. `n_features_to_select` takes integer value, basically number of features and `step` - corresponds to the number of features to remove at each iteration.

Finally, generate train test split on `X_new`, `Y` and proceed with Regressor implementation, r2_score calculation.

#### 2. Sequential Feature Selection 

`sklearn.feature_selection.SequentialFeatureSelector` library is imported. `SequentialFeatureSelector()` takes `estimator`,`n_features_to_select` and `direction` as major parameters. 

`RidgeCV` is used as estimator, `n_features_to_select` is set to `auto`, here we are relying on algorithm to choose the optimum number of feature that perform better. `direction` is taken as `forward`, sothat algorithm starts with addition of new features, to form subset of features.

Apply `fit()` and `transform()` methods on `SequentialFeatureSelector` class and obtain the X_new.

As Final step generate train test split on `X_new`, `Y` and proceed with Regressor implementation, r2_score calculation.

### Embedded Methods for Feature Selection

#### L1 Regularization

For L1 Regularization we make use `sklearn.linear_model.Lasso` class. `Aplha` is the input parameter for the `Lasso()` method. We use list of `Aplha` values ranaging from 0 to 2 (in this case) and 
note the results for each alpha case.

`Lasso()` implementation is programtically similar to Linear_Regressor. We then call upon `fit()` and `predict()` methods on `Lasso()` regressor.

`lasso` with smaller alpha’s, model coefficients are reducing to absolute zeroes. Therefore, lasso selects the only some feature while reduces the coefficients of others to zero.

R2_score is calculated on the predicted data for all values of `Alpha` and respective `model coefficients` are noted for each alpha iteration. 

Plot between the features and respective coefficients, will give us the clear understanding of feature shrinkage to zero at different `Alpha's`. 

### Dimensionality Reduction 

#### Principal Component Analysis

As part of `PCA` implementation, firstly import `sklearn.decomposition.PCA` class. Initialize `PCA` by passing the parameters like `n_components` (number of components to keep) and `svd_solver` is set to `full`.

Apply fit_transform(), resultant `X_new` will contain  4 components (as defined in the PCA params).

As Final step generate train test split on `X_new`, `Y` and proceed with Regressor implementation, r2_score calculation.





## Evaluation and Results 🔍

