# Fraud-Detection

This is code repo documenting my various attempts to predict transaction fraud using data provided by Vesta Corporation in Kaggle's IEEE-CIS Fraud Detection competition. The notebooks can be broken up into two phases based on the maturity of my analysis.

## Phase I

### EDA and feature selection

Exploratory data analysis is documented in EDA.ipynb. Correlations.ipynb shows my attempt to identify important variables through correlation with the target variable. This is not adequate as the competition metric is the AUC/ROC score, I try to imitate model selection with this metric by calculating it for each feature using a bivariate logistic regression. 

### Logistic regression and Random Forests

Logistic regression is then entertained as a candidate model in Logistic Regression.ipynb, but is found to be lacking. Random forests are then used in Random Forests.ipynb and Random Forests Redux.ipynb. Random Forests do reasonably well, I managed to submit predictions netting an AUC score of 0.89 on the public leaderboard (for reference, the top scorers at the time were generating scores of 0.94+). Parameters are initially tuned under the assumption of being embarrasingly parallel. In the second notebook, Bayesian optimisation is used to tune the parameters in conjunction with an alternative validation strategy. This does not fare well, but in hindsight I should have investigated them separately not in conjunction as I cannot identify the source of the poor performance.

### Dealing with NaNs

I also experiment with random shuffling and forward filling to deal with NaNs in fillna-random-forward.ipynb. The idea is to avoid clustering data points on one value, which is the problem with filling with the mean. This new method allows us to mimic the distribution of the non-missing values.

## Phase II

### Xgboost and experimentation with using AWS Sagemaker

In the next stage of analysis, I switch to using xgboost as the model of choice. My first attempt to train and evaluate my model is found in xgboost.ipynb. I've also decided to attempt to use Sagemaker from AWS to speed up computations as I found that tuning the random forests took a lot of computational time. If I want to comprehensively explore the hyperparameter space of xgboost, I'll probably have the same issue. 

However, while Sagemaker does seem to train models at a faster rate, it tends to run into memory problems despite theoretically having the same or more ram than my MacBook Air (4GB). I suspect it is due to poor memory management, but don't know enough about how these types of VMs work to delve into it. 

data-prep.ipynb is used to do the memory intensive data pre processing and upload it to S3 where it can be used directly by Sagemaker. I then go on to play around with xgboost in some of those notebooks and attempt to tune the parameters. I then attempt to make predictions on the test set to submit before the competition ends, but at the very last stage the notebook runs into a memory error! I give up on Sagemaker and return to training models offline.


### Validation protocol

In the notebook prototyping.ipynb, I consider various validation schema to best mimic the test conditions. A simple train-test split works quite well. I verify this using another hold-out set. Cross-validation doesn't seem to fare too well, though I haven't explored this in significant depth.

### Feature engineering

After reading some posts on the discussion forums (in particular https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575#latest-643395, and https://www.kaggle.com/c/ieee-fraud-detection/discussion/107697#latest-643868) I become interested in the idea of feature engineering. 

The first thing I look at is how to encode dummy variables, this work is documented in dummy-encoding.ipynb. Previously, I used a rather simplistic but effective approach of using the get_dummies function from Pandas to convert strings into dummies. One downside is that it creates an extra 150 variables and consumes a lot of memory. I learn about feature hashing and attempt to use it but it doesn't seem to do what I think it does. Next I use the factorize() method to assign an integer value to each unique category, this saves memory and allows models to train faster, but makes the implicit assumption that the categories are somehow ranked. I train two models using just the categorical variables with the two different methods (get_dummies vs factorize) and find that the models perform pretty similarly. I use eli5 to identify important categories, but leads to different conclusions across the two models.

In numeric-features.ipynb, I attempt to identify important numeric features by ranking them using permutation importance and standardising the weights by their associated standard deviations. These rankings serve as a basis for the new feature to be created in feature-creation.ipynb. So far I have used five techniques to aggregate continuous variables over different categories, they are counts, means, standard deviations, deviations from the mean and standardised deviations from the mean. I also create a somewhat systematic approach to evaluate all possible combinations which meet certain criteria.
