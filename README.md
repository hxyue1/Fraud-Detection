# Fraud-Detection

This is code repo documenting my various attempts to predict transaction fraud using data provided by Vesta Corporation in Kaggle's IEEE-CIS Fraud Detection competition. The notebooks can be broken up into two phases based on the maturity of my analysis.

## Phase I

Exploratory data analysis is documented in EDA.ipynb. Correlations.ipynb shows my attempt to identify important variables through correlation with the target variable. This is not adequate as the competition metric is the AUC/ROC score, I try to imitate model selection with this metric by calculating it for each feature using a bivariate logistic regression. 

Logistic regression is then entertained as a candidate model in Logistic Regression.ipynb, but is found to be lacking. Random forests are then used in Random Forests.ipynb and Random Forests Redux.ipynb. Random Forests do reasonably well, I managed to submit predictions netting an AUC score of 0.89 on the public leaderboard (for reference, the top scorers at the time were generating scores of 0.94+). Parameters are initially tuned under the assumption of being embarrasingly parallel. In the second notebook, Bayesian optimisation is used to tune the parameters in conjunction with an alternative validation strategy. This does not fare well, but in hindsight I should have investigated them separately not in conjunction as I cannot identify the source of the poor performance.

I also experiment with random shuffling and forward filling to deal with NaNs in fillna-random-forward.ipynb. The idea is to avoid clustering data points on one value, which is the problem with filling with the mean. This new method allows us to mimic the distribution of the non-missing values.
