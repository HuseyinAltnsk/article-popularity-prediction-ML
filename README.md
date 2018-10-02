# Predicting Popularity of News Articles with Machine Learning
A text-classification project from Machine Learning class on building models that can predict the popularity/momentum score 
of a given piece of news article using some relevant features.

## Dataset
The datasets used in this project, Moreover and Opoint, were provided by Quoin, the Charlotte based data science company.

## Project Summary
We found that LinearSVR performed the best in momentum score prediction, with an r2 score of over 0.8. This was true for 
both articles from “Moreover” and “Opoint”, our two major data sources. The best model used a combination of grid search
with cross-validation. We think that this project can be used to support the media and entertainment industry’s efforts in
producing contents that generate high volumes of readers, and can help many businesses better understand readers’ interest.

The final paper can be found [here](https://github.com/HuseyinAltnsk/article-popularity-prediction-ML/blob/master/Predicting%20Popularity%20of%20News%20Articles%20Using%20Machine%20Learning.pdf).
#### Detailed Conclusion
We used three regression models in the project: linear support vector regression, random forest regressor and lasso regression.
We set out to build a model that could accurately predict the popularity of news articles in terms of momentum scores. After
converting text data in to trainable, numerical values, we built and trained our models, and discovered that LinearSVR had the 
best results. Our most important discovery is that increasing the number of trees in the RandomForestRegressor model
mitigated the high variance problem we saw in our previous results by a fair amount. Using a large number of trees might be
a good idea for this problem, given the immense feature size we had. Should we have more time in the future, we would like to go 
a few steps further with pre-processing our data. First of all, we would like to add a few meta data properties to be the
feature data. Recall that for this project, we only considered the text content of individual articles as our feature. However,
meta data such as the specific website where an article is published, its author, or even the time it was published can
potentially affect its popularity. Secondly, we would like to make sure all duplicates are deleted for a less biased result.
Dropping the duplicates using Python's pandas library allowed us to throw away articles that have the same titles. However,
we found out later in our project that this approach didn’t do the entire job as some articles had the exact same contents and
momentum scores, but had different titles. Lastly, we would trim down our data set even more and take away more words that do 
not contribute to the popularity measure. For this project, we only took out common English words like “a, the, it”, etc. 
However, we believe that only a small number of words should matter for the popularity measure, and sometimes only one keyword 
such as Donald could be responsible for a high score.
