  
Anika Bhattacharya  
Travis Gillespie  
Marty Yossifov  
William Zou

DSC232R

Professor Edwin Solares  
TA Nishanth Ramesha  
Group Project Part 3 Question 4

For the first distributed model, we trained a Decision Tree Classifier to predict whether a user action would lead to buy\_comp (purchase \[1\] or non-purchase \[0\]). We use price-based and categorical features for our model, which achieved about 60% validation accuracy. This is better than random guessing for a binary classification task like buy\_comp, so it provides a useful baseline for predicting purchases, but better tuning or stronger models would most likely be needed to improve performance.

Indeed, a potential improvement for our model is testing different hyperparameters like the depth of the tree. For instance, a deeper tree could capture more complex relationships in the data, but it could produce overfitting. Beyond this improvement, we could also try different, more sophisticated, models like Random Forests. This could help because a single tree can be too simple since it’s overly relied upon to make predictions. Having a forest combines predictions from many trees which can produce better accuracy. Gradient Boosted trees are another great alternative since they have trees built sequentially in order to improve accuracy.

Distributed computing helped process this large dataset because Spark distributed the preprocessing steps as well as model training across workers. This reduces computation time to the point where we can make use of the SDSC clusters and Apache Spark in order to run this analysis in a timely way (if at all, since this would very unlikely be possible to run locally).