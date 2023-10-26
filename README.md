# advtopics_ML1
These notebooks focus on topics in machine learning and deeplearning:

The first problem explores the bias-variance tradeoff in regression, involving dataset creation, polynomial estimator fitting, and assessing model complexity tradeoffs.
Problem two involves hyperparameter tuning for the K-Nearest Neighbors (KNN) algorithm using cross-validation, aiming to identify the optimal K and distance metric.
Problem three tackles algorithm evaluation using confusion matrices, comparing metrics like balanced accuracy and F-1 score, and considering the effectiveness of different advice in choosing the best algorithm.
The final problem delves into logistic regression with regularization, examining the impact of different penalties, C values, and the L2 norm of coefficients, offering insights into the regularization's effects.

Specifically: 
Problem 1 - Bias Variance Tradeoff 

Create a dataset with 20 points using the function y(x) = x + sin(1.5x) + N(0,0.32).
Use weighted sum of polynomials as estimator functions (g1, g3, g5, g10) and plot them along with y(x) and f(x).
Generate 100 datasets, each with 50 points, and fit estimators of varying complexity (g1 to g15) to calculate squared bias, variance, and error on the testing set, showing the tradeoff with model complexity.
Identify the best model with the smallest Mean Squared Error and determine its bias and variance.

Problem 2 - KNN Hyperparameter Tuning using Cross Validation 

Train a KNN classifier with K=4 and p=2, evaluating misclassification error, accuracy, precision, recall, and F-1 score on the test set.
Perform 5-fold cross-validation to find the best value of K for p=1 and p=2, plotting misclassification error against K for both cases.
Determine the best value of K with Euclidean distance, compare it to Manhattan distance, and find the best combination of p and K for the minimum misclassification error.

Problem 3 - Which Algorithm is Better? 

Create confusion matrices for two document retrieval algorithms, counting TP, FP, FN, TN.
Decide between using Balanced accuracy or F-1 score for identifying the better algorithm, supported by calculations.
Explain whether the advice from a friend or instructor helped or not and list metrics that could aid in making the right selection.

Problem 4 - Logistic Regression with Regularization 

Explain the significance of parameters C, solver, penalty, and multi-class in the LogisticRegression class of scikit-learn.
Define 'l1' and 'l2' penalties for logistic regression.
Fit 20 logistic regression models with 'l1' and 'l2' penalties, varying C values, and multi-class='ovr'.
Collect weight coefficients for features, plot them against C values, and observe the effect of regularization.
Calculate the ratio of L2 norm for coefficients with varying degrees of regularization and plot the results.
