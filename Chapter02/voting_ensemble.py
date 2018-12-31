import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
# We finally use the voting classifier to choose the best of the three. Next we create the sub models 
voters = []
log_reg = LogisticRegression() # the logistic regression model
voters.append(('logistic', model1))
desc_tree = DecisionTreeClassifier() # the decision tree classifier model
voters.append(('cart', model2))
cup_vec_mac = SVC() # the support vector machine model
voters.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(voters)
#The final model is chosen by performing a k- fold cross validation:
results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())
