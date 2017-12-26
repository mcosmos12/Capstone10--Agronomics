import pandas as pd
import csv
import numpy as np
from sklearn.svm import SVC
import pylab
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from scipy import optimize
import math
import pandas.plotting
from sklearn.model_selection import train_test_split
from pandas.plotting import radviz
from sklearn import tree
from sklearn import svm
from scipy.stats import sem
from sklearn import cross_validation as cv
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from sklearn import preprocessing
from sklearn import utils
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer  
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.cross_validation import KFold
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
#from yellowbrick.regressor import PredictionError


dataset = pd.read_csv('5-MERGE.csv',dtype={'Year':int,'Harvested':int,'Production':int,'Price':float})

print(dataset.County.value_counts())

print(dataset.County_Code.value_counts())

print(dataset.Commodity_Code.value_counts())

print(dataset.dtypes)

print(dataset.columns)

print(dataset.shape)


from sklearn import preprocessing
le_county = preprocessing.LabelEncoder()

#to convert into numbers
dataset.County = le_county.fit_transform(dataset.County)


print(dataset.describe())

print(dataset.head())

print(dataset.tail())

features = ['Year','Harvested','Value','Grow_total_p','Grow_avg_t','Price','Percapita_Personal_Income','House_Price_Index']
target = 'Yield'


X = (dataset[features])
y = (dataset[target])


# L2 and L1 Regularization 
alphas = np.logspace(-10, 0, 200)


scaler = StandardScaler().fit(X)
X = scaler.transform(X)

# Create training and test splits 
X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.2,random_state=0)

from sklearn import cross_validation

clf = LinearRegression() 
scores = cross_validation.cross_val_score(
clf, X_train, y_train, cv=5)
print(scores) 
print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))


model = LinearRegression() 
model.fit(X_train, y_train)
yhat = model.predict(X_test)
r2 = r2_score(y_test, yhat)
me = mse(y_test, yhat)
print("r2={:0.3f} MSE={:0.3f}".format(r2,me))


from yellowbrick.regressor import PredictionError
# Instantiate the visualizer
visualizer = PredictionError(LinearRegression())
# Fit
visualizer.fit(X_train, y_train)
# Score and visualize
visualizer.score(X_test, y_test)
visualizer.poof()


from yellowbrick.regressor import ResidualsPlot
model = ResidualsPlot(LinearRegression())
model.fit(X_train, y_train)
model.score(X_test, y_test)
model.poof()


model = ElasticNetCV(alphas=alphas) 
model.fit(X_train, y_train)
yhat = model.predict(X_test)
r2 = r2_score(y_test, yhat)
me = mse(y_test, yhat)
print("r2={:0.3f} MSE={:0.3f}".format(r2,me))


from yellowbrick.regressor import PredictionError
# Instantiate the visualizer
visualizer = PredictionError(ElasticNetCV())
# Fit
visualizer.fit(X_train, y_train)
# Score and visualize
visualizer.score(X_test, y_test)
visualizer.poof()


model = LassoCV(alphas=alphas) 
model.fit(X_train, y_train)
yhat = model.predict(X_test)
r2 = r2_score(y_test, yhat)
me = mse(y_test, yhat)
print("r2={:0.3f} MSE={:0.3f}".format(r2,me))



from yellowbrick.regressor import PredictionError
# Instantiate the visualizer
visualizer = PredictionError(ElasticNetCV())
# Fit
visualizer.fit(X_train, y_train)
# Score and visualize
visualizer.score(X_test, y_test)
visualizer.poof()



model = Pipeline([
    ('poly', PolynomialFeatures(2)), 
    ('lasso', LassoCV(alphas=alphas)),
])
model.fit(X_train, y_train)
yhat = model.predict(X_test)
r2 = r2_score(y_test, yhat)
me = mse(y_test, yhat)
print("r2={:0.3f} MSE={:0.3f} alpha={:0.3f}".format(r2,me, model.named_steps['lasso'].alpha_))


from yellowbrick.regressor import PredictionError
# Instantiate the visualizer
visualizer = PredictionError(Pipeline([
    ('poly', PolynomialFeatures(2)), 
    ('lasso', LassoCV(alphas=alphas)),
]))
# Fit
visualizer.fit(X_train, y_train)
# Score and visualize
visualizer.score(X_test, y_test)
visualizer.poof()



from yellowbrick.regressor import ResidualsPlot
model = ResidualsPlot(LassoCV())
model.fit(X_train, y_train)
model.score(X_test, y_test)
model.poof()



from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators = 5, random_state = 0) 
model.fit(X_train, y_train)
yhat = model.predict(X_test)
r2 = r2_score(y_test, yhat)
me = mse(y_test, yhat)
print("r2={:0.3f} MSE={:0.3f}".format(r2,me))


from yellowbrick.regressor import PredictionError
# Instantiate the visualizer
visualizer = PredictionError(RandomForestRegressor())
# Fit
visualizer.fit(X_train, y_train)
# Score and visualize
visualizer.score(X_test, y_test)
visualizer.poof()



from sklearn import tree
from IPython.core.display import Image
from pandas.compat import StringIO

import pydotplus
# Visualize tree
dot_data = StringIO()
tree.export_graphviz(model.estimators_[0], out_file=dot_data)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
image = graph.write("random_network")


from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor() 
model.fit(X_train, y_train)
yhat = model.predict(X_test)
r2 = r2_score(y_test, yhat)
me = mse(y_test, yhat)
print("r2={:0.3f} MSE={:0.3f}".format(r2,me))


from yellowbrick.regressor import PredictionError
# Instantiate the visualizer
visualizer = PredictionError(AdaBoostRegressor())
# Fit
visualizer.fit(X_train, y_train)
# Score and visualize
visualizer.score(X_test, y_test)
visualizer.poof()


from sklearn.linear_model import BayesianRidge
model = BayesianRidge() 
model.fit(X_train, y_train)
yhat = model.predict(X_test) #predicted
r2 = r2_score(y_test, yhat)
me = mse(y_test, yhat)
print("r2={:0.3f} MSE={:0.3f}".format(r2,me))


from yellowbrick.regressor import PredictionError
# Instantiate the visualizer
visualizer = PredictionError(BayesianRidge())
# Fit
visualizer.fit(X_train, y_train)
# Score and visualize
visualizer.score(X_test, y_test)
visualizer.poof()


model = svm.SVR()
model.fit(X_train, y_train)
yhat = model.predict(X_test)
r2 = r2_score(y_test, yhat)
me = mse(y_test, yhat)
print("r2={:0.3f} MSE={:0.3f}".format(r2,me))


from yellowbrick.regressor import PredictionError
# Instantiate the visualizer
visualizer = PredictionError(svm.SVR())
# Fit
visualizer.fit(X_train, y_train)
# Score and visualize
visualizer.score(X_test, y_test)
visualizer.poof()



from sklearn import linear_model
model = linear_model.Lasso (alpha=0.2)
model.fit(X_train, y_train)
yhat = model.predict(X_test) #predicted
r2 = r2_score(y_test, yhat)
me = mse(y_test, yhat)
print("r2={:0.3f} MSE={:0.3f}".format(r2,me))



from yellowbrick.regressor import PredictionError
# Instantiate the visualizer
visualizer = PredictionError(Lasso())
# Fit
visualizer.fit(X_train, y_train)
# Score and visualize
visualizer.score(X_test, y_test)
visualizer.poof()


model = svm.NuSVR()
model.fit(X_train, y_train)
yhat = model.predict(X_test)
r2 = r2_score(y_test, yhat)
me = mse(y_test, yhat)
print("r2={:0.3f} MSE={:0.3f}".format(r2,me))



from yellowbrick.regressor import PredictionError
# Instantiate the visualizer
visualizer = PredictionError(svm.NuSVR())
# Fit
visualizer.fit(X_train, y_train)
# Score and visualize
visualizer.score(X_test, y_test)
visualizer.poof()


model = svm.LinearSVR()
model.fit(X_train, y_train)
yhat = model.predict(X_test)
r2 = r2_score(y_test, yhat)
me = mse(y_test, yhat)
print("r2={:0.3f} MSE={:0.3f}".format(r2,me))



from yellowbrick.regressor import PredictionError
# Instantiate the visualizer
visualizer = PredictionError(svm.LinearSVR())
# Fit
visualizer.fit(X_train, y_train)
# Score and visualize
visualizer.score(X_test, y_test)
visualizer.poof()


from sklearn.neighbors import KNeighborsRegressor
model= KNeighborsRegressor(n_neighbors=2)
model.fit(X_train, y_train)
yhat = model.predict(X_test)
r2 = r2_score(y_test, yhat)
me = mse(y_test, yhat)
print("r2={:0.3f} MSE={:0.3f}".format(r2,me))


from yellowbrick.regressor import PredictionError
# Instantiate the visualizer
visualizer = PredictionError(KNeighborsRegressor())
# Fit
visualizer.fit(X_train, y_train)
# Score and visualize
visualizer.score(X_test, y_test)
visualizer.poof()


model = Ridge()
model.fit(X_train, y_train)
yhat = model.predict(X_test)
r2 = r2_score(y_test, yhat)
me = mse(y_test, yhat)
print("r2={:0.3f} MSE={:0.3f}".format(r2,me))


from yellowbrick.regressor import PredictionError
# Instantiate the visualizer
visualizer = PredictionError(Ridge())
# Fit
visualizer.fit(X_train, y_train)
# Score and visualize
visualizer.score(X_test, y_test)
visualizer.poof()


# Create a histogram
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(dataset['Yield'], bins = 10, range = (dataset['Yield'].min(),dataset['Yield'].max()))
plt.title('Yield distribution')
plt.xlabel('Yield')
plt.ylabel('Year')
plt.show()


dataset[features].hist(figsize=(20,10))
plt.show()


g1 = sns.boxplot(y='Yield', x='Year', data=dataset)
plt.show()


g = sns.jointplot("Year", "Yield",kind='hex',data= dataset)
plt.show()


dataset.plot(kind='scatter', x='Grow_avg_t', y='Grow_total_p', c='Yield',figsize=[20,8])
plt.style.use('ggplot')
plt.show()

dataset.plot(kind='scatter', x='Year', y='Annual_t', c='Yield',figsize=[20,10])
plt.style.use('ggplot')
plt.show()

dataset.plot(kind='scatter', x='Year', y='Total_p', c='Yield',figsize=[20,10])
plt.style.use('ggplot')
plt.show()


dataset.plot(kind='scatter', x='Year', y='Yield', c='Price',figsize=[20,10])
plt.show()


dataset.plot(kind='scatter', x='Year', y='Yield', c='Percapita_Personal_Income',figsize=[20,10])
plt.show()


dataset.plot(kind='scatter', x='Year', y='Yield', c='Resident_Population',figsize=[20,10])
plt.show()


dataset.plot(kind='scatter', x='Year', y='Annual_t', c='Total_p',figsize=[20,10])
plt.style.use('ggplot')
plt.show()


dataset.plot(kind='scatter', x='Year', y='Annual_t', c='Total_p',figsize=[20,10])
plt.style.use('ggplot')
plt.show()


from pandas.tools.plotting import scatter_matrix
areas = dataset[['Year','Yield','Price','Grow_total_p','Grow_avg_t','House_Price_Index','Personal_Income']]
scatter_matrix(areas, alpha=0.2, figsize=(18,18), diagonal='kde')
plt.show()


sns.set()


data1 = dataset.pivot('Yield', 'Year', 'Price')

# Draw a heatmap with the numeric values in each cell
sns.heatmap(data1, annot=True, fmt='f', linewidths=1)
plt.show()


import pickle 

with open('forest-riders.pkl', 'wb') as f:
    pickle.dump(model, f)


with open('forest-riders.pkl', 'rb') as f:
    model = pickle.load(f)


from pandas.tools.plotting import radviz
plt.figure(figsize=(12,12))
radviz(dataset, 'Yield')
plt.show()


from yellowbrick.features.rankd import Rank2D 
# Instantiate the visualizer with the Covariance ranking algorithm 
visualizer = Rank2D(features=features, algorithm='covariance')

visualizer.fit(X, y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()    # Draw/show/poof the data


################################################3
#######RANDOM FOREST REGRESSOR TREE######

# features = ['Year','Harvested','Value','Grow_total_p','Grow_avg_t','Price']
# target = 'Yield'
# 
# from sklearn.datasets import make_regression
# from sklearn.ensemble import RandomForestRegressor
# 
# X = (dataset[features])
# y = (dataset[target])
# 
# # L2 and L1 Regularization 
# #alphas = np.logspace(-10, 0, 200)
# 
# X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.2,random_state=1)
# 
# 
# model = RandomForestRegressor(n_estimators = 10, random_state = 0,max_depth=3) 
# model.fit(X_train, y_train)
# yhat = model.predict(X_test)
# r2 = r2_score(y_test, yhat)
# me = mse(y_test, yhat)
# print("r2={:0.3f} MSE={:0.3f}".format(r2,me))
# 
# from sklearn import tree
# from IPython.core.display import Image
# from pandas.compat import StringIO
# 
# import pydotplus
# # Visualize tree
# dot_data = StringIO()
# tree.export_graphviz(model.estimators_[5], out_file='tree.dot')
