import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt

# in order to present the data set, we need to load the module.
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# train test split
from sklearn.model_selection import train_test_split

# cross validation
from sklearn.model_selection import cross_val_score
# random forest algorithm
from sklearn.ensemble import RandomForestClassifier

# create an iris data set object
irisDS= load_iris()

y= irisDS.target
X= irisDS.data

# keep only 2 dimentions out of the 4
pca= PCA(n_components=2)
X2= pca.fit_transform(X)

X3= X2
y3= y

# Spliting to training and testing arrays
[X3_train, X3_test, y3_train, y3_test]= train_test_split(X3, y3, test_size= 0.25, train_size= 0.75, random_state= 1)
# training graph
fig1= plt.figure(1)
ax1= fig1.add_subplot(111)
ax1.scatter(X3_train[:,0][y3_train==0], X3_train[:,1][y3_train==0], color='b')
ax1.scatter(X3_train[:,0][y3_train==1], X3_train[:,1][y3_train==1], color='r')
ax1.scatter(X3_train[:,0][y3_train==2], X3_train[:,1][y3_train==2], color='g')
ax1.scatter(X3_test[:,0], X3_test[:,1], color='k')
ax1.set_xlabel('feature 1')
ax1.set_ylabel('feature 2')
ax1.plot([0.25,2.5],[-1.25,1.5], linestyle=':', color='m')
ax1.plot([-2,-2],[-1.25,1.5], linestyle=':', color='m')
plt.show()


randForest= RandomForestClassifier()
# cross validation
cvscore= cross_val_score(randForest, X3, y3, cv=10, scoring= 'accuracy')
cvscore_mean= np.mean(cvscore)
cvscore_std= np.std(cvscore)

print('cvscore_mean= ', cvscore_mean)
print('cvscore_std= ', cvscore_std)

print('end')


