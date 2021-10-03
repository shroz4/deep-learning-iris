import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt

# in order to present the data set, we need to load the module.
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
# KNN (k-nearest neighbors) algorithm
from sklearn.neighbors import KNeighborsClassifier
# train test split
from sklearn.model_selection import train_test_split
# evaluation accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# create an iris data set object
irisDS= load_iris()

y= irisDS.target
X= irisDS.data
# keep only 2 dimentions out of the 4
pca= PCA(n_components=2)
X2= pca.fit_transform(X)
# step 10 rows:
#X3= X2[::10]
#y3= y[::10]

X3= X2
y3= y

#knn= KNeighborsClassifier(n_neighbors=5)
#knn.fit(X3,y3)

# Spliting to training and testing arrays
[X3_train, X3_test, y3_train, y3_test]= train_test_split(X3, y3, test_size= 0.25, train_size= 0.75, random_state= 1)
# training graph
fig1= plt.figure(3)
ax1= fig1.add_subplot(111)
ax1.scatter(X3_train[:,0][y3_train==0], X3_train[:,1][y3_train==0], color='b')
ax1.scatter(X3_train[:,0][y3_train==1], X3_train[:,1][y3_train==1], color='r')
ax1.scatter(X3_train[:,0][y3_train==2], X3_train[:,1][y3_train==2], color='g')
ax1.scatter(X3_test[:,0], X3_test[:,1], color='k')
ax1.set_xlabel('feature 1')
ax1.set_ylabel('feature 2')
plt.show()

knn= KNeighborsClassifier(n_neighbors=5)
knn.fit(X3_train,y3_train)
y3_pred= knn.predict(X3_test)

# checking accuracy
acc_score= accuracy_score(y3_test, y3_pred)
con_mat= confusion_matrix(y3_test, y3_pred)

# cross validation
from sklearn.model_selection import cross_val_score
cvscore= cross_val_score(knn, X3, y3, cv=10, scoring= 'accuracy')
cvscore_mean= np.mean(cvscore)
cvscore_std= np.std(cvscore)

print('cvscore_mean= ', cvscore_mean)
print('cvscore_std= ', cvscore_std)

print('end')
