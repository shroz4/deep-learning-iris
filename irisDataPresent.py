import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# in order to present the data set, we need to load the module.
from sklearn.datasets import load_iris
print("shanit")
# create an iris data set object
irisDS= load_iris()
print (type(irisDS))
# print the iris data
#print (irisDS.data)
print (type(irisDS.data))
print(irisDS.feature_names)
print(irisDS.target)
print(type(irisDS.target))
print(irisDS.target_names)

data= irisDS.data
columns= irisDS.feature_names
irisDF= pd.DataFrame(data=data, columns=columns)
print(irisDF)
#irisDF['target']= irisDS['target']
#print(irisDF)

# changing the column type to category
irisDF['target_name']= irisDS['target']
irisDF['target_name']= irisDF['target_name'].astype('category')
print(irisDF)
print('shanit')
print(irisDS['target_names'][1])

# changing the numeric value to the iris type as string
# first create array of the string. target_names column
targetSize= np.size(irisDS.target_names)
print(type(targetSize))
targetNameCol= {}
for i in range(targetSize):
    print(i)
    targetNameCol[i]=irisDS['target_names'][i]
    print(targetNameCol)
irisDF['target_name']= irisDF['target_name'].cat.rename_categories(targetNameCol)
print(irisDF)
# exporting irisDF to excel
irisDF.to_excel('iris.xlsx')
# creating graphs
irisG= sns.pairplot(irisDF, hue= 'target_name')
plt.show()
print('end')
