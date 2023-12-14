import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

"""df = pd.read_csv('amazon_products.csv')
df['revenue'] = df['price']*df['boughtInLastMonth']
for i in range(len(df)):
    if df.loc[i, 'listPrice'] == 0:
        df.loc[i, 'listPrice'] = df.loc[i, 'price']
df['discount'] = round((1 - df['price']/df['listPrice']), 2)

df.to_csv('fullDF.csv', index=False)"""

df = pd.read_csv('fullDF.csv')

import math
def DisTr(x):
    if math.isnan(x):
        return 0
    x = int(round(x, 1)*100)
    return x
df['discount'] = df['discount'].apply(DisTr)
print(df['discount'].value_counts())

"""def helperRev(x):
    x = int(x)
    if x==0:
        return 0
    elif x>0 and x<=250:
        return 125
    elif x>250 and x<=750:
        return 500
    elif x>750 and x<=1250:
        return 1000
    elif x>1000 and x<=5000:
        return 2500
    elif x>5000 and x<=10000:
        return 7500
    elif x>10000 and x<=20000:
        return 15000
    elif x>20000 and x<=30000:
        return 25000
    elif x>30000 and x<=40000:
        return 35000
    elif x>50000 and x<=60000:
        return 55000
    elif x>60000 and x<=70000:
        return 65000
    elif x>70000 and x<=80000:
        return 75000
    elif x>80000 and x<=90000:
        return 85000
    elif x>90000 and x<=100000:
        return 95000
    elif x>100000 and x<=150000:
        return 125000
    elif x>150000 and x<=250000:
        return 200000
    elif x>250000 and x<=500000:
        return 375000
    elif x>500000 and x<=1000000:
        return 750000
    else:
        return 1000000"""

def helperRev(x):
    x = int(x)
    if x==0:
        return 0
    elif x>0 and x<=1000:
        return 500
    elif x>1000 and x<=5000:
        return 2500
    elif x>5000 and x<=15000:
        return 10000
    elif x>15000 and x<=50000:
        return 35000
    elif x>50000 and x<=100000:
        return 75000
    elif x>100000 and x<=200000:
        return 150000
    elif x>200000 and x<=500000:
        return 350000
    elif x>500000 and x<=1000000:
        return 750000
    else:
        return 1000000

"""def helperRev(x):
    x = int(x)
    if x==0:
        return "None"
    elif x>0 and x<=1000:
        return "Below 1000"
    elif x>1000 and x<=10000:
        return "1,000 - 10,000"
    elif x>10000 and x<=50000:
        return "10,000 - 50,000"
    elif x>50000 and x<=100000:
        return "10,000 - 50,000"
    elif x>100000 and x<=500000:
        return "100,000 - 500,000"
    elif x>500000 and x<=1000000:
        return "500,000 - 1,000,000"
    else:
        return "More than 1000000"""
    
df['revenue'] = df['revenue'].apply(helperRev)
print(df["revenue"].value_counts())

#print(df['discount'].value_counts())

#lrX = df[["reviews", "price", "listPrice", "discount", "category_id", "isBestSeller"]]
"""lrX = df[["reviews", "price", "listPrice", "category_id", "isBestSeller"]]
lrX = lrX.to_numpy()
lrY = df["stars"]
lrY = lrY.to_numpy()
regr = linear_model.LinearRegression()

norm = Normalizer()
norm.fit_transform(lrX, lrY)

X_train, X_test, y_train, y_test = train_test_split(lrX,lrY , random_state=104,test_size=0.25, shuffle=True)

regr.fit(X_train, y_train)
yhat = regr.predict(X_test)
print("LR:")
print(mean_squared_error(y_test, yhat))
print(regr.score(X_test, y_test))"""

#grps = df.groupby('category_id')
#df = grps.get_group(1)
#rfX = df[["reviews", "price", "listPrice", "category_id", "isBestSeller", "stars"]]
#rfX = df[["reviews", "price", "listPrice", "category_id", "isBestSeller", "stars", "discount"]]
rfX = df[["reviews", "price", "category_id", "isBestSeller", "stars", "discount"]]
rfY = df["revenue"]
rfX = np.array(rfX)
rfY = np.array(rfY)
norm = Normalizer()
#norm.fit_transform(rfX, rfY)
X_train, X_test, y_train, y_test = train_test_split(rfX,rfY , random_state=104,test_size=0.25, shuffle=True)

#print("LR:")
#clf = linear_model.LinearRegression()
#clf.fit(X_train, y_train)
#yhat = clf.predict(X_test)
#print(mean_absolute_percentage_error(y_test, yhat))

"""print("RF:")
clf = RandomForestClassifier(max_depth=25, min_samples_leaf=100, random_state=0)
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
print(accuracy_score(y_test, yhat))
yhat = clf.predict(X_train)
print(accuracy_score(y_train, yhat))
print(yhat[:10])
print(y_test[:10])
#print(mean_squared_error(y_test, yhat))
#print(clf.score(X_test, y_test))
#print(mean_absolute_percentage_error(y_test, yhat))
#print(mean_absolute_error(y_test, yhat))"""

print("DT:")
clf = DecisionTreeClassifier(max_depth=25, min_samples_leaf=500)
#clf = RandomForestClassifier(max_depth=25, min_samples_leaf=100, random_state=0)
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
print(accuracy_score(y_test, yhat))
yhat2 = clf.predict(X_train)
print(accuracy_score(y_train, yhat2))
"""print(yhat[:10])
print(y_test[:10])

d = {}
print(len(yhat))
print(len(y_test))
for i in range(len(yhat)):
    if yhat[i] == y_test[i]:
        d.setdefault(yhat[i], 0)
        d[yhat[i]] += 1
for i in d.keys():
    print(i)
    print(d[i])"""

#print("Support vector machine")
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler
#from sklearn.svm import SVC
#clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
#clf.fit(X_train, y_train)
#yhat = clf.predict(X_test)
#print(accuracy_score(y_test, yhat))
#print(mean_squared_error(y_test, yhat))
#print(clf.score(X_test, y_test))
#print(mean_absolute_percentage_error(y_test, yhat))
#print(mean_absolute_error(y_test, yhat))



print("NUNET")
from sklearn.neural_network import MLPClassifier
regr = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
yhat = regr.predict(X_test)
print(accuracy_score(y_test, yhat))
#print(regr.score(X_test, y_test))
#print(mean_squared_error(y_test, yhat))
#print(mean_absolute_percentage_error(y_test, yhat))
#print(mean_absolute_error(y_test, yhat))
#print(yhat[:10])
#print(y_test[:10])

print("KNN")
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=50)
neigh.fit(X_train, y_train)
yhat = neigh.predict(X_test)
print(accuracy_score(y_test, yhat))
#print(mean_squared_error(y_test, yhat))
#print(neigh.score(X_test, y_test))
#print(mean_absolute_percentage_error(y_test, yhat))
#print(mean_absolute_error(y_test, yhat))