import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestRegressor

"""df = pd.read_csv('amazon_products.csv')
df['revenue'] = df['price']*df['boughtInLastMonth']
for i in range(len(df)):
    if df.loc[i, 'listPrice'] == 0:
        df.loc[i, 'listPrice'] = df.loc[i, 'price']
df['discount'] = round((1 - df['price']/df['listPrice']), 2)

df.to_csv('fullDF.csv', index=False)"""

df = pd.read_csv('fullDF.csv')

#print(df['discount'].value_counts())

#lrX = df[["reviews", "price", "listPrice", "discount", "category_id", "isBestSeller"]]
lrX = df[["reviews", "price", "listPrice", "category_id", "isBestSeller"]]
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
print(regr.score(X_test, y_test))

rfX = df[["reviews", "price", "listPrice", "category_id", "isBestSeller", "stars"]]
rfY = df["revenue"]
norm = Normalizer()
norm.fit_transform(rfX, rfY)
clf = RandomForestRegressor(max_depth=2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(rfX,rfY , random_state=104,test_size=0.25, shuffle=True)
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
print("RF:")
print(mean_squared_error(y_test, yhat))
print(clf.score(X_test, y_test))