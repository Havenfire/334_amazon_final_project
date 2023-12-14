import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

"""df = pd.read_csv('fullDF.csv')
df['title'] = df['title'].str.lower()
df['title'].fillna('', inplace=True)
df['title'] = df['title'].apply(lambda x: re.sub("[^\\w\\s]", "", x))
df['title'] = df['title'].apply(lambda x: re.sub('[0-9]', "", x))

df_ids = pd.read_csv('title_clusters_output.csv')
df['id'] = df_ids['category_cluster']

import math
def DisTr(x):
    if math.isnan(x):
        return 0
    x = int(round(x, 1)*100)
    return x
df['discount'] = df['discount'].apply(DisTr)
print(df['discount'].value_counts())

def helperRev(x):
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
        return 1000000
    
df['revenue'] = df['revenue'].apply(helperRev)
#print(df["revenue"].value_counts())

#print(df)
#df.to_csv('completeDF.csv', index=False)"""
"""df = pd.read_csv('completeDF.csv')
print(df)


#rfX = df[["reviews", "price", "category_id", "isBestSeller", "stars", "discount"]]
#rfY = df["revenue"]
#rfX = np.array(rfX)
#rfY = np.array(rfY)
#X_train, X_test, y_train, y_test = train_test_split(rfX,rfY , random_state=104,test_size=0.25, shuffle=True)


#--------------------------------------------

df["revenue_perceptron"] = df["revenue"]
def helperRevPerc(x):
    x = int(x)
    if x<10000:
        return 0
    return 1
    
df['revenue_perceptron'] = df['revenue_perceptron'].apply(helperRevPerc)
print(df["revenue_perceptron"].value_counts())
df = df.sample(frac=1, random_state=0)
title = df.iloc[:500000, :]["title"]
rfY = df.iloc[:500000, :]["revenue_perceptron"]
rfY = np.array(rfY)
title = np.array(title)

for i in range(len(title)):
    try:
        title[i] = title[i].split()
    except:
        print(title[i])
        title[i] = []
#print(title)

d = {}
for i in range(len(title)):
    arr = []
    for q in range(len(title[i])):
        arr.append(title[i][q])
    arr = np.array(arr)
    arr = np.unique(arr)
    for q in range(len(arr)):
        d.setdefault(arr[q], 0)
        d[arr[q]] += 1

#print(len(d.keys()))

c = []

for i in d.keys():
    if d[i] < 1000: #50
        c.append(i)

for i in c:
    d.pop(i)

print(len(d.keys()))

x = []
pbar = tqdm(total=100)
for i in range(len(title)):
    pbar.update(100 / len(title))
    ar = []
    for key in d.keys():
        if key in title[i]:
            ar.append(1)
        else:
            ar.append(0)
    x.append(ar)

#print(x)
x = np.array(x)
y = rfY

X_train, X_test, y_train, y_test = train_test_split(x,y , random_state=104,test_size=0.25, shuffle=True)

clf = Perceptron()
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
print(accuracy_score(y_test, yhat))
keys = list(d.keys())
weights = clf.coef_[0]

#print(keys)
#print(weights)

#full_array = []
fulldict = {}
for i in range(len(keys)):
    #full_array.append([keys[i], weights[i]])
    fulldict[keys[i]] = weights[i]

df["title_score"] = df["revenue"]

pbar = tqdm(total=100)

for i in range(len(df)):
    try:
        title_i = df.loc[i, "title"].split()
        title_score = 0
        for q in title_i:
            if q in fulldict:
                title_score += fulldict[q]
            else:
                title_score = 0

        df.loc[i, "title_score"] = title_score
    except:
        df.loc[i, "title_score"] = 0
    finally:
        pbar.update(100 / len(df))

print(df)
df.to_csv('SuperCompleteDF.csv', index=False)

#full_array.sort(key=lambda xx: xx[1])
#print(full_array[:30])
#print(full_array[len(full_array)-30:])

#q = full_array[:10] + full_array[len(full_array)-10:]
#q.reverse()
#q = pd.DataFrame(q)
#q = q.rename({0: "Word", 1: "Weight"}, axis="columns")
#print(q)"""


#---------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('SuperCompleteDF.csv')

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
    
df['revenue'] = df['revenue'].apply(helperRev)

#rfX = df[["reviews", "price", "category_id", "isBestSeller", "stars", "discount", "title_score", "id"]]
rfX = df[["reviews", "price", "category_id", "isBestSeller", "stars", "discount"]]
rfY = df["revenue"]
rfX = np.array(rfX)
rfY = np.array(rfY)
X_train, X_test, y_train, y_test = train_test_split(rfX,rfY , random_state=104,test_size=0.25, shuffle=True)

"""print("DT:")
mdept = [1, 5, 10, 15, 20, 25, 35]
min_leaf = [1, 5, 10, 25, 50, 100, 200, 500]
best = (0, 0)
best_acc = 0

acc_arr = []
acc_arr_dept_x = []
acc_arr_ml_x = []

for dp in mdept:
    print(dp)
    for min_l in min_leaf:
        clf = DecisionTreeClassifier(max_depth=dp, min_samples_leaf=min_l)
        clf.fit(X_train, y_train)
        yhat = clf.predict(X_test)
        ac = accuracy_score(y_test, yhat)
        if ac > best_acc:
            best = (dp, min_l)
            best_acc = ac
        acc_arr_dept_x = dp
        acc_arr_ml_x = min_l
        acc_arr = ac

print(best)
print(best_acc)"""

"""print("RF:")
accarr = []
best_acc = 0
best = 0
min_estimators = [1, 20, 40, 60, 80, 100]
for i in min_estimators:
    print(i)
    clf = RandomForestClassifier(max_depth=25, min_samples_leaf=50, n_estimators=i, max_features=None) #60 is best
    clf.fit(X_train, y_train)
    yhat = clf.predict(X_test)
    ac = accuracy_score(y_test, yhat)
    if ac > best_acc:
            best = i
            best_acc = ac
    acc_arr = ac
    accarr.append(ac)

print(best)
print(best_acc)
plt.plot(accarr)
plt.title("Accuracy VS number of trees")
plt.xlabel("Number of trees")
plt.ylabel("Accuracy")
plt.show()"""

"""accarr = []
best_acc = 0
best = 0
mdept = [1, 5, 10, 15, 20, 25, 35]
for i in mdept:
    print(i)
    clf = DecisionTreeClassifier(max_depth=i, min_samples_leaf=50)
    clf.fit(X_train, y_train)
    yhat = clf.predict(X_test)
    ac = accuracy_score(y_test, yhat)
    if ac > best_acc:
            best = i
            best_acc = ac
    acc_arr = ac
    accarr.append(ac)

print(best)
print(best_acc)
plt.plot(accarr)
plt.title("Accuracy VS Max depth")
plt.xlabel("Max depth")
plt.ylabel("Accuracy")
plt.xticks(list(range(len(mdept))), mdept)
plt.show()"""

print("DT:")
clf = DecisionTreeClassifier(max_depth=25, min_samples_leaf=50)
#clf = RandomForestClassifier(max_depth=25, min_samples_leaf=100, random_state=0)
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
print(accuracy_score(y_test, yhat))

print("RF:")
clf = RandomForestClassifier(max_depth=25, min_samples_leaf=50, random_state=0, n_estimators=60, max_features=None)
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
print(accuracy_score(y_test, yhat))

print("NUNET")
from sklearn.neural_network import MLPClassifier
regr = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
yhat = regr.predict(X_test)
print(accuracy_score(y_test, yhat))

print("KNN")
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=50)
neigh.fit(X_train, y_train)
yhat = neigh.predict(X_test)
print(accuracy_score(y_test, yhat))