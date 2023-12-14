import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
import re

df = pd.read_csv('fullDF.csv')
df['title'] = df['title'].str.lower()
df['title'].fillna('', inplace=True)
df['title'] = df['title'].apply(lambda x: re.sub("[^\\w\\s]", "", x))
df['title'] = df['title'].apply(lambda x: re.sub('[0-9]', "", x))

#grps = df.groupby('category_id')
#df = grps.get_group(110)

print(type(df))
print(len(df))

def helperRev(x):
    x = int(x)
    if x<10000:
        return 0
    return 1
    
df['revenue'] = df['revenue'].apply(helperRev)
print(df["revenue"].value_counts())

#print(len(df))
df = df.sample(frac=1, random_state=0)
print(df.head)

#rfX = df.iloc[:200000, :]["title"]
#rfY = df.iloc[:200000, :]["revenue"]

rfX = df["title"]
rfY = df["revenue"]

rfX = np.array(rfX)
rfY = np.array(rfY)

for i in range(len(rfX)):
    try:
        rfX[i] = rfX[i].split()
    except:
        print(rfX[i])
        rfX[i] = []
print(rfX)

d = {}
for i in range(len(rfX)):
    arr = []
    for q in range(len(rfX[i])):
        #d.setdefault(rfX[i][q], 0)
        #d[rfX[i][q]] += 1
        arr.append(rfX[i][q])
    arr = np.array(arr)
    arr = np.unique(arr)
    for q in range(len(arr)):
        d.setdefault(arr[q], 0)
        d[arr[q]] += 1

#print(len(d.keys()))

c = []

for i in d.keys():
    if d[i] < 10000: #50
        c.append(i)

for i in c:
    d.pop(i)

print(len(d.keys()))

x = []
pbar = tqdm(total=100)
for i in range(len(rfX)):
    #print(i, len(rfX))
    pbar.update(100 / len(rfX))
    ar = []
    for key in d.keys():
        if key in rfX[i]:
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

full_array = []
for i in range(len(keys)):
    full_array.append([keys[i], weights[i]])

full_array.sort(key=lambda xx: xx[1])
#print(full_array[:30])
#print(full_array[len(full_array)-30:])

q = full_array[:10] + full_array[len(full_array)-10:]
q.reverse()
q = pd.DataFrame(q)
q = q.rename({0: "Word", 1: "Weight"}, axis="columns")
print(q)