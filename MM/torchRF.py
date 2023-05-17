import os 
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

os.getcwd()

data = pd.read_csv("./otto_train.csv")
data.head()

nCar = data.shape[0] 
nVar = data.shape[1] 
print('nCar: %d' % nCar, 'nVar: %d' % nVar )

data= data.drop(['id'],axis=1)

mapping_dict = {'Class_1' : 1,
                'Class_2' : 2,
                'Class_3' : 3,
                'Class_4' : 4,
                'Class_5' : 5,
                'Class_6' : 6,
                'Class_7' : 7,
                'Class_8' : 8,
                'Class_9' : 9,}
after_mapping_target = data['target'].apply(lambda x : mapping_dict[x])
after_mapping_target

feature_columns = list(data.columns.difference(['target']))
X = data[feature_columns]
y = after_mapping_target

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42) 
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) 


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 

clf = RandomForestClassifier(n_estimators=20, max_depth=5,random_state=0)
clf.fit(train_x,train_y)

predict1 = clf.predict(test_x)
print(accuracy_score(test_y,predict1))


clf = RandomForestClassifier(n_estimators=100, max_depth=20,random_state=0)
clf.fit(train_x,train_y)

predict2 = clf.predict(test_x)
print(accuracy_score(test_y,predict2))


clf = RandomForestClassifier(n_estimators=300, max_depth=20,random_state=0)
clf.fit(train_x,train_y)

predict2 = clf.predict(test_x)
print(accuracy_score(test_y,predict2))

clf = RandomForestClassifier(n_estimators=100, max_depth=100,random_state=0)
clf.fit(train_x,train_y)

predict2 = clf.predict(test_x)
print(accuracy_score(test_y,predict2))