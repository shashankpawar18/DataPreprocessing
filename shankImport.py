# Data Preprocessing Template

# Importing the libraries
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# Seperating the feature matrix and the dependent variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

# Handling missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean',axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# Categorical Data Encoding
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])

# One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
oneHotEncoder_X = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder_X.fit_transform(X).toarray()
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# Train-Test Split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 42, stratify = y)

# Feature Scaling just the non-dummy variables
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train[:,3:] = sc_X.fit_transform(X_train[:,3:])
#X_test[:,3:] = sc_X.transform(X_test[:,3:])

# Feature Scaling all variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)