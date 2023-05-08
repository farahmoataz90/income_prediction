import pandas as pd
import numpy as np
import seaborn as sns

df=pd.read_csv("train_data (1).csv")
df.dtypes
#Replacing question marks which are considered null values by real null values to facilitate replacing these values afterwards
df=df.replace(' ?', None)
df.isna().sum()
df['workclass']=df['workclass'].fillna(df['workclass'].mode())
df['native-country']=df['native-country'].fillna(df['native-country'].mode())
df['occupation']=df['occupation'].fillna(df['occupation'].mode())
#Dropping unneeded columns that won't critically affect model prediction
df = df.drop("sex",axis=1)
df = df.drop("age",axis=1)
df = df.drop("marital-status",axis=1)
df = df.drop("fnlwgt",axis=1)
df = df.drop("race",axis=1)
df = df.drop('native-country',axis = 1)
#df['Income ']=df['Income '].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1)

#Encoding categorical data using hot encoding (getting dummies)
one_hot_encoded_data = pd.get_dummies(df, columns = ['workclass','education','occupation','relationship'])
one_hot_encoded_data['workclass_ Private'].head(20)
X = one_hot_encoded_data.drop('Income ', axis=1)
y = one_hot_encoded_data['Income ']

#Selecting Features using chi square that depends on p-value
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import chi2, f_classif
FeatureSelection = GenericUnivariateSelect(score_func = chi2, mode='k_best', param=5)
X = FeatureSelection.fit_transform(X, y)

#Printing columns names
for col in one_hot_encoded_data.columns:
    print(col)

#Array of features selected of 'true' and not selected of 'false':
FeatureSelection.get_support()
# Features selected are: education-num, capital-gain, capital-loss, hours-per-week

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Splitting data into data for model training and data for the model to test on
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Use the model to make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluation Phase

# Calculate the accuracy score of the model
accuracy = accuracy_score(y_test, y_pred)
# We use pos_label to tell which class has the higher label value
# Precision & Recall
precision = precision_score(y_test, y_pred, pos_label=' >50K')
recall = recall_score(y_test, y_pred, pos_label=' >50K')
# F1
f1 = f1_score(y_test, y_pred, pos_label=' >50K')
# Confusion Matrix
confusion = confusion_matrix(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion matrix:\n", confusion)

from sklearn.preprocessing import StandardScaler
# Preprocess the data using StandardScaler (Solution for SVC not ending model training)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import LinearSVC
# Create SVC model
model = SVC(kernel='linear') 
model.fit(X_train, y_train) 
# Use the model to make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluation Phase

# Calculate the accuracy score of the model
accuracy = accuracy_score(y_test, y_pred)
# We use pos_label to tell which class has the higher label value
# Precision & Recall
precision = precision_score(y_test, y_pred, pos_label=' >50K')
recall = recall_score(y_test, y_pred, pos_label=' >50K')
# F1
f1 = f1_score(y_test, y_pred, pos_label=' >50K')
# Confusion Matrix
confusion = confusion_matrix(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion matrix:\n", confusion)

from sklearn.tree import DecisionTreeClassifier

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create decision tree model
model = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluation Phase

# Calculate the accuracy score of the model
accuracy = accuracy_score(y_test, y_pred)
# We use pos_label to tell which class has the higher label value
# Precision & Recall
precision = precision_score(y_test, y_pred, pos_label=' >50K')
recall = recall_score(y_test, y_pred, pos_label=' >50K')
# F1
f1 = f1_score(y_test, y_pred, pos_label=' >50K')
# Confusion Matrix
confusion = confusion_matrix(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion matrix:\n", confusion)