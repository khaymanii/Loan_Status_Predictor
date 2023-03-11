

# Importing the libraries

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score



loan_dataset = pd.read_csv("Loan.csv")

loan_dataset.shape
loan_dataset.head()
loan_dataset.describe()
loan_dataset.isnull().sum()
loan_dataset = loan_dataset.dropna()
loan_dataset.isnull().sum()


# Label encoding

loan_dataset.replace({"Loan_Status" : {"N":0, "Y":1}}, inplace=True)

loan_dataset.head(10)

loan_dataset["Dependents"].value_counts()


# Replace the value of 3+ to 4

loan_dataset = loan_dataset.replace(to_replace = "3+", value = 4)

loan_dataset["Dependents"].value_counts()



# Data visualization

# Education and loan status

sns.countplot(x="Education", hue="Loan_Status", data = loan_dataset)



# marital status and loan status

sns.countplot(x="Married", hue="Loan_Status", data = loan_dataset)


# Converting all the categorical columns values to numerical values

loan_dataset.replace({"Married" : {"No":0, "Yes":1}, "Gender" : {"Male":1, "Female":0}, "Self_Employed" : {'No': 0, "Yes": 1}
                     , "Property_Area": {"Rural":0, "Semiurban":1, 'Urban': 2}, "Education": {"Graduate":1, "Not Graduate": 0}}, inplace=True)


loan_dataset.head(10)

X = loan_dataset.drop(columns= ["Loan_ID", "Loan_Status"], axis=1)
y = loan_dataset["Loan_Status"]
print(X)
print(y)



# Data splitting

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1, stratify = y, random_state = 2)


print(X.shape, X_train.shape, X_test.shape)


classifier = svm.SVC(kernel = 'linear')
classifier.fit(X_train, y_train)

# Model Evaluation " training data"

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)
print("Accuracy on training data : ", training_data_accuracy)



# Model Evaluation " test data"

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)


print("Accuracy on test data : ", test_data_accuracy)

