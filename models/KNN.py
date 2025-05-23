import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


dataset=pd.read_csv('datasets/diabetes.csv')

dataset.replace(0, np.nan, inplace=True)
dataset.fillna(dataset.mean(), inplace=True)

features=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
target='Outcome'

x=dataset[features]
y=dataset[target]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

k=5
model=KNeighborsClassifier(n_neighbors=k)


model.fit(x_train,y_train)

pred=model.predict(x_test)


accuracy = accuracy_score(y_test, pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
print("Classification Report:")
print(classification_report(y_test, pred))

