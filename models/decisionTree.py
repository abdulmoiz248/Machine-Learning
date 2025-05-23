import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df=load_iris()

x=df.data
y=df.target

XTrain, XTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

model=DecisionTreeClassifier()

model.fit(XTrain,yTrain)

y_pred=model.predict(XTest)

print("Accu=",accuracy_score(y_pred,yTest))

plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=df.feature_names, class_names=df.target_names)
plt.show()