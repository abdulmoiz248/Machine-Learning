import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

datasets=load_breast_cancer()
x=datasets.data
y=datasets.target
print("Shape of X:", x.shape)
print("Shape of y:", y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


print("Shape of y train:", y_train.shape)
print("Shape of y test:", y_test.shape)


model=LogisticRegression(max_iter=10000)

model.fit(x_train,y_train)

predict=model.predict(x_test)

acc=accuracy_score(predict,y_test)
cm=confusion_matrix(predict,y_test)

print("Accuracy=",acc)
print("Confusion=",cm)
































# # Step 1: Input data
# studyHours = np.array([1, 2, 3, 4, 5])
# results = np.array([0, 0, 0, 1, 1])  # 0 = Fail, 1 = Pass

# # Step 2: Sigmoid function
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))

# # Step 3: Loss function (Binary Cross-Entropy)
# def computeLoss(y_true, y_pred):
#     epsilon = 1e-10  # avoid log(0)
#     return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

# # Step 4: Training the model using Gradient Descent
# def trainLogisticRegression(X, y, lr=0.1, epochs=1000):
#     X = X.reshape(-1, 1)  # Ensure X is in a 2D column vector format (5, 1)
#     n_samples = X.shape[0]
#     weights = np.zeros(1)  # Initialize weights as a scalar for one feature
#     bias = 0

#     for _ in range(epochs):
#         # Linear equation: w*X + b
#         linear_output = weights * X.flatten() + bias
#         predictions = sigmoid(linear_output)

#         # Compute gradients
#         dw = (1/n_samples) * np.dot(X.flatten(), (predictions - y))  # Derivative wrt weights
#         db = (1/n_samples) * np.sum(predictions - y)  # Derivative wrt bias

#         # Update weights and bias
#         weights -= lr * dw
#         bias -= lr * db

#     return weights, bias

# # Step 5: Prediction function
# def predict(X, weights, bias):
#     linear_output = weights * X + bias
#     probabilities = sigmoid(linear_output)
#     return [1 if prob >= 0.5 else 0 for prob in probabilities]

# # Run Training
# weights, bias = trainLogisticRegression(studyHours, results)

# # Predict for a new student
# testHours = 30
# pred = predict(np.array([testHours]), weights, bias)
# print(f"Predicted result for {testHours} hours of study: {'Pass' if pred[0] == 1 else 'Fail'}")

