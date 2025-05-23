import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def manualApproach(study_Hours):
    hours = np.array([1, 2, 3, 4, 5])
    scores = np.array([10, 15, 12, 14, 14])
    mean_x = np.mean(hours)
    mean_y = np.mean(scores)
    slope = np.sum((hours - mean_x) * (scores - mean_y)) / np.sum((hours - mean_x) ** 2)
    b = mean_y - slope * mean_x
    return slope * study_Hours + b

def Scikit_Learn():
    dataset = pd.read_csv('datasets/linearRegression.csv')

    columnsToUse = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors',
                    'condition','sqft_above','sqft_basement','yr_built',
                    'city','statezip','country']
    
    columnsToCheck = columnsToUse + ['price']
    dataset = dataset.dropna(subset=columnsToCheck)

    features = dataset[columnsToUse]
    target = dataset['price']

    features = pd.get_dummies(features, drop_first=True)

    x_train, x_test, y_train, y_test = train_test_split(features, target, train_size=0.8, random_state=42)

    model = LinearRegression()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    mse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")
    plt.scatter(y_test, prediction, alpha=0.5)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted House Prices')
    plt.show()


Scikit_Learn()
