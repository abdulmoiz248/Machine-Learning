from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load your data
# (assume you already used pandas to load a DataFrame)

# 2. Split into input (X) and output (y)
X = df[['feature1', 'feature2']]
y = df['target']

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate the model
y_pred = model.predict(X_test)
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
