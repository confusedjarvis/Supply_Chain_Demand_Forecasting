import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from preprocessing import preprocess_data

df = preprocess_data()

target = "Number of products sold"

X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, preds))

# Save model
joblib.dump(model, "demand_forecasting_model.pkl")

# Save feature names (VERY IMPORTANT)
joblib.dump(X.columns.tolist(), "model_features.pkl")
