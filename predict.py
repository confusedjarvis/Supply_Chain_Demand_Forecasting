import joblib
import pandas as pd

model = joblib.load("demand_forecasting_model.pkl")

sample = pd.DataFrame([{
    "Price": 20,
    "Stock levels": 150,
    "Leadtime": 10,
    "Month": 6,
    "DayOfWeek": 2,
    "Quarter": 2
}])

print("Predicted Sales:", model.predict(sample)[0])
