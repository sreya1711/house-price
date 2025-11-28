from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib

app = FastAPI(title="House Price Prediction API")

# Load the model
model = joblib.load("house_model.pkl")

# Input data model for POST
class Input(BaseModel):
    data: list = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]

# POST endpoint for custom JSON
@app.post("/predict")
def predict_post(input: Input):
    prediction = model.predict([input.data])
    return {"prediction": prediction[0]}

# GET endpoint for browser testing
@app.get("/predict")
def predict_get():
    default_data = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]
    prediction = model.predict([default_data])
    return {"prediction": prediction[0]}

# Optional root endpoint
@app.get("/")
def root():
    return {"message": "House Price Prediction API is running. Use /predict for predictions."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
