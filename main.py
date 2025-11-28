from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import gradio as gr

app = FastAPI(title="House Price Prediction API")

model = joblib.load("house_model.pkl")

class Input(BaseModel):
    data: list

@app.post("/predict")
def predict_post(input: Input):
    prediction = model.predict([input.data])
    return {"prediction": float(prediction[0])}

@app.get("/predict")
def predict_get():
    default_data = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]
    prediction = model.predict([default_data])
    return {"prediction": float(prediction[0])}

@app.get("/health")
def health():
    return {"status": "ok"}

def predict_ui(median_income, house_age, rooms, bedrooms, population, households, latitude, longitude):
    data = [median_income, house_age, rooms, bedrooms, population, households, latitude, longitude]
    prediction = model.predict([data])[0]
    return float(prediction)

ui = gr.Interface(
    fn=predict_ui,
    inputs=[
        gr.Number(label="Median Income"),
        gr.Number(label="House Age"),
        gr.Number(label="Rooms"),
        gr.Number(label="Bedrooms"),
        gr.Number(label="Population"),
        gr.Number(label="Households"),
        gr.Number(label="Latitude"),
        gr.Number(label="Longitude"),
    ],
    outputs=gr.Number(label="Predicted Price"),
    title="House Price Prediction",
)

app = gr.mount_gradio_app(app, ui, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
