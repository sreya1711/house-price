from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import gradio as gr
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI(title="Styled House Price Predictor")

# Load Model
model = joblib.load("house_model.pkl")

# -------- FASTAPI ENDPOINTS --------
class Input(BaseModel):
    data: list

@app.post("/predict")
def predict_post(input: Input):
    prediction = model.predict([input.data])
    return {"prediction": float(prediction[0])}

@app.get("/health")
def health():
    return {"status": "ok"}

# -------- GRADIO HELPER (Chart Generator) --------
def create_plot(values):
    fig, ax = plt.subplots()
    ax.bar(
        ["Median Income", "House Age", "Rooms", "Bedrooms",
         "Population", "Households", "Latitude", "Longitude"],
        values
    )
    ax.set_title("Feature Visualization")
    ax.set_ylabel("Value")
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

# -------- GRADIO UI FUNCTION --------
def predict_ui(median_income, house_age, rooms, bedrooms, population, households, latitude, longitude):

    data = [
        median_income,
        house_age,
        rooms,
        bedrooms,
        population,
        households,
        latitude,
        longitude,
    ]

    prediction = model.predict([data])[0]

    # Create chart
    chart_img = create_plot(data)

    return float(prediction), chart_img


# -------- GRADIO INTERFACE (STYLED) --------
ui = gr.Interface(
    fn=predict_ui,
    inputs=[
        gr.Number(label="Median Income", value=8.3252),
        gr.Number(label="House Age", value=41.0),
        gr.Number(label="Rooms", value=6.98),
        gr.Number(label="Bedrooms", value=1.02),
        gr.Number(label="Population", value=322),
        gr.Number(label="Households", value=2.55),
        gr.Number(label="Latitude", value=37.88),
        gr.Number(label="Longitude", value=-122.23),
    ],
    outputs=[
        gr.Number(label="Predicted Price"),
        gr.Image(label="Feature Chart"),
    ],
    title="🏠 House Price Predictor (Enhanced UI)",
    description="Enter house features to get a prediction and a visual chart of input values.",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="purple",
        neutral_hue="gray"
    ),
)

# Mount at ROOT → https://yoursite.onrender.com
app = gr.mount_gradio_app(app, ui, path="/")

# Run Uvicorn on Render
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
