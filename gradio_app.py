import joblib
import gradio as gr

# Load the model
model = joblib.load("house_model.pkl")

# Prediction function
def predict_house(data1, data2, data3, data4, data5, data6, data7, data8):
    input_data = [data1, data2, data3, data4, data5, data6, data7, data8]
    prediction = model.predict([input_data])
    return prediction[0]

# Create Gradio interface
inputs = [
    gr.Number(label="MedInc (data1)", value=8.3252),
    gr.Number(label="HouseAge (data2)", value=41.0),
    gr.Number(label="AveRooms (data3)", value=6.98),
    gr.Number(label="AveBedrms (data4)", value=1.02),
    gr.Number(label="Population (data5)", value=322),
    gr.Number(label="AveOccup (data6)", value=2.55),
    gr.Number(label="Latitude (data7)", value=37.88),
    gr.Number(label="Longitude (data8)", value=-122.23)
]

outputs = gr.Textbox(label="Predicted House Price")

gr.Interface(fn=predict_house, inputs=inputs, outputs=outputs, title="House Price Predictor", description="Predict house prices using your model").launch()

