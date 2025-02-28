# Independences :
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.layers import TFSMLayer
import numpy as np
import uvicorn
from PIL import Image
import io

# Load Our NN Model :
MODEL_PATH = r"D:\Fasila - Graduation Project\CNN Model"
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]

# Classification Classes :
CLASS_LABELS = ["Early Blight", "Healthy", "Late Blight"]

# Initialize App :
app = FastAPI()

# Processing Then Predicting Step :
def preprocess_image(image: bytes):
    """Preprocess the uploaded image for the model."""
    image = Image.open(io.BytesIO(image)).convert("RGB")
    image = image.resize((256, 256))  
    image = np.array(image) / 255.0 
    image = np.expand_dims(image, axis=0)
    return tf.constant(image, dtype=tf.float32)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """API endpoint to predict plant disease from an image."""
    image = await file.read()
    image_input = preprocess_image(image)
    output = infer(keras_tensor=image_input)
    
    # Get predicted class
    predictions = output['output_0'].numpy()
    predicted_class = np.argmax(predictions)
    confidence = float(np.max(predictions))
    
    return {
        "filename": file.filename,
        "predicted_class": CLASS_LABELS[predicted_class],
        "confidence": confidence
    }