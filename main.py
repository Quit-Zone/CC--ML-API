from fastapi import FastAPI, HTTPException
import tensorflow as tf
import tensorflow_decision_forests
import numpy as np
import os
import logging

app = FastAPI()

# Define label names
label_names = ["Physically active", "Creatively engaged",
               "Relaxed and leisurely", "Socially involved"]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model during startup


@app.on_event("startup")
async def load_model():
    global model
    model_path = "model"
    try:
        logger.info(f"Loading model from: {model_path}")
        model = tf.saved_model.load(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

# Preprocess input data


def preprocess_input(prediction_data):
    processed_input = {
        'age': tf.convert_to_tensor([prediction_data['age']], dtype=tf.int64),
        'alcohol': tf.convert_to_tensor([prediction_data['alcoholConsumption']], dtype=tf.int64),
        'gender': tf.convert_to_tensor([prediction_data['gender']], dtype=tf.int64),
        'height': tf.convert_to_tensor([prediction_data['height']], dtype=tf.float32),
        'hobby_1': tf.convert_to_tensor([prediction_data['hobby_1']], dtype=tf.int64),
        'hobby_2': tf.convert_to_tensor([prediction_data['hobby_2']], dtype=tf.int64),
        'hobby_3': tf.convert_to_tensor([prediction_data['hobby_3']], dtype=tf.int64),
        'physical_activity': tf.convert_to_tensor([prediction_data['physicalActivity']], dtype=tf.int64),
        'smoke': tf.convert_to_tensor([prediction_data['smokingHabit']], dtype=tf.int64),
        'weight': tf.convert_to_tensor([prediction_data['weight']], dtype=tf.float32),
    }
    return processed_input

@app.post("/predict")
async def predict_classification(prediction_data: dict):
    try:
        processed_input = preprocess_input(prediction_data)
        # Use the model directly for prediction
        prediction = model(processed_input)
        scores = prediction.numpy()

        highest_score_index = np.argmax(scores)
        prediction_result = label_names[highest_score_index - 1]

        return {"prediction_result": prediction_result}

    except Exception as error:
        logger.error(f'Error retrieving prediction: {error}')
        raise HTTPException(status_code=500, detail="Prediction error")


@app.get("/")
def read_root():
    return {"message": "Predict Classification API started"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
