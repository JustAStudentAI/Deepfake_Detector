# This is a streamlit app that loads the trained deepfake detector model and lets the user upload an image to get a prediction
import os
import io
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
import streamlit as st
import numpy as np
import datetime   


# --------------------------
# DATABASE SETUP WITH SQLALCHEMY
# --------------------------
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///deepfake_predictions.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
base = declarative_base()

class PredictionLog(base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    prediction = Column(Float)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.now)

base.metadata.create_all(bind=engine)


# --------------------------
# STREAMLIT APP SETUP
# --------------------------
st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("Deepfake Detector")


# --------------------------
# MODEL AND PREPROCESSING SETUP
# --------------------------
MODEL_NAME = "google/vit-base-patch16-224"

processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# MODEL LOADING FUNCTION (CACHED)
# --------------------------
@st.cache(allow_output_mutation=True)
def load_model():
    """
    Loads the pre-trained Vision Transformer model and its trained weights.
    The model is moved to the correct device and set to evaluation mode.
    """
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True
        )
    model_path = "deepfake_vit_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        st.error("Model weights not found. Please train the model first.")
    model.to(device)
    model.eval()
    return model

# Load the model (cached so that it only loads once per session)
model = load_model()


# --------------------------
# STREAMLIT FRONTEND: FILE UPLOADER
# --------------------------
uploaded_file = st.file_uploader("Upload an image (.jpg, .jpeg, or .png)", type=["jpg", "jpeg", "png"])


# --------------------------
# PREDICTION AND DATABASE LOGGING
# --------------------------
if uploaded_file is not None:
    try:
        # Read the uploaded file into memory and open it as an image
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    except Exception as e:
        st.error(f"Error: {e}")

    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Running Prediction...")

    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Run the model
    with torch.no_grad():
        outputs = model(pixel_values=input_tensor).logits
        pred_val = outputs.argmax(dim=1).item()
        confidence = torch.softmax(outputs, dim=1)[0][pred_val].item()
    
    # Display the prediction
    prediction = "Fake" if pred_val == 0 else "Real"
    st.write(f"**Prediction:** {prediction}")
    st.write(f"**Confidence:** {confidence*100:.1f}%")

    # Log the prediction to the database
    db = SessionLocal()
    log_entry = PredictionLog(
        filename=uploaded_file.name,
        prediction=prediction,
        confidence=confidence,
        timestamp=datetime.datetime.now()
    )
    db.add(log_entry)
    db.commit()
    db.close()