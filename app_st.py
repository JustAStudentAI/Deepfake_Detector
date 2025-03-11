import os               # For file path operations
import io               # For handling byte streams
import torch            # PyTorch for model inference
from PIL import Image   # Pillow for image processing
from torchvision import transforms  # For image preprocessing
from transformers import ViTForImageClassification, ViTImageProcessor  # Pre-trained model and processor
import streamlit as st  # Streamlit for building the web app
import numpy as np      # For numerical operations
import datetime         # For timestamping predictions


# --------------------------
# DATABASE SETUP WITH SQLALCHEMY
# --------------------------
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

# Define the SQLite database URL (this will create a file "deepfake_predictions.db")
DATABASE_URL = "sqlite:///deepfake_predictions.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define the PredictionLog model to store prediction data.
# We changed the "prediction" column from Float to String so that it can store textual labels ("Fake" or "Real").
class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)           # Unique ID for each record
    filename = Column(String, index=True)                        # Name of the uploaded file
    prediction = Column(String)                                  # Predicted label as text ("Fake" or "Real")
    confidence = Column(Float)                                   # Confidence score (as a float)
    timestamp = Column(DateTime, default=datetime.datetime.now)  # Timestamp when the prediction was made

# Create the database table if it doesn't exist
Base.metadata.create_all(bind=engine)

# --------------------------
# STREAMLIT APP SETUP
# --------------------------
# Configure the Streamlit page and set the title
st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("Deepfake Detector")

# --------------------------
# MODEL AND PREPROCESSING SETUP
# --------------------------
# Define the model name; this must match the pre-trained model used during training.
MODEL_NAME = "google/vit-base-patch16-224"

# Load the image processor from Hugging Face to obtain normalization parameters.
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

# Define a transformation pipeline that:
#   1. Resizes images to 224x224 (the size expected by ViT)
#   2. Converts images to PyTorch tensors
#   3. Normalizes them using the processor's image_mean and image_std
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# Set the device to GPU if available; otherwise, use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# MODEL LOADING FUNCTION (CACHED)
# --------------------------
# Use st.cache_resource to cache the model so it's only loaded once per session.
@st.cache_resource
def load_model():
    """
    Loads the pre-trained Vision Transformer model and its trained weights.
    The model is moved to the proper device and set to evaluation mode.
    """
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,                    # Two classes: Fake and Real
        ignore_mismatched_sizes=True     # Reinitialize classifier head for 2 classes
    )
    model_path = "deepfake_vit_model.pth"  # Path to your saved model weights
    if os.path.exists(model_path):
        # Load the saved weights and map them to the correct device
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        st.error("Model weights not found. Please train the model first.")
    model.to(device)    # Move model to GPU or CPU
    model.eval()        # Set the model to evaluation mode
    return model

# Load the model using the caching function
model = load_model()

# --------------------------
# STREAMLIT FRONTEND: FILE UPLOADER
# --------------------------
# Create a file uploader widget that accepts images in .jpg, .jpeg, or .png formats.
uploaded_file = st.file_uploader("Upload an image (.jpg, .jpeg, or .png)", type=["jpg", "jpeg", "png"])

# --------------------------
# PREDICTION, DATABASE LOGGING, AND DISPLAY OF PREVIOUS PREDICTIONS
# --------------------------
if uploaded_file is not None:
    try:
        # Read the uploaded file into memory and open it as an RGB image
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    except Exception as e:
        st.error(f"Error reading the image: {e}")
    else:
        # Display the uploaded image using 'use_container_width' (new parameter replacing use_column_width)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("Running Prediction...")

        # Preprocess the image and add a batch dimension (unsqueeze)
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Perform model inference without calculating gradients
        with torch.no_grad():
            outputs = model(pixel_values=input_tensor).logits  # Get raw scores (logits)
            pred_val = outputs.argmax(dim=1).item()             # Choose the class with the highest score
            confidence = torch.softmax(outputs, dim=1)[0][pred_val].item()  # Compute the confidence

        # Map the predicted class index to a label using a dictionary
        class_map = {0: "Fake", 1: "Real"}
        prediction = class_map[pred_val]

        # Display the prediction results
        st.write(f"**Prediction:** {prediction}")
        st.write(f"**Confidence:** {confidence*100:.1f}%")

        # Log the prediction to the database
        db = SessionLocal()  # Create a new database session
        log_entry = PredictionLog(
            filename=uploaded_file.name,
            prediction=prediction,              # Now a string ("Fake" or "Real")
            confidence=confidence,
            timestamp=datetime.datetime.now()
        )
        db.add(log_entry)
        db.commit()  # Save the entry to the database
        db.close()
        st.write("Prediction logged to the database.")

# --------------------------
# DISPLAY PREVIOUS PREDICTIONS FROM THE DATABASE
# --------------------------
st.subheader("Previous Predictions")
db = SessionLocal()
# Query the 10 most recent prediction logs
predictions = db.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(10).all()
db.close()
if predictions:
    for pred in predictions:
        st.write(f"Filename: {pred.filename}, Prediction: {pred.prediction}, Confidence: {pred.confidence*100:.1f}%, Timestamp: {pred.timestamp}")
else:
    st.write("No previous predictions found.")
