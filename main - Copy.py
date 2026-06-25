from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import traceback

from preprocessing import preprocess_from_bytes
from gradcam import make_gradcam_heatmap, overlay_heatmap

from supabase import create_client, Client

import json
import numpy as np
import cv2
import os
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY")

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

# Folder GradCAM
GRADCAM_FOLDER = "gradcam_results"
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# Load Model
model = load_model("plantcare_tomato_model.keras")

with open("class_names.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

print(class_names["class_names"])


@app.get("/")
def home():
    return {
        "message": "PlantCare Backend Running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes
        contents = await file.read()

        # Preprocessing
        x = preprocess_from_bytes(contents)

        # Prediction
        pred = model.predict(x)

        idx = int(np.argmax(pred[0]))

        english_label = class_names["class_names"][idx]

        prediction = class_names["label_id"][english_label]

        confidence = round(float(pred[0][idx]), 2)

        # Original RGB image
        arr = np.frombuffer(contents, np.uint8)

        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image."
            )

        # Save GradCAM locally
        filename = f"{uuid.uuid4()}.png"

        gradcam_path = os.path.join(
            GRADCAM_FOLDER,
            filename
        )

        cv2.imwrite(
            gradcam_path,
            img_bgr
        )

        # Upload to Supabase Storage
        with open(gradcam_path, "rb") as f:
            supabase.storage.from_("gradcam-images").upload(
                filename,
                f,
                {
                    "content-type": "image/png"
                }
            )

        gradcam_url = (
            supabase.storage
            .from_("gradcam-images")
            .get_public_url(filename)
        )

        # Save prediction
        supabase.table("predictions").insert({
            "prediction": prediction,
            "confidence": confidence,
            "gradcam_url": gradcam_url
        }).execute()

        # Response
        return {
            "prediction": prediction,
            "confidence": confidence,
            "gradcam_url": gradcam_url
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
    