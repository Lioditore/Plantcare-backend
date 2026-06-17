from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from tensorflow.keras.models import load_model
from preprocessing import load_and_preprocess
import json
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import numpy as np
import io
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # untuk development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("best_model.keras")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

print(class_names['class_names'])

@app.get("/")
def home():
    return {
        "message": "PlantCare Backend Running"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # baca file
    contents = await file.read()

    # simpan file (opsional)
    save_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(save_path, "wb") as f:
        f.write(contents)

    # preprocessing
    x = load_and_preprocess(save_path)

    # prediksi
    pred = model.predict(x)

    idx = int(np.argmax(pred[0]))
    result = class_names['label_id'][class_names['class_names'][idx]]
    confidence = round(float(pred[0][idx]), 2)

    return {
        "filename": file.filename,
        "prediction": result,
        "confidence": confidence
    }
