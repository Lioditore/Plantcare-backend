# 🌽 PlantCare Backend

Backend API for PlantCare, a web application that classifies corn leaf diseases using a TensorFlow deep learning model.

The backend is built with **FastAPI** and deployed on **Render**, while the frontend is hosted separately.

---

## Features

- Upload corn leaf images through API
- TensorFlow-based disease classification
- MobileNetV2 image preprocessing pipeline
- Returns prediction results with confidence scores
- CORS enabled for frontend integration
- Ready for deployment on Render

---

## Supported Classes

The model currently supports 4 classes:

| English Label | Indonesian Label |
|---|---|
| Common_Rust | Karat Daun |
| Gray_Leaf_Spot | Bercak Daun Abu-abu |
| Healthy | Sehat |
| Northern_Leaf_Blight | Hawar Daun Utara |

---

## Project Structure

```text
Plantcare-backend/
│
├── main.py
├── preprocessing.py
├── best_model.keras
├── class_names.json
├── requirements.txt
├── uploads/
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Plantcare-backend.git
cd Plantcare-backend
```

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Locally

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The backend will run at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### GET /

Health check endpoint.

#### Response

```json
{
    "message": "PlantCare Backend Running"
}
```

---

### POST /predict

Predict the disease class from an uploaded image.

#### Request

Content-Type:

```text
multipart/form-data
```

Field:

| Name | Type |
|---|---|
| file | Image File |

---

#### Example using cURL

```bash
curl -X POST \
  "http://127.0.0.1:8000/predict" \
  -F "file=@corn_leaf.jpg"
```

---

#### Successful Response

```json
{
    "filename": "corn_leaf.jpg",
    "prediction": "Karat Daun",
    "confidence": 0.98
}
```

---

## Preprocessing Pipeline

The uploaded image goes through the following steps:

1. Read image using OpenCV
2. Convert BGR → RGB
3. Resize to 224 × 224 pixels
4. Apply MobileNetV2 preprocessing
5. Add batch dimension
6. Feed into TensorFlow model

Pipeline:

```text
Image Upload
    ↓
OpenCV Read
    ↓
BGR → RGB
    ↓
Resize (224×224)
    ↓
preprocess_input()
    ↓
Expand Dimension
    ↓
TensorFlow Model
    ↓
Prediction
```

---

## Model Information

- Framework: TensorFlow / Keras
- Architecture: MobileNetV2
- Input Size: 224 × 224
- Number of Classes: 4

Classes:

```json
[
    "Common_Rust",
    "Gray_Leaf_Spot",
    "Healthy",
    "Northern_Leaf_Blight"
]
```

---

## Deployment

### Backend

Hosted on Render.

Example URL:

```text
https://plantcare-backend-b8ad.onrender.com
```

Swagger Docs:

```text
https://plantcare-backend-b8ad.onrender.com/docs
```

### Frontend

Frontend is deployed separately on Vercel.

---

## Notes

Uploaded images are currently stored temporarily in the `uploads/` directory before preprocessing.

Since Render uses ephemeral storage, uploaded files may not persist after service restarts.

For production environments, consider:

- Processing images directly from memory
- Automatically deleting uploaded files after prediction
- Using cloud storage solutions if permanent storage is required

---

## License

This project was developed for educational and research purposes.
