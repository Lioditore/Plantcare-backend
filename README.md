# PlantCare Backend API

Backend API untuk klasifikasi penyakit daun tomat menggunakan model Deep Learning (TensorFlow/Keras) dengan fitur visualisasi Grad-CAM dan penyimpanan hasil prediksi ke Supabase.

---

## Overview

PlantCare Backend menerima gambar daun tomat dari client, melakukan preprocessing, menjalankan inferensi menggunakan model machine learning, menghasilkan visualisasi Grad-CAM, menyimpan hasil ke Supabase, lalu mengembalikan hasil prediksi kepada client.

### Technology Stack

* FastAPI
* TensorFlow / Keras
* OpenCV
* NumPy
* Supabase
* Python Dotenv

---

# Endpoints

## GET /

Health Check Endpoint

### Request

```http
GET /
```

### Response

```json
{
  "message": "PlantCare Backend Running"
}
```

---

## POST /predict

Melakukan prediksi penyakit daun tomat berdasarkan gambar yang diunggah.

### Request

#### Content-Type

```http
multipart/form-data
```

#### Form Data

| Field | Type       | Required |
| ----- | ---------- | -------- |
| file  | Image File | Yes      |

Contoh:

```http
POST /predict
```

Upload file gambar:

```text
leaf.jpg
```

---

# Prediction Process

Saat endpoint dipanggil, sistem akan:

1. Membaca file gambar.
2. Melakukan preprocessing.
3. Menjalankan model TensorFlow.
4. Mengambil kelas dengan probabilitas tertinggi.
5. Menghitung confidence score.
6. Membuat visualisasi Grad-CAM.
7. Menyimpan gambar Grad-CAM secara lokal.
8. Mengunggah gambar ke Supabase Storage.
9. Menyimpan hasil prediksi ke database Supabase.
10. Mengirimkan hasil ke client.

---

# Response

### Success Response

```json
{
  "prediction": "Late Blight",
  "confidence": 0.97,
  "gradcam_url": "https://xxxxx.supabase.co/storage/v1/object/public/gradcam-images/abc.png"
}
```

### Response Fields

| Field       | Description                           |
| ----------- | ------------------------------------- |
| prediction  | Label penyakit hasil klasifikasi      |
| confidence  | Tingkat keyakinan model               |
| gradcam_url | URL publik hasil visualisasi Grad-CAM |

---

# Database

## Table: predictions

Data hasil prediksi disimpan ke tabel:

```sql
predictions
```

### Columns

| Column      | Type  |
| ----------- | ----- |
| prediction  | text  |
| confidence  | float |
| gradcam_url | text  |

---

# Supabase Storage

Bucket yang digunakan:

```text
gradcam-images
```

File Grad-CAM yang dihasilkan akan diunggah ke bucket ini dan diberikan URL publik.

---

# Grad-CAM

Grad-CAM digunakan untuk menjelaskan area gambar yang paling berpengaruh terhadap keputusan model.

Output berupa heatmap yang ditumpuk di atas gambar asli sehingga pengguna dapat melihat bagian daun yang menjadi fokus model.
