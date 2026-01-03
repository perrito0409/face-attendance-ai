# Face Attendance AI (FAISS + ONNX)

## 📌 Overview
This project implements a **face recognition pipeline for attendance systems** using:

- **InsightFace (ONNX models)** for face detection & embedding extraction  
- **FAISS** for fast similarity search over face embeddings  
- Designed to be **backend-friendly** (no training, inference only)

The system takes a face image as input and returns the most similar registered student IDs with cosine similarity scores.

---

## 🧠 Architecture

```

Image
↓
InsightFace (ONNX)
↓
Face Embedding (512-d)
↓
FAISS Index (cosine similarity)
↓
Top-K matched student IDs

```

---

## 📂 Project Structure

```

face-attendance-ai/
├── assets/                 # Sample images for testing
├── index/
│   ├── faiss.index         # FAISS index (embedding database)
│   └── labels.json         # Mapping: vector → student_id
├── notebooks/
│   └── 01_faiss_face_identification.ipynb
├── tests/
│   ├── test_load_faiss.py
│   └── test_recognizer.py
├── face_pipeline.py        # End-to-end image → result pipeline
├── face_recognizer.py      # Core FAISS search logic (backend-ready)
├── requirements.txt
└── README.md

```

---

## 🤖 Models

This project uses **InsightFace `buffalo_l` ONNX models**:

- Face detection
- Face recognition (512-d embeddings)
- Gender & age (optional)

Models are automatically downloaded and cached at:

```

~/.insightface/models/buffalo_l/

```

Example ONNX files:
```

det_10g.onnx
w600k_r50.onnx
genderage.onnx

````

> No `.pth` files are used — inference is done fully with **ONNX + onnxruntime**, suitable for deployment.

---

## ⚙️ Installation

Create and activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
````

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1️⃣ Run Face Recognition Pipeline

```bash
python face_pipeline.py assets/query.jpg
```

### Example Output

```json
{
  "accept": true,
  "best_id": "SV01",
  "best_score": 0.84,
  "topk": [
    { "student_id": "SV01", "score": 0.84 },
    { "student_id": "SV01", "score": 0.76 },
    { "student_id": "SV03", "score": 0.25 }
  ]
}
```

---

## 🧪 Testing

### Test FAISS index loading

```bash
python tests/test_load_faiss.py
```

### Test recognizer logic

```bash
PYTHONPATH=. python tests/test_recognizer.py
```

---

## 🏗️ Design Notes

* FAISS uses **cosine similarity** (`IndexFlatIP` with normalized vectors)
* One person can have **multiple embeddings**
* Decision is based on:

  * best similarity score
  * configurable threshold
* Backend can call `FaceRecognizer.search()` directly

---

## 🔮 Future Work

* Integrate with FastAPI / Flask backend
* Store embeddings & metadata in database (Supabase / PostgreSQL)
* Add liveness detection (blink / motion)
* Support online index update

---

## 👤 Author

**Dat Tran**
Face Attendance AI – Prototype for backend integration

