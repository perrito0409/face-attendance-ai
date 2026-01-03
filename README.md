Face Attendance AI – Face Recognition Engine (ONNX + FAISS)
📌 Overview

This project implements the AI face recognition engine for a student attendance system.

The pipeline uses:

InsightFace (ArcFace) for face detection & embedding

ONNX Runtime for efficient inference

FAISS for fast vector similarity search

The system takes an input image and returns the most likely student identity with confidence scores.

🧠 AI Pipeline
Input Image
   ↓
Face Detection (InsightFace, ONNX)
   ↓
Face Embedding (512-D ArcFace, ONNX)
   ↓
FAISS Vector Search (cosine similarity)
   ↓
Decision: best_id / best_score / accept

📂 Project Structure
face-attendance-ai/
├── assets/                 # Demo images
│   └── query.jpg
├── index/                  # Vector database
│   ├── faiss.index
│   └── labels.json
├── notebooks/              # Experiments (Colab / Jupyter)
│   └── 01_faiss_face_identification.ipynb
├── tests/                  # Unit & integration tests
│   ├── test_load_faiss.py
│   └── test_recognizer.py
├── face_pipeline.py        # End-to-end inference (image → JSON)
├── face_recognizer.py      # FAISS search logic
├── requirements.txt
├── README.md
└── venv/

🤖 ONNX Models (InsightFace)

This project uses InsightFace buffalo_l model pack.

ONNX models are automatically downloaded and cached at:

~/.insightface/models/buffalo_l/


Example models:

det_10g.onnx – face detection

w600k_r50.onnx – ArcFace recognition (512-D embedding)

genderage.onnx

landmark_2d_106.onnx

Verification command:

find ~/.insightface -name "*.onnx"

⚙️ Installation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


Dependencies:

insightface

onnxruntime

faiss-cpu

opencv-python

numpy

🚀 Quick Start (End-to-End Demo)

Run face recognition on a demo image:

python face_pipeline.py assets/query.jpg

Example Output
{
  "accept": true,
  "best_id": "SV01",
  "best_score": 0.8492,
  "topk": [
    { "student_id": "SV01", "score": 0.8492 },
    { "student_id": "SV01", "score": 0.7664 },
    { "student_id": "SV03", "score": 0.2539 }
  ]
}

🧪 Testing

Test FAISS index loading:

python tests/test_load_faiss.py


Test recognizer logic:

PYTHONPATH=. python tests/test_recognizer.py

🔌 Backend Integration (API Contract)

Input

Image file (or base64 image from frontend)

Output

{
  "best_id": "SV01",
  "best_score": 0.85,
  "accept": true,
  "topk": [...]
}


This module is designed to be wrapped by a REST API (Flask / FastAPI) by the backend team.

✅ Current Status

 ONNX inference via InsightFace

 512-D face embeddings (ArcFace)

 FAISS vector search

 End-to-end demo working

 Ready for backend integration

📌 Notes

ONNX model files are not committed to GitHub (cached locally).

assets/ contains only small demo images.

Vector DB (faiss.index, labels.json) can be regenerated if needed.

👤 Author

AI Module – Face Attendance System
Role: AI / Face Recognition