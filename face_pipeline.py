import json
import cv2
import numpy as np
from insightface.app import FaceAnalysis

from face_recognizer import FaceRecognizer

# Init InsightFace (detector + embedding)
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

def get_embedding_from_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    faces = app.get(img)
    if len(faces) == 0:
        raise ValueError(f"No face detected in: {image_path}")

    # lấy face lớn nhất (an toàn hơn)
    faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    emb = faces[0].embedding.astype("float32")

    # L2 normalize (để dùng cosine ~ inner product)
    emb = emb / np.linalg.norm(emb)
    return emb

def recognize(image_path: str, topk=5, threshold=0.6):
    emb = get_embedding_from_image(image_path)
    rec = FaceRecognizer("index/faiss.index", "index/labels.json")
    return rec.search(emb, topk=topk, threshold=threshold)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python face_pipeline.py <image_path>")
        sys.exit(1)

    out = recognize(sys.argv[1], topk=5, threshold=0.6)
    print(json.dumps(out, indent=2, ensure_ascii=False))
