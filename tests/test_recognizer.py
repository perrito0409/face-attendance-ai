import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import faiss
import numpy as np
from face_recognizer import FaceRecognizer

def main():
    # Load index và lấy 1 vector thật từ index để query
    idx = faiss.read_index("index/faiss.index")
    real_embedding = idx.reconstruct(0)  # embedding của ảnh số 0 trong index

    rec = FaceRecognizer("index/faiss.index", "index/labels.json")
    out = rec.search(real_embedding, topk=5, threshold=0.6)

    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
