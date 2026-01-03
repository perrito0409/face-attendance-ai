import json
import faiss
import numpy as np

class FaceRecognizer:
    def __init__(self, index_path="index/faiss.index", labels_path="index/labels.json"):
        self.index = faiss.read_index(index_path)
        self.labels = json.load(open(labels_path, "r"))

    def search(self, embedding, topk=5, threshold=0.6):
        """
        embedding: numpy array (512,)
        return: dict
        """
        emb = embedding.astype("float32")
        emb = emb / np.linalg.norm(emb)
        emb = emb.reshape(1, -1)

        scores, ids = self.index.search(emb, topk)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            results.append({
                "student_id": self.labels[idx],
                "score": float(score)
            })

        best = results[0]
        return {
            "accept": best["score"] >= threshold,
            "best_id": best["student_id"],
            "best_score": best["score"],
            "topk": results
        }
