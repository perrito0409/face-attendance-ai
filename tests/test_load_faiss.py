import json
import faiss

idx = faiss.read_index("index/faiss.index")
labels = json.load(open("index/labels.json", "r"))

print("FAISS ntotal:", idx.ntotal)
print("labels len:", len(labels))
print("sample labels:", labels[:5])
