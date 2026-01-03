# Face Attendance AI – Prototype

## Overview
This repository contains a prototype AI pipeline for student attendance
using face recognition. The goal is to identify a student from an input image
and decide whether to accept attendance based on similarity.

## AI Pipeline
1. Face detection / localization
2. Face embedding extraction (512-D, ArcFace / InsightFace)
3. Vector similarity search using FAISS (cosine similarity)
4. Threshold-based decision (ACCEPT / REJECT)

## Demo Result (Leave-One-Out)
Query image: `SV01_1.jpg` (not included in index)

Result:
```json
{
  "ok": true,
  "query": "cropped/SV01_1.jpg",
  "best_id": "SV01",
  "best_score": 0.8853957056999207,
  "accept": true,
  "topk": [
    {"student_id": "SV01", "score": 0.8853957056999207},
    {"student_id": "SV01", "score": 0.8099558353424072},
    {"student_id": "SV03", "score": 0.24935826659202576}
  ]
}
