# JobGuard AI - Fake Job Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Django-4.2+-green.svg" alt="Django">
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-orange.svg" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

A professional AI-powered web application that detects fake job postings using a hybrid Machine Learning and Rule-Based detection system. Built with Django and scikit-learn, the system analyzes job descriptions in real time and identifies suspicious recruitment patterns, scam indicators, and fraudulent hiring behavior.

---

## 🚀 Live Demo

<p align="center">
  <img src="jobgaurd.gif" alt="JobGuard AI Demo" width="760">
</p>

---

## ✨ Features

- Hybrid ML + Rule-Based Detection Engine
- Real-time fake job classification
- Risk assessment (Low / Medium / High)
- Confidence scoring system
- Premium dark-themed dashboard
- Interactive visual analysis
- Developer mode for technical insights
- Pre-loaded sample test data
- Responsive UI for desktop and mobile

---

## 🧠 Detection Engine

The system combines Machine Learning predictions with suspicious keyword analysis to improve reliability.

### Machine Learning Pipeline
- TF-IDF Vectorization
- Logistic Regression Classifier
- Probability-based prediction scoring

### Rule-Based Analysis
The system scans for suspicious recruitment phrases such as:
- “No interview required”
- “Work from home”
- “Guaranteed income”
- “Processing fee”
- “Immediate hiring”

Weighted scoring is used to detect scam patterns and increase classification accuracy.

---

## 📊 Risk Classification

| Risk Level | Meaning |
|------------|---------|
| 🟢 Low Risk | Likely legitimate posting |
| 🟡 Medium Risk | Suspicious or partially risky |
| 🔴 High Risk | Potentially fraudulent job posting |

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~95% |
| Precision | ~90% |
| Recall | ~85% |
| F1-Score | ~87% |

---

## 🛠 Tech Stack

### Backend
- Django 4.2+
- Python

### Machine Learning
- scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

### Frontend
- HTML5
- CSS3
- JavaScript

### Visualization
- Chart.js
- Interactive Dashboard UI

---

## 📂 Project Structure

```bash
fake-job-detection-django/
├── django_app/
├── model/
├── requirements.txt
├── design-md/
├── README.md
└── jobgaurd.gif
