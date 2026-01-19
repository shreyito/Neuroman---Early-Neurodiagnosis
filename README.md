# Neuroman ML - Early Neuro Diagnosis 🧠

A machine learning-powered REST API for early detection of Parkinson's disease using deep neural network analysis with confidence-based predictions.

---

## 📌 Overview

Neuroman ML implements a **pre-trained deep learning model** that analyzes patient biomarker data to assess Parkinson's disease risk. The system processes structured medical data through a validated neural network and returns probability-based assessments with interpretable results.

The system is **production-ready** with CORS support, comprehensive error handling, and a simple REST interface designed for healthcare application integration.

---

## 🎯 Problem Statement

Early neurological diagnosis faces several challenges:
- Manual analysis of complex biomarker patterns
- Delayed detection reducing treatment effectiveness
- Need for accessible screening tools
- Requirement for interpretable AI predictions

**Goal:**  
Design a system that:
- Provides rapid initial screening for Parkinson's indicators
- Delivers confidence-scored predictions
- Maintains simplicity for clinical integration
- Explains risk levels clearly

---

## 🏗️ System Architecture

```
Patient Data Upload (CSV/JSON)
        ↓
File Validation & Parsing
        ↓
Data Preprocessing
        ↓
Neural Network Inference
        ↓
Threshold-Based Classification
        ↓
Confidence Score + Interpretation
        ↓
JSON Response
```

Key principles:
- Single-endpoint simplicity
- Format-agnostic input handling
- Probability-based predictions
- Clear clinical interpretation

---

## 🧠 Model Details

### Model Specifications
- **Architecture:** Deep Neural Network (Keras/TensorFlow)
- **Model File:** `best_model_fold_1.h5`
- **Training:** Cross-validated (Fold 1)
- **Output:** Single probability score (0.0 - 1.0)

### Prediction Thresholds
| Probability Range | Classification | Clinical Action |
|------------------|----------------|-----------------|
| 0.00 - 0.79 | Low Risk | Routine monitoring |
| 0.80 - 1.00 | High Risk | Clinical evaluation recommended |

This threshold ensures conservative screening appropriate for medical contexts.

---

## 📊 API Reference

### Base URL
```
http://localhost:5000
```

### Endpoints

#### `GET /`
Health check endpoint.

**Response:**
```
NeuroMan Model API is Running
```

---

#### `POST /predict`
Upload patient biomarker data for Parkinson's risk assessment.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body Parameter: `file` (CSV or JSON)

**Supported Formats:**
- CSV with feature columns
- JSON with feature key-value pairs

**Example Request (cURL):**
```bash
curl -X POST \
  -F "file=@patient_biomarkers.csv" \
  http://localhost:5000/predict
```

**Success Response (High Risk):**
```json
{
  "prediction": 0.92,
  "message": "Possibility of Parkinson Detected"
}
```

**Success Response (Low Risk):**
```json
{
  "prediction": 0.45,
  "message": "Low Possibility of Parkinson"
}
```

**Error Responses:**
```json
{
  "error": "No file uploaded"
}
```
```json
{
  "error": "Unsupported file format. Use CSV or JSON."
}
```
```json
{
  "error": "Invalid data structure or processing error"
}
```

---

## 📁 Input Data Format

The model expects patient biomarker data with features matching the training dataset.

### CSV Format
```csv
feature1,feature2,feature3,feature4,feature5
0.234,0.567,0.891,0.432,0.765
```

### JSON Format
```json
{
  "feature1": 0.234,
  "feature2": 0.567,
  "feature3": 0.891,
  "feature4": 0.432,
  "feature5": 0.765
}
```

**Note:** Ensure all required features are present and properly normalized to match training data preprocessing.

---

## 🛠️ Tech Stack

- **Framework:** Flask 2.x
- **ML Library:** TensorFlow/Keras
- **Data Processing:** Pandas, NumPy
- **CORS:** Flask-CORS
- **Language:** Python 3.7+

---

## 🚀 Installation & Setup

### Prerequisites
```bash
python --version  # 3.7 or higher required
pip --version
```

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd neuroman-ml
```

### Step 2: Install Dependencies
```bash
pip install flask flask-cors tensorflow numpy pandas
```

Or using requirements.txt:
```bash
pip install -r requirements.txt
```

### Step 3: Verify Model File
Ensure `best_model_fold_1.h5` is in the project root directory.

### Step 4: Run the Server
```bash
python app.py
```

Server starts at: `http://localhost:5000`

---

## 🧪 Testing the API

### Using cURL
```bash
# Health check
curl http://localhost:5000/

# Prediction
curl -X POST -F "file=@sample_data.csv" http://localhost:5000/predict
```

### Using Python Requests
```python
import requests

url = "http://localhost:5000/predict"
files = {"file": open("patient_data.csv", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Using Postman
1. Set method to `POST`
2. URL: `http://localhost:5000/predict`
3. Body → form-data
4. Key: `file` (type: File)
5. Select your CSV/JSON file
6. Send request

---

## ⚙️ Configuration

### Production Deployment
Modify `app.py`:
```python
if __name__ == '__main__':
    app.run(
        debug=False,        # Disable debug mode
        host='0.0.0.0',     # Allow external access
        port=5000           # Configure port
    )
```

### Environment Variables (Recommended)
```python
import os

app.run(
    debug=os.getenv('FLASK_DEBUG', 'False') == 'True',
    host=os.getenv('FLASK_HOST', '0.0.0.0'),
    port=int(os.getenv('FLASK_PORT', 5000))
)
```

### CORS Configuration
For production, restrict origins:
```python
CORS(app, origins=["https://your-frontend-domain.com"])
```

---




## 🔮 Future Enhancements

- Multi-class disease classification (Parkinson's subtypes)
- Batch prediction endpoint for multiple patients
- Model versioning and A/B testing
- Confidence interval estimates
- Feature importance visualization
- Integration with EHR systems
- Real-time model retraining pipeline
- Explainable AI (SHAP/LIME) integration
- Mobile app SDK




---

## ⚠️ Medical Disclaimer

**IMPORTANT:** This tool is intended for research and educational purposes only. It is NOT a substitute for professional medical diagnosis or treatment.

- Always consult qualified healthcare professionals for medical decisions
- Results should be interpreted by trained clinicians
- Not approved for clinical diagnostic use without proper validation
- System accuracy may vary with different patient populations

---

## 🙏 Acknowledgments

- TensorFlow/Keras team for ML framework
- Flask community for web framework
- [Dataset source/Research paper citations]
- Contributors and testers

---

**Version:** 1.0.0  
**Last Updated:** January 2026  
**Status:** Active Development
