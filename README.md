# Diabetic Retinopathy Screening Dashboard

A web application for automated diabetic retinopathy (DR) screening using deep learning. Upload retinal images to receive a diagnosis, confidence score, and lesion heatmap visualization. Results can be exported as CSV and are sorted by severity.

## Features
- Upload one or more retinal images for DR diagnosis
- Deep learning model (MobileNetV3) with Grad-CAM heatmap visualization
- Severity-based triage (Normal, Mild, Moderate, Severe, Proliferative)
- Download results as CSV
- Modern, responsive UI

---

## Setup Instructions

### 1. Clone the Repository
```sh
git clone https://github.com/Dharmendra016/Diabetic-Retinopathy-Screening.git
cd Diabetic-Retinopathy-Screening
```

### 2. Create and Activate Virtual Environment (Recommended)
**Windows (PowerShell):**
```sh
py -3.10 -m venv dr_app_venv
dr_app_venv/Scripts/activate
or
python -m venv dr_app_venv
.\dr_app_venv\Scripts\Activate.ps1
```

**(If using CMD, use `dr_app_venv\Scripts\activate.bat` instead.)**

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Download Model Weights
- Ensure `model_weight/best_mobilenet_model.pth` is present. (Already included in this repo.)

### 5. Run the Application
```sh
python app.py
```

- The app will start at [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Usage
1. Open the app in your browser.
2. Click **Choose Files** and select one or more retinal images (JPG/PNG).
3. Click **Diagnose**.
4. View results, heatmaps, and download CSV if needed.

---

## File Structure
```
app.py                  # Main Flask backend
requirements.txt        # Python dependencies
model_weight/           # Pretrained model weights
static/                 # Frontend JS/CSS
  ├─ script.js
  └─ style.css
templates/
  └─ index.html         # Main HTML template
notebook/               # (Optional) Jupyter notebooks for experiments
```

---

## Notes
- Requires Python 3.8+
- GPU is optional but recommended for faster inference
- For any issues, please open an issue on the repository