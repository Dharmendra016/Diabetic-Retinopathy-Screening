# Diabetic Retinopathy Detection Web App

This project is a Flask-based web application for automated detection and severity classification of Diabetic Retinopathy (DR) from retinal images. It uses deep learning models (MobileNetV3) for a two-stage classification pipeline and provides Grad-CAM visualizations for interpretability. Users can upload images, view predictions, download results as CSV/PDF, and visualize heatmaps.

> ⚠️ **Note:** The current model is only for testing (trained for ~10 epochs), so the results are not very accurate yet.


---

## Features
- **Image Upload:** Upload one or multiple retinal images for analysis.
- **Two-Stage Classification:**
  - **Stage 1:** Detects presence of DR (No DR vs DR).
  - **Stage 2:** If DR is present, classifies severity (Mild, Moderate, Severe, Proliferative).
- **Grad-CAM Heatmaps:** Visualize model attention for each prediction.
- **Results Table:** View predictions, confidence scores, and severity indices.
- **Export:** Download results as CSV or PDF.
- **Clear Results:** Reset session results.

---

## Local Setup Instructions


### 1. Clone the Repository

If you have a GitHub repository:

```powershell
# Open PowerShell and navigate to your desired directory
cd C:\Users\Dell\Downloads
git clone https://github.com/Dharmendra016/Diabetic-Retinopathy-Screening
```

If you have a ZIP file or folder:

1. Download and extract the ZIP file to your desired location (e.g., `C:\Users\Dell\Downloads\Diabetic-Retinopathy-Screening
`.
2. Navigate to the extracted folder:
  ```powershell
  cd C:\Users\Dell\Downloads\Diabetic-Retinopathy-Screening
  ```

### 2. Install Python (if not already installed)
- **Recommended Version:** Python 3.10 or higher
- Download from [python.org](https://www.python.org/downloads/)
- During installation, check "Add Python to PATH"


### 3. Create and Activate Virtual Environment

It is recommended to use a Python virtual environment to isolate dependencies.

#### Create a new virtual environment:

```powershell
# Make sure you are in your project directory
cd C:\Users\Dell\Downloads\Diabetic-Retinopathy-Screening

# Create a virtual environment named 'dr_app_venv'
py -3.10 -m venv dr_app_venv
        or
python -m venv dr_app_venv
```

#### Activate the virtual environment:

```powershell
# On Windows PowerShell
.\dr_app_venv\Scripts\Activate.ps1

# On Windows Command Prompt
dr_app_venv\Scripts\activate.bat

# On Git Bash
source dr_app_venv/Scripts/activate
```


### 4. Install Required Python Packages

After activating your virtual environment, install all required packages:

```powershell
pip install -r requirements.txt
            or
pip install torch torchvision flask pillow numpy opencv-python matplotlib reportlab fpdf2
```

**Package Descriptions:**
- `torch`, `torchvision`: Deep learning framework
- `flask`: Web application framework
- `pillow`: Image processing
- `numpy`: Numerical operations
- `opencv-python`: Image manipulation
- `matplotlib`: Visualization
- `reportlab`: PDF generation
- `fpdf2`: (for notebook PDF export)

### 5. Project Structure

```
hack__abcde/
├── app.py                  # Main Flask application
├── aba.ipynb               # Jupyter notebook (optional)
├── dr_app_venv/            # Python virtual environment
├── models/                 # Pretrained model weights (.pth files)
├── notebook/               # Additional notebooks
├── results/                # Evaluation results, metrics, plots
├── static/
│   ├── style.css           # Custom styles
│   └── uploads/            # Uploaded images & heatmaps
├── templates/
│   └── index.html          # Main HTML template
└── README.md               # This file
```

### 6. Download/Place Model Files
- Ensure the following files are present in the `models/` folder:
  - `stage1_best.pth` (Stage 1 model)
  - `mobilenetv3_classes1-4.pth` (Stage 2 model)
- These are required for inference. If missing, contact the project owner or retrain models.

### 7. Run the Application

```powershell
# Ensure you are in the project directory and the virtual environment is activated
python app.py
```
- The app will start in debug mode and be accessible at [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

### 8. Using the Web App
1. **Open your browser** and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
2. **Upload Images:** Use the upload form to select one or more retinal images (JPG/PNG).
3. **View Results:** Predictions, confidence scores, severity indices, and heatmaps will be displayed.
4. **Export Results:** Use the buttons to download results as CSV or PDF.
5. **Clear Results:** Use the "Clear Results" button to reset the session.



## Additional Notes
- **Image Format:** Use high-quality retinal images (JPG/PNG).
- **Session Storage:** Results are stored in the Flask session until cleared.
- **Heatmaps:** Grad-CAM visualizations are saved in `static/uploads/`.
- **Security:** For production, change the `app.secret_key` and disable debug mode.
- **Customization:** Edit `style.css` and `index.html` for UI changes.

---

## License
This project is for educational and research purposes. For commercial use, contact the author.

---

## Contact
For questions, issues, or contributions, please open an issue or contact the project maintainer.
