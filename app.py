import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import base64
import io
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Model and Device Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5
class_names = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative']
severity_map = {name: i for i, name in enumerate(class_names)}

# Re-create the model architecture as you defined it
model = mobilenet_v3_small()
in_features = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(in_features, out_features=288),
    nn.Hardswish(),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(288, out_features=num_classes)
)
model.load_state_dict(torch.load('./model_weight/best_mobilenet_model.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- Preprocessing Transformation ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Helper Functions ---
def get_prediction_and_heatmap(image_file):
    """
    Processes a single image, gets the prediction, and generates the Grad-CAM heatmap.
    """
    img_bytes = image_file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    rgb_img_np = np.float32(np.array(pil_img.resize((256, 256)))) / 255

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    
    predicted_class_id = torch.argmax(probabilities).item()
    predicted_class_name = class_names[predicted_class_id]
    confidence = probabilities[predicted_class_id].item()

    target_layer = model.features[-1]
    targets = [ClassifierOutputTarget(predicted_class_id)]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)

    visualization_pil = Image.fromarray((visualization * 255).astype(np.uint8))
    buffered = io.BytesIO()
    visualization_pil.save(buffered, format="JPEG")
    heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {
        "prediction": predicted_class_name,
        "confidence": round(confidence, 4),
        "heatmap": heatmap_base64
    }

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/diagnose', methods=['POST'])
def diagnose_api():
    if 'files[]' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400
    
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return jsonify({"error": "No selected files"}), 400
    
    results = []
    for file in files:
        if file:
            try:
                result = get_prediction_and_heatmap(file)
                result['filename'] = file.filename
                results.append(result)
            except Exception as e:
                results.append({"filename": file.filename, "error": str(e)})

    # Sort results by severity level (Proliferative first)
    sorted_results = sorted(results, key=lambda x: severity_map.get(x.get('prediction', ''), -1), reverse=True)
    
    return jsonify(sorted_results)

if __name__ == '__main__':
    app.run(debug=True)