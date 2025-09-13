import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import base64
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Device & Model Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5
# Unified class names
class_names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
severity_map = {name: i for i, name in enumerate(class_names)}

# Load MobileNetV3 with default weights
mobileNet_small = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

# Replace classifier with your custom architecture
mobileNet_small.classifier = nn.Sequential(
    nn.Linear(576, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(1024, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(512, num_classes)
)

# Load trained weights
mobileNet_small.load_state_dict(torch.load('./model_weight/best_weights_mod_new_.pth', map_location=DEVICE))
mobileNet_small.to(DEVICE)
mobileNet_small.eval()

# --- Preprocessing (match Kaggle) ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Helper Function ---
def get_prediction_and_heatmap(image_file):
    try:
        img_bytes = image_file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except:
        return {"error": "Invalid image file"}
    
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    rgb_img_np = np.float32(np.array(pil_img.resize((224, 224)))) / 255.0

    # Prediction
    with torch.no_grad():
        output = mobileNet_small(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]

    predicted_class_id = torch.argmax(probabilities).item()
    predicted_class_name = class_names[predicted_class_id]
    confidence = probabilities[predicted_class_id].item()
    print(f"Predicted: {predicted_class_name} ({confidence*100:.2f}%)")

    # Grad-CAM
    target_layer = mobileNet_small.features[-1]
    cam = GradCAM(model=mobileNet_small, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(predicted_class_id)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)

    # Encode heatmap to base64
    visualization_pil = Image.fromarray((visualization * 255).astype(np.uint8))
    buffered = io.BytesIO()
    visualization_pil.save(buffered, format="JPEG")
    heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Determine triage color
    if predicted_class_name in ['No DR', 'Mild DR']:
        color = 'green'
    elif predicted_class_name == 'Moderate DR':
        color = 'yellow'
    else:
        color = 'red'

    return {
        "prediction": predicted_class_name,
        "confidence": round(confidence, 4),
        "heatmap": heatmap_base64,
        "color": color
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
        result = get_prediction_and_heatmap(file)
        result['filename'] = file.filename
        results.append(result)

    # Sort by severity (Proliferative first)
    sorted_results = sorted(results, key=lambda x: severity_map.get(x.get('prediction', ''), -1), reverse=True)
    return jsonify(sorted_results)

# --- Offline Inference (matches Flask) ---
def inference_on_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = mobileNet_small(img_tensor)
        probabilities = nn.functional.softmax(outputs, dim=1)[0]

    predicted_class_index = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class_index].item()
    predicted_label = class_names[predicted_class_index]

    return {
        "prediction": predicted_label,
        "confidence": confidence
    }

# Example usage (offline only)
if __name__ == '__main__':
    app.run(debug=True)
