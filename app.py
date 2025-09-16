import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from flask import Flask, request, render_template, session, redirect, url_for, Response
from PIL import Image, ImageOps
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
from io import StringIO, BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors

# -------------------- Flask App --------------------
app = Flask(__name__)
app.secret_key = "super_secret_key"  # Required for session
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Padding + Resize --------------------
def padding_and_resize(img, target_size=224, fill=(0, 0, 0)):
    w, h = img.size
    if w != h:
        if w > h:
            delta = w - h
            padding = (0, delta // 2, 0, delta - delta // 2)
        else:
            delta = h - w
            padding = (delta // 2, 0, delta - delta // 2, 0)
        img = ImageOps.expand(img, padding, fill=fill)
    img = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    return img

# -------------------- Common Transform --------------------
transform = transforms.Compose([
    transforms.Lambda(lambda img: padding_and_resize(img, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------- Load Models --------------------
# Stage 1: Binary classification (No DR vs. DR)
model_stage1 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
num_ftrs1 = model_stage1.classifier[3].in_features
model_stage1.classifier[3] = nn.Linear(num_ftrs1, 2)
model_stage1.load_state_dict(torch.load("./models/stage1_best.pth", map_location=device))
model_stage1 = model_stage1.to(device)
model_stage1.eval()

# Stage 2: Severity classification (Mild, Moderate, Severe, Proliferative)
model_stage2 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
num_ftrs2 = model_stage2.classifier[3].in_features
model_stage2.classifier[3] = nn.Linear(num_ftrs2, 4)
model_stage2.load_state_dict(torch.load("./models/mobilenetv3_classes1-4.pth", map_location=device))
model_stage2 = model_stage2.to(device)
model_stage2.eval()

classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

# -------------------- Grad-CAM --------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[:, class_idx]
        loss.backward(retain_graph=True)

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.squeeze(0)

        for i in range(len(pooled_gradients)):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = activations.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= heatmap.max() + 1e-8
        return heatmap

def overlay_heatmap(img_path, heatmap, save_name):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], save_name)
    plt.imsave(save_path, overlay)
    return os.path.basename(save_path)

# -------------------- Index Route --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("images[]") or [request.files.get("image")]
        results = []

        for file in files:
            if not file or file.filename == '':
                continue
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(img_path)

            img = Image.open(img_path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)

            # Stage 1: Binary classification
            out1 = model_stage1(input_tensor)
            probs1 = torch.softmax(out1, dim=1)
            pred1 = probs1.argmax(dim=1).item()
            conf1 = float(probs1[0, pred1].item())

            gradcam1 = GradCAM(model_stage1, model_stage1.features[-1])
            heatmap1 = gradcam1.generate(input_tensor, class_idx=pred1)
            heatmap1_path = overlay_heatmap(img_path, heatmap1, f"stage1_heatmap_{file.filename}.png")

            result = {
                "image_path": file.filename,
                "prediction": "No DR",
                "confidence": conf1,
                "confidence_percent": f"{conf1 * 100:.1f}",  # Pre-format for template
                "heatmap1": heatmap1_path,
                "heatmap2": None,
                "severity_idx": 0
            }

            if pred1 == 1:  # DR present, proceed to Stage 2
                out2 = model_stage2(input_tensor)
                probs2 = torch.softmax(out2, dim=1)
                pred2 = probs2.argmax(dim=1).item()
                conf2 = float(probs2[0, pred2].item())

                gradcam2 = GradCAM(model_stage2, model_stage2.features[-1])
                heatmap2 = gradcam2.generate(input_tensor, class_idx=pred2)
                heatmap2_path = overlay_heatmap(img_path, heatmap2, f"stage2_heatmap_{file.filename}.png")

                result["prediction"] = classes[pred2 + 1]  # Map to Mild, Moderate, Severe, Proliferative
                result["confidence"] = conf2
                result["confidence_percent"] = f"{conf2 * 100:.1f}"  # Pre-format for template
                result["heatmap2"] = heatmap2_path
                result["severity_idx"] = pred2 + 1

            results.append(result)

        # Sort results by severity_idx in descending order (highest severity first)
        results = sorted(results, key=lambda x: x["severity_idx"], reverse=True)
        session["results"] = results
        return redirect(url_for("index"))

    # GET: Retrieve results from session without clearing
    results = session.get("results", [])
    return render_template("index.html", results=results)

# -------------------- Clear Results Route --------------------
@app.route("/clear", methods=["GET"])
def clear_results():
    session.pop("results", None)
    return redirect(url_for("index"))

# -------------------- Export CSV --------------------
@app.route("/export/csv")
def export_csv():
    results = session.get("results", [])
    if not results:
        return "No data available", 404

    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(["Image", "Prediction", "Confidence", "Severity Index"])
    for res in results:
        cw.writerow([res["image_path"], res["prediction"], f"{res['confidence']:.4f}", res["severity_idx"]])
    output = si.getvalue()
    return Response(output, mimetype="text/csv", headers={"Content-disposition": "attachment; filename=dr_report.csv"})

# -------------------- Export PDF --------------------
@app.route("/export/pdf")
def export_pdf():
    results = session.get("results", [])
    if not results:
        return "No data available", 404

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)

    data = [['Image', 'Prediction', 'Confidence', 'Severity Index']]
    for res in results:
        data.append([res['image_path'], res['prediction'], f"{res['confidence']:.4f}", res['severity_idx']])

    table = Table(data)
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ])

    # Add row colors based on severity (column 3: Severity Index)
    for i, row in enumerate(data[1:], start=1):
        severity = row[3]
        if severity == 0:
            style.add('BACKGROUND', (0, i), (-1, i), colors.lightgreen)
        elif severity <= 2:
            style.add('BACKGROUND', (0, i), (-1, i), colors.yellow)
        else:
            style.add('BACKGROUND', (0, i), (-1, i), colors.lightcoral)

    table.setStyle(style)
    elements = [table]
    doc.build(elements)

    pdf_output = buffer.getvalue()
    buffer.close()
    return Response(pdf_output, mimetype="application/pdf", headers={"Content-disposition": "attachment; filename=dr_report.pdf"})

if __name__ == "__main__":
    app.run(debug=True)