# Tutorial: Image Classification — Step-by-Step Guide

A **practical, detailed** guide to using Phoenix ML framework for Image Classification using Deep Learning.

> **Summary**: Classify images using a pre-trained ResNet model exported to ONNX.

---

## 📁 User Directory Structure

```
my-vision-project/
├── .env                           # ← User configures infra URLs here
├── model_configs/
│   └── image-classification.yaml  # ← Define use case
├── my_training/
│   ├── train.py                   # ← Training script
│   └── preprocessor.py            # ← Custom image preprocessor plugin!
├── data/
│   └── images/                    # ← Image structure inside DVC
│       ├── cat/
│       └── dog/
└── models/                        # ← Framework auto-creates during training
```

---

## 1️⃣ Step 1 — Define your use case (YAML Config)

```yaml
# model_configs/image-classification.yaml
model_id: image-classification
version: v1
framework: onnx
task_type: classification
model_path: models/image_classification/v1/model.onnx
train_script: my_training/train.py

# Data source
data_source:
  type: dvc                       # Highly recommended for images
  path: data/images/

# Retrain strategy
retrain:
  trigger: data_change            # Retrain when DVC tracks new images
  drift_detection: false          # Disable drift: KS/PSI are not for images!

# Monitoring
monitoring:
  primary_metric: accuracy
```

---

## 2️⃣ Step 2 — Write Preprocessor

Since raw API inputs will be base64 strings or URLs, you must register a preprocessor plugin to convert them into `NxCxHxW` NumPy tensors for ONNX!

```python
# my_training/preprocessor.py
import base64
import io
import numpy as np
from PIL import Image
from phoenix_ml.domain.inference.services.processor_plugin import IPreprocessor

class ResNetPreprocessor(IPreprocessor):
    async def preprocess(self, raw_input: dict, model_config: dict) -> list[float]:
        # 1. Read base64 image
        b64_str = raw_input.get("image_base64")
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # 2. Resize and crop to 224x224
        img = img.resize((224, 224))
        
        # 3. Normalize (ImageNet)
        np_img = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_img = (np_img - mean) / std
        
        # 4. HWC -> CHW format for PyTorch/ONNX
        np_img = np_img.transpose(2, 0, 1)
        
        # Flatten to 1D list for the framework's JSON boundary
        return np_img.flatten().tolist()
```

*(Register this inside your FastAPI `lifespan` or startup script using `PluginRegistry`)*

---

## 3️⃣ Step 3 — Write Training/Export Script

```python
# my_training/train.py
import torch
import torchvision.models as models

def train_and_export(output_path: str, metrics_path: str = None, data_path: str = None, reference_path: str = None) -> None:
    # Normally you'd train your PyTorch model here using data_path!
    # For this tutorial, we export a pre-trained ResNet18:
    model = models.resnet18(pretrained=True)
    model.eval()

    # Create dummy input that matches ResNet18's expected input shape: Batch=1, Channels=3, H=224, W=224
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export PyTorch to ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f"Exported ResNet18 ONNX to {output_path}")
```

---

## 4️⃣ Call the API

Send base64 encoded images to the API:

```bash
# Convert image to base64
IMG_B64=$(base64 -w 0 data/images/dog/sample.jpg)

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "image-classification", 
    "image_base64": "'"$IMG_B64"'"
  }'

# Response:
# {"result": 258, "confidence": 0.88, "latency_ms": 42.5}
# (258 = ImageNet class ID for Samoyed dog)
```
