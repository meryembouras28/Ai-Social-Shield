---
license: apache-2.0
language:
- en
base_model:
- google/siglip2-base-patch16-224
pipeline_tag: image-classification
library_name: transformers
tags:
- nswf
- exnrt.com
---

# NSFW Image Detection – A Top Performer

This model is fine-tuned for **NSFW image classification**. It classifies content into three safety-critical categories, making it useful for moderation, safety filtering, and compliant content handling systems.

<p>
  <a href="https://exnrt.com/blog/ai/fine-tuning-siglip2/" target="_blank">
    <img src="https://img.shields.io/badge/View%20Training%20Code-blue?style=for-the-badge&logo=readthedocs"/>
  </a>
  <a href="https://exnrt.com/blog/ai/fine-tuning-siglip2/" target="_blank">https://exnrt.com/blog/ai/fine-tuning-siglip2/</a>
</p>

---

## 🚀 Usage Example

```python
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch.nn.functional as F

model_path = "Ateeqq/nsfw-image-detection"
processor = AutoImageProcessor.from_pretrained(model_path)
model = SiglipForImageClassification.from_pretrained(model_path)

image_path = r"/content/download.jpg"
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
probabilities = F.softmax(logits, dim=1)

predicted_class_id = logits.argmax().item()
predicted_class_label = model.config.id2label[predicted_class_id]
confidence_scores = probabilities[0].tolist()

print(f"Predicted class ID: {predicted_class_id}")
print(f"Predicted class label: {predicted_class_label}\n")
for i, score in enumerate(confidence_scores):
    label = model.config.id2label[i]
    print(f"Confidence for '{label}': {score:.6f}")
```

## Output
```
Predicted class ID: 2
Predicted class label: safe_normal

Confidence for 'gore_bloodshed_violent': 0.000002
Confidence for 'nudity_pornography': 0.000005
Confidence for 'safe_normal': 0.999993
```

---

## 🧠 Model Details

* **Base model**: `google/siglip2-base-patch16-224`
* **Task**: Image Classification (NSFW/Safe detection)
* **Framework**: PyTorch / Hugging Face Transformers
* **Fine-tuned on**: Custom dataset with 3 content categories
* **Selected checkpoint**: Epoch 5
* **Batch size**: 64
* **Epochs trained**: 5

---

### 📌 Confusion Matrix

![Metrics](https://huggingface.co/Ateeqq/nsfw-image-detection/resolve/main/final-epoch-results.png)

---

### 🏷️ Categories

| ID | Label                    |Excluded|
| -- | ---------------------------|---------------|
| 0  | ✅`gore_bloodshed_violent` |❌ Fight, Accident, Angry|
| 1  | ✅`nudity_pornography`     |❌ Normal Romance, Normal Kissing|
| 2  | ✅`safe_normal`            |❌ |

### 🧾 Label Mapping

```python
label2id = {'gore_bloodshed_violent': 0, 'nudity_pornography': 1, 'safe_normal': 2}
id2label = {0: 'gore_bloodshed_violent', 1: 'nudity_pornography', 2: 'safe_normal'}
```
---

## 📊 Training Metrics (Epoch 5 Selected ✅)

| Epoch | Training Loss | Validation Loss | Accuracy   |
| ----- | ------------- | --------------- | ---------- |
| 1     | 0.0765        | 0.1166          | 95.70%     |
| 2     | 0.0719        | 0.0477          | 98.34%     |
| 3     | 0.0089        | 0.0634          | 98.05%     |
| 4     | 0.0109        | 0.0437          | 98.61%     |
| 5 ✅   | 0.0001        | 0.0389          | **99.02%** |

### 📌 Epoch Training Results

![Epoch Results](https://huggingface.co/Ateeqq/nsfw-image-detection/resolve/main/all-epochs-results.png)

- **Training runtime**: 1h 21m 40s
- **Final Training Loss**: 0.0727
- **Steps/sec**: 0.11 | **Samples/sec**: 6.99