# SweedpPOS Documents calssifier

[https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/tutorial-classification.html](https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/tutorial-classification.html)

Sagemaker + AWS zero shot

Model deploy

```jsx
from sagemaker.huggingface import HuggingFaceModel
import sagemaker

role = sagemaker.get_execution_role()
model = HuggingFaceModel(
    model_data="openai/clip-vit-large-patch14",
    role=role,
    transformers_version="4.26",
    pytorch_version="1.13",
    py_version="py39",
)
predictor = model.deploy(initial_instance_count=1, instance_type="ml.g4dn.xlarge")

```

Inference

```jsx
from PIL import Image
import requests

def classify_image(image_url, labels):
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    return processor.decode(outputs.logits.argmax(dim=-1))

```

Prediction

```python
labels = ["arctic fox", "snowy owl", "glacier", "igloo"]
result = classify_image("https://example.com/arctic.jpg", labels)
print(f"Predicted: {result}")  # Output: Predicted: snowy owl

```