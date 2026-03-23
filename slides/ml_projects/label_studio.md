## Шаг 3: Разворачиваем LabelStudio

На этом шаге нужно разметить выборку в LabelStudio. Запускаем интерфейс командой

```shell
make labelstudio
```

Далее

* на вкладке `Sign up` вводим любой логин и пароль
* создаём проект и загружаем датасет на вкладке **Data import**
* 
* на вкладке `Labeling`  `Ordered By Time`
* Label all tasks
* размечаем на positive/negative
* когда датасет размечен - нажимаем "export"

Сохраняем размеченный датасет в файл `labeled_messages.csv`

# Labeling

# Multi-Label Classification Data Preparation with Label Studio

Here's how to set up a workflow for multi-label classification using Label Studio, including zero-shot classification for pre-labeling:

## Step 1: Install and Set Up Label Studio

```bash
bash
Copy
# Install Label Studio
pip install label-studio

# Launch Label Studio
label-studio start
```

## Step 2: Create a Project with Multi-Label Configuration

1. In the Label Studio interface, create a new project
2. Configure a labeling interface for multi-label text classification:

```xml
<View>
  <Text name="text" value="$text"/>
  <Choices name="labels" toName="text" choice="multiple" showInLine="false">
    <Choice value="Category1"/>
    <Choice value="Category2"/>
    <Choice value="Category3"/>
    <!-- Add all your label categories -->
  </Choices>
</View>

```

## Step 3: Pre-label Data with Zero-Shot Classification

```python
import json
from transformers import pipeline
import pandas as pd

# Load your dataset
df = pd.read_csv("your_dataset.csv")
texts = df["text"].tolist()

# Initialize zero-shot classifier
classifier = pipeline("zero-shot-classification",
                     model="facebook/bart-large-mnli")

# Define your labels
candidate_labels = ["Category1", "Category2", "Category3"]

# Pre-label data
results = []
for i, text in enumerate(texts):
    # Get zero-shot predictions
    output = classifier(text, candidate_labels, multi_label=True)

    # Filter predictions above a confidence threshold
    predicted_labels = [label for label, score in zip(output["labels"], output["scores"])
                       if score > 0.5]  # Adjust threshold as needed

    # Create Label Studio compatible format
    item = {
        "id": i,
        "text": text,
        "predictions": [{
            "model_version": "zero-shot-bart",
            "result": [{
                "from_name": "labels",
                "to_name": "text",
                "type": "choices",
                "value": {
                    "choices": predicted_labels
                }
            }]
        }]
    }
    results.append(item)

# Save to JSON for Label Studio import
with open("prelabeled_data.json", "w") as f:
    json.dump(results, f)

```

## Step 4: Import Pre-labeled Data to Label Studio

1. In your Label Studio project, go to "Import" tab
2. Choose "Upload Files" and select your `prelabeled_data.json` file
3. Make sure to check "JSON-formatted tasks" and enable "Predictions" if you want to see the zero-shot model's suggestions

## Step 5: Manual Labeling and Correction

1. Start labeling in the Label Studio interface
2. Review and correct the pre-labels from the zero-shot classifier
3. You'll see the model's predictions as pre-selected choices

## Step 6: Export Labeled Data for Training

1. Once labeling is complete, go to "Export" tab
2. Choose "JSON" format for the export (includes both text and labels)
3. Process the exported data for your BERT training:

```python
import json
import pandas as pd

with open("labeled_data.json", "r") as f:
    data = json.load(f)

texts = []
label_lists = []

for item in data:
    texts.append(item["data"]["text"])

    if "annotations" in item and len(item["annotations"]) > 0:
        choices = item["annotations"][0]["result"][0]["value"]["choices"]
        label_lists.append(choices)
    else:
        label_lists.append([])

# Convert to multi-hot encoding format
all_labels = sorted(list(set(label for labels in label_lists for label in labels)))
encoded_labels = []

for labels in label_lists:
    encoded = [1 if label in labels else 0 for label in all_labels]
    encoded_labels.append(encoded)

# Create DataFrame with text and encoded labels
df_final = pd.DataFrame({
    "text": texts,
})

# Add label columns
for i, label in enumerate(all_labels):
    df_final[label] = [encoded[i] for encoded in encoded_labels]

# Save for BERT training
df_final.to_csv("bert_training_data.csv", index=False)

```

## Tips for Label Studio Configuration

1. **Keyboard Shortcuts**: Enable keyboard shortcuts to speed up labeling
2. **Agreement Metrics**: If using multiple annotators, set up agreement metrics
3. **Active Learning**: Consider active learning to prioritize examples for manual labeling
4. **Regional Models**: For Russian language, use "cross-encoder/nli-deberta-v3-xsmall-russian" for zero-shot classification

This workflow lets you efficiently create multi-label training data by combining the strengths of zero-shot pre-labeling and human annotation refinement, perfect for training Modern-BERT models.
