# Detoxify ONNX 🚀

This project provides an ONNX-exported and quantized version of the [Detoxify](https://github.com/unitaryai/detoxify) multilingual model, optimized for runtime inference.  
It enables faster and lighter toxicity detection using ONNX Runtime.

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 🛠️ Features

- Export Detoxify multilingual model to ONNX
- Quantize the ONNX model for faster inference
- Update model `config.json` with correct label mappings
- Evaluate the model using standard metrics (accuracy, precision, recall, F1, AUC-ROC)

---

## 🧪 ONNX Evaluation Results

### Original Model (using Detoxify lib and ONNX):

| Threshold | Accuracy | Precision | Recall | F1     | AUC-ROC |
|----------:|---------:|----------:|-------:|-------:|--------:|
| **0.2**   | 0.8408   | 0.4899    | 0.8659 | 0.6257 | 0.9345  |
| **0.4**   | 0.8723   | 0.5628    | 0.7577 | 0.6459 | 0.9345  |
| **0.5**   | 0.8845   | 0.6073    | 0.7041 | 0.6521 | 0.9345  |
| **0.7**   | 0.8954   | 0.6951    | 0.5691 | 0.6258 | 0.9345  |
| **0.9**   | 0.8941   | 0.8501    | 0.3780 | 0.5234 | 0.9345  |

Time for 1 threshold evaluation =~ 3 min 30s 

### Quantized model:

| Threshold | Accuracy | Precision | Recall | F1     | AUC-ROC |
|----------:|---------:|----------:|-------:|-------:|--------:|
| **0.2**   | 0.8581   | 0.5249    | 0.8154 | 0.6387 | 0.9306  |
| **0.4**   | 0.8809   | 0.6001    | 0.6748 | 0.6353 | 0.9306  |
| **0.5**   | 0.8880   | 0.6408    | 0.6179 | 0.6291 | 0.9306  |
| **0.7**   | 0.8969   | 0.7467    | 0.4984 | 0.5978 | 0.9306  |
| **0.9**   | 0.8869   | 0.8878    | 0.3024 | 0.4512 | 0.9306  |

Time for 1 threshold evaluation =~ 2 min 41s 

---

## 📥 Dataset: Jigsaw Toxic Comment Classification

This project uses the validation set from the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

### 🔽 How to Download:

1. Create a [Kaggle](https://www.kaggle.com/) account and accept the competition rules.
2. Download the dataset from the [Data tab](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) of the competition page.
3. Extract or locate the `validation.csv`.
4. Place the validation file under a local folder:

```bash
mkdir -p jigsaw_data
cp path/to/validation.csv jigsaw_data/
```

> Note: You may want to install the Kaggle CLI for automated downloading:

```bash
# Make sur to have kaggle.json loaded accordingly
pip install kaggle
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
unzip jigsaw-toxic-comment-classification-challenge.zip -d jigsaw_data
```

## 🧵 Usage

### Convert and Quantize

```bash
python convert_to_onnx.py --model-dir detoxify_model \
  --save-hf-model \
  --export-onnx \
  --quantize-onnx \
  --update-config
```

### Evaluate

```bash
python evaluate.py --validation-file ./jigsaw_data/validation.csv \
  --model-path ./detoxify_model/model.quant.onnx \
  --tokenizer-path ./detoxify_model \
  --thresholds 0.2 0.4 0.5 0.7 0.9 \
  --plot-roc
```

---

## 🤗 Hugging Face Model

You can find and use the quantized ONNX model from the [Gravitee.io organization on Hugging Face](https://huggingface.co/gravitee-io/detoxify-onnx):

```python
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import numpy as np

# Load model and tokenizer using optimum
model = ORTModelForSequenceClassification.from_pretrained("gravitee-io/detoxify-onnx", file_name="model.quant.onnx")
tokenizer = AutoTokenizer.from_pretrained("gravitee-io/detoxify-onnx")

# Tokenize input
text = "Your comment here"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Run inference
outputs = model(**inputs)
logits = outputs.logits

# Optional: convert to probabilities
probs = 1 / (1 + np.exp(-logits))
print(probs)
```

