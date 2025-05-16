import argparse
import json
import os

import onnx
import onnxoptimizer
import torch
from detoxify import Detoxify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType


def save_hf_model(model_dir, model_type):
    print("[*] Loading Detoxify multilingual model...")
    unitary_model = Detoxify(model_type)
    hf_model = unitary_model.model
    hf_tokenizer = unitary_model.tokenizer

    os.makedirs(model_dir, exist_ok=True)
    print(f"[*] Saving model and tokenizer to '{model_dir}'...")
    hf_model.save_pretrained(model_dir)
    hf_tokenizer.save_pretrained(model_dir)
    AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large").save_pretrained(model_dir)


def export_model_to_onnx(model_dir):
    print("[*] Exporting model to ONNX...")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer("Hello, World!", return_tensors="pt")
    torch.onnx.export(
        model,
        (inputs['input_ids'], inputs['attention_mask']),
        os.path.join(model_dir, "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=17
    )
    print("[✓] ONNX export complete.")


def optimize_onnx_model(model_dir):
    print("[*] Optimizing ONNX model...")
    model_path = os.path.join(model_dir, "model.onnx")
    optimized_path = os.path.join(model_dir, "model.optim.onnx")

    model = onnx.load(model_path)
    passes = onnxoptimizer.get_fuse_and_elimination_passes()
    optimized = onnxoptimizer.optimize(model, passes)
    onnx.save(optimized, optimized_path)
    print("[✓] Optimization complete.")


def quantize_onnx_model(model_dir):
    print("[*] Preparing to quantize ONNX model...")
    optimized_path = os.path.join(model_dir, "model.optim.onnx")
    quantized_path = os.path.join(model_dir, "model.quant.onnx")

    if not os.path.exists(optimized_path):
        print("[!] Optimized model not found. Optimizing now...")
        optimize_onnx_model(model_dir)

    quantize_dynamic(
        optimized_path,
        quantized_path,
        weight_type=QuantType.QInt8
    )
    print("[✓] Quantization complete.")


def update_config_labels(model_dir):
    print("[*] Updating config.json with id2label and label2id...")

    classes = [
        "toxicity", "severe_toxicity", "obscene", "identity_attack",
        "insult", "threat", "sexual_explicit"
    ]
    identity_classes = [
        "male", "female", "homosexual_gay_or_lesbian", "christian",
        "jewish", "muslim", "black", "white", "psychiatric_or_mental_illness"
    ]
    all_labels = classes + identity_classes

    id2label = {str(i): label for i, label in enumerate(all_labels)}
    label2id = {label: i for i, label in enumerate(all_labels)}

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["id2label"] = id2label
    config["label2id"] = label2id

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print("[✓] config.json updated.")


def main():
    parser = argparse.ArgumentParser(description="Detoxify ONNX export and quantization pipeline")
    parser.add_argument("--model-dir", type=str, default="detoxify_model", help="Directory to save/load the model")
    parser.add_argument("--model-type", type=str, default="multilingual", help="Type of model to use")
    parser.add_argument("--save-hf-model", action="store_true", help="Save Detoxify model and tokenizer")
    parser.add_argument("--export-onnx", action="store_true", help="Export model to ONNX format")
    parser.add_argument("--quantize-onnx", action="store_true", help="Quantize ONNX model (auto-optimizes if needed)")
    parser.add_argument("--update-config", action="store_true", help="Update config.json with labels")
    args = parser.parse_args()

    if args.save_hf_model and args.model_type:
        save_hf_model(args.model_dir, args.model_type)

    if args.export_onnx:
        export_model_to_onnx(args.model_dir)

    if args.quantize_onnx:
        quantize_onnx_model(args.model_dir)

    if args.update_config:
        update_config_labels(args.model_dir)


if __name__ == "__main__":
    main()
