import argparse
import pandas as pd
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc
)
from tqdm import tqdm
from transformers import AutoTokenizer


def load_model_and_tokenizer(model_path, tokenizer_path):
    print("[*] Loading ONNX model and tokenizer...")
    ort_session = ort.InferenceSession(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return ort_session, tokenizer


def predict_logits(text, tokenizer, ort_session):
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)
    ort_inputs = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask']
    }
    ort_outputs = ort_session.run(None, ort_inputs)
    logits = ort_outputs[0]
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid
    return probs[0][0]


def evaluate(validation_df, tokenizer, ort_session, thresholds, plot_roc=False):
    scores = pd.DataFrame(columns=["toxicity_threshold", "accuracy", "precision", "recall", "f1", "auc_roc"])

    for threshold in thresholds:
        predictions, probs, actual_labels = [], [], []

        print(f"\n[*] Evaluating threshold {threshold}...")
        for _, row in tqdm(validation_df.iterrows(), total=len(validation_df)):
            text = row['comment_text']
            toxic_prob = predict_logits(text, tokenizer, ort_session)
            predicted_label = toxic_prob > threshold

            predictions.append(predicted_label)
            probs.append(toxic_prob)
            actual_labels.append(int(row['toxic'] > 0))

        # Compute metrics
        accuracy = accuracy_score(actual_labels, predictions)
        precision = precision_score(actual_labels, predictions)
        recall = recall_score(actual_labels, predictions)
        f1 = f1_score(actual_labels, predictions)
        fpr, tpr, _ = roc_curve(actual_labels, probs)
        roc_auc = auc(fpr, tpr)

        print(f"--- Threshold: {threshold} ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {roc_auc:.4f}")

        scores.loc[len(scores)] = [threshold, accuracy, precision, recall, f1, roc_auc]

        if plot_roc:
            plt.figure()
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve (Threshold = {threshold})')
            plt.legend(loc="lower right")
            plt.grid()
            plt.show()

    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate ONNX Detoxify model")
    parser.add_argument("--validation-file", type=str, default="./jigsaw_data/validation.csv", help="Path to validation CSV")
    parser.add_argument("--model-path", type=str, default="./detoxify_model/model.quant.onnx", help="Path to quantized ONNX model")
    parser.add_argument("--tokenizer-path", type=str, default="./detoxify_model", help="Path to tokenizer directory")
    parser.add_argument("--thresholds", type=float, nargs='+', default=[0.2, 0.4, 0.5, 0.7, 0.9], help="List of thresholds to evaluate")
    parser.add_argument("--plot-roc", action="store_true", help="Plot ROC curves")
    args = parser.parse_args()

    validation_df = pd.read_csv(args.validation_file)
    ort_session, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_path)

    scores = evaluate(validation_df, tokenizer, ort_session, args.thresholds, args.plot_roc)

    print("\nAll scores:")
    print(scores)


if __name__ == "__main__":
    main()
