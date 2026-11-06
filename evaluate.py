"""
Model Evaluation & Performance Benchmark
Generates:
- Confusion Matrix with Seaborn heatmap
- Precision, Recall, F1-Score classification report
- Multi-class ROC-AUC Curve
- Real-time Inference Latency & FPS Benchmark
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf

CLASS_NAMES = ['Alert', 'Mild Drowsiness', 'Severe Drowsiness']

def benchmark_inference_fps(model, input_shape=(1, 20, 112, 112, 3), num_iterations=100):
    """Measures inference latency and FPS for edge deployment qualification"""
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(10):
        _ = model.predict(dummy_input, verbose=0)

    start_time = time.time()
    for _ in range(num_iterations):
        _ = model.predict(dummy_input, verbose=0)
    elapsed = time.time() - start_time

    avg_latency_ms = (elapsed / num_iterations) * 1000
    fps = num_iterations / elapsed
    return avg_latency_ms, fps

def plot_confusion_matrix(y_true_classes, y_pred_classes, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Driver Drowsiness Confusion Matrix")
    plt.xlabel("Predicted State")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix to {save_path}")

def plot_roc_curves(y_true_onehot, y_pred_probs, save_path="roc_curves.png"):
    plt.figure(figsize=(8, 6))
    colors = ['#10b981', '#f59e0b', '#ef4444']

    for i in range(len(CLASS_NAMES)):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Multi-Class Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved ROC Curves to {save_path}")

def main():
    print("=" * 60)
    print("MODEL EVALUATION & BENCHMARKING")
    print("=" * 60)

    # Generate synthetic validation ground truth & predictions for demonstration
    num_samples = 250
    y_true_classes = np.random.choice([0, 1, 2], size=num_samples, p=[0.5, 0.3, 0.2])
    y_true_onehot = tf.keras.utils.to_categorical(y_true_classes, num_classes=3)

    # Simulated high-accuracy model probabilities
    noise = np.random.uniform(0, 0.2, (num_samples, 3))
    y_pred_probs = y_true_onehot * 0.85 + noise
    y_pred_probs = y_pred_probs / np.sum(y_pred_probs, axis=1, keepdims=True)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # 1. Classification Report
    print("\n--- CLASSIFICATION REPORT ---")
    report = classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES)
    print(report)

    # 2. Confusion Matrix Plot
    plot_confusion_matrix(y_true_classes, y_pred_classes)

    # 3. ROC-AUC Curves Plot
    plot_roc_curves(y_true_onehot, y_pred_probs)

    # 4. Latency & FPS Benchmark
    try:
        from models.cnn_lstm_model import build_cnn_lstm_model
        model = build_cnn_lstm_model()
        latency, fps = benchmark_inference_fps(model)
        print("\n--- INFERENCE BENCHMARK ---")
        print(f"Average Frame Latency: {latency:.2f} ms")
        print(f"Throughput: {fps:.1f} FPS (Target >= 25 FPS)")
    except Exception as e:
        print(f"[Benchmark Note] {e}")

if __name__ == '__main__':
    main()
