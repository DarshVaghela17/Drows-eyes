"""
Grad-CAM (Gradient-Weighted Class Activation Mapping) Generator
Visualizes which facial regions (e.g. drooping eyelids, open yawning mouth)
influenced the CNN prediction. Essential for explainable AI in safety-critical ADAS.
"""

import cv2
import numpy as np
import tensorflow as tf

class GradCAMVisualizer:
    def __init__(self, model, target_layer_name='target_conv_layer'):
        """
        model: Trained CNN-LSTM model or spatial CNN backbone
        target_layer_name: Name of the last convolutional layer
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.grad_model = self._create_grad_model()

    def _create_grad_model(self):
        """Constructs a gradient extraction sub-model mapping inputs to conv layer activations & predictions"""
        # If model is CNN-LSTM, resolve the internal CNN backbone
        backbone = self.model
        if hasattr(self.model, 'get_layer') and 'timedistributed_cnn' in [l.name for l in self.model.layers]:
            backbone = self.model.get_layer('timedistributed_cnn').layer

        target_layer = backbone.get_layer(self.target_layer_name)
        return tf.keras.models.Model(
            inputs=[backbone.inputs],
            outputs=[target_layer.output, backbone.output]
        )

    def generate_heatmap(self, img_array, class_index=None):
        """
        Computes Grad-CAM activation heatmap for a given input image:
        alpha_k = (1 / Z) * sum( d y^c / d A^k )
        L_GradCAM = ReLU( sum( alpha_k * A^k ) )
        """
        # Ensure 4D tensor (1, H, W, C)
        if len(img_array.shape) == 3:
            img_tensor = np.expand_dims(img_array, axis=0)
        else:
            img_tensor = img_array

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_tensor)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            loss = predictions[:, class_index]

        # Gradients of the target class score w.r.t. target conv layer outputs
        grads = tape.gradient(loss, conv_outputs)

        # Global Average Pooling of gradients across spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the channels of the activation map by corresponding gradient importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU to focus strictly on features positively contributing to drowsiness
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    def overlay_heatmap_on_image(self, original_frame, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
        """
        Colorizes heatmap and blends it with original RGB/BGR frame.
        Red: High activation (strongest drowsiness feature)
        Blue: Low activation (neutral background)
        """
        h, w = original_frame.shape[:2]
        # Resize heatmap to match original image dimensions
        resized_heatmap = cv2.resize(heatmap, (w, h))

        # Scale to 0-255 uint8
        heatmap_uint8 = np.uint8(255 * resized_heatmap)

        # Apply JET colormap
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

        # Blend original frame with heatmap
        blended = cv2.addWeighted(original_frame, 1.0 - alpha, colored_heatmap, alpha, 0)
        return blended, colored_heatmap
