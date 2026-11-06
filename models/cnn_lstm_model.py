"""
Hybrid Spatial-Temporal Deep Learning Architecture for Driver Drowsiness
Combines:
1. Spatial Feature Extractor: TimeDistributed CNN (MobileNetV2 or Custom ConvNet)
   extracting facial spatial clues (eyelid position, mouth aperture, pupil direction).
2. Temporal Sequence Model: Bidirectional LSTM / GRU modeling chronological dynamics
   (slow micro-sleep eye closure vs fast alert blinks, yawning progression, head nodding).
3. Dense Classification Head: 3-class Softmax (Alert, Mild Drowsiness, Severe Drowsiness).
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_custom_cnn_backbone(input_shape=(112, 112, 3)):
    """
    Lightweight, highly efficient CNN feature extractor for real-time edge execution.
    Input: Single cropped face frame (112, 112, 3)
    Output: 128-dimensional spatial feature embedding vector
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1: Low-level edge & contour features (eyelids, lip outlines)
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 56 x 56

    # Block 2: Intermediate facial component representation
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 28 x 28

    # Block 3: Deep semantic facial landmarks & fatigue expressions
    x = layers.Conv2D(128, (3, 3), padding='same', name='target_conv_layer')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 14 x 14

    # Spatial Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    return models.Model(inputs=inputs, outputs=x, name='spatial_cnn_backbone')

def build_cnn_lstm_model(sequence_length=20, frame_size=(112, 112, 3), num_classes=3):
    """
    End-to-End CNN-LSTM Temporal Architecture
    Sequence Length: 20 consecutive frames (~0.66 seconds at 30 FPS)
    Classes:
      0: Alert
      1: Mild Drowsiness
      2: Severe Drowsiness
    """
    cnn_backbone = build_custom_cnn_backbone(input_shape=frame_size)

    # Sequential Video Frame Input: (batch, sequence_length, height, width, channels)
    sequence_input = layers.Input(shape=(sequence_length, *frame_size), name='video_sequence_input')

    # Apply CNN across each temporal frame slice
    spatial_features = layers.TimeDistributed(cnn_backbone, name='timedistributed_cnn')(sequence_input)

    # Temporal Sequence Modeling with Bidirectional LSTM
    # Captures both past context (eye closing) and future transitions (recovery vs micro-sleep)
    lstm_out = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        name='bilstm_layer_1'
    )(spatial_features)

    lstm_out = layers.Bidirectional(
        layers.LSTM(32, return_sequences=False, dropout=0.3),
        name='bilstm_layer_2'
    )(lstm_out)

    # Dense Classification Head
    dense = layers.Dense(64, activation='relu')(lstm_out)
    dense = layers.BatchNormalization()(dense)
    dense = layers.Dropout(0.4)(dense)

    output = layers.Dense(num_classes, activation='softmax', name='drowsiness_state_output')(dense)

    model = models.Model(inputs=sequence_input, outputs=output, name='Driver_Drowsiness_CNN_LSTM')
    return model

def compile_model(model, learning_rate=1e-4):
    """Compiles the model with Adam optimizer, Categorical Crossentropy, and Accuracy metrics"""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    return model

if __name__ == '__main__':
    model = build_cnn_lstm_model()
    model.summary()
