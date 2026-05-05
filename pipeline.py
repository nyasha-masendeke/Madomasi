import tensorflow as tf
from pathlib import Path
import numpy as np
from PIL import Image

def train_model(base_model="mobilenetv3small", epochs=10, batch_size=16, 
                lr=0.001, freeze_base=False, output_path="models/best.keras", 
                callbacks=None):
    """Core training logic (replaces CLI train() function)"""
    # 1. Build/Load model
    if base_model == "mobilenetv3small":
        base = tf.keras.applications.MobileNetV3Small(input_shape=(224, 224, 3), 
                                                       include_top=False, 
                                                       weights="imagenet")
    base.trainable = not freeze_base
    
    inputs = tf.keras.Input((224, 224, 3))
    x = base(inputs, training=not freeze_base)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)  # Adjust classes as needed
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    # 2. Load dataset (replace with your actual data pipeline)
    # train_ds, val_ds = load_datasets(batch_size)
    
    # 3. Train
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        # train_ds, 
        epochs=epochs, 
        validation_data=None,  # Replace with val_ds
        callbacks=callbacks
    )
    
    model.save(output_path)
    return model

def predict_image(model_path, image_data, confidence_threshold=0.5):
    """Core inference logic (replaces CLI predict() function)"""
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess uploaded image
    img = Image.open(image_data).resize((224, 224)).convert("RGB")
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    
    # Predict
    preds = model.predict(arr)[0]
    class_idx = np.argmax(preds)
    confidence = float(preds[class_idx])
    
    # Map index to disease name (adjust to your config)
    from config import DISEASE_CLASSES
    disease_name = DISEASE_CLASSES[class_idx] if class_idx < len(DISEASE_CLASSES) else "Unknown"
    
    return {
        "disease": disease_name,
        "confidence": confidence,
        "passes_threshold": confidence >= confidence_threshold,
        "all_probs": {DISEASE_CLASSES[i]: float(p) for i, p in enumerate(preds)}
    }
