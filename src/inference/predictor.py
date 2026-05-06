"""Model inference for Keras and TFLite."""
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf

def load_model(path: str):
    """Load Keras or TFLite model."""
    path = Path(path)
    
    if path.suffix == '.h5' or path.suffix == '.keras':
        # Load Keras model
        return tf.keras.models.load_model(path)
    elif path.suffix == '.tflite':
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=str(path))
        interpreter.allocate_tensors()
        return interpreter
    else:
        raise ValueError(f"Unsupported model format: {path.suffix}. Use .h5, .keras, or .tflite")

def predict(model, image_path: str, class_names: list):
    """Run inference and return predictions."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    # Resize to standard size (adjust based on your model's expected input)
    image = image.resize((224, 224))
    
    # Convert to array and normalize
    input_array = np.array(image, dtype=np.float32)
    input_array = input_array / 255.0  # Normalize to [0, 1]
    
    # Add batch dimension
    input_array = np.expand_dims(input_array, axis=0)
    
    # Check if it's TFLite interpreter or Keras model
    if isinstance(model, tf.lite.Interpreter):
        # Get input and output details
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        # Set input tensor
        model.set_tensor(input_details[0]['index'], input_array)
        
        # Run inference
        model.invoke()
        
        # Get output
        predictions = model.get_tensor(output_details[0]['index'])
    else:
        # Keras model
        predictions = model.predict(input_array, verbose=0)
    
    # Postprocess: get top predictions
    top_indices = np.argsort(predictions[0])[::-1][:5]  # Top 5
    results = []
    
    for idx in top_indices:
        results.append({
            'class': class_names[idx] if idx < len(class_names) else f"Class_{idx}",
            'confidence': float(predictions[0][idx])
        })
    
    return results
