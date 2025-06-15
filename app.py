import os
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this for production

# Configuration
app.config.update({
    'UPLOAD_FOLDER': 'static/uploads',
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg'},
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB limit
    'MODEL_PATH': 'models/final_model.h5',
    'IMAGE_SIZE': (380, 380)
})

# Load model with error handling
try:
    model = load_model(app.config['MODEL_PATH'])
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {str(e)}")
    raise RuntimeError("Could not load the AI model")

# Class names (replace with your actual classes)
CLASS_NAMES = [
    'Normal', 'Diabetic Retinopathy - Mild', 'Diabetic Retinopathy - Moderate',
    'Diabetic Retinopathy - Severe', 'Diabetic Retinopathy - Proliferative',
    'Glaucoma - Early', 'Glaucoma - Moderate', 'Glaucoma - Severe',
    # ... add all 39 classes
]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_image(filepath):
    try:
        img = Image.open(filepath)
        img.verify()
        img.close()
        return True
    except (UnidentifiedImageError, IOError):
        return False

def preprocess_image(image_path):
    try:
        img = image.load_img(image_path, target_size=app.config['IMAGE_SIZE'])
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")

def generate_grad_cam(img_path, model, pred_index):
    try:
        # Load image
        img = image.load_img(img_path, target_size=app.config['IMAGE_SIZE'])
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Find last conv layer
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                last_conv_layer = layer
                break
        else:
            raise ValueError("No convolutional layer found")

        # Create gradient model
        grad_model = tf.keras.models.Model(
            [model.inputs], [last_conv_layer.output, model.output]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Generate heatmap
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        # Process visualization
        heatmap = cv2.resize(heatmap, (img.width, img.height))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(
            cv2.cvtColor(np.uint8(img_array[0]*255), cv2.COLOR_RGB2BGR), 
            0.6, 
            heatmap, 
            0.4, 
            0
        )
        return superimposed_img
    except Exception as e:
        raise RuntimeError(f"Grad-CAM failed: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('⚠️ No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        
        # Check for empty filename
        if file.filename == '':
            flash('⚠️ No file selected')
            return redirect(request.url)
        
        # Validate file type
        if not allowed_file(file.filename):
            flash('❌ Invalid file type. Only PNG, JPG, JPEG allowed.')
            return redirect(request.url)
        
        try:
            # Save file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Validate image
            if not validate_image(filepath):
                os.remove(filepath)
                flash('❌ Invalid image file')
                return redirect(request.url)
            
            # Process and predict
            processed_image = preprocess_image(filepath)
            predictions = model.predict(processed_image)
            
            # Handle predictions
            if predictions.shape[1] == 1:  # Binary
                predicted_class = int(predictions[0][0] > 0.5)
                confidence = predictions[0][0] if predicted_class else 1 - predictions[0][0]
            else:  # Multi-class
                predicted_class = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
            
            result = CLASS_NAMES[predicted_class]
            confidence_percent = f"{confidence * 100:.2f}%"
            
            # Generate visualization
            try:
                grad_cam_img = generate_grad_cam(filepath, model, predicted_class)
                grad_cam_filename = f"gradcam_{filename}"
                grad_cam_path = os.path.join(app.config['UPLOAD_FOLDER'], grad_cam_filename)
                cv2.imwrite(grad_cam_path, grad_cam_img)
            except Exception as e:
                print(f"Visualization error: {str(e)}")
                grad_cam_filename = None
            
            return render_template('result.html',
                                filename=filename,
                                prediction=result,
                                confidence=confidence_percent,
                                gradcam_image=grad_cam_filename)
            
        except Exception as e:
            flash(f'❌ Error: {str(e)}')
            return redirect(request.url)
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0')