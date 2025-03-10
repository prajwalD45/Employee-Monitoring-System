import time
import os
import shutil
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Configure folders
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
CUSTOM_MODELS_FOLDER = 'custom_models'

for folder in [UPLOAD_FOLDER, MODELS_FOLDER, CUSTOM_MODELS_FOLDER, 'templates']:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
webcam = None
current_detections = []
selected_video_path = None
processing_active = False

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
MODEL_ALLOWED_EXTENSIONS = {'pt'}

def allowed_file(filename, allowed_ext):
    """Check if uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_ext

def handle_model_selection(request):
    """Handle model selection and validation."""
    try:
        model_name = request.form.get('model')
        custom_model = request.files.get('custom_model')

        if custom_model and custom_model.filename:
            if not allowed_file(custom_model.filename, MODEL_ALLOWED_EXTENSIONS):
                return jsonify({'success': False, 'error': 'Invalid model file type'}), 400
            
            model_filename = secure_filename(custom_model.filename)
            model_path = os.path.join(CUSTOM_MODELS_FOLDER, model_filename)
            custom_model.save(model_path)
        elif model_name:
            model_path = os.path.join(MODELS_FOLDER, model_name)
        else:
            return jsonify({'success': False, 'error': 'No model specified'}), 400

        if not os.path.exists(model_path):
            return jsonify({'success': False, 'error': 'Model file not found'}), 400

        return model_path

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Routes for different pages
@app.route('/')
def index():
    """Redirect to train page as home."""
    return redirect(url_for('train_page'))

@app.route('/train-page')
def train_page():
    """Page for training YOLO model."""
    return render_template('train.html')

@app.route('/image-predict')
def image_predict():
    """Page for image prediction."""
    models = [f for f in os.listdir(MODELS_FOLDER) if f.endswith('.pt')]
    return render_template('image_predict.html', models=models)

@app.route('/video-predict')
def video_predict():
    """Page for video prediction."""
    models = [f for f in os.listdir(MODELS_FOLDER) if f.endswith('.pt')]
    return render_template('video_predict.html', models=models)

@app.route('/webcam-detect')
def webcam_detect():
    """Page for webcam detection."""
    models = [f for f in os.listdir(MODELS_FOLDER) if f.endswith('.pt')]
    return render_template('webcam_detect.html', models=models)

@app.route('/train', methods=['POST'])
def train():
    """Train a YOLOv8 model with user-provided YAML file."""
    try:
        if 'yaml_file' not in request.files:
            return jsonify({'success': False, 'error': 'No YAML file provided'}), 400

        yaml_file = request.files['yaml_file']
        if yaml_file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400

        epochs = int(request.form.get('epochs', 10))
        batch_size = int(request.form.get('batch_size', 16))
        imgsz = int(request.form.get('imgsz', 640))

        yaml_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(yaml_file.filename))
        yaml_file.save(yaml_path)

        model = YOLO('yolov8n.pt')
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            project='runs/detect',
            name='weightsTraining'
        )

        best_model_path = 'runs/detect/weightsTraining/weights/best.pt'
        last_model_path = 'runs/detect/weightsTraining/weights/last.pt'

        if os.path.exists(best_model_path):
            model_path = best_model_path
        elif os.path.exists(last_model_path):
            model_path = last_model_path
        else:
            return jsonify({'success': False, 'error': 'No model file found'}), 500

        new_model_name = f'model_e{epochs}_b{batch_size}_{int(time.time())}.pt'
        os.makedirs(MODELS_FOLDER, exist_ok=True)
        shutil.copy2(model_path, os.path.join(MODELS_FOLDER, new_model_name))

        return jsonify({'success': True, 'model_name': new_model_name})

    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict-image', methods=['POST'])
def predict_image():
    """Handle image prediction requests."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400

        if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400

        model_path = handle_model_selection(request)
        if isinstance(model_path, tuple):
            return model_path

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Failed to save image'}), 500

        return detect_image(filepath, model_path)

    except Exception as e:
        logging.error(f"Image prediction error: {str(e)}")
        return jsonify({'success': False, 'error': f'Prediction error: {str(e)}'}), 500

def detect_image(filepath, model_path):
    """Perform detection on an image."""
    try:
        model = YOLO(model_path)
        results = model(filepath)

        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    'class': result.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()
                })

        output_filename = f"output_{os.path.basename(filepath)}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        for result in results:
            img = result.plot()
            cv2.imwrite(output_path, img)

        return jsonify({
            'success': True,
            'image_url': f'/uploads/{output_filename}',
            'detections': detections
        })

    except Exception as e:
        logging.error(f"Image detection error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict-video', methods=['POST'])
def predict_video():
    """Handle video prediction requests."""
    global selected_video_path
    try:
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400

        if not allowed_file(file.filename, {'mp4', 'avi', 'mov'}):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400

        model_path = handle_model_selection(request)
        if isinstance(model_path, tuple):
            return model_path

        filename = secure_filename(file.filename)
        selected_video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(selected_video_path)
        
        return jsonify({'success': True})

    except Exception as e:
        logging.error(f"Video prediction error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    """Stream video frames with detections."""
    global selected_video_path, processing_active
    model_name = request.args.get('model', 'best.pt')
    return Response(
        generate_video_frames(model_name),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

def generate_video_frames(model_name):
    """Generate video frames with detections."""
    global selected_video_path, processing_active, current_detections
    
    if not selected_video_path or not os.path.exists(selected_video_path):
        return
        
    try:
        model_path = os.path.join(app.config['MODELS_FOLDER'], model_name)
        model = YOLO(model_path)
        cap = cv2.VideoCapture(selected_video_path)
        
        while cap.isOpened() and processing_active:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                continue
                
            results = model(frame)
            current_detections = []
            for result in results:
                for box in result.boxes:
                    current_detections.append({
                        'class': result.names[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist()
                    })
                    
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
    except Exception as e:
        logging.error(f"Video frame generation error: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()

@app.route('/start_processing', methods=['POST'])
def start_processing():
    global processing_active
    processing_active = True
    return jsonify({'success': True})

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    global processing_active
    processing_active = False
    return jsonify({'success': True})

@app.route('/predict-webcam', methods=['POST'])
def predict_webcam():
    """Handle webcam detection requests."""
    global webcam
    try:
        model_path = handle_model_selection(request)
        if isinstance(model_path, tuple):
            return model_path
            
        if webcam is None or not webcam.isOpened():
            webcam = cv2.VideoCapture(0)
            if not webcam.isOpened():
                return jsonify({'success': False, 'error': 'Could not start webcam'}), 500
            webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            webcam.set(cv2.CAP_PROP_FPS, 30)
        
        return Response(
            detect_webcam(model_path),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
        
    except Exception as e:
        logging.error(f"Webcam prediction error: {str(e)}")
        if webcam:
            webcam.release()
            webcam = None
        return jsonify({'success': False, 'error': str(e)}), 500

def detect_webcam(model_path):
    """Perform real-time detection using the webcam."""
    global webcam, current_detections
    try:
        model = YOLO(model_path)
        
        while webcam.isOpened():
            success, frame = webcam.read()
            if not success:
                time.sleep(0.1)
                continue
                
            results = model(frame, conf=0.5)
            current_detections = []
            for result in results:
                for box in result.boxes:
                    current_detections.append({
                        'class': result.names[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist()
                    })
                    
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
    except Exception as e:
        logging.error(f"Webcam detection error: {str(e)}")
        if webcam:
            webcam.release()
            webcam = None

@app.route('/stop-webcam', methods=['POST'])
def stop_webcam():
    """Stop the webcam stream."""
    global webcam, current_detections
    try:
        if webcam:
            webcam.release()
            webcam = None
        current_detections = []
        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Stop webcam error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get-detection-results', methods=['GET'])
def get_detection_results():
    """Get current detection results from webcam or video stream."""
    global current_detections
    try:
        return jsonify({
            'success': True,
            'detections': current_detections
        })
    except Exception as e:
        logging.error(f"Get detection results error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded and processed files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/test-webcam')
def test_webcam():
    """Basic webcam test page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Webcam Test</title>
    </head>
    <body>
        <h2>Basic Webcam Test</h2>
        <button id="startButton">Start Webcam</button>
        <button id="stopButton" style="display:none;">Stop Webcam</button>
        <br><br>
        <video id="videoElement" width="640" height="480" style="border: 1px solid black;" autoplay></video>

        <script>
            const videoElement = document.getElementById('videoElement');
            const startButton = document.getElementById('startButton');
            const stopButton = document.getElementById('stopButton');
            let stream = null;

            startButton.onclick = async function() {
                try {
                    stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { 
                            width: 640,
                            height: 480
                        }
                    });
                    videoElement.srcObject = stream;
                    startButton.style.display = 'none';
                    stopButton.style.display = 'block';
                } catch(err) {
                    console.error("Error: " + err);
                    alert("Error accessing webcam: " + err.message);
                }
            };

            stopButton.onclick = function() {
                if(stream) {
                    stream.getTracks().forEach(track => track.stop());
                    videoElement.srcObject = null;
                    startButton.style.display = 'block';
                    stopButton.style.display = 'none';
                }
            };
        </script>
    </body>
    </html>
    """

@app.teardown_appcontext
def cleanup(exception=None):
    """Clean up webcam on application shutdown."""
    global webcam
    if webcam:
        webcam.release()
        webcam = None

if __name__ == '__main__':
    app.run(debug=True)