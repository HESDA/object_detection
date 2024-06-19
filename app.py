from flask import Flask, request, jsonify, render_template, Response
from ultralytics import YOLO
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Muat model
model = YOLO('models/best.pt')

# Direktori untuk menyimpan file yang diunggah
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def process_frame(frame):
    # Konversi frame dari BGR (OpenCV) ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Deteksi objek
    results = model(rgb_frame)
    
    # Gambar bounding boxes pada frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Konversi tensor ke list kemudian int
            label = int(box.cls)  # Konversi tensor ke int
            confidence = float(box.conf)  # Konversi tensor ke float
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        video = cv2.VideoCapture(filepath)
        frames = []

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            processed_frame = process_frame(frame)
            frames.append(processed_frame)
        
        video.release()
        
        height, width, layers = frames[0].shape
        video_output_path = os.path.join(UPLOAD_FOLDER, 'output.mp4')
        video_output = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
        
        for frame in frames:
            video_output.write(frame)
        
        video_output.release()

        return jsonify({'result': 'Video processed', 'video_path': video_output_path})

@app.route('/video_feed')
def video_feed():
    def generate():
        video_capture = cv2.VideoCapture(0)  # Menggunakan kamera laptop

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            processed_frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
